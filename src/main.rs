use std::fs::File;
use std::io;
use std::io::Write;
use std::io::{stdout, BufReader, BufWriter, IsTerminal, Read};
use std::path::{Path, PathBuf};
use std::process::exit;

use clap::{arg, Args, Parser, Subcommand, ValueEnum};
use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use image::{ColorType, EncodableLayout, GenericImageView, ImageEncoder, Rgba};
use itertools::Itertools;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use thiserror::Error;

const MAGIC: u16 = 0xd2d;
const VERSION_1: u8 = 1;
const VERSION_2: u8 = 2;

type Seed = String;

#[derive(Parser)]
#[command(author, version, about, long_about)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Injects a file into an image. If the image is not PNG/RGBA8, it will be converted
    Inject(InjectArgs),

    /// Extracts a file from an image
    Extract(ExtractArgs),

    /// Inspects an image if it has a file inside and prints the results.
    /// Also tells how large a file can be injected inside
    Inspect(InspectArgs),
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Compression {
    Default,
    Fast,
    Best,
}

impl From<Compression> for CompressionType {
    fn from(val: Compression) -> Self {
        match val {
            Compression::Default => CompressionType::Default,
            Compression::Fast => CompressionType::Fast,
            Compression::Best => CompressionType::Best,
        }
    }
}

#[derive(Args)]
struct InjectArgs {
    /// The file to inject
    cargo: PathBuf,

    /// The image (container)
    container: PathBuf,

    /// Destination, where the injected file is placed
    #[arg(short, long)]
    destination: Option<PathBuf>,

    /// Whether to write metadata. If not set, extracting would require --read-meta=false
    /// and the exact file size in bytes (--read-size)
    #[arg(short, long, default_value_t = true, action = clap::ArgAction::Set)]
    write_meta: bool,

    /// Compression level used to compress PNG
    #[arg(value_enum, long, default_value_t = Compression::Default)]
    compression: Compression,

    /// Use seed to place bits in pseudorandom pixel positions.
    /// The same seed must be provided during extraction or inspection
    /// to correctly recover the data.
    #[arg(long)]
    seed: Option<Seed>,
}

#[derive(Args)]
struct ExtractArgs {
    /// Container that contains a file
    container: PathBuf,

    /// Where to save the extracted file. If not set, the filename will be read
    /// from metadata (if any). If none are set, defaults to "cargo"
    #[arg(short, long)]
    destination: Option<PathBuf>,

    /// Whether to read metadata. If metadata was not written and --read-meta=true,
    /// extraction will fail. If metadata was written and --read-meta=false,
    /// the extracted file will be broken
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    read_meta: bool,

    /// How many bytes of the cargo file to read. Defaults to the value in metadata.
    /// If none, defaults to the maximum (until the container ends)
    #[arg(long)]
    read_size: Option<u32>,

    /// Seed used to pseudorandomly locate embedded data.
    /// Must match the seed used during injection, if any.
    #[arg(long)]
    seed: Option<Seed>,
}

#[derive(Args)]
struct InspectArgs {
    /// Container file
    path: PathBuf,

    /// Seed used to pseudorandomly locate embedded data.
    /// Must match the seed used during injection, if any.
    #[arg(long)]
    seed: Option<Seed>,
}

const KB: u32 = 1024;
const MB: u32 = 1024 * 1024;
const MB_MINUS_1: u32 = MB - 1;
const GB: u32 = MB * 1024;
const GB_MINUS_1: u32 = GB - 1;

#[inline]
fn format_size(size: u32) -> String {
    match size {
        (GB..=u32::MAX) => format!("{:.2} GB", (size as f32) / (GB as f32)),
        (MB..=GB_MINUS_1) => format!("{:.2} MB", (size as f32) / (MB as f32)),
        (KB..=MB_MINUS_1) => format!("{:.2} KB", (size as f32) / (KB as f32)),
        _ => format!("{size} bytes"),
    }
}

#[inline]
fn to_bits(val: u8) -> [u8; 8] {
    [
        (val >> 7) & 1,
        (val >> 6) & 1,
        (val >> 5) & 1,
        (val >> 4) & 1,
        (val >> 3) & 1,
        (val >> 2) & 1,
        (val >> 1) & 1,
        val & 1,
    ]
}

#[inline]
fn iter_dots(w: u32, h: u32) -> impl Iterator<Item = (u32, u32)> {
    (0..w).cartesian_product(0..h)
}

fn seed_to_u64(seed: &str) -> u64 {
    let hash = blake3::hash(seed.as_bytes());
    u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap())
}

fn pseudo_shuffle_coords(w: u32, h: u32, seed: &Seed) -> impl Iterator<Item = (u32, u32)> {
    let mut coords: Vec<(u32, u32)> = iter_dots(w, h).collect();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed_to_u64(seed));
    coords.shuffle(&mut rng);
    coords.into_iter()
}

fn gen_dots(w: u32, h: u32, seed: Option<&Seed>) -> Box<dyn Iterator<Item = (u32, u32)>> {
    match seed {
        Some(seed) => Box::new(pseudo_shuffle_coords(w, h, seed)),
        None => Box::new(iter_dots(w, h)),
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct Meta {
    pub version: u8,
    pub size: u32,
    pub filename: Option<String>,
    pub hash: Option<u32>,
}

impl Meta {
    pub fn to_bytes(&self) -> Vec<u8> {
        // 2 bytes signature + version
        // TLV for version 2
        let signature = (MAGIC << 3) | (VERSION_2 as u16);
        let mut result = Vec::with_capacity(32);
        result.extend(signature.to_le_bytes());
        // TLV: size (tag=1, len=4)
        result.push(u8::from(MetaTag::Size));
        result.push(4); // length
        result.extend(self.size.to_le_bytes());
        // TLV: filename (tag=2, len=N)
        if let Some(name) = &self.filename {
            let name_bytes = name.as_bytes();
            if name_bytes.len() > 255 {
                result.push(u8::from(MetaTag::Filename));
                result.push(0x00); // extended length marker
                let len = name_bytes.len() as u16;
                result.extend(len.to_le_bytes());
                result.extend(name_bytes);
            } else {
                result.push(u8::from(MetaTag::Filename));
                result.push(name_bytes.len() as u8);
                result.extend(name_bytes);
            }
        }
        // TLV: hash (tag=3, len=4)
        if let Some(hash) = self.hash {
            result.push(u8::from(MetaTag::Hash));
            result.push(4);
            result.extend(hash.to_le_bytes());
        }
        // TLV: end marker (tag=0, len=0)
        result.push(0);
        result.push(0);
        result
    }

    pub fn to_bits(&self) -> impl Iterator<Item = u8> {
        self.to_bytes().into_iter().flat_map(to_bits)
    }

    pub fn make(size: u32, filename: Option<String>, hash: Option<u32>) -> Self {
        Self {
            version: VERSION_2,
            size,
            filename,
            hash,
        }
    }

    pub fn read<T>(value: &mut T) -> Result<Self, MetaError>
    where
        T: Iterator<Item = u8>,
    {
        // Read only 2 bytes for signature
        let sig_bytes: Vec<u8> = value.take(2).collect();
        if sig_bytes.len() != 2 {
            return Err(MetaError::NoBytes);
        }
        let signature = u16::from_le_bytes([sig_bytes[0], sig_bytes[1]]);
        let sign = signature >> 3;
        if sign != MAGIC {
            return Err(MetaError::SignatureMismatch);
        }
        let version = (signature & 0b111) as u8;
        match version {
            VERSION_1 => {
                // Read 5 more bytes for old format
                let header_rest: Vec<u8> = value.take(5).collect();
                if header_rest.len() != 5 {
                    return Err(MetaError::NoBytes);
                }
                let mut header = sig_bytes;
                header.extend(header_rest);
                let size = u32::from_le_bytes(header[2..6].try_into().unwrap());
                let filename_size = match header[6] {
                    0b11111111 => None,
                    sz => Some(sz),
                };
                let mut filename = None;
                if let Some(sz) = filename_size {
                    let filename_vec = value.take(sz as usize).collect_vec();
                    if filename_vec.len() as u8 != sz {
                        return Err(MetaError::MalformedFilename);
                    }
                    let filename_lossy = String::from_utf8_lossy(&filename_vec);
                    filename = Some(filename_lossy.to_string());
                }
                Ok(Meta {
                    version,
                    size,
                    filename,
                    hash: None,
                })
            }
            VERSION_2 => {
                // TLV format, value is at TLV start
                let mut size = None;
                let mut filename = None;
                let mut hash = None;
                loop {
                    let tag = value.next().ok_or(MetaError::NoBytes)?;
                    let len = value.next().ok_or(MetaError::NoBytes)?;
                    if tag == 0 && len == 0 {
                        break;
                    }
                    let actual_len = if len == 0x00 {
                        u16::from_le_bytes(read_vec(2, value)?.try_into().unwrap()) as usize
                    } else {
                        len as usize
                    };
                    match MetaTag::try_from(tag) {
                        Ok(MetaTag::Size) if actual_len == 4 => {
                            size =
                                Some(u32::from_le_bytes(read_vec(4, value)?.try_into().unwrap()));
                        }
                        Ok(MetaTag::Filename) => {
                            filename = Some(
                                String::from_utf8_lossy(&read_vec(actual_len, value)?).to_string(),
                            );
                        }
                        Ok(MetaTag::Hash) if actual_len == 4 => {
                            hash =
                                Some(u32::from_le_bytes(read_vec(4, value)?.try_into().unwrap()));
                        }
                        _ => {
                            let _ = read_vec(actual_len, value)?;
                        }
                    }
                }
                let size = size.ok_or(MetaError::NoBytes)?;
                Ok(Meta {
                    version,
                    size,
                    filename,
                    hash,
                })
            }
            v => Err(MetaError::UnsupportedVersion(v)),
        }
    }
}

#[derive(Debug, Error)]
pub enum MetaError {
    #[error("Insufficient bytes to parse metadata")]
    NoBytes,
    #[error("Invalid metadata signature")]
    SignatureMismatch,
    #[error("Unsupported metadata version: {0}")]
    UnsupportedVersion(u8),
    #[error("Invalid or corrupted filename in metadata")]
    MalformedFilename,
}

#[derive(Debug, Error)]
enum InspectError {
    #[error("File not found")]
    FileNotExist,
    #[error("Not a valid image file")]
    NotAnImage,
    #[error("Not a regular file")]
    NotAFile,
}

#[derive(Debug, Error)]
enum ExtractError {
    #[error("Failed to open container file")]
    ContainerOpen,
    #[error("Failed to save output file")]
    Save,
    #[error("Invalid metadata: {0}")]
    BrokenMeta(String),
    #[error("Failed to verify hash")]
    HashMismatch,
}

#[derive(Debug, Error)]
enum InjectError {
    #[error("Failed to open container file")]
    CannotOpenContainer,
    #[error("Failed to open input file")]
    CannotOpenCargo,
    #[error("File size exceeds container capacity: available {available}, file {cargo_size}, metadata {meta_size}")]
    ExceededSize {
        available: u32,
        cargo_size: u32,
        meta_size: u32,
    },
    #[error("Failed to save output file: {0}")]
    CannotSave(String),
    #[error("Filename is too long (maximum 255 bytes)")]
    FilenameOverflow,
}

macro_rules! meta_tag_enum {
    ($( $name:ident = $val:expr ),* $(,)?) => {
        #[repr(u8)]
        #[derive(Debug, Copy, Clone, PartialEq, Eq)]
        enum MetaTag {
            $( $name = $val, )*
        }
        impl From<MetaTag> for u8 {
            fn from(tag: MetaTag) -> Self {
                tag as u8
            }
        }
        impl std::convert::TryFrom<u8> for MetaTag {
            type Error = ();
            fn try_from(value: u8) -> Result<Self, Self::Error> {
                match value {
                    $( $val => Ok(MetaTag::$name), )*
                    _ => Err(()),
                }
            }
        }
    };
}

meta_tag_enum! {
    Size = 1,
    Filename = 2,
    Hash = 3,
}

fn inspect(args: InspectArgs) -> Result<(), InspectError> {
    if !args.path.exists() {
        return Err(InspectError::FileNotExist);
    }

    if !args.path.is_file() {
        return Err(InspectError::NotAFile);
    }

    let filename = args.path.file_name().unwrap().to_string_lossy().to_string();
    let img = image::open(&args.path).map_err(|_| InspectError::NotAnImage)?;
    let (w, h) = img.dimensions();
    let max_cargo_size = format_size((w * h * 4) / 8);
    let bytes = gen_dots(w, h, args.seed.as_ref())
        .flat_map(|(x, y)| img.get_pixel(x, y).0)
        .map(|v| v & 1)
        .chunks(8);
    let mut content = bytes.into_iter().map(|chunk| {
        chunk
            .zip((0..8).rev())
            .map(|(bit, shift)| bit << shift)
            .sum()
    });
    let meta = Meta::read(&mut content).ok();
    println!("Image file: {filename}");
    println!("Dimensions: {w}x{h}");
    println!("Maximum embeddable file size: {max_cargo_size}");
    match meta {
        None => println!("No embedded data detected or metadata is missing."),
        Some(ref v) => {
            println!("Metadata version: {}", v.version);
            let cargo_filename = v
                .filename
                .clone()
                .unwrap_or_else(|| String::from("<unnamed>"));
            let cargo_size = format_size(v.size);
            println!("Embedded file name: {cargo_filename}");
            println!("Embedded file size: {cargo_size}");
            if let Some(hash) = v.hash {
                println!("Embedded file CRC32: {hash:08x}");
            }
        }
    }
    Ok(())
}

fn make_writer(dest: Option<&Path>, default: impl AsRef<Path>) -> Result<Box<dyn Write>, String> {
    let writer = if !stdout().is_terminal() && dest.is_none() {
        Box::new(stdout()) as Box<dyn Write>
    } else {
        let dest = dest.unwrap_or(default.as_ref());
        let write_file = BufWriter::new(File::create(dest).map_err(|e| e.to_string())?);
        Box::new(write_file) as Box<dyn Write>
    };
    Ok(writer)
}

fn extract(args: ExtractArgs) -> Result<(), ExtractError> {
    let img = image::open(&args.container).map_err(|_| ExtractError::ContainerOpen)?;
    let (w, h) = img.dimensions();
    let bytes = gen_dots(w, h, args.seed.as_ref())
        .flat_map(|(x, y)| img.get_pixel(x, y).0)
        .map(|v| v & 1)
        .chunks(8);
    let mut content = bytes.into_iter().map(|chunk| {
        chunk
            .zip((0..8).rev())
            .map(|(bit, shift)| bit << shift)
            .sum()
    });
    let meta = if args.read_meta {
        Some(Meta::read(&mut content).map_err(|e| ExtractError::BrokenMeta(e.to_string()))?)
    } else {
        None
    };

    let (meta_filename, size) = if let Some(Meta { filename, size, .. }) = &meta {
        (filename.as_ref().map(PathBuf::from), Some(*size))
    } else {
        (None, None)
    };

    let read_size = args.read_size.or(size).unwrap_or(u32::MAX);
    let mut writer = make_writer(
        args.destination.as_deref(),
        meta_filename.unwrap_or(PathBuf::from("cargo")),
    )
    .map_err(|_| ExtractError::Save)?;
    let mut crc = crc32fast::Hasher::new();
    let mut buffer = [0u8; 8192];
    let mut remaining = read_size as usize;
    while remaining > 0 {
        let to_read = buffer.len().min(remaining);
        let mut filled = 0;
        while filled < to_read {
            match content.next() {
                Some(b) => {
                    buffer[filled] = b;
                    filled += 1;
                }
                None => break,
            }
        }
        if filled == 0 {
            break;
        }
        writer
            .write_all(&buffer[..filled])
            .map_err(|_| ExtractError::Save)?;
        crc.update(&buffer[..filled]);
        remaining -= filled;
    }
    writer.flush().map_err(|_| ExtractError::Save)?;
    if let Some(meta) = &meta {
        if let Some(expected_hash) = meta.hash {
            let calculated_hash = crc.finalize();
            if calculated_hash != expected_hash {
                return Err(ExtractError::HashMismatch);
            }
        }
    }
    Ok(())
}

fn inject(args: InjectArgs) -> Result<(), InjectError> {
    let mut img = image::open(&args.container)
        .map_err(|_| InjectError::CannotOpenContainer)?
        .into_rgba8();

    let cargo = File::open(&args.cargo).map_err(|_| InjectError::CannotOpenCargo)?;
    let (w, h) = img.dimensions();
    let max_cargo_size = (w * h * 4) / 8;
    let cargo_meta = cargo.metadata().map_err(|_| InjectError::CannotOpenCargo)?;
    let cargo_size = cargo_meta.len() as u32;
    let mut meta_bits = vec![];

    let hash;
    if args.write_meta {
        let filename = args
            .cargo
            .file_name()
            .map(|v| String::from(v.to_string_lossy()));
        if let Some(v) = &filename {
            if v.len() >= 255 {
                return Err(InjectError::FilenameOverflow);
            }
        }
        // Calculate crc32 on a separate file descriptor
        let mut hasher = crc32fast::Hasher::new();
        let mut crc_file = File::open(&args.cargo).map_err(|_| InjectError::CannotOpenCargo)?;
        let mut buf = [0u8; 8192];
        loop {
            let n = std::io::Read::read(&mut crc_file, &mut buf)
                .map_err(|_| InjectError::CannotOpenCargo)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        hash = Some(hasher.finalize());
        let meta = Meta::make(cargo_size, filename, hash);
        meta_bits.extend(meta.to_bits());
    }

    let meta_size = (meta_bits.len() / 8) as u32;
    let total_size = cargo_size + meta_size;
    if total_size > max_cargo_size {
        return Err(InjectError::ExceededSize {
            available: max_cargo_size,
            cargo_size,
            meta_size,
        });
    }

    let cargo_bits = BufReader::new(cargo)
        .bytes()
        .flat_map(|x| to_bits(x.unwrap()));
    let bits = meta_bits.into_iter().chain(cargo_bits);
    let color_coords = gen_dots(w, h, args.seed.as_ref());
    let mut bit_iter = bits;
    // Iterate over all coordinates, modifying only the required number of pixels
    let mut changed = 0;
    let total_bits = total_size * 8;
    for (x, y) in color_coords {
        if changed >= total_bits {
            break;
        }
        let mut px = img.get_pixel(x, y).0;
        for channel in &mut px {
            if changed >= total_bits {
                break;
            }
            if let Some(bit) = bit_iter.next() {
                *channel = (*channel & 0b11111110) | bit;
                changed += 1;
            }
        }
        img.put_pixel(x, y, Rgba(px));
    }
    let writer = make_writer(args.destination.as_deref(), "modified.png")
        .map_err(InjectError::CannotSave)?;
    let encoder =
        PngEncoder::new_with_quality(writer, args.compression.into(), FilterType::default());
    encoder
        .write_image(img.as_bytes(), w, h, ColorType::Rgba8.into())
        .map_err(|e| InjectError::CannotSave(e.to_string()))
}

fn main() -> io::Result<()> {
    let cli = Cli::parse();
    if let Err(e) = match cli.command {
        Commands::Inject(args) => inject(args).map_err(|e| e.to_string()),
        Commands::Extract(args) => extract(args).map_err(|e| e.to_string()),
        Commands::Inspect(args) => inspect(args).map_err(|e| e.to_string()),
    } {
        eprintln!("{e}");
        exit(1);
    }
    Ok(())
}

fn read_vec<I>(len: usize, iter: &mut I) -> Result<Vec<u8>, MetaError>
where
    I: Iterator<Item = u8>,
{
    let mut v = Vec::with_capacity(len);
    for _ in 0..len {
        v.push(iter.next().ok_or(MetaError::NoBytes)?);
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use std::assert_eq;

    use super::{MAGIC, VERSION_1};
    use crate::Meta;

    #[test]
    fn test_meta_v2_roundtrip() {
        // Test version 2 (TLV): serialize and parse
        let meta = Meta::make(1231234, Some("hello.zip".to_string()), None);
        let meta_bytes = meta.to_bytes();
        println!("meta v2 bytes: {:02x?}", meta_bytes);
        let mut bytes = meta_bytes.into_iter();
        match Meta::read(&mut bytes) {
            Err(_) => panic!("meta wasn't read (v2)"),
            Ok(v) => assert_eq!(v, meta, "received meta differs (v2)"),
        }
        assert_eq!(bytes.next(), None);
    }

    #[test]
    fn test_meta_v1_parsing() {
        // Test version 1 (legacy): manual bytes, only parse
        let mut v1_bytes = Vec::new();
        let signature = (MAGIC << 3) | (VERSION_1 as u16);
        v1_bytes.extend(signature.to_le_bytes());
        v1_bytes.extend(1231234u32.to_le_bytes());
        let filename = b"hello.zip";
        v1_bytes.push(filename.len() as u8);
        v1_bytes.extend(filename);
        let mut v1_iter = v1_bytes.into_iter();
        let meta_v1 = Meta::read(&mut v1_iter).expect("meta v1 should parse");
        assert_eq!(meta_v1.version, VERSION_1);
        assert_eq!(meta_v1.size, 1231234);
        assert_eq!(meta_v1.filename.as_deref(), Some("hello.zip"));
        assert_eq!(v1_iter.next(), None);
    }
}
