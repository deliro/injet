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
const VERSION: u8 = 1;

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
struct Meta {
    version: u8,
    size: u32,
    filename_size: Option<u8>,
    filename: Option<String>,
}

impl Meta {
    fn to_bytes(&self) -> Vec<u8> {
        // 2 bytes signature + version
        // 4 bytes file size
        // 1 byte filename size (if any)
        // X bytes filename (X < 255)
        let signature = (MAGIC << 3) | (self.version as u16);
        let (filename_size, has_filename): (u8, _) = match &self.filename {
            None => (0b11111111, false),
            Some(v) => (v.len() as u8, true),
        };
        let mut result = Vec::with_capacity(7);
        result.extend(signature.to_le_bytes());
        result.extend(self.size.to_le_bytes());
        result.push(filename_size);
        if has_filename {
            result.extend(self.filename.clone().unwrap().into_bytes())
        }
        result
    }

    fn to_bits(&self) -> impl Iterator<Item = u8> {
        self.to_bytes().into_iter().flat_map(to_bits)
    }

    fn make(size: u32, filename: Option<String>) -> Self {
        let filename_size = filename.as_ref().map(|v| v.len() as u8);
        Self {
            version: VERSION,
            size,
            filename_size,
            filename,
        }
    }
}

// https://github.com/Ixrec/rust-orphan-rules
struct W<T>(T);
impl<T> TryFrom<W<&mut T>> for Meta
where
    T: Iterator<Item = u8>,
{
    type Error = MetaError;

    fn try_from(W(value): W<&mut T>) -> Result<Self, Self::Error> {
        let header = value.take(7).collect_vec();
        if header.len() != 7 {
            return Err(MetaError::NoBytes);
        }
        let signature = u16::from_le_bytes(header[0..2].try_into().unwrap());
        let sign = signature >> 3;
        if sign != MAGIC {
            return Err(MetaError::SignatureMismatch);
        }

        let version = (signature & 0b111) as u8;
        if version != VERSION {
            return Err(MetaError::UnsupportedVersion(version));
        }

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

        Ok(Self {
            version,
            size,
            filename_size,
            filename,
        })
    }
}

#[derive(Debug, Error)]
enum MetaError {
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
    let meta = Meta::try_from(W(&mut content)).ok();
    println!("Image file: {filename}");
    println!("Dimensions: {w}x{h}");
    println!("Maximum embeddable file size: {max_cargo_size}");
    match meta {
        None => println!("No embedded data detected or metadata is missing."),
        Some(v) => {
            let cargo_filename = v.filename.unwrap_or_else(|| String::from("<unnamed>"));
            let cargo_size = format_size(v.size);
            println!("Embedded file name: {cargo_filename}");
            println!("Embedded file size: {cargo_size}");
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
        Some(Meta::try_from(W(&mut content)).map_err(|e| ExtractError::BrokenMeta(e.to_string()))?)
    } else {
        None
    };

    let (meta_filename, size) = if let Some(Meta { filename, size, .. }) = meta {
        (filename.map(PathBuf::from), Some(size))
    } else {
        (None, None)
    };

    let read_size = args.read_size.or(size).unwrap_or(u32::MAX);
    let mut writer = make_writer(
        args.destination.as_deref(),
        meta_filename.unwrap_or(PathBuf::from("cargo")),
    )
    .map_err(|_| ExtractError::Save)?;
    for b in content.take(read_size as usize) {
        writer.write_all(&[b]).map_err(|_| ExtractError::Save)?
    }
    writer.flush().map_err(|_| ExtractError::Save)?;
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
        let meta = Meta::make(cargo_size, filename);
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

#[cfg(test)]
mod tests {
    use std::assert_eq;

    use crate::{Meta, W};

    #[test]
    fn meta_test() {
        let meta = Meta::make(1231234, Some("hello.zip".to_string()));
        let mut bytes = meta.to_bytes().into_iter();

        match Meta::try_from(W(&mut bytes)) {
            Err(_) => panic!("meta wasn't read"),
            Ok(v) => assert_eq!(v, meta, "received meta differs"),
        }

        assert_eq!(bytes.next(), None);
    }
}
