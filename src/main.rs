use std::borrow::Cow;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io;
use std::io::Write;
use std::io::{stdout, BufReader, BufWriter, IsTerminal, Read};
use std::path::PathBuf;
use std::process::exit;

use ascii_table::AsciiTable;
use clap::{arg, Args, Parser, Subcommand, ValueEnum};
use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use image::{ColorType, EncodableLayout, GenericImageView, ImageEncoder, Rgba};
use itertools::Itertools;
use rand::seq::SliceRandom;
use rand::SeedableRng;

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

enum MetaError {
    NoBytes,
    SignatureMismatch,
    UnsupportedVersion(u8),
    MalformedFilename,
}

impl Display for MetaError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            MetaError::NoBytes => f.write_str("not enough bytes to build the meta"),
            MetaError::SignatureMismatch => f.write_str("signature mismatch"),
            MetaError::UnsupportedVersion(v) => {
                f.write_str(&format!("version {} isn't supported", v))
            }
            MetaError::MalformedFilename => f.write_str("error reading cargo filename"),
        }
    }
}

enum InspectError {
    FileNotExist,
    NotAnImage,
    NotAFile,
}

impl Display for InspectError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            InspectError::FileNotExist => f.write_str("file doesn't exist"),
            InspectError::NotAnImage => f.write_str("file is not an image"),
            InspectError::NotAFile => f.write_str("not a file"),
        }
    }
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
    let ascii_table = AsciiTable::default();
    let dimensions_fmt = format!("{w}x{h}");
    let mut table_data = vec![
        ["filename".into(), Cow::from(&filename)],
        ["dimensions".into(), dimensions_fmt.into()],
        ["max cargo size".into(), max_cargo_size.into()],
    ];

    match meta {
        None => table_data.push(["doesn't seem it contains any cargo".into(), "".into()]),
        Some(v) => {
            let cargo_filename = v.filename.unwrap_or(String::from("<unnamed>"));
            let cargo_size = format_size(v.size);
            table_data.push(["cargo filename".into(), cargo_filename.into()]);
            table_data.push(["cargo size".into(), cargo_size.into()]);
        }
    }
    ascii_table.print(table_data);
    Ok(())
}

enum ExtractError {
    ContainerOpen,
    Save,
    BrokenMeta(String),
}

impl Display for ExtractError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ExtractError::ContainerOpen => f.write_str("cannot open the container file"),
            ExtractError::Save => f.write_str("cannot save to the destination"),
            ExtractError::BrokenMeta(v) => f.write_str(&format!("meta is broken: {v}")),
        }
    }
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

    let dest = args
        .destination
        .or(meta_filename)
        .unwrap_or(PathBuf::from("cargo"));

    let read_size = args.read_size.or(size).unwrap_or(u32::MAX);
    let file = File::create(dest).map_err(|_| ExtractError::Save)?;
    let mut writer = BufWriter::new(file);
    for b in content.take(read_size as usize) {
        writer.write_all(&[b]).map_err(|_| ExtractError::Save)?
    }
    writer.flush().map_err(|_| ExtractError::Save)?;
    Ok(())
}

enum InjectError {
    CannotOpenContainer,
    CannotOpenCargo,
    ExceededSize {
        available: u32,
        cargo_size: u32,
        meta_size: u32,
    },
    CannotSave(String),
    FilenameOverflow,
}

impl Display for InjectError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            InjectError::CannotOpenContainer => f.write_str("cannot open container file"),
            InjectError::CannotOpenCargo => f.write_str("cannot open cargo file"),
            InjectError::ExceededSize {
                available,
                cargo_size,
                meta_size,
            } => f.write_str(&format!(
                "cannot inject cargo. available size: {}, cargo size: {} + meta size: {}",
                format_size(*available),
                format_size(*cargo_size),
                format_size(*meta_size),
            )),
            InjectError::CannotSave(v) => {
                f.write_str(&format!("cannot save destination file: {v}"))
            }
            InjectError::FilenameOverflow => {
                f.write_str("filename must be less than 255 bytes length")
            }
        }
    }
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
    let total_size = cargo_size + meta_size; // todo: add crc32 check
    let required_pixels = ((total_size * 8) / 4) as usize;
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
    // [(r, g, b, a), (r, g, b, a)] turns into flat [r, g, b, a, r, g, b, a]
    let colors = gen_dots(w, h, args.seed.as_ref())
        .take(required_pixels)
        .flat_map(|(x, y)| img.get_pixel(x, y).0)
        .collect_vec();
    let binding = meta_bits
        .into_iter()
        .chain(cargo_bits)
        .zip(colors)
        .map(|(bit, color)| ((color & 0b11111110) | bit))
        .chunks(4);

    let new_pixels = binding.into_iter().map(|chunk| {
        let colors: [u8; 4] = chunk.collect_vec().try_into().unwrap();
        Rgba::from(colors)
    });

    gen_dots(w, h, args.seed.as_ref())
        .zip(new_pixels)
        .for_each(|((x, y), pixel)| img.put_pixel(x, y, pixel));

    let writer = make_writer(&args)?;
    let encoder =
        PngEncoder::new_with_quality(writer, args.compression.into(), FilterType::default());
    encoder
        .write_image(img.as_bytes(), w, h, ColorType::Rgba8)
        .map_err(|e| InjectError::CannotSave(e.to_string()))
}

fn make_writer(args: &InjectArgs) -> Result<Box<dyn Write>, InjectError> {
    let writer = if !stdout().is_terminal() && args.destination.is_none() {
        Box::new(stdout()) as Box<dyn Write>
    } else {
        let dest = args
            .destination
            .clone()
            .unwrap_or(PathBuf::from("modified.png"));
        let write_file =
            BufWriter::new(File::create(dest).map_err(|e| InjectError::CannotSave(e.to_string()))?);
        Box::new(write_file) as Box<dyn Write>
    };
    Ok(writer)
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
            Err(_) => assert!(false, "meta wasn't read"),
            Ok(v) => assert_eq!(v, meta, "received meta differs"),
        }

        assert_eq!(bytes.next(), None);
    }
}
