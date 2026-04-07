use std::fs::File;
use std::io;
use std::io::Write;
use std::io::{stdout, BufReader, BufWriter, IsTerminal, Read, Seek};
use std::path::{Path, PathBuf};
use std::process::exit;

use clap::{arg, Args, Parser, Subcommand, ValueEnum};
use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use image::{ColorType, EncodableLayout, GenericImageView, ImageEncoder};
use itertools::Itertools;
use thiserror::Error;

use injet::meta::Meta;

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
const MAX_FILENAME_LEN: usize = 255;

#[inline]
fn format_size(size: u32) -> String {
    match size {
        (GB..=u32::MAX) => format!("{:.2} GB", (size as f32) / (GB as f32)),
        (MB..=GB_MINUS_1) => format!("{:.2} MB", (size as f32) / (MB as f32)),
        (KB..=MB_MINUS_1) => format!("{:.2} KB", (size as f32) / (KB as f32)),
        _ => format!("{size} bytes"),
    }
}

fn display_path(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string_lossy().into_owned())
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

fn seed_to_array(seed: &str) -> [u8; 32] {
    *blake3::hash(seed.as_bytes()).as_bytes()
}

fn pseudo_shuffle_coords(w: u32, h: u32, seed: &Seed) -> impl Iterator<Item = (u32, u32)> {
    let n = u64::from(w) * u64::from(h);
    // Round total bits up to an even number so Feistel halves are symmetric.
    let bits_min = if n <= 1 { 2 } else { 64 - (n - 1).leading_zeros() };
    let bits_even = if bits_min % 2 == 0 {
        bits_min
    } else {
        bits_min + 1
    };
    let bits = bits_even.max(2);
    let half = bits / 2;
    let mask = if half >= 64 { u64::MAX } else { (1u64 << half) - 1 };
    let total = if bits >= 64 { u64::MAX } else { 1u64 << bits };
    let key = seed_to_array(seed);

    let round_fn = move |x: u64, r: u32| -> u64 {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&key);
        hasher.update(&[r as u8]);
        hasher.update(&x.to_le_bytes());
        let bytes = hasher.finalize();
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&bytes.as_bytes()[..8]);
        u64::from_le_bytes(buf) & mask
    };

    let feistel = move |x: u64| -> u64 {
        let mut l = (x >> half) & mask;
        let mut r = x & mask;
        for round in 0..4u32 {
            let new_l = r;
            let new_r = l ^ round_fn(r, round);
            l = new_l;
            r = new_r;
        }
        (l << half) | r
    };

    (0..total)
        .map(feistel)
        .filter(move |v| *v < n)
        .map(move |v| {
            let x = (v / u64::from(h)) as u32;
            let y = (v % u64::from(h)) as u32;
            (x, y)
        })
}

fn gen_dots(w: u32, h: u32, seed: Option<&Seed>) -> impl Iterator<Item = (u32, u32)> {
    match seed {
        Some(seed) => itertools::Either::Left(pseudo_shuffle_coords(w, h, seed)),
        None => itertools::Either::Right(iter_dots(w, h)),
    }
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
    #[error("Filename is too long (maximum {} bytes)", MAX_FILENAME_LEN)]
    FilenameOverflow,
}

fn inspect(args: InspectArgs) -> Result<(), InspectError> {
    if !args.path.exists() {
        return Err(InspectError::FileNotExist);
    }

    if !args.path.is_file() {
        return Err(InspectError::NotAFile);
    }

    let filename = display_path(&args.path);
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
            let cargo_filename = v.filename().unwrap_or("<unnamed>");
            let cargo_size = v
                .size()
                .map(format_size)
                .unwrap_or_else(|| "<unknown>".to_string());
            println!("Embedded file name: {cargo_filename}");
            println!("Embedded file size: {cargo_size}");
            if let Some(hash) = v.hash() {
                println!("Embedded file CRC32: {hash:08x}");
            }
            if let Some(meta_hash) = v.meta_hash() {
                println!("Header CRC32: {meta_hash:08x}");
            }
            if let Some(expected_hash) = v.hash() {
                let read_size = v.size().unwrap_or(u32::MAX) as usize;
                let mut crc = crc32fast::Hasher::new();
                let mut remaining = read_size;
                let mut chunk = [0u8; 8192];
                while remaining > 0 {
                    let to_read = chunk.len().min(remaining);
                    let mut filled = 0;
                    while filled < to_read {
                        match content.next() {
                            Some(b) => {
                                chunk[filled] = b;
                                filled += 1;
                            }
                            None => break,
                        }
                    }
                    if filled == 0 {
                        break;
                    }
                    crc.update(&chunk[..filled]);
                    remaining -= filled;
                }
                let calculated = crc.finalize();
                if calculated == expected_hash {
                    println!("Payload CRC32: ok");
                } else {
                    println!("Payload CRC32: mismatch");
                }
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

/// Where an `extract` run streams its bytes to.
///
/// For file targets we write to a `<final_path>.partial` sidecar and atomically
/// rename to the final path only after the CRC verification has passed. This
/// guarantees the destination path is either absent or contains a fully
/// verified payload.
enum ExtractTarget {
    Stdout(Box<dyn Write>),
    File {
        partial_path: PathBuf,
        final_path: PathBuf,
        writer: BufWriter<File>,
    },
}

impl ExtractTarget {
    fn writer_mut(&mut self) -> &mut dyn Write {
        match self {
            ExtractTarget::Stdout(w) => w.as_mut(),
            ExtractTarget::File { writer, .. } => writer,
        }
    }

    fn finalize(self) -> Result<(), ExtractError> {
        match self {
            ExtractTarget::Stdout(_) => Ok(()),
            ExtractTarget::File {
                partial_path,
                final_path,
                writer,
            } => {
                drop(writer); // flush + close before rename
                std::fs::rename(&partial_path, &final_path).map_err(|_| ExtractError::Save)
            }
        }
    }

    fn abort(self) {
        if let ExtractTarget::File {
            partial_path,
            writer,
            ..
        } = self
        {
            drop(writer);
            let _ = std::fs::remove_file(&partial_path);
        }
    }
}

fn make_extract_target(
    dest: Option<&Path>,
    default: impl AsRef<Path>,
) -> Result<ExtractTarget, String> {
    if !stdout().is_terminal() && dest.is_none() {
        return Ok(ExtractTarget::Stdout(Box::new(stdout()) as Box<dyn Write>));
    }
    let final_path = dest
        .map(PathBuf::from)
        .unwrap_or_else(|| default.as_ref().to_path_buf());
    // Append ".partial" to the final path (preserving any existing extension).
    let partial_path = {
        let mut p = final_path.as_os_str().to_owned();
        p.push(".partial");
        PathBuf::from(p)
    };
    let file = File::create(&partial_path).map_err(|e| e.to_string())?;
    let writer = BufWriter::new(file);
    Ok(ExtractTarget::File {
        partial_path,
        final_path,
        writer,
    })
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

    let (meta_filename, size) = if let Some(meta) = &meta {
        (meta.filename().map(PathBuf::from), meta.size())
    } else {
        (None, None)
    };

    let read_size = args.read_size.or(size).unwrap_or(u32::MAX);
    let mut target = make_extract_target(
        args.destination.as_deref(),
        meta_filename.unwrap_or(PathBuf::from("cargo")),
    )
    .map_err(|_| ExtractError::Save)?;

    let result = (|| -> Result<(), ExtractError> {
        let mut crc = crc32fast::Hasher::new();
        let mut buffer = [0u8; 8192];
        let mut remaining = read_size as usize;
        let writer = target.writer_mut();
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
            if let Some(expected_hash) = meta.hash() {
                if crc.finalize() != expected_hash {
                    return Err(ExtractError::HashMismatch);
                }
            }
        }
        Ok(())
    })();

    match result {
        Ok(()) => target.finalize(),
        Err(e) => {
            target.abort();
            Err(e)
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
            if v.len() > MAX_FILENAME_LEN {
                return Err(InjectError::FilenameOverflow);
            }
        }
        // Calculate crc32 by streaming the cargo, then rewind for the bit pass
        let mut hasher = crc32fast::Hasher::new();
        let mut reader = BufReader::new(&cargo);
        let mut buf = [0u8; 8192];
        loop {
            let n = reader
                .read(&mut buf)
                .map_err(|_| InjectError::CannotOpenCargo)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        drop(reader);
        (&cargo)
            .rewind()
            .map_err(|_| InjectError::CannotOpenCargo)?;
        let hash = hasher.finalize();
        let meta = Meta::make_v3(Some(cargo_size), filename, Some(hash));
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
        let px = img.get_pixel_mut(x, y);
        for channel in &mut px.0 {
            if changed >= total_bits {
                break;
            }
            if let Some(bit) = bit_iter.next() {
                *channel = (*channel & 0b1111_1110) | bit;
                changed += 1;
            }
        }
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

    #[test]
    fn test_display_path_handles_dot_dot() {
        use std::path::Path;
        use crate::display_path;
        // `..` has no file_name component — must not panic, must produce something printable.
        let s = display_path(Path::new(".."));
        assert!(!s.is_empty());

        // A normal file_name path keeps the last component.
        let s = display_path(Path::new("/tmp/hello.bin"));
        assert_eq!(s, "hello.bin");
    }

    #[test]
    fn test_pseudo_shuffle_is_deterministic_for_same_seed() {
        use crate::pseudo_shuffle_coords;
        let a: Vec<_> = pseudo_shuffle_coords(20, 20, &"abc".to_string()).collect();
        let b: Vec<_> = pseudo_shuffle_coords(20, 20, &"abc".to_string()).collect();
        assert_eq!(a, b, "same seed must yield identical order");
        let c: Vec<_> = pseudo_shuffle_coords(20, 20, &"abd".to_string()).collect();
        assert_ne!(a, c, "different seed must yield different order");
        // Permutation property: every coordinate appears exactly once.
        let mut sorted = a.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 400);
    }
}
