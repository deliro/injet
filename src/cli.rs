use std::path::PathBuf;

use clap::{arg, Args, Parser, Subcommand, ValueEnum};
use image::codecs::png::CompressionType;

use crate::lsb::Seed;

#[derive(Parser)]
#[command(author, version, about, long_about)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Injects a file into an image. If the image is not PNG/RGBA8, it will be converted
    Inject(InjectArgs),

    /// Extracts a file from an image
    Extract(ExtractArgs),

    /// Inspects an image if it has a file inside and prints the results.
    /// Also tells how large a file can be injected inside
    Inspect(InspectArgs),
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
pub enum Compression {
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
pub struct InjectArgs {
    /// The file to inject
    pub payload: PathBuf,

    /// The image (container)
    pub container: PathBuf,

    /// Destination, where the injected file is placed
    #[arg(short, long)]
    pub destination: Option<PathBuf>,

    /// Whether to write metadata. If not set, extracting would require --read-meta=false
    /// and the exact file size in bytes (--read-size)
    #[arg(short, long, default_value_t = true, action = clap::ArgAction::Set)]
    pub write_meta: bool,

    /// Compression level used to compress PNG
    #[arg(value_enum, long, default_value_t = Compression::Default)]
    pub compression: Compression,

    /// Use seed to place bits in pseudorandom pixel positions.
    /// The same seed must be provided during extraction or inspection
    /// to correctly recover the data.
    #[arg(long)]
    pub seed: Option<Seed>,
}

#[derive(Args)]
pub struct ExtractArgs {
    /// Container that contains a file
    pub container: PathBuf,

    /// Where to save the extracted file. If not set, the filename will be read
    /// from metadata (if any). If none are set, defaults to "payload"
    #[arg(short, long)]
    pub destination: Option<PathBuf>,

    /// Whether to read metadata. If metadata was not written and --read-meta=true,
    /// extraction will fail. If metadata was written and --read-meta=false,
    /// the extracted file will be broken
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub read_meta: bool,

    /// How many bytes of the payload file to read. Defaults to the value in metadata.
    /// If none, defaults to the maximum (until the container ends)
    #[arg(long)]
    pub read_size: Option<u32>,

    /// Seed used to pseudorandomly locate embedded data.
    /// Must match the seed used during injection, if any.
    #[arg(long)]
    pub seed: Option<Seed>,
}

#[derive(Args)]
pub struct InspectArgs {
    /// Container file
    pub path: PathBuf,

    /// Seed used to pseudorandomly locate embedded data.
    /// Must match the seed used during injection, if any.
    #[arg(long)]
    pub seed: Option<Seed>,
}
