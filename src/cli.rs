use std::path::PathBuf;

use clap::{arg, Args, Parser, Subcommand, ValueEnum};
use image::codecs::png::CompressionType;

use crate::lsb::Seed;

#[derive(Debug, Args, Clone)]
pub struct PassphraseFlags {
    /// Read PSK passphrase from a file (one trailing newline stripped, max 4 KiB).
    #[arg(long)]
    pub psk_file: Option<PathBuf>,

    /// Read PSK passphrase from an environment variable.
    #[arg(long)]
    pub psk_env: Option<String>,

    /// Prompt for PSK passphrase on a TTY.
    #[arg(long)]
    pub psk_prompt: bool,
}

impl PassphraseFlags {
    #[must_use]
    pub fn into_source(self) -> Option<crate::auth::passphrase::PassphraseSource> {
        if let Some(p) = self.psk_file {
            Some(crate::auth::passphrase::PassphraseSource::File(p))
        } else if let Some(v) = self.psk_env {
            Some(crate::auth::passphrase::PassphraseSource::Env(v))
        } else if self.psk_prompt {
            Some(crate::auth::passphrase::PassphraseSource::Prompt)
        } else {
            None
        }
    }
}

#[derive(Debug, Args, Clone)]
pub struct SignKeyPassphraseFlags {
    /// Read the Ed25519 private key passphrase from a file.
    #[arg(long)]
    pub sign_key_passphrase_file: Option<PathBuf>,

    /// Read the Ed25519 private key passphrase from an environment variable.
    #[arg(long)]
    pub sign_key_passphrase_env: Option<String>,

    /// Prompt for the Ed25519 private key passphrase on a TTY.
    #[arg(long)]
    pub sign_key_passphrase_prompt: bool,
}

impl SignKeyPassphraseFlags {
    #[must_use]
    pub fn into_source(self) -> Option<crate::auth::passphrase::PassphraseSource> {
        if let Some(p) = self.sign_key_passphrase_file {
            Some(crate::auth::passphrase::PassphraseSource::File(p))
        } else if let Some(v) = self.sign_key_passphrase_env {
            Some(crate::auth::passphrase::PassphraseSource::Env(v))
        } else if self.sign_key_passphrase_prompt {
            Some(crate::auth::passphrase::PassphraseSource::Prompt)
        } else {
            None
        }
    }
}

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

    #[command(flatten)]
    pub psk: PassphraseFlags,

    /// Sign the container with an OpenSSH Ed25519 private key.
    #[arg(long, conflicts_with_all = ["psk_file", "psk_env", "psk_prompt"])]
    pub sign_key: Option<PathBuf>,

    #[command(flatten)]
    pub sign_key_passphrase: SignKeyPassphraseFlags,
}

impl InjectArgs {
    #[must_use]
    pub fn auth_spec(&self) -> Option<crate::auth::AuthSpec> {
        if let Some(src) = self.psk.clone().into_source() {
            return Some(crate::auth::AuthSpec::Psk(src));
        }
        self.sign_key.clone().map(|key_path| crate::auth::AuthSpec::Ed25519 {
            key_path,
            key_passphrase: self.sign_key_passphrase.clone().into_source(),
        })
    }
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

    #[command(flatten)]
    pub psk: PassphraseFlags,

    /// OpenSSH Ed25519 public key file for verification.
    #[arg(long, conflicts_with_all = ["psk_file", "psk_env", "psk_prompt"])]
    pub verify_key: Option<PathBuf>,

    /// Inline OpenSSH public key string from environment variable.
    #[arg(long)]
    pub verify_key_env: Option<String>,

    /// Skip signature verification on a signed container. Always prints a stderr warning.
    #[arg(long)]
    pub insecure_skip_verify: bool,
}

impl ExtractArgs {
    #[must_use]
    pub fn verify_spec(&self) -> Option<crate::auth::VerifySpec> {
        if let Some(src) = self.psk.clone().into_source() {
            return Some(crate::auth::VerifySpec::Psk(src));
        }
        if let Some(p) = self.verify_key.clone() {
            return Some(crate::auth::VerifySpec::Ed25519Path(p));
        }
        if let Some(name) = &self.verify_key_env {
            if let Ok(value) = std::env::var(name) {
                return Some(crate::auth::VerifySpec::Ed25519Inline(value));
            }
        }
        None
    }
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
