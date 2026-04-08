use std::fs::File;
use std::io::{stdout, BufWriter, IsTerminal, Write};
use std::path::Path;

use image::codecs::png::{FilterType, PngEncoder};
use image::{ColorType, EncodableLayout, ImageEncoder};
use thiserror::Error;

use crate::cli::InjectArgs;
use crate::lsb::{gen_dots, to_bits};
use crate::meta::{Meta, MetaError};

const MAX_FILENAME_LEN: usize = 255;

#[derive(Debug, Error)]
pub enum InjectError {
    #[error("Failed to open container file")]
    CannotOpenContainer,
    #[error("Failed to open input file")]
    CannotOpenPayload,
    #[error("Payload file exceeds maximum supported size of {}", u32::MAX)]
    PayloadTooLarge,
    #[error("File size exceeds container capacity: available {available}, file {payload_size}, metadata {meta_size}")]
    ExceededSize {
        available: u32,
        payload_size: u32,
        meta_size: u32,
    },
    #[error("Failed to save output file: {0}")]
    CannotSave(String),
    #[error("Filename is too long (maximum {} bytes)", MAX_FILENAME_LEN)]
    FilenameOverflow,
    #[error("Failed to encode metadata: {0}")]
    Meta(#[from] MetaError),
    #[error("Signing requires metadata; use --write-meta=true (the default) when passing --psk-* or --sign-key")]
    SigningRequiresMetadata,
    #[error("Key error: {0}")]
    Key(#[from] crate::auth::KeyError),
}

impl InjectError {
    #[must_use]
    pub fn exit_code(&self) -> i32 {
        match self {
            InjectError::SigningRequiresMetadata | InjectError::Key(_) => 2,
            InjectError::CannotOpenContainer
            | InjectError::CannotOpenPayload
            | InjectError::PayloadTooLarge
            | InjectError::ExceededSize { .. }
            | InjectError::CannotSave(_)
            | InjectError::FilenameOverflow
            | InjectError::Meta(_) => 1,
        }
    }
}

fn make_writer(dest: Option<&Path>, default: impl AsRef<Path>) -> Result<Box<dyn Write>, String> {
    let writer: Box<dyn Write> = if !stdout().is_terminal() && dest.is_none() {
        Box::new(stdout())
    } else {
        let dest = dest.unwrap_or_else(|| default.as_ref());
        let write_file = BufWriter::new(File::create(dest).map_err(|e| e.to_string())?);
        Box::new(write_file)
    };
    Ok(writer)
}

/// Injects a file into a PNG container using LSB encoding.
///
/// # Errors
///
/// Returns an [`InjectError`] when the container or payload cannot be opened, the payload
/// does not fit, the filename is invalid, or the output cannot be written.
pub fn inject(args: &InjectArgs) -> Result<(), InjectError> {
    if !args.write_meta && args.auth_spec().is_some() {
        return Err(InjectError::SigningRequiresMetadata);
    }

    let mut img = image::open(&args.container)
        .map_err(|_| InjectError::CannotOpenContainer)?
        .into_rgba8();

    let (width, height) = img.dimensions();
    let pixel_count = u64::from(width).saturating_mul(u64::from(height));
    let max_payload_bytes_u64 = pixel_count
        .saturating_mul(4)
        .checked_div(8)
        .ok_or(InjectError::CannotOpenContainer)?;
    let max_payload_size = u32::try_from(max_payload_bytes_u64).unwrap_or(u32::MAX);

    let payload_bytes: Vec<u8> =
        std::fs::read(&args.payload).map_err(|_| InjectError::CannotOpenPayload)?;
    let payload_size =
        u32::try_from(payload_bytes.len()).map_err(|_| InjectError::PayloadTooLarge)?;

    let meta_bytes: Vec<u8> = if args.write_meta {
        let filename = args
            .payload
            .file_name()
            .map(|v| String::from(v.to_string_lossy()));
        if let Some(v) = filename.as_ref() {
            if v.len() > MAX_FILENAME_LEN {
                return Err(InjectError::FilenameOverflow);
            }
        }
        let hash = crc32fast::hash(&payload_bytes);
        let meta = Meta::make_v3(Some(payload_size), filename, Some(hash));
        match args.auth_spec() {
            None => meta.to_bytes()?,
            Some(crate::auth::AuthSpec::Psk(src)) => {
                let pass = crate::auth::passphrase::load(&src)?;
                let signer = crate::auth::PskSigner::new(pass.as_slice())?;
                meta.to_bytes_with_auth(&payload_bytes, &signer)?
            }
            Some(crate::auth::AuthSpec::Ed25519 {
                key_path,
                key_passphrase,
            }) => {
                let signing =
                    crate::auth::ed25519::load_signing_key(&key_path, key_passphrase.as_ref())?;
                let signer = crate::auth::Ed25519Signer::new(signing);
                meta.to_bytes_with_auth(&payload_bytes, &signer)?
            }
        }
    } else {
        Vec::new()
    };

    let meta_size = u32::try_from(meta_bytes.len()).map_err(|_| InjectError::PayloadTooLarge)?;
    let total_size = payload_size
        .checked_add(meta_size)
        .ok_or(InjectError::PayloadTooLarge)?;
    if total_size > max_payload_size {
        return Err(InjectError::ExceededSize {
            available: max_payload_size,
            payload_size,
            meta_size,
        });
    }

    let meta_bits = meta_bytes.into_iter().flat_map(to_bits);
    let payload_bits = payload_bytes.into_iter().flat_map(to_bits);
    let mut bit_iter = meta_bits.chain(payload_bits);
    let color_coords = gen_dots(width, height, args.seed.as_ref());
    // Iterate over all coordinates, modifying only the required number of pixels.
    let mut changed: u64 = 0;
    let total_bits = u64::from(total_size).saturating_mul(8);
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
                changed = changed.saturating_add(1);
            }
        }
    }
    let writer = make_writer(args.destination.as_deref(), "modified.png")
        .map_err(InjectError::CannotSave)?;
    let encoder =
        PngEncoder::new_with_quality(writer, args.compression.into(), FilterType::default());
    encoder
        .write_image(img.as_bytes(), width, height, ColorType::Rgba8.into())
        .map_err(|e| InjectError::CannotSave(e.to_string()))
}
