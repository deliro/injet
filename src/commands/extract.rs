use std::fs::File;
use std::io::{stdout, BufWriter, IsTerminal, Write};
use std::path::{Path, PathBuf};

use image::GenericImageView;
use itertools::Itertools;
use thiserror::Error;

use crate::cli::ExtractArgs;
use crate::lsb::gen_dots;
use crate::meta::Meta;

#[derive(Debug, Error)]
pub enum ExtractError {
    #[error("Failed to open container file")]
    ContainerOpen,
    #[error("Failed to save output file")]
    Save,
    #[error("Invalid metadata: {0}")]
    BrokenMeta(String),
    #[error("Failed to verify hash")]
    HashMismatch,
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

pub fn extract(args: ExtractArgs) -> Result<(), ExtractError> {
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
