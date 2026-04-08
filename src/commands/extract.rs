use std::fs::File;
use std::io::{stdout, BufWriter, IsTerminal, Write};
use std::path::{Path, PathBuf};

use image::GenericImageView;
use itertools::Itertools;
use thiserror::Error;

use crate::cli::ExtractArgs;
use crate::lsb::gen_dots;
use crate::meta::Meta;

const READ_BUFFER_SIZE: usize = 8192;
const INITIAL_PAYLOAD_CAPACITY: usize = 4 * 1024 * 1024;

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
    #[error("Auth error: {0}")]
    Auth(#[from] crate::auth::AuthError),
    #[error("Key error: {0}")]
    Key(#[from] crate::auth::KeyError),
}

impl ExtractError {
    /// Returns the process exit code for this error.
    #[must_use]
    pub fn exit_code(&self) -> i32 {
        match self {
            ExtractError::Auth(_) | ExtractError::Key(_) => 2,
            ExtractError::ContainerOpen
            | ExtractError::Save
            | ExtractError::BrokenMeta(_)
            | ExtractError::HashMismatch => 1,
        }
    }
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
        let writer: Box<dyn Write> = Box::new(stdout());
        return Ok(ExtractTarget::Stdout(writer));
    }
    let final_path = dest.map_or_else(|| default.as_ref().to_path_buf(), PathBuf::from);
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

fn fill_buffer<I: Iterator<Item = u8>>(content: &mut I, buffer: &mut [u8]) -> usize {
    let mut filled: usize = 0;
    for slot in buffer.iter_mut() {
        match content.next() {
            Some(b) => {
                *slot = b;
                filled = filled.saturating_add(1);
            }
            None => break,
        }
    }
    filled
}

/// Run the auth verification pipeline given a resolved decision.
///
/// Called only when `decision == Verify`. Builds `auth_input`, loads the
/// verify job via `pair_up`, and runs the appropriate MAC / signature check.
///
/// # Errors
///
/// Returns [`ExtractError`] wrapping the relevant [`crate::auth::AuthError`]
/// or [`crate::auth::KeyError`] on any failure.
fn run_verify(
    args: &ExtractArgs,
    meta: Option<&Meta>,
    payload_bytes: &[u8],
) -> Result<(), ExtractError> {
    let prefix = meta
        .and_then(|m| m.auth_prefix.as_deref())
        .ok_or(crate::auth::AuthError::MissingAuthPrefix)?;
    let mut auth_input = Vec::with_capacity(prefix.len().saturating_add(payload_bytes.len()));
    auth_input.extend_from_slice(prefix);
    auth_input.extend_from_slice(payload_bytes);
    let spec = args
        .verify_spec()
        .ok_or(crate::auth::AuthError::KeyRequired)?;
    let fields_slice: &[crate::meta::MetaField] =
        meta.map_or(&[], |m| m.fields.as_slice());
    let job = crate::auth::pair_up(&spec, fields_slice)?;
    match job {
        crate::auth::VerifyJob::Psk { passphrase, salt, mac } => {
            let key = crate::auth::psk::derive_psk(passphrase.as_slice(), &salt)
                .map_err(|_| crate::auth::AuthError::VerificationFailed)?;
            if !crate::auth::psk::verify_psk(&key, &auth_input, &mac) {
                return Err(crate::auth::AuthError::VerificationFailed.into());
            }
        }
        crate::auth::VerifyJob::Ed25519 { pubkey, sig } => {
            if !crate::auth::ed25519::verify_ed25519(&pubkey, &auth_input, &sig) {
                return Err(crate::auth::AuthError::VerificationFailed.into());
            }
        }
    }
    Ok(())
}

/// Extracts a previously injected file from a PNG container.
///
/// # Errors
///
/// Returns an [`ExtractError`] for unreadable containers, malformed metadata,
/// hash mismatches, auth verification failures, or filesystem errors while
/// writing the output.
pub fn extract(args: &ExtractArgs) -> Result<(), ExtractError> {
    let img = image::open(&args.container).map_err(|_| ExtractError::ContainerOpen)?;
    let (width, height) = img.dimensions();
    let bytes = gen_dots(width, height, args.seed.as_ref())
        .flat_map(|(x, y)| img.get_pixel(x, y).0)
        .map(|v| v & 1)
        .chunks(8);
    let mut content = bytes.into_iter().map(|chunk| {
        chunk
            .zip((0_u8..8).rev())
            .map(|(bit, shift)| bit << shift)
            .sum::<u8>()
    });

    let meta = if args.read_meta {
        Some(Meta::read(&mut content).map_err(|e| ExtractError::BrokenMeta(e.to_string()))?)
    } else {
        None
    };

    let (meta_filename, meta_size) = match &meta {
        Some(m) => (m.filename().map(PathBuf::from), m.size()),
        None => (None, None),
    };

    let read_size = args.read_size.or(meta_size).unwrap_or(u32::MAX);
    let read_size_usize = usize::try_from(read_size).unwrap_or(usize::MAX);

    // Read all payload bytes into memory so we can verify before writing.
    let mut payload_bytes: Vec<u8> =
        Vec::with_capacity(read_size_usize.min(INITIAL_PAYLOAD_CAPACITY));
    let mut buffer = [0_u8; READ_BUFFER_SIZE];
    let mut remaining = read_size_usize;
    while remaining > 0 {
        let to_read = buffer.len().min(remaining);
        let target_slice = buffer.get_mut(..to_read).ok_or(ExtractError::Save)?;
        let filled = fill_buffer(&mut content, target_slice);
        if filled == 0 {
            break;
        }
        let chunk = buffer.get(..filled).ok_or(ExtractError::Save)?;
        payload_bytes.extend_from_slice(chunk);
        remaining = remaining.saturating_sub(filled);
    }

    // Existing CRC32 check — now acts on the in-memory Vec.
    if let Some(m) = &meta {
        if let Some(expected_hash) = m.hash() {
            if crc32fast::hash(&payload_bytes) != expected_hash {
                return Err(ExtractError::HashMismatch);
            }
        }
    }

    // Auth pipeline — runs after CRC check, before writing the file.
    let container_auth = meta
        .as_ref()
        .and_then(|m| crate::auth::detect_auth(&m.fields));
    let user_algo = args
        .verify_spec()
        .as_ref()
        .map(crate::auth::VerifySpec::algorithm);
    let decision = crate::auth::classify(container_auth, user_algo, args.insecure_skip_verify);

    match decision {
        crate::auth::ExtractDecision::Unsigned => {}
        crate::auth::ExtractDecision::RejectUnsignedKeyProvided => {
            return Err(crate::auth::AuthError::ContainerNotSigned.into());
        }
        crate::auth::ExtractDecision::RejectSignedNoKey => {
            return Err(crate::auth::AuthError::KeyRequired.into());
        }
        crate::auth::ExtractDecision::RejectAlgorithmMismatch { expected, actual } => {
            return Err(crate::auth::AuthError::AlgorithmMismatch { expected, actual }.into());
        }
        crate::auth::ExtractDecision::SkipVerifyInsecure => {
            eprintln!("WARNING: extracting signed container without verification");
        }
        crate::auth::ExtractDecision::Verify => {
            run_verify(args, meta.as_ref(), &payload_bytes)?;
        }
    }

    // Write the payload only after all checks pass.
    let mut target = make_extract_target(
        args.destination.as_deref(),
        meta_filename.unwrap_or_else(|| PathBuf::from("payload")),
    )
    .map_err(|_| ExtractError::Save)?;

    let write_result = (|| -> Result<(), ExtractError> {
        let writer = target.writer_mut();
        writer
            .write_all(&payload_bytes)
            .map_err(|_| ExtractError::Save)?;
        writer.flush().map_err(|_| ExtractError::Save)?;
        Ok(())
    })();

    match write_result {
        Ok(()) => target.finalize(),
        Err(e) => {
            target.abort();
            Err(e)
        }
    }
}
