use std::path::Path;

use image::GenericImageView;
use itertools::Itertools;
use thiserror::Error;

use crate::cli::InspectArgs;
use crate::lsb::gen_dots;
use crate::meta::Meta;

const KB: u32 = 1024;
const MB: u32 = 1024 * 1024;
const MB_MINUS_1: u32 = MB - 1;
const GB: u32 = MB * 1024;
const GB_MINUS_1: u32 = GB - 1;
const READ_BUFFER_SIZE: usize = 8192;

fn fmt_with_unit(size: u32, divisor: u32, unit: &str) -> String {
    let whole = size.checked_div(divisor).unwrap_or(0);
    let remainder = size.checked_rem(divisor).unwrap_or(0);
    let scaled = u64::from(remainder).saturating_mul(100);
    let frac = u32::try_from(scaled.checked_div(u64::from(divisor)).unwrap_or(0)).unwrap_or(0);
    format!("{whole}.{frac:02} {unit}")
}

#[inline]
fn format_size(size: u32) -> String {
    match size {
        GB..=u32::MAX => fmt_with_unit(size, GB, "GB"),
        MB..=GB_MINUS_1 => fmt_with_unit(size, MB, "MB"),
        KB..=MB_MINUS_1 => fmt_with_unit(size, KB, "KB"),
        _ => format!("{size} bytes"),
    }
}

pub(crate) fn display_path(path: &Path) -> String {
    path.file_name().map_or_else(
        || path.to_string_lossy().into_owned(),
        |n| n.to_string_lossy().into_owned(),
    )
}

#[derive(Debug, Error)]
pub enum InspectError {
    #[error("File not found")]
    FileNotExist,
    #[error("Not a valid image file")]
    NotAnImage,
    #[error("Not a regular file")]
    NotAFile,
}

/// Inspects a container image, printing its dimensions, capacity, and any embedded metadata.
///
/// # Errors
///
/// Returns an [`InspectError`] when the path does not exist, is not a regular file, or
/// cannot be decoded as an image.
pub fn inspect(args: &InspectArgs) -> Result<(), InspectError> {
    if !args.path.exists() {
        return Err(InspectError::FileNotExist);
    }

    if !args.path.is_file() {
        return Err(InspectError::NotAFile);
    }

    let filename = display_path(&args.path);
    let img = image::open(&args.path).map_err(|_| InspectError::NotAnImage)?;
    let (width, height) = img.dimensions();
    let pixel_count = u64::from(width).saturating_mul(u64::from(height));
    let max_payload_bytes = pixel_count.saturating_mul(4).checked_div(8).unwrap_or(0);
    let max_payload_size = format_size(u32::try_from(max_payload_bytes).unwrap_or(u32::MAX));
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
    let meta = Meta::read(&mut content).ok();
    println!("Image file: {filename}");
    println!("Dimensions: {width}x{height}");
    println!("Maximum embeddable file size: {max_payload_size}");
    let Some(meta) = meta.as_ref() else {
        println!("No embedded data detected or metadata is missing.");
        return Ok(());
    };
    print_meta(meta, &mut content);
    Ok(())
}

fn print_meta<I: Iterator<Item = u8>>(meta: &Meta, content: &mut I) {
    println!("Metadata version: {}", meta.version);
    let payload_filename = meta.filename().unwrap_or("<unnamed>");
    let payload_size = meta
        .size()
        .map_or_else(|| "<unknown>".to_string(), format_size);
    println!("Embedded file name: {payload_filename}");
    println!("Embedded file size: {payload_size}");
    if let Some(hash) = meta.hash() {
        println!("Embedded file CRC32: {hash:08x}");
    }
    if let Some(meta_hash) = meta.meta_hash() {
        println!("Header CRC32: {meta_hash:08x}");
    }
    if let Some(algo) = crate::auth::detect_auth(&meta.fields) {
        println!("Signed: {algo}");
    }
    let Some(expected_hash) = meta.hash() else {
        return;
    };
    let read_size = usize::try_from(meta.size().unwrap_or(u32::MAX)).unwrap_or(usize::MAX);
    let mut crc = crc32fast::Hasher::new();
    let mut remaining = read_size;
    let mut buffer = [0_u8; READ_BUFFER_SIZE];
    while remaining > 0 {
        let to_read = buffer.len().min(remaining);
        let Some(slice) = buffer.get_mut(..to_read) else {
            break;
        };
        let mut filled: usize = 0;
        for slot in slice.iter_mut() {
            match content.next() {
                Some(b) => {
                    *slot = b;
                    filled = filled.saturating_add(1);
                }
                None => break,
            }
        }
        if filled == 0 {
            break;
        }
        let Some(written) = buffer.get(..filled) else {
            break;
        };
        crc.update(written);
        remaining = remaining.saturating_sub(filled);
    }
    let calculated = crc.finalize();
    if calculated == expected_hash {
        println!("Payload CRC32: ok");
    } else {
        println!("Payload CRC32: mismatch");
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::display_path;
    use std::path::Path;

    #[test]
    fn test_display_path_handles_dot_dot() {
        // `..` has no file_name component — must not panic, must produce something printable.
        let s = display_path(Path::new(".."));
        assert!(!s.is_empty());

        // A normal file_name path keeps the last component.
        let s = display_path(Path::new("/tmp/hello.bin"));
        assert_eq!(s, "hello.bin");
    }
}
