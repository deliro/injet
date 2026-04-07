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

#[inline]
fn format_size(size: u32) -> String {
    match size {
        (GB..=u32::MAX) => format!("{:.2} GB", (size as f32) / (GB as f32)),
        (MB..=GB_MINUS_1) => format!("{:.2} MB", (size as f32) / (MB as f32)),
        (KB..=MB_MINUS_1) => format!("{:.2} KB", (size as f32) / (KB as f32)),
        _ => format!("{size} bytes"),
    }
}

pub(crate) fn display_path(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string_lossy().into_owned())
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

pub fn inspect(args: InspectArgs) -> Result<(), InspectError> {
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

#[cfg(test)]
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
