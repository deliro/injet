use std::fs::File;
use std::io::{stdout, BufReader, BufWriter, IsTerminal, Read, Seek, Write};
use std::path::Path;

use image::codecs::png::{FilterType, PngEncoder};
use image::{ColorType, EncodableLayout, ImageEncoder};
use thiserror::Error;

use crate::cli::InjectArgs;
use crate::lsb::{gen_dots, to_bits};
use crate::meta::Meta;

const MAX_FILENAME_LEN: usize = 255;

#[derive(Debug, Error)]
pub enum InjectError {
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

pub fn inject(args: InjectArgs) -> Result<(), InjectError> {
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
