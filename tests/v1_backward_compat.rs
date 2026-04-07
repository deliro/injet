use assert_cmd::Command;
use image::{ImageBuffer, Rgba};
use std::path::PathBuf;
use tempfile::tempdir;

fn embed_bits_into_png(png: &mut ImageBuffer<Rgba<u8>, Vec<u8>>, bits: &[u8]) {
    let (w, h) = (png.width(), png.height());
    let mut idx = 0;
    'outer: for x in 0..w {
        for y in 0..h {
            let p = png.get_pixel_mut(x, y);
            for c in 0..4 {
                if idx >= bits.len() {
                    break 'outer;
                }
                p.0[c] = (p.0[c] & 0xFE) | (bits[idx] & 1);
                idx += 1;
            }
        }
    }
}

fn bits(bytes: &[u8]) -> Vec<u8> {
    bytes
        .iter()
        .flat_map(|b| (0..8).rev().map(move |i| (b >> i) & 1))
        .collect()
}

#[test]
fn extract_reads_legacy_v1_format() {
    const MAGIC: u16 = 0xd2d;
    const VERSION_1: u8 = 1;
    let dir = tempdir().unwrap();
    let png_path: PathBuf = dir.path().join("v1.png");
    let extracted = dir.path().join("out.bin");

    // Build a v1 meta byte stream:
    //   signature = (MAGIC << 3) | VERSION_1   (2 bytes, little-endian)
    //   size       = u32 little-endian (4 bytes)
    //   filename_len (u8, 0xFF means "no filename")
    //   filename bytes
    let signature = (MAGIC << 3) | (VERSION_1 as u16);
    let payload = b"hello";
    let mut meta = Vec::new();
    meta.extend(signature.to_le_bytes());
    meta.extend((payload.len() as u32).to_le_bytes());
    let filename = b"hi.bin";
    meta.push(filename.len() as u8);
    meta.extend(filename);

    // Concatenate meta and payload bits (MSB-first per byte, matching the production encoder).
    let mut bit_stream = bits(&meta);
    bit_stream.extend(bits(payload));

    // Build a 50x50 white PNG and embed the bits.
    let mut img: ImageBuffer<Rgba<u8>, Vec<u8>> =
        ImageBuffer::from_fn(50, 50, |_, _| Rgba([255, 255, 255, 255]));
    embed_bits_into_png(&mut img, &bit_stream);
    img.save(&png_path).unwrap();

    // Run extract and verify the payload round-trips.
    let mut cmd = Command::cargo_bin("injet").unwrap();
    cmd.args([
        "extract",
        png_path.to_str().unwrap(),
        "-d",
        extracted.to_str().unwrap(),
    ]);
    cmd.assert().success();
    assert_eq!(std::fs::read(&extracted).unwrap(), payload);
}
