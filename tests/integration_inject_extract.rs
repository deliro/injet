use assert_cmd::Command;
use image::open;
use image::ImageBuffer;
use image::Rgba;
use predicates::prelude::*;
use rstest::rstest;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::{tempdir, TempDir};

const TEST_PAYLOAD: &[u8] = include_bytes!("test_payload_2kb.bin");

struct TestEnv {
    dir: TempDir,
    png_path: PathBuf,
    bin_path: PathBuf,
    out_png_path: PathBuf,
    extracted_bin_path: PathBuf,
}

fn setup_env() -> TestEnv {
    let dir = tempdir().unwrap();
    let png_path = dir.path().join("test.png");
    let bin_path = dir.path().join("payload.bin");
    let out_png_path = dir.path().join("out.png");
    let extracted_bin_path = dir.path().join("extracted.bin");
    // Always create fresh PNG and test file
    let img: ImageBuffer<Rgba<u8>, Vec<u8>> =
        ImageBuffer::from_fn(100, 100, |_x, _y| Rgba([255, 255, 255, 255]));
    img.save(&png_path).unwrap();
    std::fs::write(&bin_path, TEST_PAYLOAD).unwrap();
    TestEnv {
        dir,
        png_path,
        bin_path,
        out_png_path,
        extracted_bin_path,
    }
}

fn corrupt_metadata_bit(src_png: &Path, dst_png: &Path) {
    let mut img = open(src_png).unwrap().into_rgba8();
    img.get_pixel_mut(0, 0).0[0] ^= 1;
    img.save(dst_png).unwrap();
}

fn corrupt_payload_bit(src_png: &Path, dst_png: &Path) {
    let mut img = open(src_png).unwrap().into_rgba8();
    img.get_pixel_mut(32, 0).0[0] ^= 1;
    img.save(dst_png).unwrap();
}

fn assert_extract_fails_with_error(png_path: &Path, extracted_path: &Path, expected: &str) {
    let mut cmd = Command::cargo_bin("injet").unwrap();
    cmd.args([
        "extract",
        png_path.to_str().unwrap(),
        "-d",
        extracted_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains(expected));
}

#[test]
fn extract_fails_on_corrupted_metadata() {
    let env = setup_env();
    inject_file_into_png(&env.bin_path, &env.png_path, &env.out_png_path, true, None);
    let corrupted_png_path = env.dir.path().join("corrupted.png");
    corrupt_metadata_bit(&env.out_png_path, &corrupted_png_path);
    assert_extract_fails_with_error(
        &corrupted_png_path,
        &env.extracted_bin_path,
        "Invalid metadata signature",
    );
}

#[test]
fn extract_fails_on_crc32_data() {
    let env = setup_env();
    inject_file_into_png(&env.bin_path, &env.png_path, &env.out_png_path, true, None);
    let corrupted_png_path = env.dir.path().join("corrupted_data.png");
    corrupt_payload_bit(&env.out_png_path, &corrupted_png_path);
    assert_extract_fails_with_error(
        &corrupted_png_path,
        &env.extracted_bin_path,
        "Failed to verify hash",
    );
}

fn inject_file_into_png(
    cargo: &Path,
    container: &Path,
    out_png: &Path,
    write_meta: bool,
    seed: Option<&str>,
) {
    let mut cmd = Command::cargo_bin("injet").unwrap();
    cmd.args([
        "inject",
        cargo.to_str().unwrap(),
        container.to_str().unwrap(),
        "-d",
        out_png.to_str().unwrap(),
    ]);
    if !write_meta {
        cmd.args(["--write-meta", "false"]);
    }
    if let Some(seed_val) = seed {
        cmd.args(["--seed", seed_val]);
    }
    cmd.assert().success();
}

fn extract_file_from_png(
    container: &Path,
    out_file: &Path,
    write_meta: bool,
    read_size: Option<usize>,
    seed: Option<&str>,
    expect_success: bool,
) -> Option<Vec<u8>> {
    let mut cmd = Command::cargo_bin("injet").unwrap();
    cmd.args([
        "extract",
        container.to_str().unwrap(),
        "-d",
        out_file.to_str().unwrap(),
    ]);
    if !write_meta {
        if let Some(size) = read_size {
            cmd.args(["--read-meta", "false", "--read-size", &size.to_string()]);
        } else {
            cmd.args(["--read-meta", "false"]);
        }
    }
    if let Some(seed_val) = seed {
        cmd.args(["--seed", seed_val]);
    }
    if expect_success {
        cmd.assert().success();
        Some(fs::read(out_file).unwrap())
    } else {
        cmd.assert()
            .failure()
            .stderr(predicate::str::contains("Invalid metadata"));
        None
    }
}

#[rstest]
#[case(true, None)]
#[case(false, None)]
#[case(true, Some("myseed"))]
#[case(false, Some("myseed"))]
fn inject_and_extract_parametrized(#[case] write_meta: bool, #[case] seed: Option<&str>) {
    let env = setup_env();
    inject_file_into_png(
        &env.bin_path,
        &env.png_path,
        &env.out_png_path,
        write_meta,
        seed,
    );
    let extracted_bytes = extract_file_from_png(
        &env.out_png_path,
        &env.extracted_bin_path,
        write_meta,
        Some(TEST_PAYLOAD.len()),
        seed,
        true,
    )
    .unwrap();
    assert_eq!(extracted_bytes, TEST_PAYLOAD);
}

#[rstest]
#[case(false)] // extraction without seed
#[case(true)] // extraction with wrong seed
fn extract_with_wrong_or_missing_seed_fails(#[case] use_wrong_seed: bool) {
    let env = setup_env();
    let seed = Some("myseed");
    inject_file_into_png(&env.bin_path, &env.png_path, &env.out_png_path, true, seed);
    let wrong_seed = if use_wrong_seed {
        Some("wrongseed")
    } else {
        None
    };
    extract_file_from_png(
        &env.out_png_path,
        &env.extracted_bin_path,
        true,
        Some(TEST_PAYLOAD.len()),
        wrong_seed,
        false,
    );
}
