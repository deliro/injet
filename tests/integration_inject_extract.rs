use assert_cmd::Command;
use image::{ImageBuffer, Rgba};
use predicates::prelude::*;
use rstest::rstest;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::{tempdir, TempDir};

fn create_temp_png(dir: &TempDir, name: &str) -> PathBuf {
    let path = dir.path().join(name);
    let img: ImageBuffer<Rgba<u8>, Vec<u8>> =
        ImageBuffer::from_fn(100, 100, |_x, _y| Rgba([255, 255, 255, 255]));
    img.save(&path).unwrap();
    path
}

fn create_temp_file(dir: &TempDir, name: &str, content: &[u8]) -> PathBuf {
    let path = dir.path().join(name);
    fs::write(&path, content).unwrap();
    path
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
    let dir = tempdir().unwrap();
    let png_path = create_temp_png(&dir, "test.png");
    let txt_content = format!(
        "Hello, integration test! write_meta={write_meta} seed={:?}",
        seed
    )
    .into_bytes();
    let txt_path = create_temp_file(&dir, "test.txt", &txt_content);
    let out_png_path = dir.path().join("out.png");
    let extracted_txt_path = dir.path().join("extracted.txt");

    inject_file_into_png(&txt_path, &png_path, &out_png_path, write_meta, seed);
    let extracted_bytes = extract_file_from_png(
        &out_png_path,
        &extracted_txt_path,
        write_meta,
        Some(txt_content.len()),
        seed,
        true,
    )
    .unwrap();
    assert_eq!(extracted_bytes, txt_content);
}

#[rstest]
#[case(false)] // extraction without seed
#[case(true)] // extraction with wrong seed
fn extract_with_wrong_or_missing_seed_fails(#[case] use_wrong_seed: bool) {
    let seed = Some("myseed");
    let dir = tempdir().unwrap();
    let png_path = create_temp_png(&dir, "test.png");
    let txt_content = b"Negative scenario test".to_vec();
    let txt_path = create_temp_file(&dir, "test.txt", &txt_content);
    let out_png_path = dir.path().join("out.png");
    let fail_txt_path = dir.path().join("fail.txt");

    inject_file_into_png(&txt_path, &png_path, &out_png_path, true, seed);
    let wrong_seed = if use_wrong_seed {
        Some("wrongseed")
    } else {
        None
    };
    extract_file_from_png(
        &out_png_path,
        &fail_txt_path,
        true,
        Some(txt_content.len()),
        wrong_seed,
        false,
    );
}
