#![allow(clippy::unwrap_used)]

use assert_cmd::Command;
use image::open;
use image::ImageBuffer;
use image::Rgba;
use predicates::prelude::*;
use rstest::rstest;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::{tempdir, TempDir};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/keys")
        .join(name)
}

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

fn flip_lsb_at_meta_byte(png: &Path, meta_byte: usize) {
    let mut img = open(png).unwrap().into_rgba8();
    let (_w, h) = img.dimensions();
    // iter_dots iterates (0..w).cartesian_product(0..h) → x outer, y inner.
    // Each metadata byte contributes 8 bits to 8 consecutive channels (RGBA).
    // We flip the LSB of the first bit of the target meta byte.
    let bit_index = meta_byte.checked_mul(8).unwrap();
    let pixel_index = bit_index.checked_div(4).unwrap();
    let channel_in_pixel = bit_index.checked_rem(4).unwrap();
    let pixel_index_u32 = u32::try_from(pixel_index).unwrap();
    let x = pixel_index_u32.checked_div(h).unwrap();
    let y = pixel_index_u32.checked_rem(h).unwrap();
    let pixel = img.get_pixel_mut(x, y);
    let channel = pixel.0.get_mut(channel_in_pixel).unwrap();
    *channel ^= 1;
    img.save(png).unwrap();
}

fn corrupt_metadata_bit(src_png: &Path, dst_png: &Path) {
    let mut img = open(src_png).unwrap().into_rgba8();
    img.get_pixel_mut(0, 0).0[0] ^= 1;
    img.save(dst_png).unwrap();
}

fn corrupt_payload_bit(src_png: &Path, dst_png: &Path) {
    let mut img = open(src_png).unwrap().into_rgba8();
    // The v3 meta header for the default fixture is ~33 bytes = 264 bits =
    // 66 channels = 16.5 pixels. Pixel (1, 0) sits at linear index 100
    // (since iter_dots is x-outer, y-inner with h=100), bit index 400 — well
    // past the meta region and well inside the 2 KB payload bit stream.
    // The choice survives any reasonable v3 meta growth (up to ~50 pixels).
    img.get_pixel_mut(1, 0).0[0] ^= 1;
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

fn inject_file_into_png(
    payload: &Path,
    container: &Path,
    out_png: &Path,
    write_meta: bool,
    seed: Option<&str>,
) {
    let mut cmd = Command::cargo_bin("injet").unwrap();
    cmd.args([
        "inject",
        payload.to_str().unwrap(),
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

/// Checks that inject fails if the payload is too large for the PNG.
///
/// # Parameters
/// - `payload_size = 6000`: Payload is larger than the PNG can hold (100x100 RGBA8 PNG can hold
///   5000 bytes max, so 6000 is always too much).
/// - `payload_size = 5000`: Payload fits exactly into the PNG if there is no meta, but with meta it
///   does not fit. This checks the edge case when meta pushes the total size over the limit.
#[rstest]
#[case(6000)]
#[case(5000)]
fn inject_fails_on_size_limits(#[case] payload_size: usize) {
    let env = setup_env();
    let payload = vec![0u8; payload_size];
    let bin_path = env.dir.path().join("payload_param.bin");
    std::fs::write(&bin_path, &payload).unwrap();
    let mut cmd = Command::cargo_bin("injet").unwrap();
    cmd.args([
        "inject",
        bin_path.to_str().unwrap(),
        env.png_path.to_str().unwrap(),
        "-d",
        env.out_png_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("exceeds container capacity"));
}

type CorruptFn = fn(&Path, &Path);

#[rstest]
#[case::meta(corrupt_metadata_bit, "Invalid metadata signature")]
#[case::payload(corrupt_payload_bit, "Failed to verify hash")]
fn extract_fails_on_corruption(#[case] corrupt_fn: CorruptFn, #[case] expected_error: &str) {
    let env = setup_env();
    inject_file_into_png(&env.bin_path, &env.png_path, &env.out_png_path, true, None);
    let corrupted_png_path = env.dir.path().join("corrupted_param.png");
    corrupt_fn(&env.out_png_path, &corrupted_png_path);
    assert_extract_fails_with_error(&corrupted_png_path, &env.extracted_bin_path, expected_error);
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

/// Layout for v3 meta with default fixture filename `payload.bin` (11 bytes):
///   `sig`:        2 bytes  → offsets 0..2
///   `Size` TLV:   6 bytes  → offsets 2..8   (value at 4..8)
///   `Filename`:   13 bytes → offsets 8..21  (value at 10..21)
///   `Hash` TLV:   6 bytes  → offsets 21..27
///   `MetaHash`:   6 bytes  → offsets 27..33
///   `end`:        2 bytes  → offsets 33..35
const META_OFFSET_INSIDE_SIZE: usize = 5;
const META_OFFSET_INSIDE_FILENAME: usize = 14;

#[rstest]
#[case::size_field(META_OFFSET_INSIDE_SIZE)]
#[case::filename_field(META_OFFSET_INSIDE_FILENAME)]
fn extract_detects_header_corruption(#[case] meta_byte_offset: usize) {
    let env = setup_env();
    inject_file_into_png(&env.bin_path, &env.png_path, &env.out_png_path, true, None);
    flip_lsb_at_meta_byte(&env.out_png_path, meta_byte_offset);

    let mut cmd = Command::cargo_bin("injet").unwrap();
    cmd.args([
        "extract",
        env.out_png_path.to_str().unwrap(),
        "-d",
        env.extracted_bin_path.to_str().unwrap(),
    ]);
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("Metadata header CRC mismatch"));
}

#[test]
fn inject_emits_v3_header_with_meta_hash() {
    let env = setup_env();
    inject_file_into_png(&env.bin_path, &env.png_path, &env.out_png_path, true, None);

    // Inspect must report version 3.
    let mut cmd = Command::cargo_bin("injet").unwrap();
    cmd.args(["inspect", env.out_png_path.to_str().unwrap()]);
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Metadata version: 3"));
}

#[test]
fn inject_accepts_filename_of_exactly_255_bytes() {
    let env = setup_env();
    // 255 ASCII bytes total, including ".bin"
    let stem: String = "a".repeat(251);
    let filename = format!("{stem}.bin");
    assert_eq!(filename.len(), 255);
    let payload_path = env.dir.path().join(&filename);
    std::fs::write(&payload_path, b"hello").unwrap();

    let mut cmd = Command::cargo_bin("injet").unwrap();
    cmd.args([
        "inject",
        payload_path.to_str().unwrap(),
        env.png_path.to_str().unwrap(),
        "-d",
        env.out_png_path.to_str().unwrap(),
    ]);
    cmd.assert().success();

    // Round-trip: extract via -d (metadata-driven default filename is covered by Task 12).
    // The 255-byte filename is still embedded in metadata and parsed end-to-end here.
    let restored = env.dir.path().join("restored.bin");
    let mut extract = Command::cargo_bin("injet").unwrap();
    extract.args([
        "extract",
        env.out_png_path.to_str().unwrap(),
        "-d",
        restored.to_str().unwrap(),
    ]);
    extract.assert().success();

    assert_eq!(std::fs::read(&restored).unwrap(), b"hello");
}

#[rstest]
#[case::meta_yes_size_none(true, None)]
#[case::meta_no_size_some(false, Some(2048))]
#[case::meta_no_size_none(false, None)]
fn extract_read_mode_combinations(#[case] write_meta: bool, #[case] read_size: Option<usize>) {
    let env = setup_env();
    inject_file_into_png(
        &env.bin_path,
        &env.png_path,
        &env.out_png_path,
        write_meta,
        None,
    );
    let bytes = extract_file_from_png(
        &env.out_png_path,
        &env.extracted_bin_path,
        write_meta,
        read_size,
        None,
        true,
    )
    .unwrap();
    // For the meta_no_size_none case, extract reads to EOF and may include extra bytes
    // beyond the payload. Truncate before comparing.
    let truncated_to = read_size.unwrap_or(TEST_PAYLOAD.len()).min(bytes.len());
    let payload_slice = TEST_PAYLOAD
        .get(..truncated_to.min(TEST_PAYLOAD.len()))
        .unwrap();
    let bytes_slice = bytes.get(..truncated_to).unwrap();
    assert_eq!(bytes_slice, payload_slice);
}

#[rstest]
#[case("default")]
#[case("fast")]
#[case("best")]
fn round_trip_with_compression(#[case] level: &str) {
    let env = setup_env();
    let mut cmd = Command::cargo_bin("injet").unwrap();
    cmd.args([
        "inject",
        env.bin_path.to_str().unwrap(),
        env.png_path.to_str().unwrap(),
        "-d",
        env.out_png_path.to_str().unwrap(),
        "--compression",
        level,
    ]);
    cmd.assert().success();

    let bytes = extract_file_from_png(
        &env.out_png_path,
        &env.extracted_bin_path,
        true,
        Some(TEST_PAYLOAD.len()),
        None,
        true,
    )
    .unwrap();
    assert_eq!(bytes, TEST_PAYLOAD);
}

#[test]
fn extract_to_stdout_when_piped() {
    let env = setup_env();
    inject_file_into_png(&env.bin_path, &env.png_path, &env.out_png_path, true, None);
    // assert_cmd captures stdout via Stdio::piped → make_writer's `is_terminal()` returns false → writes to stdout.
    let out = Command::cargo_bin("injet")
        .unwrap()
        .args(["extract", env.out_png_path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(out.stdout.get(..TEST_PAYLOAD.len()).unwrap(), TEST_PAYLOAD);
}

#[test]
fn inject_succeeds_at_exact_capacity_without_meta() {
    let env = setup_env();
    // 100x100 RGBA8 → 5000 bytes max payload, no meta.
    let payload = vec![0xABu8; 5000];
    let bin_path = env.dir.path().join("exact.bin");
    std::fs::write(&bin_path, &payload).unwrap();
    let mut cmd = Command::cargo_bin("injet").unwrap();
    cmd.args([
        "inject",
        bin_path.to_str().unwrap(),
        env.png_path.to_str().unwrap(),
        "-d",
        env.out_png_path.to_str().unwrap(),
        "--write-meta",
        "false",
    ]);
    cmd.assert().success();

    let bytes = extract_file_from_png(
        &env.out_png_path,
        &env.extracted_bin_path,
        false,
        Some(5000),
        None,
        true,
    )
    .unwrap();
    assert_eq!(bytes, payload);
}

#[test]
fn extract_does_not_leave_corrupt_file_on_hash_mismatch() {
    let env = setup_env();
    inject_file_into_png(&env.bin_path, &env.png_path, &env.out_png_path, true, None);
    let corrupted = env.dir.path().join("corrupt.png");
    corrupt_payload_bit(&env.out_png_path, &corrupted);

    let mut cmd = Command::cargo_bin("injet").unwrap();
    cmd.args([
        "extract",
        corrupted.to_str().unwrap(),
        "-d",
        env.extracted_bin_path.to_str().unwrap(),
    ]);
    cmd.assert().failure();

    assert!(
        !env.extracted_bin_path.exists(),
        "extract must remove the partial output on hash mismatch (found leftover at {:?})",
        env.extracted_bin_path
    );
    // Also no .partial file lingering
    let partial = env.extracted_bin_path.with_extension("partial");
    assert!(!partial.exists(), "no .partial file should remain");
}

// ─── Auth helpers ────────────────────────────────────────────────────────────

fn inject_psk(env: &TestEnv, psk_filename: &str) {
    let mut cmd = Command::cargo_bin("injet").unwrap();
    cmd.args([
        "inject",
        env.bin_path.to_str().unwrap(),
        env.png_path.to_str().unwrap(),
        "-d",
        env.out_png_path.to_str().unwrap(),
        "--psk-file",
        fixture(psk_filename).to_str().unwrap(),
    ]);
    cmd.assert().success();
}

fn inject_ed25519(env: &TestEnv, key_filename: &str) {
    let mut cmd = Command::cargo_bin("injet").unwrap();
    cmd.args([
        "inject",
        env.bin_path.to_str().unwrap(),
        env.png_path.to_str().unwrap(),
        "-d",
        env.out_png_path.to_str().unwrap(),
        "--sign-key",
        fixture(key_filename).to_str().unwrap(),
    ]);
    cmd.assert().success();
}

// ─── PSK roundtrip tests ─────────────────────────────────────────────────────

#[test]
fn psk_roundtrip_happy() {
    let env = setup_env();
    let mut inject = Command::cargo_bin("injet").unwrap();
    inject.args([
        "inject",
        env.bin_path.to_str().unwrap(),
        env.png_path.to_str().unwrap(),
        "-d",
        env.out_png_path.to_str().unwrap(),
        "--psk-file",
        fixture("psk.txt").to_str().unwrap(),
    ]);
    inject.assert().success();

    let mut extract = Command::cargo_bin("injet").unwrap();
    extract.args([
        "extract",
        env.out_png_path.to_str().unwrap(),
        "-d",
        env.extracted_bin_path.to_str().unwrap(),
        "--psk-file",
        fixture("psk.txt").to_str().unwrap(),
    ]);
    extract.assert().success();
    assert_eq!(fs::read(&env.extracted_bin_path).unwrap(), TEST_PAYLOAD);
}

#[test]
fn psk_roundtrip_wrong_passphrase() {
    let env = setup_env();
    let mut inject = Command::cargo_bin("injet").unwrap();
    inject.args([
        "inject",
        env.bin_path.to_str().unwrap(),
        env.png_path.to_str().unwrap(),
        "-d",
        env.out_png_path.to_str().unwrap(),
        "--psk-file",
        fixture("psk.txt").to_str().unwrap(),
    ]);
    inject.assert().success();

    let mut extract = Command::cargo_bin("injet").unwrap();
    extract.args([
        "extract",
        env.out_png_path.to_str().unwrap(),
        "-d",
        env.extracted_bin_path.to_str().unwrap(),
        "--psk-file",
        fixture("wrong_psk.txt").to_str().unwrap(),
    ]);
    extract
        .assert()
        .failure()
        .code(2_i32)
        .stderr(predicate::str::contains("verification failed"));
    assert!(!env.extracted_bin_path.exists());
}

#[test]
fn psk_signed_extract_no_key() {
    let env = setup_env();
    inject_psk(&env, "psk.txt");
    let mut extract = Command::cargo_bin("injet").unwrap();
    extract.args([
        "extract",
        env.out_png_path.to_str().unwrap(),
        "-d",
        env.extracted_bin_path.to_str().unwrap(),
    ]);
    extract
        .assert()
        .failure()
        .code(2_i32)
        .stderr(predicate::str::contains("no verification key was provided"));
}

#[test]
fn psk_signed_extract_insecure_skip() {
    let env = setup_env();
    inject_psk(&env, "psk.txt");
    let mut extract = Command::cargo_bin("injet").unwrap();
    extract.args([
        "extract",
        env.out_png_path.to_str().unwrap(),
        "-d",
        env.extracted_bin_path.to_str().unwrap(),
        "--insecure-skip-verify",
    ]);
    extract
        .assert()
        .success()
        .stderr(predicate::str::contains("WARNING:"));
    assert_eq!(fs::read(&env.extracted_bin_path).unwrap(), TEST_PAYLOAD);
}

#[test]
fn psk_roundtrip_tampered_payload() {
    let env = setup_env();
    inject_psk(&env, "psk.txt");
    let corrupted = env.dir.path().join("corrupt.png");
    corrupt_payload_bit(&env.out_png_path, &corrupted);
    let mut extract = Command::cargo_bin("injet").unwrap();
    extract.args([
        "extract",
        corrupted.to_str().unwrap(),
        "-d",
        env.extracted_bin_path.to_str().unwrap(),
        "--psk-file",
        fixture("psk.txt").to_str().unwrap(),
    ]);
    extract.assert().failure(); // CRC32 fires first → HashMismatch (exit 1)
    assert!(!env.extracted_bin_path.exists());
}

// ─── Ed25519 roundtrip tests ──────────────────────────────────────────────────

#[test]
fn ed25519_roundtrip_happy() {
    let env = setup_env();
    inject_ed25519(&env, "signer_ed25519");
    let mut extract = Command::cargo_bin("injet").unwrap();
    extract.args([
        "extract",
        env.out_png_path.to_str().unwrap(),
        "-d",
        env.extracted_bin_path.to_str().unwrap(),
        "--verify-key",
        fixture("signer_ed25519.pub").to_str().unwrap(),
    ]);
    extract.assert().success();
    assert_eq!(fs::read(&env.extracted_bin_path).unwrap(), TEST_PAYLOAD);
}

#[test]
fn ed25519_roundtrip_wrong_pubkey() {
    let env = setup_env();
    inject_ed25519(&env, "signer_ed25519");
    let mut extract = Command::cargo_bin("injet").unwrap();
    extract.args([
        "extract",
        env.out_png_path.to_str().unwrap(),
        "-d",
        env.extracted_bin_path.to_str().unwrap(),
        "--verify-key",
        fixture("wrong_signer_ed25519.pub").to_str().unwrap(),
    ]);
    extract
        .assert()
        .failure()
        .code(2_i32)
        .stderr(predicate::str::contains("verification failed"));
    assert!(!env.extracted_bin_path.exists());
}

#[test]
fn ed25519_signed_extract_no_key() {
    let env = setup_env();
    inject_ed25519(&env, "signer_ed25519");
    let mut extract = Command::cargo_bin("injet").unwrap();
    extract.args([
        "extract",
        env.out_png_path.to_str().unwrap(),
        "-d",
        env.extracted_bin_path.to_str().unwrap(),
    ]);
    extract
        .assert()
        .failure()
        .code(2_i32)
        .stderr(predicate::str::contains("no verification key was provided"));
}

#[test]
fn ed25519_signed_extract_insecure_skip() {
    let env = setup_env();
    inject_ed25519(&env, "signer_ed25519");
    let mut extract = Command::cargo_bin("injet").unwrap();
    extract.args([
        "extract",
        env.out_png_path.to_str().unwrap(),
        "-d",
        env.extracted_bin_path.to_str().unwrap(),
        "--insecure-skip-verify",
    ]);
    extract
        .assert()
        .success()
        .stderr(predicate::str::contains("WARNING:"));
}

#[test]
fn ed25519_signed_extract_with_psk_key_rejected() {
    let env = setup_env();
    inject_ed25519(&env, "signer_ed25519");
    let mut extract = Command::cargo_bin("injet").unwrap();
    extract.args([
        "extract",
        env.out_png_path.to_str().unwrap(),
        "-d",
        env.extracted_bin_path.to_str().unwrap(),
        "--psk-file",
        fixture("psk.txt").to_str().unwrap(),
    ]);
    extract
        .assert()
        .failure()
        .code(2_i32)
        .stderr(predicate::str::contains("ed25519").and(predicate::str::contains("psk")));
}

#[test]
fn ed25519_encrypted_key_with_passphrase_file() {
    let env = setup_env();
    let passphrase_file = env.dir.path().join("passphrase.txt");
    std::fs::write(&passphrase_file, b"test-passphrase\n").unwrap();
    let mut inject = Command::cargo_bin("injet").unwrap();
    inject.args([
        "inject",
        env.bin_path.to_str().unwrap(),
        env.png_path.to_str().unwrap(),
        "-d",
        env.out_png_path.to_str().unwrap(),
        "--sign-key",
        fixture("signer_ed25519_encrypted").to_str().unwrap(),
        "--sign-key-passphrase-file",
        passphrase_file.to_str().unwrap(),
    ]);
    inject.assert().success();

    let mut extract = Command::cargo_bin("injet").unwrap();
    extract.args([
        "extract",
        env.out_png_path.to_str().unwrap(),
        "-d",
        env.extracted_bin_path.to_str().unwrap(),
        "--verify-key",
        fixture("signer_ed25519_encrypted.pub").to_str().unwrap(),
    ]);
    extract.assert().success();
    assert_eq!(fs::read(&env.extracted_bin_path).unwrap(), TEST_PAYLOAD);
}

#[test]
fn ed25519_encrypted_key_wrong_passphrase() {
    let env = setup_env();
    let bad_pass = env.dir.path().join("bad.txt");
    std::fs::write(&bad_pass, b"wrong\n").unwrap();
    let mut inject = Command::cargo_bin("injet").unwrap();
    inject.args([
        "inject",
        env.bin_path.to_str().unwrap(),
        env.png_path.to_str().unwrap(),
        "-d",
        env.out_png_path.to_str().unwrap(),
        "--sign-key",
        fixture("signer_ed25519_encrypted").to_str().unwrap(),
        "--sign-key-passphrase-file",
        bad_pass.to_str().unwrap(),
    ]);
    inject
        .assert()
        .failure()
        .code(2_i32)
        .stderr(predicate::str::contains("decrypt"));
}

// ─── Cross-cutting negatives ──────────────────────────────────────────────────

#[test]
fn unsigned_extract_with_verify_key_rejected() {
    let env = setup_env();
    inject_file_into_png(&env.bin_path, &env.png_path, &env.out_png_path, true, None);
    let mut extract = Command::cargo_bin("injet").unwrap();
    extract.args([
        "extract",
        env.out_png_path.to_str().unwrap(),
        "-d",
        env.extracted_bin_path.to_str().unwrap(),
        "--verify-key",
        fixture("signer_ed25519.pub").to_str().unwrap(),
    ]);
    extract
        .assert()
        .failure()
        .code(2_i32)
        .stderr(predicate::str::contains("not signed"));
}

#[test]
fn inject_two_auth_modes_rejected_by_clap() {
    let env = setup_env();
    let mut cmd = Command::cargo_bin("injet").unwrap();
    cmd.args([
        "inject",
        env.bin_path.to_str().unwrap(),
        env.png_path.to_str().unwrap(),
        "-d",
        env.out_png_path.to_str().unwrap(),
        "--psk-file",
        fixture("psk.txt").to_str().unwrap(),
        "--sign-key",
        fixture("signer_ed25519").to_str().unwrap(),
    ]);
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("cannot be used with"));
}

#[test]
fn inject_psk_with_no_meta_rejected() {
    let env = setup_env();
    let mut cmd = Command::cargo_bin("injet").unwrap();
    cmd.args([
        "inject",
        env.bin_path.to_str().unwrap(),
        env.png_path.to_str().unwrap(),
        "-d",
        env.out_png_path.to_str().unwrap(),
        "--write-meta",
        "false",
        "--psk-file",
        fixture("psk.txt").to_str().unwrap(),
    ]);
    cmd.assert()
        .failure()
        .code(2_i32)
        .stderr(predicate::str::contains("Signing requires metadata"));
}

#[test]
fn inject_sign_key_with_no_meta_rejected() {
    let env = setup_env();
    let mut cmd = Command::cargo_bin("injet").unwrap();
    cmd.args([
        "inject",
        env.bin_path.to_str().unwrap(),
        env.png_path.to_str().unwrap(),
        "-d",
        env.out_png_path.to_str().unwrap(),
        "--write-meta",
        "false",
        "--sign-key",
        fixture("signer_ed25519").to_str().unwrap(),
    ]);
    cmd.assert()
        .failure()
        .code(2_i32)
        .stderr(predicate::str::contains("Signing requires metadata"));
}

// ─── Per-container distinct salts ────────────────────────────────────────────

#[test]
fn psk_two_containers_have_distinct_salts() {
    let env = setup_env();
    let png_a = env.dir.path().join("a.png");
    let png_b = env.dir.path().join("b.png");

    for out in [&png_a, &png_b] {
        let mut cmd = Command::cargo_bin("injet").unwrap();
        cmd.args([
            "inject",
            env.bin_path.to_str().unwrap(),
            env.png_path.to_str().unwrap(),
            "-d",
            out.to_str().unwrap(),
            "--psk-file",
            fixture("psk.txt").to_str().unwrap(),
        ]);
        cmd.assert().success();
    }

    let salt_a = read_salt_from_png(&png_a);
    let salt_b = read_salt_from_png(&png_b);
    assert_ne!(salt_a, salt_b, "two PSK containers must have distinct salts");
}

fn read_salt_from_png(png_path: &Path) -> [u8; 16] {
    use image::GenericImageView;
    use injet::lsb::gen_dots;
    use injet::meta::{Meta, MetaField};
    use itertools::Itertools;

    let img = image::open(png_path).unwrap();
    let (w, h) = img.dimensions();
    let bytes = gen_dots(w, h, None)
        .flat_map(|(x, y)| img.get_pixel(x, y).0)
        .map(|v| v & 1)
        .chunks(8);
    let mut content = bytes.into_iter().map(|chunk| {
        chunk
            .zip((0_u8..8).rev())
            .map(|(bit, shift)| bit << shift)
            .sum::<u8>()
    });
    let meta = Meta::read(&mut content).unwrap();
    meta.fields
        .iter()
        .find_map(|f| match f {
            MetaField::Mac { salt, .. } => Some(*salt),
            _ => None,
        })
        .unwrap()
}
