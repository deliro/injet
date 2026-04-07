use assert_cmd::Command;
use image::{ImageBuffer, Rgba};
use rstest::rstest;
use std::path::PathBuf;
use tempfile::tempdir;

fn run_inspect(args: &[&str]) -> String {
    let out = Command::cargo_bin("injet")
        .unwrap()
        .args(args)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    // Strip the "Image file:" line — it embeds a temp directory path that varies between runs.
    stdout
        .lines()
        .filter(|l| !l.starts_with("Image file:"))
        .collect::<Vec<_>>()
        .join("\n")
}

#[rstest]
#[case::empty_png("empty")]
#[case::with_meta("with_meta")]
#[case::with_seed("with_seed")]
fn inspect_output(#[case] case: &str) {
    let dir = tempdir().unwrap();
    let png: PathBuf = dir.path().join("c.png");
    let img: ImageBuffer<Rgba<u8>, Vec<u8>> =
        ImageBuffer::from_fn(80, 80, |_, _| Rgba([255, 255, 255, 255]));
    img.save(&png).unwrap();

    if case != "empty" {
        let cargo_path = dir.path().join("hi.bin");
        std::fs::write(&cargo_path, b"hello world").unwrap();
        let mut inj = vec![
            "inject".to_string(),
            cargo_path.to_string_lossy().into_owned(),
            png.to_string_lossy().into_owned(),
            "-d".into(),
            png.to_string_lossy().into_owned(),
        ];
        if case == "with_seed" {
            inj.push("--seed".into());
            inj.push("s".into());
        }
        let inj_refs: Vec<&str> = inj.iter().map(|s| s.as_str()).collect();
        Command::cargo_bin("injet")
            .unwrap()
            .args(&inj_refs)
            .assert()
            .success();
    }

    let mut args: Vec<String> = vec!["inspect".into(), png.to_string_lossy().into_owned()];
    if case == "with_seed" {
        args.push("--seed".into());
        args.push("s".into());
    }
    let arg_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    let stdout = run_inspect(&arg_refs);
    insta::assert_snapshot!(case, stdout);
}
