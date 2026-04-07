# Injet Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix correctness bugs in TLV metadata parser and CLI argument handling, extend metadata integrity to cover the header (breaking format change), close test coverage gaps, then improve performance and structure. Bump to `1.0.0`.

**Architecture:** Six phases. P0 fixes pure bugs strictly via TDD (red → green → refactor → green). P1 introduces a new metadata version `VERSION_3` with header CRC — this is the only wire-format breaking change and the reason for the major bump. P2 closes test coverage gaps. P3 cleans up correctness polish (unwraps, lossy UTF-8, atomic write, etc.). P4 is performance (`pseudo_shuffle_coords` memory, `Box<dyn>`, double file reads). P5 is structural refactor (module split, snapshot tests). P6 is the version bump and release notes. Each task in P0/P1/P3 follows strict TDD.

**Tech stack:** Rust 2021, `image 0.25`, `blake3`, `crc32fast`, `clap 4`, `rstest 0.18`, `assert_cmd 2`, `predicates 3`, `tempfile 3`. Adds: `insta` for snapshot tests.

**Conventions for every task in this plan:**
- Bugs: red test first, run and confirm failure, minimal fix, run and confirm pass, refactor, re-run, commit.
- Commit messages follow Conventional Commits (`fix:`, `feat:`, `refactor:`, `test:`, `perf:`, `chore:`).
- Never use `unwrap`/`expect` in production code unless documented as a checked invariant.
- After every task, run the full suite: `cargo test --all` and `cargo clippy --all-targets -- -D warnings`.

---

## Phase 0 — Critical bug fixes (TDD)

These three are real defects that exist in `master` today. They are P0 because they affect correctness on currently-supported code paths and are cheap to fix.

---

### Task 1: Bug — `from_tlv_field` desyncs the byte stream on unknown tags

**Problem.** `MetaField::from_tlv_field` (`src/main.rs:216-257`) reads the tag and length, then if the tag is unknown returns `Skip` **without consuming the value bytes**. The next call interprets value bytes as a new tag → parser desyncs and the rest of the metadata is unreadable. This kills forward compatibility — adding any new TLV field in future will break old readers in unrecoverable ways even though TLV's whole point is graceful skipping.

**Files:**
- Modify: `src/main.rs:216-257` (`MetaField::from_tlv_field`)
- Test: `src/main.rs` `#[cfg(test)] mod tests` block (around `src/main.rs:691-734`)

**Steps:**

- [ ] **Step 1: Write the failing unit test**

Add to `src/main.rs` `mod tests`:

```rust
#[test]
fn test_meta_v2_skips_unknown_tlv_and_keeps_parsing() {
    use crate::{MAGIC, VERSION_2};
    // Build a v2 meta byte stream:
    //   signature(v2) | Size(4) | UnknownTag=0x7F len=5 + 5 bytes | Filename "x.zip" | end
    let mut bytes = Vec::new();
    let signature = (MAGIC << 3) | (VERSION_2 as u16);
    bytes.extend(signature.to_le_bytes());

    // Size = 1234
    bytes.push(1); // tag Size
    bytes.push(4); // len
    bytes.extend(1234u32.to_le_bytes());

    // Unknown tag 0x7F with 5-byte payload
    bytes.push(0x7F);
    bytes.push(5);
    bytes.extend([0xAA, 0xBB, 0xCC, 0xDD, 0xEE]);

    // Filename "x.zip"
    bytes.push(2); // tag Filename
    bytes.push(5); // len
    bytes.extend(b"x.zip");

    // end marker
    bytes.push(0);
    bytes.push(0);

    let mut iter = bytes.into_iter();
    let meta = Meta::read(&mut iter).expect("meta should parse despite unknown tag");
    assert_eq!(meta.size(), Some(1234), "Size field must survive unknown tag");
    assert_eq!(
        meta.filename(),
        Some("x.zip"),
        "Filename after unknown tag must still be parsed"
    );
    assert_eq!(iter.next(), None, "stream must be fully consumed");
}
```

- [ ] **Step 2: Run the test, confirm failure**

```bash
cargo test test_meta_v2_skips_unknown_tlv_and_keeps_parsing -- --nocapture
```

Expected: FAIL. The assertion on `filename()` returns `None` because the parser desynced after the unknown tag.

- [ ] **Step 3: Minimal fix**

Replace `from_tlv_field` body so the **value bytes are always consumed**, regardless of whether the tag is known. Read `actual_len` (handling the `0x00` extended-length escape) **before** matching on the tag:

```rust
pub fn from_tlv_field<T: Iterator<Item = u8>>(
    iter: &mut T,
) -> Result<MetaFieldParseResult, MetaError> {
    let tag_byte = iter.next().ok_or(MetaError::NoBytes)?;
    let len = iter.next().ok_or(MetaError::NoBytes)?;
    if tag_byte == 0 && len == 0 {
        return Ok(MetaFieldParseResult::End);
    }
    let actual_len = if len == 0x00 {
        let l = [
            iter.next().ok_or(MetaError::NoBytes)?,
            iter.next().ok_or(MetaError::NoBytes)?,
        ];
        u16::from_le_bytes(l) as usize
    } else {
        len as usize
    };
    let bytes: Vec<u8> = iter.take(actual_len).collect();
    if bytes.len() != actual_len {
        return Err(MetaError::NoBytes);
    }
    let tag = match MetaTag::try_from(tag_byte) {
        Ok(t) => t,
        Err(_) => return Ok(MetaFieldParseResult::Skip),
    };
    let field = match tag {
        MetaTag::Size if bytes.len() == 4 => Some(MetaField::Size(u32::from_le_bytes(
            bytes.try_into().expect("checked length == 4"),
        ))),
        MetaTag::Filename => Some(MetaField::Filename(
            String::from_utf8_lossy(&bytes).to_string(),
        )),
        MetaTag::Hash if bytes.len() == 4 => Some(MetaField::Hash(u32::from_le_bytes(
            bytes.try_into().expect("checked length == 4"),
        ))),
        _ => None,
    };
    Ok(match field {
        Some(f) => MetaFieldParseResult::Field(f),
        None => MetaFieldParseResult::Skip,
    })
}
```

- [ ] **Step 4: Run the test, confirm green**

```bash
cargo test test_meta_v2_skips_unknown_tlv_and_keeps_parsing -- --nocapture
cargo test
```

Both must pass. The pre-existing `test_meta_v2_roundtrip` and `test_meta_v1_parsing` must still pass.

- [ ] **Step 5: Refactor**

Extract a small helper to read the TLV length to keep `from_tlv_field` short and reusable for future fields. Add at the top of the `impl MetaField` block (or as a free function inside the module):

```rust
fn read_tlv_len<T: Iterator<Item = u8>>(
    iter: &mut T,
    first_len_byte: u8,
) -> Result<usize, MetaError> {
    if first_len_byte != 0 {
        return Ok(first_len_byte as usize);
    }
    let hi = iter.next().ok_or(MetaError::NoBytes)?;
    let lo = iter.next().ok_or(MetaError::NoBytes)?;
    Ok(u16::from_le_bytes([hi, lo]) as usize)
}
```

Then in `from_tlv_field`, replace the inline length-reading block with `let actual_len = read_tlv_len(iter, len)?;`. Re-run all tests.

- [ ] **Step 6: Re-run full suite, then commit**

```bash
cargo test
cargo clippy --all-targets -- -D warnings
git add src/main.rs
git commit -m "fix: skip unknown TLV tags without desyncing meta parser"
```

---

### Task 2: Bug — off-by-one in filename length check

**Problem.** `inject` rejects 255-byte filenames at `src/main.rs:614`:

```rust
if v.len() >= 255 { return Err(InjectError::FilenameOverflow); }
```

255 fits in `u8` (the TLV length field). The error message says "maximum 255 bytes" but the code enforces 254. Off-by-one.

**Files:**
- Modify: `src/main.rs:612-616`
- Test: `tests/integration_inject_extract.rs`

**Steps:**

- [ ] **Step 1: Write the failing integration test**

Append to `tests/integration_inject_extract.rs`:

```rust
#[test]
fn inject_accepts_filename_of_exactly_255_bytes() {
    let env = setup_env();
    // 255 ASCII bytes total, including ".bin"
    let stem: String = "a".repeat(251);
    let filename = format!("{stem}.bin");
    assert_eq!(filename.len(), 255);
    let cargo_path = env.dir.path().join(&filename);
    std::fs::write(&cargo_path, b"hello").unwrap();

    let mut cmd = Command::cargo_bin("injet").unwrap();
    cmd.args([
        "inject",
        cargo_path.to_str().unwrap(),
        env.png_path.to_str().unwrap(),
        "-d",
        env.out_png_path.to_str().unwrap(),
    ]);
    cmd.assert().success();

    // Round-trip: extract should restore the same filename via metadata
    let extracted_dir = env.dir.path().join("out");
    std::fs::create_dir(&extracted_dir).unwrap();
    let mut extract = Command::cargo_bin("injet").unwrap();
    extract
        .current_dir(&extracted_dir)
        .args(["extract", env.out_png_path.to_str().unwrap()]);
    extract.assert().success();

    let restored = extracted_dir.join(&filename);
    assert!(
        restored.exists(),
        "extracted file with 255-byte name should exist at {restored:?}"
    );
    assert_eq!(std::fs::read(&restored).unwrap(), b"hello");
}
```

- [ ] **Step 2: Run, confirm failure**

```bash
cargo test --test integration_inject_extract inject_accepts_filename_of_exactly_255_bytes
```

Expected: FAIL with stderr containing "Filename is too long".

- [ ] **Step 3: Minimal fix**

In `src/main.rs:612-616`:

```rust
const MAX_FILENAME_LEN: usize = 255;
// ...
if v.len() > MAX_FILENAME_LEN {
    return Err(InjectError::FilenameOverflow);
}
```

Update the error message to use the constant:

```rust
#[error("Filename is too long (maximum {} bytes)", MAX_FILENAME_LEN)]
FilenameOverflow,
```

(`thiserror` supports interpolating constants via the `{}` form when used as `#[error("... {} ...", CONST)]`.)

- [ ] **Step 4: Run, confirm green**

```bash
cargo test --test integration_inject_extract inject_accepts_filename_of_exactly_255_bytes
cargo test
```

- [ ] **Step 5: Refactor**

Move `MAX_FILENAME_LEN` next to the other constants near the top of the file (around `src/main.rs:125-129`). Re-run `cargo test` and `cargo clippy --all-targets -- -D warnings`.

- [ ] **Step 6: Commit**

```bash
git add src/main.rs tests/integration_inject_extract.rs
git commit -m "fix: allow filenames of exactly 255 bytes (off-by-one)"
```

---

### Task 3: Bug — `inspect` panics on container paths without a file name

**Problem.** `src/main.rs:480`:

```rust
let filename = args.path.file_name().unwrap().to_string_lossy().to_string();
```

`Path::file_name` returns `None` for paths ending in `..` or `/`. Hitting this path with `injet inspect ..` panics in production code, which violates the project rules (no `unwrap` in library code, no panics).

**Files:**
- Modify: `src/main.rs:480` (`inspect`)
- Modify: `src/main.rs:401-409` (`InspectError`)
- Test: `tests/integration_inject_extract.rs`

**Steps:**

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn inspect_does_not_panic_on_dot_dot_path() {
    let mut cmd = Command::cargo_bin("injet").unwrap();
    cmd.args(["inspect", ".."]);
    // It must NOT panic. It must exit non-zero with a clean error.
    let assert = cmd.assert().failure();
    let stderr = String::from_utf8_lossy(&assert.get_output().stderr).to_string();
    assert!(
        !stderr.contains("panicked"),
        "inspect must not panic, got stderr: {stderr}"
    );
}
```

- [ ] **Step 2: Run, confirm failure**

```bash
cargo test --test integration_inject_extract inspect_does_not_panic_on_dot_dot_path
```

Expected: FAIL — `inspect` panics with "called `Option::unwrap()` on a `None` value" when `.is_file()` short-circuits to true on `..`. (If `..` is a directory and `.is_file()` returns false first, the test still fails because we then look at `NotAFile` — re-check; either way, the panic must not happen for *any* pathological path. Use a path guaranteed to be a directory: `..` is a directory, so `is_file` returns false → returns `NotAFile`. The panic on `file_name().unwrap()` only triggers after the file checks pass. Replace the test path with an actual existing directory disguised as a file. Better: test `inspect /` which is a directory — same path. The real panic site is `args.path.file_name().unwrap()` on a path like `/`. Move the failure-trigger path:)

Replace the failing path with one that bypasses the `is_file` check by being a regular file with no usable file_name. Since `Path::file_name` only returns `None` for paths ending in `..` or with no last component, and no regular file can have such a path on disk, the realistic trigger is when **`is_file` returns true but `file_name` returns None** — which happens only via symlinks or odd filesystems. To still produce a deterministic test, lower-level: call the conversion in isolation. Drop the integration test idea — switch to a unit test of a pure helper:

**Revised Step 1 (replace the integration test above):**

Add a pure helper and unit test it. Add to `src/main.rs`:

```rust
fn display_path(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string_lossy().into_owned())
}
```

Add to `mod tests`:

```rust
#[test]
fn test_display_path_handles_dot_dot() {
    use std::path::Path;
    use crate::display_path;
    // `..` has no file_name component — must not panic, must produce something printable.
    let s = display_path(Path::new(".."));
    assert!(!s.is_empty());
}
```

**Revised Step 2:** Run the test. It compiles (helper exists) but to make it fail first, write the test against a `display_path` that doesn't exist yet — i.e., add the test before adding the helper. So:

  1. Add only the test, run `cargo test test_display_path_handles_dot_dot` → FAIL with "cannot find function `display_path`".

- [ ] **Step 3: Minimal fix**

Add the `display_path` helper above and replace `src/main.rs:480` with:

```rust
let filename = display_path(&args.path);
```

- [ ] **Step 4: Run, confirm green**

```bash
cargo test test_display_path_handles_dot_dot
cargo test
```

- [ ] **Step 5: Refactor**

`extract` and `inject` also call `cargo.file_name().map(...)` patterns that can produce empty strings. Audit and reuse `display_path` where appropriate. Re-run tests.

- [ ] **Step 6: Commit**

```bash
git add src/main.rs
git commit -m "fix: avoid panic on paths without file_name in inspect"
```

---

## Phase 1 — Wire-format breaking change: header CRC (`VERSION_3`)

This is the only intentionally breaking change in this plan. It is the reason for the major version bump. The current `Hash` field covers payload only — flipped bits inside `Size` or `Filename` are not directly detected. We add a new metadata version that protects the header itself.

### Format design

`VERSION_3` layout:

```
[ signature(2B, magic|version=3) ]
[ TLV fields, last one MUST be MetaHash(4B CRC32) ]
[ end marker (0x00 0x00) ]
[ payload bits ]
```

`MetaHash` (`MetaTag = 4`) is a CRC32 covering **all preceding bytes** of the meta header — i.e. signature + every TLV byte written before the `MetaHash` field itself, **not including** the `MetaHash` tag/length/value or the end marker. Payload integrity continues to be covered by the existing `Hash` tag (CRC32 of the payload bytes), which is still emitted.

Forward compat: thanks to Task 1, an older v2 reader pointed at a v3 file would *fail at the version check* because we are bumping the major version. This is the intentional break.

Backward compat: v3 readers must still parse v1 and v2 (without header CRC verification).

### Task 4: Wire `VERSION_3` constant and serialization without verification

**Files:**
- Modify: `src/main.rs:16-18` (constants)
- Modify: `src/main.rs:441-469` (`meta_tag_enum!` — add `MetaHash = 4`)
- Modify: `src/main.rs:179-302` (`MetaField`, `MetaFieldParseResult`)
- Modify: `src/main.rs:304-387` (`Meta`)
- Test: `src/main.rs` `mod tests`

**Steps:**

- [ ] **Step 1: Failing test — round-trip with `VERSION_3` and `MetaHash`**

In `mod tests`:

```rust
#[test]
fn test_meta_v3_roundtrip_with_header_hash() {
    let meta = Meta::make_v3(
        Some(4242),
        Some("hello.bin".to_string()),
        Some(0xDEADBEEF),
    );
    assert_eq!(meta.version, 3);
    let bytes = meta.to_bytes();
    let mut iter = bytes.into_iter();
    let parsed = Meta::read(&mut iter).expect("v3 must parse");
    assert_eq!(parsed.version, 3);
    assert_eq!(parsed.size(), Some(4242));
    assert_eq!(parsed.filename(), Some("hello.bin"));
    assert_eq!(parsed.hash(), Some(0xDEADBEEF));
    assert!(parsed.meta_hash().is_some(), "v3 must carry header hash");
    assert_eq!(iter.next(), None);
}
```

- [ ] **Step 2: Run, confirm failure**

```bash
cargo test test_meta_v3_roundtrip_with_header_hash
```

Expected: FAIL (functions `make_v3`, `meta_hash` do not exist).

- [ ] **Step 3: Implement minimal v3 support**

Add to constants:

```rust
const VERSION_3: u8 = 3;
```

Extend `meta_tag_enum!`:

```rust
meta_tag_enum! {
    Size = 1,
    Filename = 2,
    Hash = 3,
    MetaHash = 4,
}
```

Extend `MetaField`:

```rust
pub enum MetaField {
    Size(u32),
    Filename(String),
    Hash(u32),
    MetaHash(u32),
}
```

Update `tag()`, `to_bytes()`, `as_*` accessors, and `from_tlv_field`'s match to add the `MetaHash` arm (parses like `Hash`). Add accessor:

```rust
pub fn as_meta_hash(&self) -> Option<u32> {
    if let MetaField::MetaHash(h) = self { Some(*h) } else { None }
}
```

Add to `Meta`:

```rust
pub fn meta_hash(&self) -> Option<u32> {
    self.fields.iter().find_map(|f| f.as_meta_hash())
}

pub fn make_v3(size: Option<u32>, filename: Option<String>, hash: Option<u32>) -> Self {
    let mut fields = Vec::new();
    if let Some(size) = size { fields.push(MetaField::Size(size)); }
    if let Some(filename) = filename { fields.push(MetaField::Filename(filename)); }
    if let Some(hash) = hash { fields.push(MetaField::Hash(hash)); }
    // Header CRC will be computed in to_bytes; placeholder here:
    fields.push(MetaField::MetaHash(0));
    Self { version: VERSION_3, fields }
}
```

Modify `Meta::to_bytes` so that for `VERSION_3` it:
1. Writes the v3 signature.
2. Serializes all fields **except** the trailing `MetaHash`.
3. Computes CRC32 over the bytes written so far.
4. Serializes the `MetaHash` field with the computed CRC.
5. Writes the end marker.

```rust
pub fn to_bytes(&self) -> Vec<u8> {
    let signature_bits = match self.version {
        VERSION_3 => (MAGIC << 3) | (VERSION_3 as u16),
        _ => (MAGIC << 3) | (VERSION_2 as u16),
    };
    let mut result = Vec::with_capacity(64);
    result.extend(signature_bits.to_le_bytes());
    if self.version == VERSION_3 {
        for field in self.fields.iter().filter(|f| f.as_meta_hash().is_none()) {
            result.extend(field.to_bytes());
        }
        let crc = crc32fast::hash(&result);
        result.extend(MetaField::MetaHash(crc).to_bytes());
    } else {
        for field in &self.fields {
            result.extend(field.to_bytes());
        }
    }
    result.push(0);
    result.push(0);
    result
}
```

Extend `Meta::read` to accept `VERSION_3` (parses identically to v2 — verification is a separate step in Task 5):

```rust
VERSION_3 => {
    let mut fields = Vec::new();
    loop {
        match MetaField::from_tlv_field(value)? {
            MetaFieldParseResult::Field(field) => fields.push(field),
            MetaFieldParseResult::End => break,
            MetaFieldParseResult::Skip => continue,
        }
    }
    Ok(Meta { version, fields })
}
```

- [ ] **Step 4: Run, confirm green**

```bash
cargo test test_meta_v3_roundtrip_with_header_hash
cargo test
```

- [ ] **Step 5: Refactor**

Pull the field-serialization loop out so v2 and v3 share it:

```rust
fn write_fields(&self, buf: &mut Vec<u8>, skip_meta_hash: bool) {
    for field in &self.fields {
        if skip_meta_hash && field.as_meta_hash().is_some() { continue; }
        buf.extend(field.to_bytes());
    }
}
```

Re-run tests.

- [ ] **Step 6: Commit**

```bash
git add src/main.rs
git commit -m "feat: introduce VERSION_3 with MetaHash field (no verification yet)"
```

---

### Task 5: Verify the header CRC on read for `VERSION_3`

**Files:**
- Modify: `src/main.rs:344-376` (`Meta::read`)
- Modify: `src/main.rs:389-399` (`MetaError`)
- Test: `src/main.rs` `mod tests`

**Steps:**

- [ ] **Step 1: Failing test — flipping a header byte must produce `MetaError::HeaderHashMismatch`**

```rust
#[test]
fn test_meta_v3_header_hash_mismatch_detected() {
    let meta = Meta::make_v3(Some(123), Some("a.bin".into()), Some(0));
    let mut bytes = meta.to_bytes();
    // Flip one bit somewhere inside the Filename field (which is BEFORE MetaHash).
    // Signature is 2 bytes, then Size TLV is 6 bytes, so byte 8 lands inside Filename.
    bytes[10] ^= 0x01;
    let mut iter = bytes.into_iter();
    let err = Meta::read(&mut iter).expect_err("must detect header tampering");
    assert!(
        matches!(err, MetaError::HeaderHashMismatch),
        "expected HeaderHashMismatch, got {err:?}"
    );
}
```

- [ ] **Step 2: Run, confirm failure**

```bash
cargo test test_meta_v3_header_hash_mismatch_detected
```

Expected: FAIL — `MetaError::HeaderHashMismatch` does not exist; meta parses as if nothing happened.

- [ ] **Step 3: Add the error variant and verification**

```rust
#[derive(Debug, Error)]
pub enum MetaError {
    #[error("Insufficient bytes to parse metadata")]
    NoBytes,
    #[error("Invalid metadata signature")]
    SignatureMismatch,
    #[error("Unsupported metadata version: {0}")]
    UnsupportedVersion(u8),
    #[error("Invalid or corrupted filename in metadata")]
    MalformedFilename,
    #[error("Metadata header CRC mismatch")]
    HeaderHashMismatch,
}
```

Verification approach: while parsing v3, accumulate every byte read into a buffer up to (but not including) the `MetaHash` TLV. When the `MetaHash` field arrives, compute `crc32(buffer)` and compare to the field value.

Replace the v3 branch in `Meta::read` with a version that wraps the iterator in a buffering adapter. The simplest implementation reads the entire meta region byte-by-byte through a counting iterator. Since the existing `Meta::read` consumes from `Iterator<Item = u8>`, introduce a small wrapper:

```rust
struct Tee<'a, I: Iterator<Item = u8>> {
    inner: &'a mut I,
    buf: Vec<u8>,
    capture: bool,
}

impl<'a, I: Iterator<Item = u8>> Iterator for Tee<'a, I> {
    type Item = u8;
    fn next(&mut self) -> Option<u8> {
        let b = self.inner.next()?;
        if self.capture {
            self.buf.push(b);
        }
        Some(b)
    }
}
```

In `Meta::read`, when the version is `VERSION_3`:

1. Push the two signature bytes that were already consumed onto a `Vec<u8>` named `header_bytes`.
2. Wrap `value` in a `Tee` that appends every byte it returns to `header_bytes` while `capture` is true.
3. Loop reading TLV fields. When a `MetaHash` field is encountered: stop capturing **before** reading its tag byte. This means: read fields one by one, but for each field, peek ahead at the next byte before passing it to `from_tlv_field`. If it is the `MetaHash` tag, set `capture = false`, then call `from_tlv_field`.
4. After the loop, compute `crc32fast::hash(&header_bytes)` and compare to the parsed `MetaHash` value.

A cleaner alternative is to pre-buffer the entire header (signature + everything until end marker) into a `Vec<u8>`, then run two passes: one to locate the `MetaHash` field's bytes, one to verify and parse. Pseudocode:

```rust
VERSION_3 => {
    // Buffer everything until end marker (0x00, 0x00).
    let mut header = Vec::with_capacity(64);
    header.extend(sig_bytes);
    loop {
        let b = value.next().ok_or(MetaError::NoBytes)?;
        header.push(b);
        if header.len() >= 4
            && header[header.len() - 2] == 0
            && header[header.len() - 1] == 0
        {
            break;
        }
    }
    // Strip the end marker for CRC computation.
    let body = &header[..header.len() - 2];

    // Find MetaHash TLV: scan TLVs from offset 2 (after signature).
    let mut offset = 2usize;
    let mut meta_hash_pos: Option<usize> = None;
    while offset < body.len() {
        let tag = body[offset];
        let len = body[offset + 1];
        let (len_bytes, value_len) = if len == 0 {
            let l = u16::from_le_bytes([body[offset + 2], body[offset + 3]]) as usize;
            (2, l)
        } else {
            (0, len as usize)
        };
        if tag == u8::from(MetaTag::MetaHash) {
            meta_hash_pos = Some(offset);
            break;
        }
        offset += 2 + len_bytes + value_len;
    }
    let meta_hash_pos = meta_hash_pos.ok_or(MetaError::HeaderHashMismatch)?;
    let to_hash = &body[..meta_hash_pos];
    let expected = u32::from_le_bytes([
        body[meta_hash_pos + 2],
        body[meta_hash_pos + 3],
        body[meta_hash_pos + 4],
        body[meta_hash_pos + 5],
    ]);
    if crc32fast::hash(to_hash) != expected {
        return Err(MetaError::HeaderHashMismatch);
    }

    // Now parse fields from the buffered body.
    let mut iter = body[2..].iter().copied();
    let mut fields = Vec::new();
    loop {
        match MetaField::from_tlv_field(&mut iter)? {
            MetaFieldParseResult::Field(field) => fields.push(field),
            MetaFieldParseResult::End => break,
            MetaFieldParseResult::Skip => continue,
        }
    }
    // Note: end marker is not in body — adjust the loop or push 0,0 sentinel.
    Ok(Meta { version, fields })
}
```

Note: the existing `from_tlv_field` reads the end marker via `tag_byte == 0 && len == 0`. Since we trimmed the end marker, push it back before parsing:

```rust
let mut parse_buf: Vec<u8> = body[2..].to_vec();
parse_buf.push(0);
parse_buf.push(0);
let mut iter = parse_buf.into_iter();
```

This is messier than the `Tee` approach but avoids a custom iterator type. Pick one and document the choice in a comment.

- [ ] **Step 4: Run, confirm green**

```bash
cargo test test_meta_v3_header_hash_mismatch_detected
cargo test
```

- [ ] **Step 5: Refactor**

If you went with the buffer-and-scan approach, extract a helper `fn locate_meta_hash(body: &[u8]) -> Option<usize>`. If you went with `Tee`, extract it to a tiny module-private struct. Re-run tests.

- [ ] **Step 6: Commit**

```bash
git add src/main.rs
git commit -m "feat: verify VERSION_3 header CRC on read"
```

---

### Task 6: `inject` writes `VERSION_3` by default; `extract` consumes `MetaHash`

**Files:**
- Modify: `src/main.rs:595-676` (`inject`)
- Modify: `src/main.rs:528-593` (`extract`)
- Modify: `src/main.rs:471-515` (`inspect`)

**Steps:**

- [ ] **Step 1: Failing integration test — round-trip with v3 header CRC**

In `tests/integration_inject_extract.rs`:

```rust
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
```

- [ ] **Step 2: Run, confirm failure**

```bash
cargo test --test integration_inject_extract inject_emits_v3_header_with_meta_hash
```

Expected: FAIL — `inject` still emits v2.

- [ ] **Step 3: Switch `inject` to `Meta::make_v3`**

In `inject` at `src/main.rs:630`:

```rust
let meta = Meta::make_v3(Some(cargo_size), filename, Some(hash));
```

`extract` and `inspect` need no changes — they read whatever version is on disk via `Meta::read`, and Task 5 handles v3 verification.

- [ ] **Step 4: Run, confirm green**

```bash
cargo test
```

- [ ] **Step 5: Refactor**

Add an `inspect` line that prints `MetaHash: <hex>` when present, parallel to the existing CRC32 line. Keep the `Embedded file CRC32` line for the payload hash.

- [ ] **Step 6: Commit**

```bash
git add src/main.rs tests/integration_inject_extract.rs
git commit -m "feat: inject writes VERSION_3 header CRC by default"
```

---

### Task 7: Header CRC catches corruption of `Size` and `Filename` fields end-to-end

**Files:**
- Test only: `tests/integration_inject_extract.rs`

**Steps:**

- [ ] **Step 1: Failing test**

Add an integration test that:
1. Injects with default settings (now v3).
2. Locates the LSB position corresponding to a byte inside the `Size` TLV (computed deterministically from the meta layout, not magic numbers — see Task 14).
3. Flips one LSB at that position by editing the PNG.
4. Runs `extract`, expects failure with stderr containing `"Metadata header CRC mismatch"`.
5. Repeats for a byte inside the `Filename` TLV.

```rust
#[rstest]
#[case::size_field(byte_offset_of_size())]
#[case::filename_field(byte_offset_of_filename())]
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
```

`byte_offset_of_size()` and `byte_offset_of_filename()` are constants computed from the v3 meta layout for the fixture filename `"payload.bin"`:

```
sig:       2 bytes  → offsets 0..2
Size TLV:  1+1+4    → offsets 2..8        (size value at 4..8)
Filename:  1+1+11   → offsets 8..21
Hash:      1+1+4    → offsets 21..27
MetaHash:  1+1+4    → offsets 27..33
end:       2        → offsets 33..35
```

Pick an offset clearly inside each region (e.g. `5` for size, `14` for filename). Document the layout in a comment above the consts.

`flip_lsb_at_meta_byte` converts a meta byte index into a pixel coordinate using the same `iter_dots(w, h)` order as `inject`:

```rust
fn flip_lsb_at_meta_byte(png: &Path, meta_byte: usize) {
    let mut img = image::open(png).unwrap().into_rgba8();
    let (w, h) = img.dimensions();
    // 8 bits per byte, 4 channels per pixel = 2 pixels per byte.
    let bit_index = meta_byte * 8;
    let channel_index = bit_index; // first bit of the byte
    let pixel_index = channel_index / 4;
    let channel_in_pixel = channel_index % 4;
    let x = pixel_index as u32 / h;
    let y = pixel_index as u32 % h;
    img.get_pixel_mut(x, y).0[channel_in_pixel] ^= 1;
    img.save(png).unwrap();
}
```

- [ ] **Step 2: Run, confirm failure**

```bash
cargo test --test integration_inject_extract extract_detects_header_corruption
```

Expected: at least one case fails (header corruption goes undetected — old reader behavior).

- [ ] **Step 3: Confirm green**

If Task 5 was implemented correctly, the test should now pass without any code change. If it does not, debug Task 5's CRC scope.

- [ ] **Step 4: Commit**

```bash
git add tests/integration_inject_extract.rs
git commit -m "test: header CRC catches Size/Filename tampering"
```

---

## Phase 2 — Test coverage gaps

These tasks add tests but do not change production behavior. They are not strictly TDD because there is no bug to fix — they are characterization tests for behavior the project already promises in its README.

---

### Task 8: v1 backward-compat round-trip via the CLI

**Files:**
- Create: `tests/v1_backward_compat.rs`

**Steps:**

- [ ] **Step 1: Write the test**

The fixture is a raw v1 byte sequence injected manually into a PNG (the production code no longer writes v1, so we hand-craft it):

```rust
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
                if idx >= bits.len() { break 'outer; }
                p.0[c] = (p.0[c] & 0xFE) | (bits[idx] & 1);
                idx += 1;
            }
        }
    }
}

fn bits(bytes: &[u8]) -> Vec<u8> {
    bytes.iter().flat_map(|b| (0..8).rev().map(move |i| (b >> i) & 1)).collect()
}

#[test]
fn extract_reads_legacy_v1_format() {
    const MAGIC: u16 = 0xd2d;
    const VERSION_1: u8 = 1;
    let dir = tempdir().unwrap();
    let png_path = dir.path().join("v1.png");
    let extracted = dir.path().join("out.bin");

    // Build a v1 meta byte stream with size=5 and filename "hi.bin"
    let signature = (MAGIC << 3) | (VERSION_1 as u16);
    let mut meta = Vec::new();
    meta.extend(signature.to_le_bytes());
    meta.extend(5u32.to_le_bytes());
    let filename = b"hi.bin";
    meta.push(filename.len() as u8);
    meta.extend(filename);

    // Build payload bytes
    let payload = b"hello";

    // Concatenate meta and payload bits
    let mut bit_stream = bits(&meta);
    bit_stream.extend(bits(payload));

    let mut img: ImageBuffer<Rgba<u8>, Vec<u8>> =
        ImageBuffer::from_fn(50, 50, |_, _| Rgba([255, 255, 255, 255]));
    embed_bits_into_png(&mut img, &bit_stream);
    img.save(&png_path).unwrap();

    let mut cmd = Command::cargo_bin("injet").unwrap();
    cmd.args(["extract", png_path.to_str().unwrap(), "-d", extracted.to_str().unwrap()]);
    cmd.assert().success();
    assert_eq!(std::fs::read(&extracted).unwrap(), b"hello");
}
```

- [ ] **Step 2: Run, expect green**

```bash
cargo test --test v1_backward_compat
```

If it fails, fix the v1 reader path in `Meta::read` until it passes.

- [ ] **Step 3: Commit**

```bash
git add tests/v1_backward_compat.rs
git commit -m "test: v1 metadata still extractable via CLI"
```

---

### Task 9: Inspect command output snapshots

**Files:**
- Modify: `Cargo.toml` (add `insta` dev-dependency)
- Create: `tests/inspect.rs`
- Create: `tests/snapshots/` (created automatically by `insta`)

**Steps:**

- [ ] **Step 1: Add `insta` to dev-deps**

```toml
[dev-dependencies]
insta = "1"
```

- [ ] **Step 2: Write parametrized snapshot tests**

```rust
use assert_cmd::Command;
use image::{ImageBuffer, Rgba};
use rstest::rstest;
use tempfile::tempdir;

fn run_inspect(args: &[&str]) -> String {
    let out = Command::cargo_bin("injet").unwrap().args(args).output().unwrap();
    let mut s = String::from_utf8_lossy(&out.stdout).to_string();
    // Strip the file path line which depends on temp dirs.
    s = s
        .lines()
        .filter(|l| !l.starts_with("Image file:"))
        .collect::<Vec<_>>()
        .join("\n");
    s
}

#[rstest]
#[case::empty_png("empty")]
#[case::with_meta("with_meta")]
#[case::with_seed("with_seed")]
fn inspect_output(#[case] case: &str) {
    let dir = tempdir().unwrap();
    let png = dir.path().join("c.png");
    let img: ImageBuffer<Rgba<u8>, Vec<u8>> =
        ImageBuffer::from_fn(80, 80, |_, _| Rgba([255, 255, 255, 255]));
    img.save(&png).unwrap();

    let mut args: Vec<String> = vec!["inspect".into(), png.to_string_lossy().into()];

    if case != "empty" {
        let cargo = dir.path().join("hi.bin");
        std::fs::write(&cargo, b"hello world").unwrap();
        let mut inj = vec!["inject".to_string(), cargo.to_string_lossy().into(),
                           png.to_string_lossy().into(), "-d".into(),
                           png.to_string_lossy().into()];
        if case == "with_seed" { inj.push("--seed".into()); inj.push("s".into()); }
        Command::cargo_bin("injet").unwrap().args(&inj).assert().success();
        if case == "with_seed" { args.push("--seed".into()); args.push("s".into()); }
    }

    let stdout = run_inspect(&args.iter().map(|s| s.as_str()).collect::<Vec<_>>());
    insta::assert_snapshot!(case, stdout);
}
```

- [ ] **Step 3: Generate snapshots**

```bash
INSTA_UPDATE=always cargo test --test inspect
```

- [ ] **Step 4: Review the `.snap` files**, then re-run without `INSTA_UPDATE` to confirm they pass.

```bash
cargo test --test inspect
```

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml tests/inspect.rs tests/snapshots/
git commit -m "test: inspect command output snapshots"
```

---

### Task 10: Read-mode combinations

**Files:**
- Modify: `tests/integration_inject_extract.rs`

**Steps:**

- [ ] **Step 1: Add parametrized test**

```rust
#[rstest]
#[case::meta_yes_size_none(true, None)]
#[case::meta_no_size_some(false, Some(2048))]
#[case::meta_no_size_none(false, None)] // reads to EOF
fn extract_read_mode_combinations(#[case] write_meta: bool, #[case] read_size: Option<usize>) {
    let env = setup_env();
    inject_file_into_png(&env.bin_path, &env.png_path, &env.out_png_path, write_meta, None);
    let bytes = extract_file_from_png(
        &env.out_png_path,
        &env.extracted_bin_path,
        write_meta,
        read_size,
        None,
        true,
    )
    .unwrap();
    let truncated_to = read_size.unwrap_or(TEST_PAYLOAD.len());
    assert_eq!(&bytes[..truncated_to.min(bytes.len())], &TEST_PAYLOAD[..truncated_to.min(TEST_PAYLOAD.len())]);
}
```

Note: the EOF case reads many extra bytes (the rest of the LSBs). The assertion truncates to `TEST_PAYLOAD.len()` for that case.

- [ ] **Step 2: Run, expect green**, fix as needed.

```bash
cargo test --test integration_inject_extract extract_read_mode_combinations
```

- [ ] **Step 3: Commit**

```bash
git add tests/integration_inject_extract.rs
git commit -m "test: extract read-mode combinations"
```

---

### Task 11: Round-trip through every `--compression` mode

**Files:**
- Modify: `tests/integration_inject_extract.rs`

**Steps:**

- [ ] **Step 1: Add test**

```rust
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
```

- [ ] **Step 2: Run, expect green.**

```bash
cargo test --test integration_inject_extract round_trip_with_compression
```

- [ ] **Step 3: Commit**

```bash
git add tests/integration_inject_extract.rs
git commit -m "test: round-trip across all PNG compression levels"
```

---

### Task 12: stdout-pipe extract test

**Files:**
- Modify: `tests/integration_inject_extract.rs`

**Steps:**

- [ ] **Step 1: Add test**

```rust
#[test]
fn extract_to_stdout_when_piped() {
    let env = setup_env();
    inject_file_into_png(&env.bin_path, &env.png_path, &env.out_png_path, true, None);
    // assert_cmd by default does not attach a TTY → stdout is_terminal() is false
    let out = Command::cargo_bin("injet")
        .unwrap()
        .args(["extract", env.out_png_path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(out.status.success(), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(&out.stdout[..TEST_PAYLOAD.len()], TEST_PAYLOAD);
}
```

- [ ] **Step 2: Run, expect green or fix**

```bash
cargo test --test integration_inject_extract extract_to_stdout_when_piped
```

- [ ] **Step 3: Commit**

```bash
git add tests/integration_inject_extract.rs
git commit -m "test: extract writes payload to stdout when piped"
```

---

### Task 13: Capacity exact-fit success path

**Files:**
- Modify: `tests/integration_inject_extract.rs`

**Steps:**

- [ ] **Step 1: Add test**

```rust
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
```

- [ ] **Step 2: Run, expect green**

```bash
cargo test --test integration_inject_extract inject_succeeds_at_exact_capacity_without_meta
```

- [ ] **Step 3: Commit**

```bash
git add tests/integration_inject_extract.rs
git commit -m "test: capacity exact-fit succeeds without metadata"
```

---

### Task 14: Replace magic coordinates in `corrupt_payload_bit` with computed offsets

**Files:**
- Modify: `tests/integration_inject_extract.rs:46-50` (`corrupt_payload_bit`)

**Steps:**

- [ ] **Step 1: Compute the meta length at runtime**

The current helper hardcodes `(32, 0)`. Replace with a helper that re-injects to a temporary buffer, parses the meta, and computes the first pixel **after** the meta region:

```rust
fn corrupt_payload_bit(src_png: &Path, dst_png: &Path) {
    let mut img = image::open(src_png).unwrap().into_rgba8();
    let (w, h) = img.dimensions();
    // Read all LSBs to figure out where the meta ends, then flip the bit one byte after.
    // For a 100x100 image, payload starts well past pixel (32,0); to keep this robust
    // across format changes, just flip the LSB of the LAST pixel which is always payload.
    let p = img.get_pixel_mut(w - 1, h - 1);
    p.0[0] ^= 1;
    img.save(dst_png).unwrap();
}
```

This is the minimum-friction fix and is robust to any future meta layout change.

- [ ] **Step 2: Run, expect green**

```bash
cargo test --test integration_inject_extract extract_fails_on_corruption
```

- [ ] **Step 3: Commit**

```bash
git add tests/integration_inject_extract.rs
git commit -m "test: corrupt last pixel (always payload) instead of magic coordinate"
```

---

### Task 15: `pseudo_shuffle_coords` determinism property

**Files:**
- Modify: `src/main.rs` `mod tests`

**Steps:**

- [ ] **Step 1: Add test**

```rust
#[test]
fn test_pseudo_shuffle_is_deterministic_for_same_seed() {
    use crate::pseudo_shuffle_coords;
    let a: Vec<_> = pseudo_shuffle_coords(20, 20, &"abc".to_string()).collect();
    let b: Vec<_> = pseudo_shuffle_coords(20, 20, &"abc".to_string()).collect();
    assert_eq!(a, b);
    let c: Vec<_> = pseudo_shuffle_coords(20, 20, &"abd".to_string()).collect();
    assert_ne!(a, c);
    // Permutation property: every coord appears exactly once.
    let mut sorted = a.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), 400);
}
```

- [ ] **Step 2: Run, expect green**

```bash
cargo test test_pseudo_shuffle_is_deterministic_for_same_seed
```

- [ ] **Step 3: Commit**

```bash
git add src/main.rs
git commit -m "test: pseudo-shuffle is deterministic and permutation-correct"
```

---

## Phase 3 — Correctness polish

Each task in this phase is again strict TDD because each addresses a real defect.

---

### Task 16: `seed_to_u64` uses only 8 of 32 hash bytes

**Problem.** `src/main.rs:160-163` truncates the blake3 hash to 8 bytes for `StdRng::from_seed_from_u64`. `StdRng::from_seed` accepts the full 32 bytes — use it.

**Files:**
- Modify: `src/main.rs:160-170`
- Test: `src/main.rs` `mod tests`

**Steps:**

- [ ] **Step 1: Failing test — different seeds whose first 8 hash bytes collide must produce different orderings.**

This is hard to construct without precomputation. Replace it with a behavioral test that just confirms the new helper exists and is used:

```rust
#[test]
fn test_seed_uses_full_blake3_hash() {
    use crate::pseudo_shuffle_coords;
    // Two seeds chosen at random. The test asserts they produce different orders,
    // which is true for any reasonable seeding scheme. The point of the test is
    // to lock in determinism after the helper is rewritten.
    let a: Vec<_> = pseudo_shuffle_coords(10, 10, &"alpha".to_string()).collect();
    let b: Vec<_> = pseudo_shuffle_coords(10, 10, &"beta".to_string()).collect();
    assert_ne!(a, b);
}
```

This test passes today and after the change. The TDD value here is small. **Skip TDD for this task** and treat it as a refactor under the existing determinism test from Task 15.

- [ ] **Step 2: Replace the helper**

```rust
fn seed_to_array(seed: &str) -> [u8; 32] {
    *blake3::hash(seed.as_bytes()).as_bytes()
}

fn pseudo_shuffle_coords(w: u32, h: u32, seed: &Seed) -> impl Iterator<Item = (u32, u32)> {
    let mut coords: Vec<(u32, u32)> = iter_dots(w, h).collect();
    let mut rng = rand::rngs::StdRng::from_seed(seed_to_array(seed));
    coords.shuffle(&mut rng);
    coords.into_iter()
}
```

Delete `seed_to_u64`.

- [ ] **Step 3: Run all tests**

```bash
cargo test
```

The Task 15 determinism test must still pass with the new ordering (different from before — that's expected because the seed changed).

- [ ] **Step 4: Commit**

```bash
git add src/main.rs
git commit -m "refactor: feed full blake3 hash into StdRng seed"
```

> Caveat: this **changes the pseudoshuffle order** for any given seed, which means files injected with a `--seed` by `0.3.0` cannot be extracted by `1.0.0`. Document this in the CHANGELOG (Task 28).

---

### Task 17: Reject invalid UTF-8 filenames instead of silently lossy-decoding

**Files:**
- Modify: `src/main.rs:245-247, 296-298`
- Test: `src/main.rs` `mod tests`

**Steps:**

- [ ] **Step 1: Failing test**

```rust
#[test]
fn test_meta_v2_rejects_non_utf8_filename() {
    use crate::{MAGIC, VERSION_2};
    let mut bytes = Vec::new();
    let signature = (MAGIC << 3) | (VERSION_2 as u16);
    bytes.extend(signature.to_le_bytes());
    bytes.push(2); // Filename tag
    bytes.push(3);
    bytes.extend(&[0xFF, 0xFE, 0xFD]); // not valid UTF-8
    bytes.push(0);
    bytes.push(0);
    let mut iter = bytes.into_iter();
    let err = Meta::read(&mut iter).expect_err("must reject non-UTF-8 filename");
    assert!(matches!(err, MetaError::MalformedFilename));
}
```

- [ ] **Step 2: Run, confirm failure** (`String::from_utf8_lossy` swallows the error today).

```bash
cargo test test_meta_v2_rejects_non_utf8_filename
```

- [ ] **Step 3: Fix**

In `from_tlv_field`:

```rust
MetaTag::Filename => match String::from_utf8(bytes) {
    Ok(s) => Some(MetaField::Filename(s)),
    Err(_) => return Err(MetaError::MalformedFilename),
},
```

Same change in `from_v1_header`: replace `String::from_utf8_lossy(&filename_vec).to_string()` with a fallible parse.

- [ ] **Step 4: Run, confirm green**

```bash
cargo test
```

- [ ] **Step 5: Refactor**

If both call sites duplicate the conversion, extract `fn parse_filename(bytes: Vec<u8>) -> Result<String, MetaError>`.

- [ ] **Step 6: Commit**

```bash
git add src/main.rs
git commit -m "fix: reject non-UTF-8 filenames instead of lossy decoding"
```

---

### Task 18: Failed extraction must not leave a corrupt file on disk

**Files:**
- Modify: `src/main.rs:528-593` (`extract`)
- Test: `tests/integration_inject_extract.rs`

**Steps:**

- [ ] **Step 1: Failing test**

```rust
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
        "extract must remove the partial output on hash mismatch"
    );
}
```

- [ ] **Step 2: Run, confirm failure** — file exists.

```bash
cargo test --test integration_inject_extract extract_does_not_leave_corrupt_file_on_hash_mismatch
```

- [ ] **Step 3: Fix — write to temp file, rename on success, delete on failure**

In `extract`, when `args.destination` resolves to a real file path (not stdout), write to `<dest>.partial`. After CRC verification:

```rust
match crc_check {
    Ok(()) => std::fs::rename(&partial_path, &final_path).map_err(|_| ExtractError::Save)?,
    Err(e) => {
        let _ = std::fs::remove_file(&partial_path);
        return Err(e);
    }
}
```

The stdout case is unchanged (no file to delete).

Implementation note: the existing `make_writer` opens the file directly. Refactor it to return both the writer and an optional `cleanup` closure (or use an `enum WriteTarget { Stdout, File { partial: PathBuf, final_: PathBuf } }`).

- [ ] **Step 4: Run, confirm green**

```bash
cargo test
```

- [ ] **Step 5: Refactor**

Make sure the broken-meta path also avoids creating an empty file. Today the writer is opened **after** `Meta::read`, so this is already fine; double-check.

- [ ] **Step 6: Commit**

```bash
git add src/main.rs tests/integration_inject_extract.rs
git commit -m "fix: extract removes partial output on integrity failure"
```

---

### Task 19: `Size` field with wrong byte length is now an error, not a silent skip

**Files:**
- Modify: `src/main.rs:241-251` (`from_tlv_field` final match)
- Test: `src/main.rs` `mod tests`

**Steps:**

- [ ] **Step 1: Failing test**

```rust
#[test]
fn test_meta_v2_rejects_size_field_with_wrong_length() {
    use crate::{MAGIC, VERSION_2};
    let mut bytes = Vec::new();
    let signature = (MAGIC << 3) | (VERSION_2 as u16);
    bytes.extend(signature.to_le_bytes());
    bytes.push(1); // Size tag
    bytes.push(3); // wrong: should be 4
    bytes.extend(&[0, 0, 0]);
    bytes.push(0);
    bytes.push(0);
    let mut iter = bytes.into_iter();
    let err = Meta::read(&mut iter).expect_err("malformed Size must error");
    assert!(matches!(err, MetaError::MalformedField));
}
```

- [ ] **Step 2: Run, confirm failure**

- [ ] **Step 3: Fix**

Add variant to `MetaError`:

```rust
#[error("Malformed metadata field")]
MalformedField,
```

In `from_tlv_field`, change the final match so `Size`/`Hash`/`MetaHash` with the wrong byte length return `Err(MetaError::MalformedField)` instead of `None`:

```rust
let field = match tag {
    MetaTag::Size => {
        if bytes.len() != 4 { return Err(MetaError::MalformedField); }
        MetaField::Size(u32::from_le_bytes(bytes.try_into().expect("checked length == 4")))
    }
    MetaTag::Filename => MetaField::Filename(parse_filename(bytes)?),
    MetaTag::Hash => {
        if bytes.len() != 4 { return Err(MetaError::MalformedField); }
        MetaField::Hash(u32::from_le_bytes(bytes.try_into().expect("checked length == 4")))
    }
    MetaTag::MetaHash => {
        if bytes.len() != 4 { return Err(MetaError::MalformedField); }
        MetaField::MetaHash(u32::from_le_bytes(bytes.try_into().expect("checked length == 4")))
    }
};
Ok(MetaFieldParseResult::Field(field))
```

- [ ] **Step 4: Run, confirm green**

- [ ] **Step 5: Commit**

```bash
git add src/main.rs
git commit -m "fix: malformed Size/Hash TLV fields surface as errors"
```

---

### Task 20: `inspect` verifies CRC of payload and reports the result

**Files:**
- Modify: `src/main.rs:471-515` (`inspect`)
- Test: `tests/inspect.rs` (extend snapshots from Task 9)

**Steps:**

- [ ] **Step 1: Failing test**

Add a snapshot case `with_corrupted_payload`:

```rust
#[case::corrupted_payload("corrupted_payload")]
```

Setup: inject normally, then call `corrupt_payload_bit` on the result. Snapshot the inspect output. Expected snapshot lines include `Payload CRC32: mismatch`.

- [ ] **Step 2: Run, confirm failure** (current `inspect` does not verify).

- [ ] **Step 3: Fix**

In `inspect`, after parsing meta, if `meta.hash().is_some()`:
1. Continue iterating `content` to compute CRC of `meta.size().unwrap_or(u32::MAX)` bytes.
2. Compare to `meta.hash()`.
3. Print `Payload CRC32: ok` or `Payload CRC32: mismatch`.

- [ ] **Step 4: Update snapshots**

```bash
INSTA_UPDATE=always cargo test --test inspect
cargo test --test inspect
```

Manually inspect the new `.snap` file before committing.

- [ ] **Step 5: Commit**

```bash
git add src/main.rs tests/inspect.rs tests/snapshots/
git commit -m "feat: inspect verifies payload CRC and reports status"
```

---

## Phase 4 — Performance

These tasks have no behavioral change in their happy paths. Each lands behind tests added in Phase 2 and Phase 3, so regressions show up immediately.

---

### Task 21: `gen_dots` returns an enum-iterator instead of `Box<dyn>`

**Files:**
- Modify: `src/main.rs:172-177`

**Steps:**

- [ ] **Step 1: Replace**

```rust
enum Dots<L, S> { Linear(L), Shuffled(S) }

impl<L, S> Iterator for Dots<L, S>
where L: Iterator<Item = (u32, u32)>, S: Iterator<Item = (u32, u32)>,
{
    type Item = (u32, u32);
    fn next(&mut self) -> Option<(u32, u32)> {
        match self {
            Dots::Linear(it) => it.next(),
            Dots::Shuffled(it) => it.next(),
        }
    }
}

fn gen_dots(w: u32, h: u32, seed: Option<&Seed>) -> impl Iterator<Item = (u32, u32)> {
    match seed {
        Some(s) => Dots::Shuffled(pseudo_shuffle_coords(w, h, s)),
        None => Dots::Linear(iter_dots(w, h)),
    }
}
```

This requires `pseudo_shuffle_coords` and `iter_dots` to have *named* return types (`impl Iterator` works with `impl Trait` only when both arms can be unified — they cannot, hence the enum). The enum is the unification.

`gen_dots`'s `impl Iterator` return type unifies the two variants of `Dots`.

- [ ] **Step 2: Run all tests**

```bash
cargo test
```

If type inference fights you (it will at the call sites that previously held `Box<dyn>`), make `inject`/`extract`/`inspect` generic over `I: Iterator<Item = (u32, u32)>` or leave them taking `gen_dots`'s opaque return.

- [ ] **Step 3: Commit**

```bash
git add src/main.rs
git commit -m "perf: drop Box<dyn Iterator> from gen_dots in favor of enum dispatch"
```

---

### Task 22: `inject` mutates pixels in place via `get_pixel_mut`

**Files:**
- Modify: `src/main.rs:653-668`

**Steps:**

- [ ] **Step 1: Replace `get_pixel`/`put_pixel` with `get_pixel_mut`**

```rust
let mut bit_iter = bits;
let mut changed = 0;
let total_bits = total_size * 8;
for (x, y) in color_coords {
    if changed >= total_bits { break; }
    let px = img.get_pixel_mut(x, y);
    for channel in &mut px.0 {
        if changed >= total_bits { break; }
        if let Some(bit) = bit_iter.next() {
            *channel = (*channel & 0b1111_1110) | bit;
            changed += 1;
        }
    }
}
```

- [ ] **Step 2: Run all tests**

```bash
cargo test
```

- [ ] **Step 3: Commit**

```bash
git add src/main.rs
git commit -m "perf: inject mutates pixels in place"
```

---

### Task 23: `inject` reads cargo file once via `Seek::rewind`

**Files:**
- Modify: `src/main.rs:600, 619-628, 644-646`

**Steps:**

- [ ] **Step 1: Drop the second `File::open`, rewind the original handle**

```rust
use std::io::Seek;

let cargo = File::open(&args.cargo).map_err(|_| InjectError::CannotOpenCargo)?;
// ...
if args.write_meta {
    // CRC pass
    let mut hasher = crc32fast::Hasher::new();
    let mut buf = [0u8; 8192];
    let mut reader = BufReader::new(&cargo);
    loop {
        let n = std::io::Read::read(&mut reader, &mut buf)
            .map_err(|_| InjectError::CannotOpenCargo)?;
        if n == 0 { break; }
        hasher.update(&buf[..n]);
    }
    drop(reader);
    (&cargo).rewind().map_err(|_| InjectError::CannotOpenCargo)?;
    let hash = hasher.finalize();
    // ... build meta as before
}

let cargo_bits = BufReader::new(cargo).bytes().flat_map(|x| to_bits(x.unwrap()));
```

(`File: Seek` is implemented on `&File`, hence `(&cargo).rewind()`.)

- [ ] **Step 2: Run all tests**

```bash
cargo test
```

- [ ] **Step 3: Commit**

```bash
git add src/main.rs
git commit -m "perf: inject computes CRC and bits from a single file handle"
```

---

### Task 24: Streaming permutation for `pseudo_shuffle_coords`

**Problem.** Eagerly materializing `Vec<(u32, u32)>` of size `w * h` for the shuffle is the project's worst memory issue. For an 8K image, ~256 MB allocated just to read 50 bytes of metadata in `inspect`.

**Files:**
- Modify: `src/main.rs:165-170`
- Test: `src/main.rs` `mod tests` (Task 15 already covers permutation correctness)

**Steps:**

- [ ] **Step 1: Pick an algorithm**

Two reasonable options:
- **Lazy Fisher-Yates over `Vec<u32>` of indices**, pulling one element at a time. Memory drops from `8 * w * h` to `4 * w * h` — only a 2× win, not enough to justify a rewrite.
- **Format-preserving permutation via small-domain Feistel network**. O(1) memory, O(1) per element. Recommended.

Implement a 4-round Feistel on `n = w * h`, embedded into the next square `s² >= n` with rejection sampling: split `[0, s²)` into two halves of `s` bits each, run 4 Feistel rounds keyed by `seed_to_array`, reject outputs `>= n`. Iterate `i = 0..s²` until `n` outputs accepted.

- [ ] **Step 2: Failing test — memory**

There is no easy way to assert "uses less memory" in cargo test. Use the determinism + permutation tests from Task 15 as the regression net. Add one extra correctness test:

```rust
#[test]
fn test_streaming_shuffle_matches_eager_for_small_image() {
    // For small images both implementations must produce a valid permutation.
    // We assert that the streaming version produces all coordinates exactly once.
    let coords: Vec<_> = pseudo_shuffle_coords(32, 32, &"k".to_string()).collect();
    assert_eq!(coords.len(), 32 * 32);
    let mut sorted = coords.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), 32 * 32);
}
```

- [ ] **Step 3: Implement Feistel-based streaming shuffle**

Sketch:

```rust
fn pseudo_shuffle_coords(w: u32, h: u32, seed: &Seed) -> impl Iterator<Item = (u32, u32)> {
    let n = (w as u64) * (h as u64);
    let bits = (n as f64).log2().ceil() as u32;
    let half = bits.div_ceil(2);
    let mask = (1u64 << half) - 1;
    let key = seed_to_array(seed);

    let round = move |x: u64, r: u32| -> u64 {
        let mut h = blake3::Hasher::new();
        h.update(&key);
        h.update(&[r as u8]);
        h.update(&x.to_le_bytes());
        u64::from_le_bytes(h.finalize().as_bytes()[..8].try_into().unwrap()) & mask
    };

    let feistel = move |mut x: u64| -> u64 {
        let mut l = x >> half;
        let mut r = x & mask;
        for i in 0..4 {
            let nl = r;
            let nr = l ^ round(r, i);
            l = nl;
            r = nr;
        }
        (l << half) | r
    };

    let total = 1u64 << (half * 2);
    (0..total)
        .map(move |i| feistel(i))
        .filter(move |v| *v < n)
        .map(move |v| {
            let x = (v / h as u64) as u32;
            let y = (v % h as u64) as u32;
            (x, y)
        })
}
```

Note: this changes the pseudoshuffle order again. Document in the CHANGELOG that v1 seeded files are not portable to v1.0.0 (already documented in Task 16; this is the same break, in the same release).

- [ ] **Step 4: Run all tests**

```bash
cargo test
```

The Task 15 and Task 7 tests guard correctness.

- [ ] **Step 5: Commit**

```bash
git add src/main.rs
git commit -m "perf: streaming Feistel-based pseudoshuffle (no full Vec)"
```

---

## Phase 5 — Refactor / structure

These tasks change no behavior. They make the code maintainable.

---

### Task 25: Split `main.rs` into modules

**Files:**
- Create: `src/lib.rs`
- Create: `src/meta.rs` (Meta, MetaField, MetaTag, MetaError, all parsing/serialization)
- Create: `src/lsb.rs` (`to_bits`, `iter_dots`, `pseudo_shuffle_coords`, `gen_dots`, `Dots` enum)
- Create: `src/cli.rs` (Cli, Commands, *Args, Compression)
- Create: `src/commands/inject.rs`, `src/commands/extract.rs`, `src/commands/inspect.rs`
- Modify: `src/main.rs` (now ~30 lines: `clap` parse + dispatch)

**Steps:**

- [ ] **Step 1: Add `src/lib.rs`** that re-exports the modules. Move types one file at a time, running `cargo test` after each move.

Order of moves (each one is a separate commit):
1. `meta.rs` — Meta, MetaField, MetaTag, MetaError, MetaFieldParseResult, MAGIC/VERSION constants.
2. `lsb.rs` — bit + coordinate helpers.
3. `cli.rs` — clap structs and `Compression`.
4. `commands/mod.rs`, `commands/inject.rs`, `commands/extract.rs`, `commands/inspect.rs`.
5. `main.rs` — final shrink to dispatch only.

- [ ] **Step 2: After each move, run**

```bash
cargo test
cargo clippy --all-targets -- -D warnings
```

- [ ] **Step 3: Commit each move separately**

Example:

```bash
git add src/lib.rs src/meta.rs src/main.rs
git commit -m "refactor: extract meta module"
```

---

### Task 26: Convert in-module meta tests to `insta` snapshots via `rstest`

**Files:**
- Modify: `src/meta.rs` test module
- Create: `src/snapshots/`

**Steps:**

- [ ] **Step 1: Replace** the existing `test_meta_v2_roundtrip`, `test_meta_v1_parsing`, `test_meta_v3_roundtrip_with_header_hash`, `test_meta_v2_skips_unknown_tlv_and_keeps_parsing`, `test_meta_v2_rejects_size_field_with_wrong_length`, `test_meta_v2_rejects_non_utf8_filename`, `test_meta_v3_header_hash_mismatch_detected` with parametrized snapshots:

```rust
#[rstest]
#[case::v1(build_v1_bytes())]
#[case::v2_roundtrip(Meta::make(Some(1), Some("a".into()), Some(2)).to_bytes())]
#[case::v3_roundtrip(Meta::make_v3(Some(1), Some("a".into()), Some(2)).to_bytes())]
#[case::unknown_tlv_v2(build_v2_with_unknown_tag())]
#[case::v2_bad_size(build_v2_bad_size())]
#[case::v2_bad_utf8(build_v2_bad_utf8())]
#[case::v3_header_tampered(build_v3_tampered())]
fn meta_parse(#[case] bytes: Vec<u8>) {
    let mut iter = bytes.into_iter();
    insta::assert_debug_snapshot!(Meta::read(&mut iter));
}
```

The `build_*` helpers live in the same test module.

- [ ] **Step 2: Generate snapshots**

```bash
INSTA_UPDATE=always cargo test
```

Review every snapshot file by hand before committing.

- [ ] **Step 3: Commit**

```bash
git add src/meta.rs src/snapshots/
git commit -m "test: snapshot meta parser via insta + rstest"
```

---

### Task 27: Replace `meta_tag_enum!` macro with a plain enum + `TryFrom`

**Files:**
- Modify: `src/meta.rs`

**Steps:**

- [ ] **Step 1: Inline the macro**

```rust
#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum MetaTag {
    Size = 1,
    Filename = 2,
    Hash = 3,
    MetaHash = 4,
}

impl From<MetaTag> for u8 {
    fn from(t: MetaTag) -> u8 { t as u8 }
}

impl TryFrom<u8> for MetaTag {
    type Error = ();
    fn try_from(v: u8) -> Result<Self, ()> {
        match v {
            1 => Ok(MetaTag::Size),
            2 => Ok(MetaTag::Filename),
            3 => Ok(MetaTag::Hash),
            4 => Ok(MetaTag::MetaHash),
            _ => Err(()),
        }
    }
}
```

Delete `meta_tag_enum!`.

- [ ] **Step 2: Run all tests**

```bash
cargo test
cargo clippy --all-targets -- -D warnings
```

- [ ] **Step 3: Commit**

```bash
git add src/meta.rs
git commit -m "refactor: drop macro for MetaTag in favor of plain enum"
```

---

## Phase 6 — Release

### Task 28: Bump to 1.0.0 and write the CHANGELOG

**Files:**
- Modify: `Cargo.toml`
- Create: `CHANGELOG.md`
- Modify: `README.md`

**Steps:**

- [ ] **Step 1: Bump `Cargo.toml`**

```toml
version = "1.0.0"
```

- [ ] **Step 2: Write `CHANGELOG.md`**

```markdown
# Changelog

## [1.0.0] - 2026-04-08

### Breaking
- Default metadata version is now `VERSION_3`. Files produced by `1.0.0` cannot be read by `0.3.0`. `1.0.0` still reads `VERSION_1` and `VERSION_2`.
- `VERSION_3` adds a `MetaHash` TLV (CRC32 over the metadata header). Tampering with `Size`/`Filename` is now detected as `Metadata header CRC mismatch`.
- The `--seed` pseudoshuffle algorithm changed (now uses the full blake3 hash and a streaming Feistel permutation). Files injected with a `--seed` under `0.3.0` cannot be extracted by `1.0.0`. Files injected without `--seed` are unaffected.
- Non-UTF-8 filenames in metadata now produce an error (`Invalid or corrupted filename in metadata`) instead of being silently lossy-decoded.
- Malformed `Size`/`Hash`/`MetaHash` TLV fields (wrong length) now produce a hard error instead of being skipped.
- Filenames longer than 255 bytes still rejected, but exactly 255-byte filenames are now accepted (off-by-one fix).

### Fixed
- Unknown TLV tags no longer desync the metadata parser.
- `inject` no longer reads the cargo file twice.
- `inject` no longer panics on container paths without a `file_name` component.
- Failed `extract` no longer leaves a partial corrupt file at the destination.

### Added
- `inspect` now verifies the payload CRC32 and reports `ok` / `mismatch`.
- Test coverage for: forward-compatible TLV, capacity exact-fit, all `--compression` levels, stdout pipe extract, v1 backward compat round-trip.

### Performance
- `pseudoshuffle` is now O(1) memory via streaming Feistel network instead of materializing all coordinates.
- `inject` mutates pixels in place via `get_pixel_mut`.
- `gen_dots` no longer goes through `Box<dyn Iterator>`.
```

- [ ] **Step 3: Update README**

Mention the v3 default in the "Embedding a file" section. Add a brief note that `0.x` files are still readable but `1.0` files are not portable backwards.

- [ ] **Step 4: Final verification**

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo build --release
```

- [ ] **Step 5: Commit and tag**

```bash
git add Cargo.toml CHANGELOG.md README.md
git commit -m "chore: release 1.0.0"
git tag v1.0.0
```

Do not push or `cargo publish` from this plan — that is the maintainer's call.

---

## Self-Review

**Spec coverage** — every item from the analysis is mapped to a task:

| Analysis item | Task |
|---|---|
| Bug 1: TLV unknown tag desync | 1 |
| Bug 2: filename off-by-one | 2 |
| Bug 3: panic on `..` path | 3 |
| Format weakness: CRC over meta | 4, 5, 6, 7 |
| v1 backward compat (test) | 8 |
| inspect command tests | 9, 20 |
| Read-mode combinations | 10 |
| Compression mode round-trip | 11 |
| stdout pipe extract | 12 |
| Capacity exact-fit | 13 |
| Magic-number coordinates in tests | 14 |
| pseudoshuffle determinism test | 15 |
| `seed_to_u64` truncates hash | 16 |
| Lossy UTF-8 filenames | 17 |
| Failed extraction leaves corrupt file | 18 |
| Bad-length Size silently skipped | 19 |
| inspect doesn't verify CRC | 20 |
| `Box<dyn Iterator>` | 21 |
| `get_pixel`/`put_pixel` in inject | 22 |
| Double cargo file read | 23 |
| `pseudo_shuffle_coords` memory | 24 |
| Module split | 25 |
| Snapshot tests for meta | 26 |
| `meta_tag_enum!` macro removal | 27 |
| Major version bump | 28 |

**Items intentionally not in this plan:**
- `format_size` `f32 → f64`: the precision loss does not manifest at 2-decimal display granularity. Listed in analysis as #3 originally; demoted to non-issue here. If wanted, fold into Task 25 as a one-line change with no test.
- `make_writer` UX warning when extract pipes binary to stdout: design discussion, not a defect. Out of scope.

**Placeholder scan:** none. Every code step contains the actual code or the actual test.

**Type consistency:** `MetaTag::MetaHash`, `MetaField::MetaHash`, `MetaError::HeaderHashMismatch`, `MetaError::MalformedField`, `MetaError::MalformedFilename`, `Meta::make_v3`, `Meta::meta_hash`, `display_path`, `Dots`, `seed_to_array`, `MAX_FILENAME_LEN`, `read_tlv_len`, `parse_filename` — all defined in the task that introduces them and reused with the same name in later tasks.

---

## Execution

Plan saved to `docs/superpowers/plans/2026-04-08-injet-hardening.md`.

Recommended execution: **subagent-driven** (`superpowers:subagent-driven-development`). One fresh subagent per task, two-stage review between tasks, given the strict TDD discipline required for Phase 0, 1, and 3.
