# Changelog

## [1.1.0] - 2026-04-08

### Added
- Optional cryptographic authentication for containers via two opt-in modes:
  pre-shared passphrase (Argon2id key derivation + BLAKE3 keyed-hash MAC,
  48-byte Mac TLV embedding a per-container 16-byte salt) and Ed25519
  signatures (64-byte Signature TLV).
- New `inject` flags: `--psk-file`, `--psk-env`, `--psk-prompt`, `--sign-key`,
  `--sign-key-passphrase-{file,env,prompt}`. PSK and Ed25519 flags are
  mutually exclusive.
- New `extract` flags: `--psk-file`, `--psk-env`, `--psk-prompt`,
  `--verify-key`, `--verify-key-env`, `--insecure-skip-verify`.
- `inspect` reports `Signed: psk` / `Signed: ed25519` for signed containers.
- Strict-by-default verification policy: `extract` refuses to write a signed
  container's payload unless the supplied key verifies; explicit
  `--insecure-skip-verify` opt-out always prints a stderr warning.
- Exit code 2 for all auth and key errors (existing errors stay at 1).
- Test fixtures under `tests/fixtures/keys/` (Ed25519 keypairs, encrypted
  variant, wrong-key, PSK passphrase files).

### Compatibility
- Unsigned containers from any prior version (v1/v2/v3) still extract.
- Unsigned `inject` behavior is unchanged.
- `--write-meta=false` is now mutually exclusive with any signing flag
  (signing requires the metadata header to carry the auth TLV).
- Older `injet` binaries (before this release) silently skip the new auth
  TLVs and extract signed containers without verification. **Both ends must
  use this release or newer for signature verification to be effective.**

### Dependencies
- New: `ed25519-dalek 2`, `ssh-key 0.6` (with `encryption` feature),
  `argon2 0.5`, `rpassword 7`, `zeroize 1`, `getrandom 0.2`. All pure Rust,
  no C bindings.

## [1.0.0] - 2026-04-08

### Breaking
- Default metadata version is now `VERSION_3`. Files produced by `1.0.0` cannot be read by `0.3.0`. `1.0.0` still reads `VERSION_1` and `VERSION_2`.
- `VERSION_3` adds a `MetaHash` TLV (CRC32 over the metadata header). Tampering with `Size` or `Filename` is now detected as `Metadata header CRC mismatch`.
- The `--seed` pseudoshuffle algorithm changed twice: (1) it now uses the full 32-byte blake3 hash instead of an 8-byte truncation, and (2) it is now a streaming Feistel network instead of a `Vec`-materializing shuffle. Files injected with a `--seed` under `0.3.0` cannot be extracted by `1.0.0`. Files injected without `--seed` are unaffected.
- Non-UTF-8 filenames in metadata now produce an error (`Invalid or corrupted filename in metadata`) instead of being silently lossy-decoded.
- Malformed `Size` / `Hash` / `MetaHash` TLV fields (wrong length) now produce a hard error (`Malformed metadata field`) instead of being silently skipped.
- Filenames longer than 255 bytes are still rejected, but exactly 255-byte filenames are now accepted (off-by-one fix).
- Renamed the embedded-file concept from `cargo` to `payload` everywhere it was user-visible:
  - The first positional argument of `inject` is now `<PAYLOAD>` (was `<CARGO>`). Positional ordering is unchanged, so `injet inject file.bin image.png` still works.
  - The default extracted filename when no metadata and no `-d` is given is now `payload` (was `cargo`).
  - `InjectError` variants renamed: `CannotOpenCargo` → `CannotOpenPayload`, `CargoTooLarge` → `PayloadTooLarge`, `ExceededSize { cargo_size }` → `ExceededSize { payload_size }`. Error messages updated accordingly.

### Fixed
- Unknown TLV tags no longer desync the metadata parser. Forward-compatible field skipping now actually works.
- `inject` no longer reads the payload file twice (single-pass via `Seek::rewind`).
- `inject` no longer panics on container paths without a `file_name` component (e.g. `..` or `/`).
- Failed `extract` no longer leaves a partial corrupt file at the destination — writes go to `<dest>.partial` and rename atomically only on integrity success.

### Added
- `inspect` now verifies the payload CRC32 and reports `Payload CRC32: ok` or `mismatch`.
- `inspect` reports the v3 header CRC: `Header CRC32: <hex>`.
- New `MetaError` variants: `HeaderHashMismatch`, `MetaHashMissing`, `MalformedField`.
- Test coverage for: unknown-tag forward compat, exact-size filename, header CRC tampering of Size/Filename, capacity exact-fit, all `--compression` levels, stdout pipe extract, `--read-meta` × `--read-size` combinations, v1 backward compat round-trip via CLI, pseudoshuffle determinism + permutation property, snapshot tests for `inspect` and `Meta::read`.

### Performance
- Pseudoshuffle is now O(1) memory via streaming Feistel network instead of materializing all coordinates as a `Vec<(u32, u32)>`. Memory usage at injection time drops dramatically for large containers.
- `inject` mutates pixels in place via `get_pixel_mut` instead of `get_pixel` + `put_pixel`.
- `inject` reads the payload file once instead of twice.
- `gen_dots` no longer goes through `Box<dyn Iterator>` — uses `itertools::Either` for static dispatch.
- `mimalloc` is now wired in as the global allocator on glibc Linux, macOS, Windows, and FreeBSD via a `cfg`-gated `[target.'cfg(...)'.dependencies]` entry. Excluded targets (musl, Android, iOS, illumos, the BSDs not in the allow-list, WASI, embedded) silently fall back to the system allocator — no portability regressions.
- `MetaField` TLV serialization no longer allocates a per-field staging `Vec<u8>`. The new `MetaField::write_into(&mut Vec<u8>)` writes the length header from a precomputed `value_len()` and appends value bytes straight into the destination buffer. `Meta::to_bytes` and `Meta::write_*_fields` use it; the public `MetaField::to_bytes` is now a thin wrapper.
- `Meta::read_v3` no longer clones the buffered header to re-parse it — it now chains an iterator over the verified body with the synthetic end marker.
- `Meta::read` and `MetaField::from_v1_header` use `[u8; N]` stack buffers instead of `Vec::take(N).collect()` for fixed-size headers.
- `inject` no longer materializes the metadata bit stream as a separate `Vec<u8>`; the meta byte buffer is reused via `flat_map(to_bits)` and chained directly with the payload bit stream. `meta_size` is now derived from the byte length, dropping the divmod-by-8.

### Refactoring
- `src/main.rs` split from ~1200 lines into focused modules: `meta.rs`, `lsb.rs`, `cli.rs`, `commands/{inject,extract,inspect}.rs`. `main.rs` itself is now ~20 lines (clap parse + dispatch).
- Crate is now both a library (`src/lib.rs`) and a binary.
- `MetaTag` is now a plain enum instead of being macro-generated; `From<MetaTag> for u8` is an explicit `match` (no `repr(u8)` cast).
- Meta parser tests converted to `insta` snapshots via `rstest` parametrization.
- `inject::inject`, `extract::extract`, `inspect::inspect` now take their `Args` by reference instead of by value.

### Hardening
- Project-wide strict lint policy enforced via `[lints]` in `Cargo.toml`:
  - `rust`: `warnings = "deny"`, `unsafe_code = "deny"`.
  - `clippy`: `all`, `pedantic`, `cargo` set to `deny`, plus explicit denies for `unwrap_used`, `expect_used`, `panic`, `unreachable`, `indexing_slicing`, `string_slice`, `arithmetic_side_effects`, `cast_possible_truncation`, `cast_sign_loss`, `cast_possible_wrap`, `cast_precision_loss`, `float_cmp`, `lossy_float_literal`, `as_conversions`, `default_numeric_fallback`. The codebase compiles cleanly under this policy with **zero `#[allow]` overrides outside of `#[cfg(test)]` modules** (where only `clippy::unwrap_used` is allowed).
- Library code is now panic-free under the policy: no `unwrap`/`expect`, no `panic!`/`unreachable!`, no raw indexing, no `as` conversions. All overflow-prone arithmetic uses `checked_*` / `saturating_*`. Numeric narrowing goes through `TryFrom` with explicit error variants (`PayloadTooLarge`, `MetaError::FieldTooLong`, etc.).
- `Meta::to_bytes` no longer panics on unsupported writer versions — returns `MetaError::UnsupportedWriteVersion(v)` instead of `unreachable!`.
- `inspect`'s `format_size` now formats with integer arithmetic (no `f32`, no precision loss).
