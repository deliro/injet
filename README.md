# Injet

## Overview

Injet is a command-line tool that allows embedding arbitrary files into PNG images using the Least Significant Bit (LSB) method.  
It can also retrieve embedded files and inspect image capacity for embedded data.

This tool is intended for educational, archival, and personal data embedding use cases.

## Installation

To install Injet using Cargo:

```bash
cargo install injet
```

## Usage

### Embedding a file

To embed a file into a PNG image:

```bash
injet inject some_file.txt some_image.png > output.png
```

*If the image is not a PNG or uses a different color format than RGBA8,  
it will be automatically converted.*

> **Note:** Starting with `1.0.0`, `inject` writes a new `VERSION_3` metadata
> format that includes a CRC32 over the metadata header (filename + size),
> allowing tampering to be detected on extract. Files written by `1.0.0` are
> **not** readable by older binaries (`0.3.0` and earlier). `1.0.0` still reads
> files produced by older versions (`VERSION_1` and `VERSION_2`).

#### inject options

| Flag/Option                | Description                                                                                 | Default           |
|----------------------------|---------------------------------------------------------------------------------------------|-------------------|
| `-d`, `--destination PATH` | Where to save the resulting image. If not set, writes to stdout (if not terminal), else "modified.png" | stdout or file    |
| `-w`, `--write-meta BOOL`  | Whether to write metadata (filename and size). If false, extraction requires `--read-meta=false` and `--read-size`. | true              |
| `--compression LEVEL`      | PNG compression: `default`, `fast`, `best`.                                                 | default           |
| `--seed SEED`              | Use a seed string for pseudorandom bit placement. Must match during extraction/inspection.   | (none)            |
| `--psk-file PATH`          | PSK passphrase file (one trailing newline stripped, max 4 KiB). Mutually exclusive with `--sign-key`. | (none) |
| `--psk-env VAR`            | PSK passphrase from environment variable.                                                   | (none)            |
| `--psk-prompt`             | Prompt for PSK passphrase on a TTY.                                                         | (none)            |
| `--sign-key PATH`          | OpenSSH Ed25519 private key path. Mutually exclusive with `--psk-*`.                        | (none)            |
| `--sign-key-passphrase-file PATH` | Passphrase file for an encrypted Ed25519 private key.                              | (none)            |
| `--sign-key-passphrase-env VAR`   | Passphrase env var for an encrypted Ed25519 private key.                           | (none)            |
| `--sign-key-passphrase-prompt`    | Prompt for the encrypted Ed25519 private key passphrase.                           | (none)            |

### Extracting a file

To extract an embedded file from an image:

```bash
injet extract output.png
```

This will create the original file (e.g., `some_file.txt`) in the current directory.

#### extract options

| Flag/Option                | Description                                                                                 | Default           |
|----------------------------|---------------------------------------------------------------------------------------------|-------------------|
| `-d`, `--destination PATH` | Where to save the extracted file. If not set, uses filename from metadata or "payload".    | metadata/payload  |
| `--read-meta BOOL`         | Whether to read metadata. If false, you must specify `--read-size`.                         | true              |
| `--read-size BYTES`        | How many bytes to extract. If not set, uses metadata or extracts as much as possible.        | metadata/max      |
| `--seed SEED`              | Seed string for pseudorandom data location. Must match the one used during injection.        | (none)            |
| `--psk-file PATH` / `--psk-env VAR` / `--psk-prompt` | PSK passphrase source (same semantics as inject).                   | (none)            |
| `--verify-key PATH`        | OpenSSH Ed25519 public key for verification.                                                | (none)            |
| `--verify-key-env VAR`     | Inline OpenSSH public key from environment variable (`ssh-ed25519 AAAA... [comment]`).      | (none)            |
| `--insecure-skip-verify`   | Extract a signed container without verifying its signature. Always prints a stderr `WARNING:`. | false          |

### Inspecting an image

To check whether an image contains embedded data and to see the maximum supported file size:

```bash
injet inspect image.png
```

#### inspect options

| Flag/Option   | Description                                                                 |
|---------------|-----------------------------------------------------------------------------|
| `--seed SEED` | Seed string for pseudorandom data location. Must match the one used during injection. |

### Authentication

The existing `Hash` TLV is a CRC32 that detects accidental corruption only — it cannot detect adversarial tampering. Authentication is an opt-in feature that adds cryptographic integrity and authenticity to containers. Two modes are available:

- **Pre-shared passphrase (PSK)**: symmetric. Both the injecting and extracting side share the same passphrase. Argon2id derives a key from the passphrase (m=64 MiB, t=3, p=4), and a BLAKE3 keyed-hash MAC is computed over the container. A 48-byte `Mac` TLV is appended to the header (16-byte per-container random salt + 32-byte MAC). Best suited for "send to self" or "trusted peer" scenarios.
- **Ed25519 signatures**: asymmetric. The injecting side signs with an OpenSSH Ed25519 private key; the extracting side verifies with the public key. A 64-byte `Signature` TLV is appended. Best suited for "publish to many verifiers" scenarios where the signing key must remain private.

At most one auth TLV is present per container. PSK and Ed25519 flags are mutually exclusive.

#### Generating an Ed25519 keypair

```bash
ssh-keygen -t ed25519 -f ~/.injet/injet_key -C "injet@laptop"
```

The private key can optionally be encrypted with a passphrase by passing `-N <passphrase>` (or omit `-N` and enter one interactively). The resulting `injet_key.pub` is the public key used for verification.

#### Inject with PSK

```bash
injet inject payload.txt cover.png --psk-file ~/.injet/passphrase.txt -d signed.png
```

Alternative passphrase sources: `--psk-env VAR` reads from an environment variable; `--psk-prompt` prompts interactively on a TTY.

> **Note:** The first PSK inject (and extract) pays a one-shot Argon2id cost of approximately 80–150 ms on a typical laptop. This cost is intentional — it defends against offline brute-force attacks on weak passphrases.

#### Inject with Ed25519

```bash
injet inject payload.txt cover.png --sign-key ~/.injet/injet_key -d signed.png
```

If the private key is encrypted, provide its passphrase via `--sign-key-passphrase-file PATH`, `--sign-key-passphrase-env VAR`, or `--sign-key-passphrase-prompt`.

#### Strict-by-default verification on extract

On extract, `injet` enforces a strict policy:

| Container | Key supplied | Outcome |
|-----------|-------------|---------|
| Unsigned  | None        | Extract normally. |
| Unsigned  | Any key     | Reject (`ContainerNotSigned`), exit 2. |
| Signed    | None        | Reject (`KeyRequired`), exit 2. |
| Signed    | Wrong algorithm | Reject (`AlgorithmMismatch`), exit 2. |
| Signed    | Correct key | Verify; reject on failure (`VerificationFailed`), exit 2. Payload is never written on failure. |
| Signed    | `--insecure-skip-verify` | Extract and print `WARNING:` to stderr, exit 0. |

#### Extract examples

```bash
# PSK
injet extract signed.png --psk-file ~/.injet/passphrase.txt -d extracted.txt

# Ed25519
injet extract signed.png --verify-key ~/.injet/injet_key.pub -d extracted.txt

# Skip verification (always prints WARNING to stderr)
injet extract signed.png --insecure-skip-verify -d extracted.txt

# Tampered container — exits 2, no file written
injet extract tampered.png --verify-key ~/.injet/injet_key.pub
```

#### `--write-meta=false` interaction

Signing requires the metadata header to carry the auth TLV. Combining `--write-meta=false` with any signing flag (`--psk-*` or `--sign-key`) is rejected immediately with exit code 2 (`SigningRequiresMetadata`).

#### Inspect output for signed containers

```bash
injet inspect signed.png
```

For a PSK-signed container the output includes `Signed: psk`; for an Ed25519-signed container it includes `Signed: ed25519`. Unsigned containers show nothing for this field.

#### Downgrade-protection limitation

> **Warning:** Older `injet` binaries (anything released before this authentication feature — e.g. `1.0.0`) silently skip unknown TLVs and will extract a signed container **without performing any verification**. Both ends must use this release or newer for signature verification to be effective.

#### Exit codes

| Code | Meaning |
|------|---------|
| 0    | Success |
| 1    | Generic error (legacy, unchanged) |
| 2    | Auth or key error (`KeyError`, `AuthError`, `SigningRequiresMetadata`) |

#### Complete worked example

```bash
# Generate a keypair (no passphrase on the private key)
ssh-keygen -t ed25519 -f ~/.injet/injet_key -N "" -C "injet@laptop"

# Prepare a payload
echo "secret" > /tmp/payload.txt

# Sign and inject
injet inject /tmp/payload.txt cover.png --sign-key ~/.injet/injet_key -d signed.png

# Inspect — should show "Signed: ed25519"
injet inspect signed.png

# Extract and verify
injet extract signed.png --verify-key ~/.injet/injet_key.pub -d /tmp/extracted.txt
diff /tmp/payload.txt /tmp/extracted.txt   # empty diff — payloads match

# Tampered scenario
echo "lol modified" > signed.png   # destructive tamper
injet extract signed.png --verify-key ~/.injet/injet_key.pub
# → exits 2, "signature verification failed" on stderr, no file written
```

### Compatibility

Injet uses a versioned metadata format. The policy is:

- `1.0.0` writes `VERSION_3` by default. `VERSION_3` adds a `MetaHash` TLV
  (CRC32 over the header) so corruption or tampering of the filename/size
  fields is detected at extract time.
- `1.0.0` reads `VERSION_1`, `VERSION_2`, and `VERSION_3`, so containers
  produced by older versions of `injet` continue to extract correctly.
- Older `injet` binaries (`0.3.0` and earlier) **cannot** read containers
  written by `1.0.0`. Pin to `0.3.0` if you need to interoperate with an
  older reader.
- The `--seed` pseudoshuffle algorithm changed in `1.0.0`: containers
  injected with `--seed` under `0.3.0` cannot be extracted by `1.0.0`.
  Containers injected without `--seed` are unaffected.
- The first positional argument of `inject` is now `<PAYLOAD>` (was
  `<CARGO>`), and the default extracted filename when no metadata is
  present is now `payload` (was `cargo`). Positional ordering is
  unchanged, so existing scripts that pass arguments by position keep
  working without modification.
- Unsigned containers from any prior version (`VERSION_1`, `VERSION_2`,
  `VERSION_3`) still extract correctly; unsigned `inject` behavior is
  unchanged.
- The new `Mac` (TLV tag 5) and `Signature` (TLV tag 6) TLVs live inside
  `VERSION_3` — no version bump. The wire format extension is backward
  compatible for readers that implement proper unknown-tag skipping.
- **Older `injet` binaries (anything before this authentication release,
  including `1.0.0`) silently skip unknown TLVs and will extract a signed
  container without performing any verification.** Both ends must use the
  new release for signature verification to be effective.
- `--write-meta=false` is now mutually exclusive with any signing flag
  (`--psk-*`, `--sign-key`). Combining them is rejected with exit code 2.

### Command-line help

You can always see available commands and options using:

```bash
injet --help
```