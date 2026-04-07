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

### Inspecting an image

To check whether an image contains embedded data and to see the maximum supported file size:

```bash
injet inspect image.png
```

#### inspect options

| Flag/Option   | Description                                                                 |
|---------------|-----------------------------------------------------------------------------|
| `--seed SEED` | Seed string for pseudorandom data location. Must match the one used during injection. |

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

### Command-line help

You can always see available commands and options using:

```bash
injet --help
```