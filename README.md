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
| `-d`, `--destination PATH` | Where to save the extracted file. If not set, uses filename from metadata or "cargo".      | metadata/cargo    |
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

### Command-line help

You can always see available commands and options using:

```bash
injet --help
```