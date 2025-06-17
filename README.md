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

### Extracting a file

To extract an embedded file from an image:

```bash
injet extract output.png
```

This will create the original file (e.g., `some_file.txt`) in the current directory.

### Inspecting an image

To check whether an image contains embedded data and to see the maximum supported file size:

```bash
injet inspect image.png
```

### Command-line help

You can always see available commands and options using:

```bash
injet --help
```