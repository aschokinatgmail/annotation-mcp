# annotation-mcp

[![CI](https://github.com/aschokinatgmail/annotation-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/aschokinatgmail/annotation-mcp/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platforms](https://img.shields.io/badge/platforms-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey)](#supported-platforms)

MCP (Model Context Protocol) server for image annotation. Draws bounding boxes, arrows, highlights, callouts, text, and circles on images. Also detects barcodes and text regions, and crops image regions for iterative inspection.

## Supported Platforms

| OS | Versions | Python | Notes |
|---|---|---|---|
| **macOS** | 12+ (Monterey) | 3.12+ | Tested on Apple Silicon and Intel |
| **Linux** | Ubuntu 22.04+, Debian 12+, Fedora 38+ | 3.12+ | glibc-based, x86_64 and aarch64 |
| **Windows** | 10, 11, Server 2019+ | 3.12+ | x86_64 only |

## Tools

| Tool | Purpose |
|---|---|
| `get_image_info` | Read image metadata (dimensions, format, EXIF orientation). |
| `annotate` | Draw multiple annotations on one image (bbox, arrow, highlight, callout, text, circle). |
| `draw_bounding_box` | Single bounding box (convenience wrapper around `annotate`). |
| `highlight_region` | Semi-transparent rectangular highlight. |
| `draw_numbered_callouts` | Numbered circle callouts. |
| `detect_barcodes` | Detect 1D/2D barcodes (EAN, UPC, QR, etc.) with exact pixel bboxes. |
| `detect_text_regions` | Detect text regions with Tesseract OCR. Returns text + bbox + confidence. |
| `crop_for_inspection` | Crop a region to a new file. Used for iterative AI inspection. |

## Installation

### 1. System dependencies

Tesseract OCR and zbar barcode library are required for `detect_text_regions` and `detect_barcodes`. The other 5 tools (`annotate`, `draw_bounding_box`, etc.) work without them.

#### macOS

```bash
brew install tesseract tesseract-lang zbar
```

This installs:
- `tesseract` (OCR engine)
- `tesseract-lang` (163 language packs including `eng`, `rus`, `jpn`, `chi_sim`, `deu`, `fra`, etc.)
- `zbar` (barcode scanning library, including `libzbar.dylib`)

Verify:
```bash
tesseract --version
tesseract --list-langs   # should show 'eng' and 'rus'
```

#### Linux (Debian/Ubuntu)

```bash
sudo apt-get update
sudo apt-get install -y \
  tesseract-ocr \
  tesseract-ocr-eng \
  tesseract-ocr-rus \
  libzbar0 \
  zbar-tools
```

For Fedora / RHEL:
```bash
sudo dnf install -y tesseract tesseract-langpack-eng tesseract-langpack-rus zbar
```

For Arch:
```bash
sudo pacman -S tesseract tesseract-data-eng zbar
```

Verify:
```bash
tesseract --version
tesseract --list-langs
```

#### Windows

The recommended install is via [Chocolatey](https://chocolatey.org/):

```powershell
choco install --no-progress tesseract -y
choco install --no-progress python3 -y  # if you don't have Python yet
```

After install, ensure `C:\Program Files\Tesseract-OCR` is on your PATH (the choco package should do this automatically; you may need to restart your terminal).

`pyzbar` ships its own copy of `libzbar-0.dll` bundled in the wheel on Windows, so no separate zbar install is needed.

Verify in PowerShell:
```powershell
tesseract --version
tesseract --list-langs
```

### 2. Python package

The recommended Python install uses [uv](https://docs.astral.sh/uv/) (fast, hermetic).

#### macOS / Linux

```bash
# Install via uv (creates an isolated tool install)
uv tool install annotation-mcp

# Or with pipx (similar to uv tool, classic alternative)
pipx install annotation-mcp

# Or with pip
pip install annotation-mcp
```

The `annotation-mcp` command will be installed and the server can be registered in your MCP client config (e.g. `~/.config/opencode/opencode.json`):

```json
{
  "mcp": {
    "annotation": {
      "type": "local",
      "command": ["annotation-mcp"],
      "enabled": true
    }
  }
}
```

#### Windows

```powershell
# With uv (recommended)
uv tool install annotation-mcp

# With pip
pip install annotation-mcp
```

The `annotation-mcp.exe` will be installed. Same MCP client config works.

### 3. Install from source (development)

```bash
git clone https://github.com/aschokinatgmail/annotation-mcp.git
cd annotation-mcp
uv sync --all-extras --dev
uv tool install --force --reinstall .
```

## Detection tools — workflow

The detection tools are designed to solve the "where is X in this image?" problem. They return structured text + bbox data so the AI can match semantic labels to coordinates deterministically, rather than guessing from approximate vision.

### Typical workflow

1. **Call `detect_text_regions`** to get all text regions in the image with their content, bounding boxes, and confidence scores.
2. **Match semantic label to region** by inspecting the `text` field. E.g., for "find the brand title", look for the region whose `text` matches the brand name.
3. **Refine if needed** — if a region is split across multiple detections (e.g., the brand title is "ФИТОС" + "ПОРИН"), use the `crop_for_inspection` tool to zoom in and re-detect, or extend the bbox using a y-band heuristic.
4. **Call `annotate`** with the discovered bboxes.

### Example: annotating a product package

```python
# Step 1: Find all text regions
text_matches = detect_text_regions(image_path, lang="eng+rus", detail="word", min_confidence=30)
# Returns 400+ regions with text content + bbox_pixels

# Step 2: Find the brand title region
brand_region = next(r for r in text_matches if "ФИТО" in r["text"].upper())
# Returns: {text: "ФИТОС", bbox_pixels: [1019, 786, 1300, 881], confidence: 88, ...}

# Step 3: Find the barcode (deterministic decoding)
barcodes = detect_barcodes(image_path)
# Returns: [{type: "EAN13", data: "4607026420155", bbox_pixels: [2366, 3403, 2622, 3994], ...}]

# Step 4: Find the dosage table by keyword
table_regions = [r for r in text_matches if any(kw in r["text"].lower() for kw in ["мл", "м²", "капля", "растен"])]
# Take the union bbox

# Step 5: Annotate
annotate(image_path, output_path="result.png", normalized=True, annotations=[
    {"type": "bbox", "bbox": brand_region["bbox_normalized"], "label": "Brand", "color": "#FF0000"},
    {"type": "bbox", "bbox": barcodes[0]["bbox_normalized"], "label": "Barcode", "color": "#00AA00"},
    {"type": "bbox", "bbox": table_bbox_normalized, "label": "Table", "color": "#0066FF"},
])
```

### When tesseract quality is poor: re-recognize with a vision model

Tesseract produces reliable output on clean documents but can return
high-confidence nonsense on real-world photos (small, low-contrast, curved,
embossed text). The `detect_text_regions` tool has two features for
handling this:

- **`filter_garbage`** (default `True`): rejects OCR-noise regions
  (e.g. `VLOAOL`, `yOLO`, `||||`) based on a heuristic that combines
  text length, character-class diversity, vowel ratio, dictionary
  presence, and part-number pattern. Catches ~98% of typical Tesseract
  garbage on real photos.

- **`crop_regions`** (default `False`): when set to `True`, the tool
  also writes per-region image crops to disk and returns their paths in
  the result. A vision model can then re-recognize each crop with
  higher accuracy. The workflow:

```python
# Step 1: get tesseract bboxes + cropped images for each region
result = handle_detect_text_regions(
    photo_path,
    lang="eng", detail="word", preprocess="clahe",
    crop_regions=True,
    crop_padding=0.15,
)
regions = result.structuredContent["regions"]
crops = result.structuredContent["region_crops"]

# Step 2: for each crop, send to vision model for re-recognition
for crop in crops:
    vision_text = recognize_with_vision_model(crop["crop_path"])
    # vision_text is the actual text in that region
```

### When a region is split across multiple detections

OCR can split a single visual line into multiple text regions (especially with stylized fonts or low-resolution scans). To recover the full region:

1. Find a single word region whose text you recognize.
2. Compute the y-band: `y_band = (bbox_bottom - bbox_top) * 0.6` around the y-center.
3. Find all regions in the same y-band.
4. Take the union bbox (min/max of x1,y1,x2,y2).
5. Optionally add padding.

### Preprocessing modes for `detect_text_regions`

| Mode | Use case |
|---|---|
| `none` | Clean scans, screenshots, high-contrast documents. |
| `clahe` | **Default.** Photos with uneven lighting (most product photos). |
| `adaptive` | High-contrast text on flat backgrounds (signs, screenshots). |
| `otsu` | Clean black-and-white text (book pages, printed labels). |

If opencv-python is not installed, only `none` is available; the others fall back to PIL autocontrast.

## Graceful degradation

If system dependencies are missing, the detection tools return clear error messages instead of crashing:

- `detect_barcodes` without pyzbar/zbar: `"pyzbar not installed. Install with: pip install pyzbar; and the zbar shared library: brew install zbar (macOS) or apt install libzbar0 (Linux)"`
- `detect_text_regions` without tesseract: `"tesseract binary not found. Install with: brew install tesseract tesseract-lang"`
- `detect_text_regions` with missing language pack: `"Tesseract language pack(s) not installed: ['rus']. Install with: brew install tesseract-lang"`

The 5 original annotation tools (`annotate`, `draw_bounding_box`, `highlight_region`, `draw_numbered_callouts`, `get_image_info`) work without any system dependencies.

## HEIC / iPhone photo support

iPhone photos are stored as `.HEIC` files. The `image_io` module
registers `pillow_heif` at import time so PIL can decode HEIC directly.
This works on all three platforms with no extra setup.

HEIC test fixtures are **not** committed to the repo (HEIC files can
contain EXIF metadata with GPS coordinates and device info). The HEIC
tests skip cleanly when no local HEIC file is found. To exercise them
locally, place a HEIC file at any of:

- `tests/fixtures/exif_orientation.heic`
- `tests/fixtures/sample.heic`
- `~/Pictures/sample.heic`

## Coordinate systems

All tools accept either pixel coordinates (`[x1, y1, x2, y2]` in absolute pixels) or normalized coordinates (`[0.0, 0.0, 1.0, 1.0]` fractions of image dimensions). Set the `normalized` flag accordingly.

Output bboxes are always provided in **both** pixel and normalized form for the caller's convenience.

## Development

```bash
# Clone and install dev dependencies
git clone https://github.com/aschokinatgmail/annotation-mcp.git
cd annotation-mcp
uv sync --all-extras --dev

# Run tests (cross-platform)
uv run pytest

# Run tests with coverage
uv run pytest --cov=annotation_mcp

# Type check (strict mode)
uv run basedpyright

# Lint
uv run ruff check
```

CI runs on every push and PR across **macOS, Linux, and Windows** with Python 3.12:
[`.github/workflows/ci.yml`](.github/workflows/ci.yml)

## Architecture

- `image_io.py` — Image loading with EXIF orientation handling (all 8 spec values), HEIC support.
- `process.py` / `render.py` — Annotation validation and PNG rendering.
- `coords.py` — Coordinate conversion (pixel ↔ normalized).
- `handlers.py` / `handlers_detection.py` — MCP tool handlers.
- `server.py` — FastMCP server registration (8 tools).
- `detection/` — Barcode (pyzbar), OCR (tesseract + garbage filter), and crop modules.

## Privacy & security notes

- **EXIF data**: Real photos often contain GPS coordinates, device info, and timestamps. The test fixtures in this repo have all such metadata stripped. The `crop_for_inspection` and `annotate` tools DO NOT strip EXIF from output images by default — if you save annotated images, the EXIF from the source is preserved. Strip it explicitly with PIL or `exiftool` if needed.
- **No telemetry**: This server does not collect or transmit any data. All processing is local.
- **No file uploads**: The server reads images from the local filesystem at the path you provide. It does not fetch images from URLs.

## License

MIT.
