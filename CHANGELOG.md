# Changelog

All notable changes to annotation-mcp are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **OCR garbage filter** (`detection/garbage_filter.py`): heuristic
  rejection of high-confidence Tesseract noise (e.g. `VLOAOL`, `yOLO`,
  `||||`) on real-world photos. Default-on in `detect_text_regions`;
  set `filter_garbage=False` to disable. Catches ~98% of typical
  Tesseract garbage.
- **PSM-6 fallback** in `detect_text_regions`: if the default
  page-segmentation mode returns 0 regions, automatically retry with
  `--psm 6` (uniform block of text). Fixes the silent-failure trap
  where dense text images returned nothing.
- **Vision-model OCR fallback** via `crop_regions=True` on
  `detect_text_regions`: writes per-region image crops to disk and
  returns their paths in the structured output. The AI caller (or a
  vision-capable agent) can then re-recognize the cropped text with
  higher accuracy when tesseract is unreliable.
- **HEIC / iPhone photo support**: `image_io` registers `pillow_heif`
  so PIL decodes HEIC directly. Works on all three platforms (macOS,
  Linux, Windows). HEIC test fixtures are NOT committed (they may
  contain EXIF metadata); tests skip cleanly when no local HEIC is
  found.
- **EXIF orientation spec-compliance fix**: corrected
  `_apply_exif_orientation` to follow the EXIF 2.3 spec for all 8
  orientation values. Previously, `orient=6` was rotated 90° CCW
  instead of the spec-mandated 90° CW. Added 18 regression tests
  covering all 8 orientations + pixel-level rotation direction
  verification + equivalence with PIL's reference
  `ImageOps.exif_transpose`.
- **Cross-platform support**: Linux (Debian/Ubuntu/Fedora/Arch),
  Windows 10/11/Server 2019, and macOS 12+ all supported.
  Cross-platform CI on all 3 OSes with Python 3.12.
- **`detect_text_regions` parameters**: `filter_garbage` (bool, default
  True), `crop_regions` (bool, default False), `crop_padding` (float,
  default 0.1), `crop_output_dir` (str, default temp dir).

### Changed
- **Test fixtures**: `garden_photo.jpg` and `garden_photo2.jpg` have
  been EXIF-stripped (GPS, device info, timestamps removed) for safe
  public distribution. The orientation tag is preserved so EXIF
  rotation tests still pass.
- **`render.py` auto-output path**: now uses
  `tempfile.NamedTemporaryFile` (cross-platform temp dir) instead of
  hardcoded `/tmp/...`. Windows users no longer see paths under
  `C:\Users\...\AppData\Local\Temp\...` mismatches.
- **`pyproject.toml`**: added OS classifiers (macOS, Linux, Windows),
  project URLs, license (MIT), keywords, and authors metadata.

### Removed
- Removed `tests/fixtures/real_world/toyota_engine_sideways.jpg` (a
  private project photo not for public distribution). The real_world/
  test fixtures directory is now in `.gitignore` — public-domain
  samples can be added locally but should not be committed.

### Fixed
- **EXIF rotation** (`image_io.py`): orient=6 now rotates 90° CW
  per EXIF 2.3 spec (was 90° CCW).
- **ArrowAnnotation from_ alias** (`handlers.py`): the Python field
  name `from_` is now properly aliased to the Pydantic `from` field
  during MCP serialization, fixing the "Field required: from" error
  when calling `annotate` with arrow annotations.
- **Windows path safety**: removed all hardcoded `/tmp/...` paths
  from the source and tests; replaced with cross-platform
  `tempfile`/`tmp_path` fixtures.

### Security
- Test fixtures that contained real EXIF metadata (GPS coordinates,
  device serial numbers, timestamps) have been sanitized. The `.gitignore`
  now excludes all real photos, HEIC files, and originals to prevent
  accidental commits of private data.
- The `crop_for_inspection` and `annotate` tools preserve source EXIF
  in their output PNGs. If you need to strip it, use
  `PIL.Image.save(path, exif=b"")` or `exiftool -all= image.png`.

## [0.1.0] — Initial release

- 5 core annotation tools: `annotate`, `draw_bounding_box`,
  `highlight_region`, `draw_numbered_callouts`, `get_image_info`.
- 3 detection tools: `detect_barcodes` (pyzbar), `detect_text_regions`
  (Tesseract), `crop_for_inspection`.
- EXIF orientation handling (buggy for orient=6, fixed in Unreleased).
- HEIC support via pillow_heif.
- Pyproject.toml-based packaging with hatchling, installable via
  `uv tool install .` or `pip install .`.

[Unreleased]: https://github.com/aschokinatgmail/annotation-mcp/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/aschokinatgmail/annotation-mcp/releases/tag/v0.1.0
