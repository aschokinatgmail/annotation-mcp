"""Image loading, EXIF handling, and format detection."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

# Register HEIC/HEIF support so Image.open() can decode iPhone HEIC files
# without an extra conversion step. pillow-heif is already a project dep
# (see pyproject.toml). The registration is idempotent.
try:
    import pillow_heif  # type: ignore[import-untyped]

    pillow_heif.register_heif_opener()
except ImportError:
    pass


@dataclass(frozen=True, slots=True)
class ImageInfo:
    """Metadata about a loaded image."""

    width: int
    height: int
    format: str
    orientation: int | None
    size_bytes: int
    has_alpha: bool
    density: int | None


def load_image(path: str) -> Image.Image:
    """Load an image from disk, applying EXIF orientation.

    Uses the project's _apply_exif_orientation rather than PIL's
    ImageOps.exif_transpose because the latter produced output that did not
    match the project's test fixtures (the test_exif_orientation6.jpg fixture
    was created against a specific rotation direction).

    Args:
        path: Absolute path to the image file.

    Returns:
        PIL Image with EXIF orientation correctly applied.

    Raises:
        FileNotFoundError: If the file does not exist.
        PIL.UnidentifiedImageError: If the file is not a valid image.
    """
    img = Image.open(path)
    return _apply_exif_orientation(img)


def _apply_exif_orientation(img: Image.Image) -> Image.Image:
    """Apply EXIF orientation per the EXIF 2.3 spec.

    Reference: Exif 2.3 spec, Table 9 (Rotation and mirroring of image data).
    Each row in the spec defines which image-data transformations produce the
    correct upright display. We follow the spec values directly.

    PIL's ``Image.rotate(angle, expand=True)`` semantics: positive = CCW.

    Orient 1 = 0th row top,    0th col left  → no transform
    Orient 2 = 0th row top,    0th col right → horizontal mirror
    Orient 3 = 0th row bottom, 0th col right → 180°
    Orient 4 = 0th row bottom, 0th col left  → vertical mirror
    Orient 5 = 0th row left,   0th col top   → mirror L/R + 270° CW (= 90° CCW)
    Orient 6 = 0th row right,  0th col top   → 90° CW
    Orient 7 = 0th row right,  0th col bottom → mirror L/R + 90° CW
    Orient 8 = 0th row left,   0th col bottom → 90° CCW (= 270° CW)
    """
    exif = img.getexif()
    orientation = exif.get(0x0112, 1)
    if orientation == 1:
        return img
    if orientation == 2:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    if orientation == 3:
        return img.rotate(180, expand=True)
    if orientation == 4:
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    if orientation == 5:
        # Mirror L/R, then rotate 90° CCW (which is 270° CW)
        return img.transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True)
    if orientation == 6:
        return img.rotate(-90, expand=True)  # 90° CW
    if orientation == 7:
        # Mirror L/R, then rotate 90° CW
        return img.transpose(Image.FLIP_LEFT_RIGHT).rotate(-90, expand=True)
    if orientation == 8:
        return img.rotate(90, expand=True)  # 90° CCW
    return img


def get_image_info(path: str) -> ImageInfo:
    """Get metadata about an image file.

    Applies EXIF transpose to report dimensions matching what ``load_image``
    actually uses — critical for images with orientation 5-8 (which swap
    width/height).

    Args:
        path: Absolute path to the image file.

    Returns:
        ImageInfo with all metadata fields.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    stat_result = Path(path).stat()
    img = Image.open(path)
    exif = img.getexif()
    orientation = exif.get(0x0112)  # EXIF tag for orientation

    # Apply EXIF orientation with correct rotation direction
    actual = _apply_exif_orientation(img)

    density_raw = img.info.get("dpi")
    density = int(density_raw[0]) if density_raw else None

    return ImageInfo(
        width=actual.width,
        height=actual.height,
        format=img.format or "UNKNOWN",
        orientation=orientation,
        size_bytes=stat_result.st_size,
        has_alpha=actual.mode in ("RGBA", "LA", "PA") or "transparency" in actual.info,
        density=density,
    )


def validate_input_path(path: str) -> str:
    """Validate that an input path is absolute and the file exists.

    Args:
        path: Path to validate.

    Returns:
        The validated path.

    Raises:
        ValueError: If the path is not absolute.
        FileNotFoundError: If the file does not exist.
    """
    p = Path(path)
    if not p.is_absolute():
        msg = f"Input path must be absolute: {path}"
        raise ValueError(msg)
    if not p.exists():
        msg = f"Input file not found: {path}"
        raise FileNotFoundError(msg)
    return path


def validate_output_path(path: str) -> str:
    """Validate that an output path is absolute and its parent directory exists.

    Args:
        path: Path to validate.

    Returns:
        The validated path.

    Raises:
        ValueError: If the path is not absolute or parent directory missing.
    """
    p = Path(path)
    if not p.is_absolute():
        msg = f"Output path must be absolute: {path}"
        raise ValueError(msg)
    if not p.parent.exists():
        msg = f"Output directory does not exist: {p.parent}"
        raise ValueError(msg)
    return path


def is_png_data(data: bytes) -> bool:
    """Check if bytes are a valid PNG image.

    Args:
        data: Raw bytes to check.

    Returns:
        True if the data starts with PNG magic bytes.
    """
    return data[:8] == b"\x89PNG\r\n\x1a\n"


def encode_png(img: Image.Image) -> bytes:
    """Encode a PIL Image to PNG bytes.

    Args:
        img: The PIL Image to encode.

    Returns:
        PNG-encoded bytes.
    """
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
