"""Region cropping for iterative AI inspection.

Crops a sub-region of an image (with optional padding) and saves it to
a new file. This is the workflow accelerator: the AI can ask for a
high-res crop of a candidate region, then call ``look_at`` on the
crop to give more precise coordinates.

This is intentionally simple — no ML, no heuristics. Just crop and save.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from annotation_mcp.image_io import load_image, validate_input_path, validate_output_path


@dataclass(frozen=True, slots=True)
class CropResult:
    """Result of a region crop operation."""

    output_path: str
    width: int
    height: int
    x: int  # top-left x in source image
    y: int  # top-left y in source image
    bbox_pixels: tuple[int, int, int, int]  # (x, y, x2, y2) — region actually extracted
    source_width: int
    source_height: int

    def to_dict(self) -> dict[str, object]:
        return {
            "output_path": self.output_path,
            "width": self.width,
            "height": self.height,
            "x": self.x,
            "y": self.y,
            "bbox_pixels": list(self.bbox_pixels),
            "source_width": self.source_width,
            "source_height": self.source_height,
        }


class CropError(ValueError):
    """Raised when a crop operation fails (bad path, invalid bbox, etc.)."""


def crop_region(  # noqa: PLR0913
    path: str,
    bbox: tuple[float, float, float, float],
    *,
    output_path: str | None = None,
    normalized: bool = False,
    padding: float = 0.0,
    auto_output: bool = True,
) -> CropResult:
    """Crop a region from an image and save it to a new file.

    Args:
        path: Absolute path to source image.
        bbox: ``[x1, y1, x2, y2]`` region to crop. May be in pixels or
            normalized [0, 1] coordinates (set ``normalized=True``).
        output_path: Where to save the cropped image. If None and ``auto_output`` is
            True, a name like ``{stem}_crop_{x}_{y}_{w}_{h}{suffix}`` is generated
            in the same directory as the source.
        normalized: If True, bbox is in [0, 1] coordinates.
        padding: Fractional padding to add around the bbox (e.g. 0.1 = 10% on each side).
            Useful for capturing context around a tight region.
        auto_output: If True and ``output_path`` is None, auto-generate a filename.

    Returns:
        :class:`CropResult` with the saved file path and crop geometry.

    Raises:
        CropError: If paths are invalid, the bbox is malformed, or the image
            cannot be loaded.
    """
    try:
        _validated = validate_input_path(path)
    except (FileNotFoundError, ValueError) as e:
        raise CropError(str(e)) from e
    del _validated

    try:
        img = load_image(path)
    except (FileNotFoundError, OSError) as e:
        raise CropError(f"Cannot load image: {e}") from e

    src_w, src_h = img.size

    # Convert bbox to pixel coords with clamp.
    if normalized:
        x1 = bbox[0] * src_w
        y1 = bbox[1] * src_h
        x2 = bbox[2] * src_w
        y2 = bbox[3] * src_h
    else:
        x1, y1, x2, y2 = bbox

    if x2 <= x1 or y2 <= y1:
        raise CropError(
            f"Invalid bbox: x2 ({x2}) must be > x1 ({x1}) and y2 ({y2}) must be > y1 ({y1})"
        )

    # Apply padding (as a fraction of the bbox dimensions).
    if padding:
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        pad_x = bbox_w * padding
        pad_y = bbox_h * padding
        x1 -= pad_x
        y1 -= pad_y
        x2 += pad_x
        y2 += pad_y

    # Clamp to image bounds and convert to int.
    x1_i = max(0, int(round(x1)))
    y1_i = max(0, int(round(y1)))
    x2_i = min(src_w, int(round(x2)))
    y2_i = min(src_h, int(round(y2)))

    if x2_i <= x1_i or y2_i <= y1_i:
        raise CropError(
            f"Crop region is empty after clamping: ({x1_i}, {y1_i}) -> ({x2_i}, {y2_i})"
        )

    cropped: Image.Image = img.crop((x1_i, y1_i, x2_i, y2_i))

    # Resolve output path.
    if output_path is None and auto_output:
        src = Path(path)
        suffix = src.suffix or ".png"
        output_path = str(
            src.parent / f"{src.stem}_crop_{x1_i}_{y1_i}_{x2_i - x1_i}_{y2_i - y1_i}{suffix}"
        )
    elif output_path is None:
        raise CropError("output_path is required when auto_output=False")

    try:
        _validated = validate_output_path(output_path)
    except ValueError as e:
        raise CropError(str(e)) from e
    del _validated

    # Preserve source format if possible.
    src_format = (img.format or "PNG").upper()
    save_format = src_format if src_format in ("PNG", "JPEG", "WEBP", "BMP", "TIFF") else "PNG"
    save_kwargs: dict[str, object] = {}
    if save_format == "JPEG" and cropped.mode in ("RGBA", "LA", "P"):
        cropped = cropped.convert("RGB")
    if save_format == "JPEG":
        save_kwargs["quality"] = 95

    try:
        cropped.save(output_path, format=save_format, **save_kwargs)
    except OSError as e:
        raise CropError(f"Failed to save crop: {e}") from e

    return CropResult(
        output_path=output_path,
        width=cropped.width,
        height=cropped.height,
        x=x1_i,
        y=y1_i,
        bbox_pixels=(x1_i, y1_i, x2_i, y2_i),
        source_width=src_w,
        source_height=src_h,
    )
