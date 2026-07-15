"""Barcode and QR code detection via pyzbar.

Detects 1D barcodes (EAN-13, EAN-8, Code 128, Code 39, UPC, etc.) and
2D codes (QR, DataMatrix, PDF417, Aztec). Returns deterministic pixel
bounding boxes without ML — the zbar library uses classical image
processing (edge detection + scanline analysis).

Design notes:
    - Lazy import pyzbar so the rest of the MCP server works even if
      pyzbar/zbar is not installed.
    - Bounding boxes are returned in BOTH pixel and normalized
      [x1, y1, x2, y2] format so the caller can pass them straight to
      ``annotation_annotate`` without re-scaling.
    - Confidence is reported as a float in [0, 1] when zbar provides one;
      some barcode types don't carry confidence (we default to 1.0).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from PIL import Image

from annotation_mcp.image_io import load_image

if TYPE_CHECKING:
    from pyzbar.pyzbar import Decoded


@dataclass(frozen=True, slots=True)
class BarcodeMatch:
    """A single detected barcode/QR code."""

    type: str  # e.g. "EAN13", "QRCODE", "CODE128"
    data: str  # decoded payload
    bbox_pixels: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    bbox_normalized: tuple[float, float, float, float]
    confidence: float

    def to_dict(self, image_width: int, image_height: int) -> dict[str, object]:
        """Serialize to a JSON-safe dict (bbox_pixels re-derived for the caller's image)."""
        return {
            "type": self.type,
            "data": self.data,
            "bbox_pixels": list(self.bbox_pixels),
            "bbox_normalized": list(self.bbox_normalized),
            "confidence": self.confidence,
            "image_width": image_width,
            "image_height": image_height,
        }


class BarcodeDetectionError(RuntimeError):
    """Raised when barcode detection cannot run (missing deps, bad image, etc.)."""


def detect_barcodes(  # noqa: C901
    path: str,
    *,
    min_confidence: float = 0.0,
    allowed_types: list[str] | None = None,
) -> list[BarcodeMatch]:
    """Detect barcodes and QR codes in an image.

    Args:
        path: Absolute path to image file.
        min_confidence: Discard matches with confidence below this threshold (0-1).
        allowed_types: If provided, only return barcodes whose ``type`` is in this list
            (e.g. ``["EAN13", "QRCODE"]``). Case-insensitive match.

    Returns:
        List of :class:`BarcodeMatch` with pixel and normalized bounding boxes.

    Raises:
        BarcodeDetectionError: If pyzbar/zbar is not installed or the image cannot be loaded.
    """
    try:
        from pyzbar.pyzbar import decode
    except ImportError as e:
        raise BarcodeDetectionError(
            "pyzbar not installed."
            + " Install with: pip install pyzbar;"
            + " and the zbar shared library: brew install zbar (macOS)"
            + " or apt install libzbar0 (Linux)"
        ) from e

    try:
        img = load_image(path)
    except (FileNotFoundError, OSError) as e:
        raise BarcodeDetectionError(f"Cannot load image: {e}") from e

    width, height = img.size

    # pyzbar works on RGB or L mode PIL images.
    pil_img: Image.Image = img if img.mode in ("RGB", "L") else img.convert("RGB")

    decoded: list[Decoded] = decode(pil_img)

    allowed_set: set[str] | None = None
    if allowed_types is not None:
        allowed_set = {t.upper() for t in allowed_types}

    matches: list[BarcodeMatch] = []
    for d in decoded:
        type_name = _zbar_type_name(d.type)
        if allowed_set is not None and type_name.upper() not in allowed_set:
            continue

        # pyzbar rect: (x, y, w, h) — origin top-left.
        rect = cast("tuple[int, int, int, int]", d.rect)
        x, y, w, h = rect
        bbox_pixels: tuple[int, int, int, int] = (x, y, x + w, y + h)
        bbox_normalized: tuple[float, float, float, float] = (
            x / width,
            y / height,
            (x + w) / width,
            (y + h) / height,
        )

        # pyzbar doesn't expose a single confidence value; some barcodes
        # carry a quality attribute. Fall back to 1.0.
        quality = getattr(d, "quality", None)
        if quality is not None:
            confidence = float(cast("float", quality)) / 100.0
        else:
            confidence = 1.0
        if confidence < min_confidence:
            continue

        matches.append(
            BarcodeMatch(
                type=type_name,
                data=d.data.decode("utf-8", errors="replace"),
                bbox_pixels=bbox_pixels,
                bbox_normalized=bbox_normalized,
                confidence=confidence,
            )
        )

    return matches


def _zbar_type_name(type_value: object) -> str:
    """Map a pyzbar type value to a human-readable name like 'EAN13'."""
    # pyzbar.type can be either an int (ZBarSymbol enum) or a string in
    # newer versions. Handle both.
    if isinstance(type_value, str):
        return type_value
    # Map known ZBarSymbol constants to names
    name_map: dict[int, str] = {
        0: "EAN2",
        1: "EAN5",
        2: "EAN8",
        3: "UPCE",
        4: "ISBN10",
        5: "UPCA",
        6: "EAN13",
        7: "ISBN13",
        8: "COMPOSITE",
        9: "I25",
        10: "DATABAR",
        11: "DATABAR_EXP",
        12: "CODABAR",
        13: "CODE39",
        14: "PDF417",
        15: "QRCODE",
        16: "CODE93",
        17: "CODE128",
    }
    try:
        return name_map.get(int(str(type_value)), str(type_value))
    except (ValueError, TypeError):
        return str(type_value)
