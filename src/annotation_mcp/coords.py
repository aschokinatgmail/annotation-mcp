"""Coordinate normalization, validation, and clamping."""

from __future__ import annotations

from dataclasses import dataclass

NORMALIZED_EPSILON = 0.001
"""Tolerance for normalized coordinate bounds checks."""


class CoordinateError(ValueError):
    """Raised when coordinates are invalid."""


@dataclass(frozen=True, slots=True)
class NormalizedBounds:
    """Validated normalized coordinate bounds [0-1]."""

    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True, slots=True)
class PixelBounds:
    """Pixel coordinate bounds after denormalization and clamping."""

    x1: int
    y1: int
    x2: int
    y2: int


def validate_normalized(value: float, name: str) -> None:
    """Validate a single normalized coordinate value is within [-epsilon, 1+epsilon].

    Args:
        value: The coordinate value to validate.
        name: The name of the coordinate (for error messages).

    Raises:
        CoordinateError: If the value is outside the acceptable range.
    """
    if value < -NORMALIZED_EPSILON or value > 1.0 + NORMALIZED_EPSILON:
        msg = (
            f"Normalized coordinate {name}={value} is outside "
            f"[-{NORMALIZED_EPSILON}, {1 + NORMALIZED_EPSILON}]"
        )
        raise CoordinateError(msg)


def validate_normalized_bbox(
    x1: float, y1: float, x2: float, y2: float
) -> NormalizedBounds:
    """Validate a normalized bounding box.

    Args:
        x1: Top-left x (normalized 0-1).
        y1: Top-left y (normalized 0-1).
        x2: Bottom-right x (normalized 0-1).
        y2: Bottom-right y (normalized 0-1).

    Returns:
        Validated NormalizedBounds.

    Raises:
        CoordinateError: If any coordinate is out of range.
    """
    validate_normalized(x1, "x1")
    validate_normalized(y1, "y1")
    validate_normalized(x2, "x2")
    validate_normalized(y2, "y2")
    return NormalizedBounds(x1=x1, y1=y1, x2=x2, y2=y2)


def validate_normalized_point(x: float, y: float) -> None:
    """Validate a normalized point coordinate.

    Args:
        x: X coordinate (normalized 0-1).
        y: Y coordinate (normalized 0-1).

    Raises:
        CoordinateError: If any coordinate is out of range.
    """
    validate_normalized(x, "x")
    validate_normalized(y, "y")


def denormalize_bbox(
    bounds: NormalizedBounds, width: int, height: int
) -> PixelBounds:
    """Convert normalized bbox to pixel coordinates.

    Args:
        bounds: Normalized bounds (0-1 fractions).
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        PixelBounds with clamped integer coordinates.
    """
    x1 = round(bounds.x1 * width)
    y1 = round(bounds.y1 * height)
    x2 = round(bounds.x2 * width)
    y2 = round(bounds.y2 * height)
    return clamp_bbox(x1, y1, x2, y2, width, height)


def clamp_bbox(  # noqa: PLR0913
    x1: int, y1: int, x2: int, y2: int, width: int, height: int
) -> PixelBounds:
    """Clamp pixel coordinates to image dimensions and fix swapped corners.

    Args:
        x1: Top-left x candidate.
        y1: Top-left y candidate.
        x2: Bottom-right x candidate.
        y2: Bottom-right y candidate.
        width: Image width.
        height: Image height.

    Returns:
        Clamped PixelBounds with x1 <= x2 and y1 <= y2.
    """
    # Auto-swap if corners are inverted
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    # Clamp to image bounds
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    return PixelBounds(x1=x1, y1=y1, x2=x2, y2=y2)


def clamp_point(
    x: int, y: int, width: int, height: int
) -> tuple[int, int]:
    """Clamp a point to image dimensions.

    Args:
        x: X coordinate.
        y: Y coordinate.
        width: Image width.
        height: Image height.

    Returns:
        Clamped (x, y) tuple.
    """
    return (max(0, min(x, width)), max(0, min(y, height)))


def denormalize_point(
    x: float, y: float, width: int, height: int
) -> tuple[int, int]:
    """Convert a normalized point to pixel coordinates and clamp.

    Args:
        x: Normalized x (0-1).
        y: Normalized y (0-1).
        width: Image width.
        height: Image height.

    Returns:
        Clamped pixel (x, y) tuple.
    """
    px = round(x * width)
    py = round(y * height)
    return clamp_point(px, py, width, height)


def normalize_bbox(  # noqa: PLR0913
    x1: int, y1: int, x2: int, y2: int, width: int, height: int
) -> tuple[float, float, float, float]:
    """Convert pixel bbox to normalized 0-1 fractions.

    Args:
        x1: Top-left x in pixels.
        y1: Top-left y in pixels.
        x2: Bottom-right x in pixels.
        y2: Bottom-right y in pixels.
        width: Image width.
        height: Image height.

    Returns:
        (nx1, ny1, nx2, ny2) as 0-1 floats.
    """
    return (
        x1 / width if width > 0 else 0.0,
        y1 / height if height > 0 else 0.0,
        x2 / width if width > 0 else 0.0,
        y2 / height if height > 0 else 0.0,
    )
