"""Unit tests for coordinate normalization, validation, and clamping."""

from __future__ import annotations

import pytest

from annotation_mcp.coords import (
    CoordinateError,
    NormalizedBounds,
    PixelBounds,
    clamp_bbox,
    clamp_point,
    denormalize_bbox,
    denormalize_point,
    normalize_bbox,
    validate_normalized_bbox,
    validate_normalized_point,
)


class TestValidateNormalized:
    """Tests for normalized coordinate validation."""

    def test_valid_normalized_values(self) -> None:
        """Given valid normalized values, when validated, then no error."""
        bounds = validate_normalized_bbox(0.0, 0.0, 1.0, 1.0)
        assert isinstance(bounds, NormalizedBounds)
        assert bounds.x1 == 0.0
        assert bounds.y1 == 0.0
        assert bounds.x2 == 1.0
        assert bounds.y2 == 1.0

    def test_valid_within_epsilon(self) -> None:
        """Given values within epsilon tolerance, when validated, then no error."""
        bounds = validate_normalized_bbox(-0.001, 0.0, 1.0, 1.001)
        assert isinstance(bounds, NormalizedBounds)

    def test_out_of_bounds_positive(self) -> None:
        """Given value > 1+epsilon, when validated, then CoordinateError."""
        with pytest.raises(CoordinateError, match="outside"):
            validate_normalized_bbox(0.0, 0.0, 1.5, 1.0)

    def test_out_of_bounds_negative(self) -> None:
        """Given value < -epsilon, when validated, then CoordinateError."""
        with pytest.raises(CoordinateError, match="outside"):
            validate_normalized_bbox(-0.1, 0.0, 1.0, 1.0)

    def test_valid_normalized_point(self) -> None:
        """Given valid point, when validated, then no error."""
        validate_normalized_point(0.5, 0.5)

    def test_invalid_normalized_point(self) -> None:
        """Given invalid point, when validated, then CoordinateError."""
        with pytest.raises(CoordinateError):
            validate_normalized_point(2.0, 0.5)


class TestDenormalize:
    """Tests for denormalization of coordinates."""

    def test_denormalize_bbox(self) -> None:
        """Given normalized bbox, when denormalized, then correct pixel values."""
        bounds = NormalizedBounds(x1=0.0, y1=0.0, x2=1.0, y2=1.0)
        result = denormalize_bbox(bounds, 100, 200)
        assert result == PixelBounds(x1=0, y1=0, x2=100, y2=200)

    def test_denormalize_midpoint(self) -> None:
        """Given midpoint normalized values, when denormalized, then correct."""
        bounds = NormalizedBounds(x1=0.25, y1=0.5, x2=0.75, y2=1.0)
        result = denormalize_bbox(bounds, 800, 600)
        assert result == PixelBounds(x1=200, y1=300, x2=600, y2=600)

    def test_denormalize_point(self) -> None:
        """Given normalized point, when denormalized, then correct pixel values."""
        x, y = denormalize_point(0.5, 0.5, 100, 200)
        assert x == 50
        assert y == 100


class TestClamp:
    """Tests for coordinate clamping."""

    def test_clamp_within_bounds(self) -> None:
        """Given coords within bounds, when clamped, then unchanged."""
        result = clamp_bbox(10, 20, 100, 200, 500, 500)
        assert result == PixelBounds(x1=10, y1=20, x2=100, y2=200)

    def test_clamp_negative(self) -> None:
        """Given negative coords, when clamped, then set to 0."""
        result = clamp_bbox(-10, -20, 100, 200, 500, 500)
        assert result == PixelBounds(x1=0, y1=0, x2=100, y2=200)

    def test_clamp_beyond_dimensions(self) -> None:
        """Given coords beyond image, when clamped, then set to max."""
        result = clamp_bbox(0, 0, 600, 700, 500, 500)
        assert result == PixelBounds(x1=0, y1=0, x2=500, y2=500)

    def test_clamp_swapped_x(self) -> None:
        """Given x1 > x2, when clamped, then auto-swapped."""
        result = clamp_bbox(100, 0, 10, 200, 500, 500)
        assert result.x1 == 10
        assert result.x2 == 100

    def test_clamp_swapped_y(self) -> None:
        """Given y1 > y2, when clamped, then auto-swapped."""
        result = clamp_bbox(0, 200, 100, 10, 500, 500)
        assert result.y1 == 10
        assert result.y2 == 200

    def test_clamp_point_negative(self) -> None:
        """Given negative point, when clamped, then set to 0."""
        x, y = clamp_point(-5, -10, 100, 100)
        assert x == 0
        assert y == 0

    def test_clamp_point_beyond(self) -> None:
        """Given point beyond image, when clamped, then set to max."""
        x, y = clamp_point(200, 300, 100, 100)
        assert x == 100
        assert y == 100


class TestNormalize:
    """Tests for normalization of pixel coordinates."""

    def test_normalize_bbox(self) -> None:
        """Given pixel bbox, when normalized, then correct 0-1 fractions."""
        result = normalize_bbox(0, 0, 100, 200, 200, 400)
        assert result == (0.0, 0.0, 0.5, 0.5)

    def test_normalize_zero_dimensions(self) -> None:
        """Given zero dimensions, when normalized, then returns 0.0."""
        result = normalize_bbox(10, 20, 30, 40, 0, 0)
        assert result == (0.0, 0.0, 0.0, 0.0)
