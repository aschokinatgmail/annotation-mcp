"""Tests with real photographs and EXIF-rotated images.

Uses fixture images from tests/fixtures/ — real photos and EXIF-oriented
synthetic images — to verify the annotation pipeline handles real-world
inputs correctly (EXIF transpose, high-res, pixel accuracy).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from annotation_mcp.image_io import get_image_info
from annotation_mcp.models import (
    ArrowAnnotation,
    BboxAnnotation,
    CalloutAnnotation,
    CircleAnnotation,
    HighlightAnnotation,
    TextAnnotation,
)
from annotation_mcp.render import render_annotations

FIXTURES = Path(__file__).parent / "fixtures"
GARDEN = str(FIXTURES / "garden_photo.jpg")
GARDEN2 = str(FIXTURES / "garden_photo2.jpg")
EXIF6 = str(FIXTURES / "exif_orientation6.jpg")


def _require_garden_photo() -> None:
    if not Path(GARDEN).exists():
        pytest.skip("garden_photo.jpg not present (real photo, not committed)")


def _require_garden_photo2() -> None:
    if not Path(GARDEN2).exists():
        pytest.skip("garden_photo2.jpg not present (real photo, not committed)")


def _load_rgb(path: str) -> Image.Image:
    """Load an image as RGB for pixel checking."""
    return Image.open(path).convert("RGB")


class TestGetImageInfoRealPhoto:
    """get_image_info on real photos reports EXIF-transposed dimensions."""

    def test_garden_photo_reports_transposed_dimensions(self) -> None:
        """Given garden_photo.jpg with EXIF orientation=6,
        when get_image_info is called,
        then width=3024, height=4032 (transposed from raw 4032x3024)."""
        _require_garden_photo()
        info = get_image_info(GARDEN)
        assert info.width == 3024
        assert info.height == 4032
        assert info.orientation == 6
        assert info.format == "JPEG"

    def test_exif6_reports_transposed_dimensions(self) -> None:
        """Given exif_orientation6.jpg (raw 300x200, orientation=6),
        when get_image_info is called,
        then width=200, height=300."""
        info = get_image_info(EXIF6)
        assert info.width == 200
        assert info.height == 300
        assert info.orientation == 6
        assert info.format == "JPEG"


class TestAnnotateRealPhotoNormalized:
    """Annotate real photo with normalized coords — output is valid."""

    def test_normalized_annotations_produce_valid_output(self, tmp_path: Path) -> None:
        """Given garden_photo.jpg,
        when annotate with normalized bbox + highlight + text,
        then no error, output PNG exists, manifest has 3 annotations,
        and output image dimensions match transposed input (3024x4032)."""
        _require_garden_photo()
        output = str(tmp_path / "annotated.png")
        annotations = [
            BboxAnnotation(bbox=(0.1, 0.1, 0.4, 0.4), color="#FF0000"),
            HighlightAnnotation(bbox=(0.6, 0.6, 0.9, 0.9), color="#FFFF00", opacity=0.3),
            TextAnnotation(position=(0.05, 0.95), text="Test", color="#FFFFFF"),
        ]
        png_bytes, output_path, manifest = render_annotations(
            GARDEN, annotations, normalized=True, output_path=output,
        )
        try:
            assert len(png_bytes) > 0
            assert output_path == output
            assert len(manifest.annotations) == 3
            assert manifest.image.width == 3024
            assert manifest.image.height == 4032
            img = _load_rgb(output)
            assert img.size == (3024, 4032)
        finally:
            if Path(output).exists():
                Path(output).unlink()


class TestPixelCheckRealPhoto:
    """Pixel check on real photo annotation."""

    def test_red_bbox_edge_pixel_is_reddish(self, tmp_path: Path) -> None:
        """Given garden_photo.jpg,
        when draw red bbox at normalized [0.1, 0.1, 0.15, 0.15],
        then pixel on the top edge of the bbox is reddish (R > G and R > B)."""
        _require_garden_photo()
        output = str(tmp_path / "pixel_check.png")
        ann = BboxAnnotation(bbox=(0.1, 0.1, 0.15, 0.15), color="#FF0000", thickness=3)
        _, output_path, _ = render_annotations(
            GARDEN, [ann], normalized=True, output_path=output,
        )
        try:
            img = _load_rgb(output_path)
            # Denormalized: x1=302, y1=403, x2=454, y2=605
            # Top edge at y=403, 404, 405 (thickness=3)
            px = img.getpixel((363, 404))
            assert px[0] > px[1], f"Expected R > G at edge pixel, got {px}"
            assert px[0] > px[2], f"Expected R > B at edge pixel, got {px}"
        finally:
            if Path(output).exists():
                Path(output).unlink()


class TestExifOrientationAnnotation:
    """EXIF orientation — annotation at normalized center appears at visual center."""

    def test_circle_at_center_after_exif_transpose(self, tmp_path: Path) -> None:
        """Given exif_orientation6.jpg (200x300 after transpose),
        when annotate with red circle at normalized [0.5, 0.5] radius 20,
        then pixel on the circle edge at the top is reddish."""
        output = str(tmp_path / "exif_center.png")
        ann = CircleAnnotation(center=(0.5, 0.5), radius=20, color="#FF0000", thickness=2)
        _, output_path, manifest = render_annotations(
            EXIF6, [ann], normalized=True, output_path=output,
        )
        try:
            assert manifest.image.width == 200
            assert manifest.image.height == 300
            img = _load_rgb(output_path)
            # Center at (100, 150), top edge of circle at (100, 150-20) = (100, 130)
            # With thickness=2, pixels at y=130, 131 are on the edge
            px = img.getpixel((100, 131))
            assert px[0] > px[1], f"Expected R > G at circle edge, got {px}"
            assert px[0] > px[2], f"Expected R > B at circle edge, got {px}"
        finally:
            if Path(output).exists():
                Path(output).unlink()


class TestSecondRealPhotoVariety:
    """Second real photo — all 6 annotation types."""

    def test_all_six_types_on_second_photo(self, tmp_path: Path) -> None:
        """Given garden_photo2.jpg,
        when annotate with all 6 annotation types using normalized coords,
        then no error, manifest has 6 entries, output PNG valid."""
        _require_garden_photo2()
        output = str(tmp_path / "all_types.png")
        annotations = [
            BboxAnnotation(bbox=(0.05, 0.05, 0.25, 0.25), color="#FF0000", label="Box"),
            HighlightAnnotation(bbox=(0.3, 0.05, 0.5, 0.25), color="#FFFF00", opacity=0.4),
            CalloutAnnotation(point=(0.7, 0.1), number=1, color="#0066FF", label="Point"),
            TextAnnotation(position=(0.05, 0.9), text="Hello", color="#FFFFFF"),
            ArrowAnnotation(**{"from": (0.8, 0.8), "to": (0.6, 0.6), "color": "#00FF00"}),
            CircleAnnotation(center=(0.5, 0.5), radius=30, color="#FF00FF", thickness=3),
        ]
        png_bytes, output_path, manifest = render_annotations(
            GARDEN2, annotations, normalized=True, output_path=output,
        )
        try:
            assert len(png_bytes) > 0
            assert len(manifest.annotations) == 6
            types = {a.type for a in manifest.annotations}
            assert types == {"bbox", "highlight", "callout", "text", "arrow", "circle"}
            img = _load_rgb(output_path)
            assert img.size == (3024, 4032)
        finally:
            if Path(output).exists():
                Path(output).unlink()


class TestHighResImage:
    """High-res image doesn't crash."""

    def test_ten_annotations_on_high_res_photo(self, tmp_path: Path) -> None:
        """Given garden_photo.jpg (12MP),
        when annotate with 10 annotations (various types),
        then no crash, manifest has 10 entries."""
        _require_garden_photo()
        output = str(tmp_path / "high_res.png")
        annotations = [
            BboxAnnotation(bbox=(0.0, 0.0, 0.1, 0.1), color="#FF0000"),
            BboxAnnotation(bbox=(0.2, 0.2, 0.3, 0.3), color="#00FF00"),
            BboxAnnotation(bbox=(0.4, 0.4, 0.5, 0.5), color="#0000FF"),
            HighlightAnnotation(bbox=(0.6, 0.6, 0.7, 0.7), color="#FFFF00", opacity=0.3),
            HighlightAnnotation(bbox=(0.8, 0.0, 0.9, 0.1), color="#00FFFF", opacity=0.3),
            CalloutAnnotation(point=(0.1, 0.5), number=1, color="#FF0000"),
            CalloutAnnotation(point=(0.3, 0.7), number=2, color="#00FF00"),
            TextAnnotation(position=(0.5, 0.9), text="High-res test", color="#FFFFFF"),
            ArrowAnnotation(**{"from": (0.9, 0.9), "to": (0.7, 0.7), "color": "#FF0000"}),
            CircleAnnotation(center=(0.5, 0.3), radius=40, color="#FF00FF", thickness=3),
        ]
        png_bytes, output_path, manifest = render_annotations(
            GARDEN, annotations, normalized=True, output_path=output,
        )
        try:
            assert len(png_bytes) > 0
            assert len(manifest.annotations) == 10
            img = _load_rgb(output_path)
            assert img.size == (3024, 4032)
        finally:
            if Path(output).exists():
                Path(output).unlink()
