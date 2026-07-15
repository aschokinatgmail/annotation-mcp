"""Coverage gap tests for edge cases not covered by other test files."""

from __future__ import annotations

import os
import tempfile

import pytest
from PIL import Image

from annotation_mcp.draw_primitives import (
    draw_arrow,
    draw_bbox,
    draw_callout,
    draw_circle,
    draw_text,
)
from annotation_mcp.draw_utils import get_font, parse_color
from annotation_mcp.image_io import (
    ImageInfo,
    encode_png,
    get_image_info,
    is_png_data,
    load_image,
    validate_input_path,
    validate_output_path,
)
from annotation_mcp.models import (
    ArrowAnnotation,
    BboxAnnotation,
    CalloutAnnotation,
    CircleAnnotation,
    TextAnnotation,
)
from annotation_mcp.coords import PixelBounds


@pytest.fixture
def test_image() -> str:
    """Create a simple test image and return its path."""
    img = Image.new("RGB", (200, 100), color=(255, 255, 255))
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp.name, format="PNG")
    tmp.close()
    return tmp.name


class TestImageIOGaps:
    """Coverage gaps for image_io.py."""

    def test_validate_input_path_relative_raises(self) -> None:
        """Given relative path, when validated, then raises ValueError."""
        with pytest.raises(ValueError, match="Input path must be absolute"):
            validate_input_path("relative/path.png")

    def test_validate_input_path_nonexistent_raises(self) -> None:
        """Given nonexistent path, when validated, then raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Input file not found"):
            validate_input_path("/nonexistent/file.png")

    def test_validate_output_path_relative_raises(self) -> None:
        """Given relative output path, when validated, then raises ValueError."""
        with pytest.raises(ValueError, match="Output path must be absolute"):
            validate_output_path("relative/output.png")

    def test_validate_output_path_bad_parent_raises(self) -> None:
        """Given output path with nonexistent parent, when validated, then raises ValueError."""
        with pytest.raises(ValueError, match="Output directory does not exist"):
            validate_output_path("/nonexistent/dir/output.png")

    def test_encode_png_returns_png_bytes(self) -> None:
        """Given PIL Image, when encoded, then returns valid PNG bytes."""
        img = Image.new("RGB", (10, 10))
        data = encode_png(img)
        assert is_png_data(data)

    def test_get_image_info_returns_correct_info(self, test_image: str) -> None:
        """Given valid image, when get_image_info called, then returns ImageInfo."""
        info = get_image_info(test_image)
        assert isinstance(info, ImageInfo)
        assert info.width == 200
        assert info.height == 100
        assert info.format == "PNG"
        assert info.has_alpha is False
        assert info.size_bytes > 0

    def test_load_image_returns_image(self, test_image: str) -> None:
        """Given valid image path, when loaded, then returns PIL Image."""
        img = load_image(test_image)
        assert img.width == 200
        assert img.height == 100


class TestDrawPrimitivesGaps:
    """Coverage gaps for draw_primitives.py."""

    def test_draw_arrow_degenerate_length(self) -> None:
        """Given arrow with zero-length (from==to), when drawn, then no error (early return)."""
        img = Image.new("RGBA", (100, 100), (255, 255, 255, 255))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        ann = ArrowAnnotation(**{"from": (50, 50), "to": (50, 50)})
        # Should not raise — early return at length < 1
        draw_arrow(draw, ann, 100, 100)

    def test_draw_bbox_without_label(self) -> None:
        """Given bbox without label, when drawn, then no label background drawn."""
        img = Image.new("RGBA", (100, 100), (255, 255, 255, 255))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        ann = BboxAnnotation(bbox=(10, 10, 50, 50), label=None)
        draw_bbox(draw, img, PixelBounds(x1=10, y1=10, x2=50, y2=50), ann)

    def test_draw_callout_without_label(self) -> None:
        """Given callout without label, when drawn, then no label background drawn."""
        img = Image.new("RGBA", (100, 100), (255, 255, 255, 255))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        ann = CalloutAnnotation(point=(50, 50), number=1, label=None)
        draw_callout(draw, img, ann, 100, 100)

    def test_draw_circle_without_label(self) -> None:
        """Given circle without label, when drawn, then no label background drawn."""
        img = Image.new("RGBA", (100, 100), (255, 255, 255, 255))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        ann = CircleAnnotation(center=(50, 50), radius=20, label=None)
        draw_circle(draw, img, ann, 100, 100)

    def test_draw_text_without_background(self) -> None:
        """Given text with empty background, when drawn, then no background rectangle drawn."""
        img = Image.new("RGBA", (100, 100), (255, 255, 255, 255))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        ann = TextAnnotation(position=(10, 10), text="test", background="")
        draw_text(draw, img, ann, 100, 100)

    def test_draw_text_with_background(self) -> None:
        """Given text with background, when drawn, then background rectangle drawn."""
        img = Image.new("RGBA", (100, 100), (255, 255, 255, 255))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        ann = TextAnnotation(
            position=(10, 10), text="test", background="rgba(0,0,0,0.5)"
        )
        draw_text(draw, img, ann, 100, 100)
