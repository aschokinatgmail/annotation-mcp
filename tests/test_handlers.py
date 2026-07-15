"""Direct unit tests for handler functions (not through subprocess)."""

from __future__ import annotations

import os
import tempfile

import pytest
from PIL import Image

from annotation_mcp.handlers import (
    handle_annotate,
    handle_draw_bounding_box,
    handle_draw_numbered_callouts,
    handle_get_image_info,
    handle_highlight_region,
)


@pytest.fixture
def test_image() -> str:
    """Create a simple test image and return its path."""
    img = Image.new("RGB", (200, 100), color=(255, 255, 255))
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp.name, format="PNG")
    tmp.close()
    return tmp.name


class TestHandleGetImageInfo:
    """Tests for handle_get_image_info."""

    def test_valid_path_returns_structured_content(self, test_image: str) -> None:
        """Given valid image path, when handle_get_image_info called, then returns structuredContent."""
        result = handle_get_image_info(test_image)
        assert result.isError is False
        assert result.structuredContent is not None
        sc = result.structuredContent
        assert sc["width"] == 200
        assert sc["height"] == 100
        assert sc["format"] == "PNG"
        assert sc["has_alpha"] is False
        assert isinstance(sc["size_bytes"], int)
        assert sc["size_bytes"] > 0

    def test_nonexistent_path_returns_error(self) -> None:
        """Given nonexistent path, when called, then isError=True."""
        result = handle_get_image_info("/nonexistent/path.png")
        assert result.isError is True
        assert "File not found" in str(result.content)

    def test_corrupt_image_returns_os_error(self) -> None:
        """Given corrupt image path, when called, then OSError caught."""
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.write(b"not a real image")
        tmp.close()
        try:
            result = handle_get_image_info(tmp.name)
            assert result.isError is True
            assert "Error reading image" in str(result.content)
        finally:
            os.unlink(tmp.name)


class TestHandleAnnotate:
    """Tests for handle_annotate."""

    def test_valid_bbox_returns_image_and_manifest(self, test_image: str) -> None:
        """Given valid bbox annotation, when called, then returns ImageContent + manifest."""
        result = handle_annotate(
            path=test_image,
            annotations=[{"type": "bbox", "bbox": [10, 10, 100, 80]}],
        )
        assert result.isError is False
        assert result.structuredContent is not None
        sc = result.structuredContent
        assert sc["version"] == "1.0"
        assert len(sc["annotations"]) == 1
        assert sc["annotations"][0]["type"] == "bbox"

    def test_normalized_bbox_has_normalized_coords(self, test_image: str) -> None:
        """Given normalized bbox, when called, then manifest has normalized coords."""
        result = handle_annotate(
            path=test_image,
            annotations=[{"type": "bbox", "bbox": [0.0, 0.0, 0.5, 0.5]}],
            normalized=True,
        )
        assert result.isError is False
        sc = result.structuredContent
        ann = sc["annotations"][0]
        assert ann["bbox_normalized"] == (0.0, 0.0, 0.5, 0.5)
        assert ann["bbox_pixels"] is not None

    def test_nonexistent_path_returns_error(self) -> None:
        """Given nonexistent path, when called, then isError=True."""
        result = handle_annotate("/nonexistent/path.png", [])
        assert result.isError is True
        assert "File not found" in str(result.content)

    def test_unknown_type_returns_error(self, test_image: str) -> None:
        """Given unknown annotation type, when called, then isError=True."""
        result = handle_annotate(
            test_image,
            [{"type": "unknown"}],
        )
        assert result.isError is True
        assert "Unknown annotation type" in str(result.content)

    def test_out_of_bounds_bbox_clamps_with_warning(self, test_image: str) -> None:
        """Given out-of-bounds bbox, when called, then clamped with warning in manifest."""
        result = handle_annotate(
            test_image,
            [{"type": "bbox", "bbox": [0, 0, 99999, 99999]}],
        )
        assert result.isError is False
        sc = result.structuredContent
        assert len(sc["warnings"]) >= 1
        assert any("clamped" in w for w in sc["warnings"])

    def test_arrow_annotation_works(self, test_image: str) -> None:
        """Given arrow annotation, when called, then manifest has arrow entry."""
        result = handle_annotate(
            test_image,
            [{"type": "arrow", "from": [10, 10], "to": [100, 80]}],
        )
        assert result.isError is False
        sc = result.structuredContent
        assert sc["annotations"][0]["type"] == "arrow"

    def test_callout_annotation_works(self, test_image: str) -> None:
        """Given callout annotation, when called, then manifest has callout entry."""
        result = handle_annotate(
            test_image,
            [{"type": "callout", "point": [50, 50], "number": 1}],
        )
        assert result.isError is False
        sc = result.structuredContent
        assert sc["annotations"][0]["type"] == "callout"

    def test_text_annotation_works(self, test_image: str) -> None:
        """Given text annotation, when called, then manifest has text entry."""
        result = handle_annotate(
            test_image,
            [{"type": "text", "position": [10, 10], "text": "hello"}],
        )
        assert result.isError is False
        sc = result.structuredContent
        assert sc["annotations"][0]["type"] == "text"

    def test_circle_annotation_works(self, test_image: str) -> None:
        """Given circle annotation, when called, then manifest has circle entry."""
        result = handle_annotate(
            test_image,
            [{"type": "circle", "center": [50, 50], "radius": 20}],
        )
        assert result.isError is False
        sc = result.structuredContent
        assert sc["annotations"][0]["type"] == "circle"

    def test_highlight_annotation_works(self, test_image: str) -> None:
        """Given highlight annotation, when called, then manifest has highlight entry."""
        result = handle_annotate(
            test_image,
            [{"type": "highlight", "bbox": [10, 10, 50, 50]}],
        )
        assert result.isError is False
        sc = result.structuredContent
        assert sc["annotations"][0]["type"] == "highlight"

    def test_invalid_bbox_raises_value_error(self, test_image: str) -> None:
        """Given invalid bbox (non-numeric), when called, then ValueError caught."""
        result = handle_annotate(
            test_image,
            [{"type": "bbox", "bbox": "not-a-bbox"}],
        )
        assert result.isError is True
        assert "Validation error" in str(result.content)

    def test_invalid_path_raises_os_error(self) -> None:
        """Given path to corrupt file, when called, then OSError caught."""
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.write(b"not a real image")
        tmp.close()
        try:
            result = handle_annotate(
                tmp.name,
                [{"type": "bbox", "bbox": [0, 0, 10, 10]}],
            )
            assert result.isError is True
            assert "Error" in str(result.content)
        finally:
            os.unlink(tmp.name)


class TestHandleDrawBoundingBox:
    """Tests for handle_draw_bounding_box."""

    def test_basic_bbox_works(self, test_image: str) -> None:
        """Given bbox params, when called, then returns annotated image."""
        result = handle_draw_bounding_box(test_image, (10, 10, 100, 100))
        assert result.isError is False
        assert result.structuredContent is not None
        assert result.structuredContent["annotations"][0]["type"] == "bbox"

    def test_with_label_in_manifest(self, test_image: str) -> None:
        """Given bbox with label, when called, then manifest has label."""
        result = handle_draw_bounding_box(
            test_image, (10, 10, 100, 100), label="test-label"
        )
        assert result.isError is False
        ann = result.structuredContent["annotations"][0]
        assert ann["label"] == "test-label"


class TestHandleHighlightRegion:
    """Tests for handle_highlight_region."""

    def test_basic_highlight_works(self, test_image: str) -> None:
        """Given highlight params, when called, then returns highlighted image."""
        result = handle_highlight_region(test_image, (10, 10, 100, 80))
        assert result.isError is False
        assert result.structuredContent["annotations"][0]["type"] == "highlight"


class TestHandleDrawNumberedCallouts:
    """Tests for handle_draw_numbered_callouts."""

    def test_valid_callouts_work(self, test_image: str) -> None:
        """Given valid callouts, when called, then returns annotated image."""
        result = handle_draw_numbered_callouts(
            test_image,
            [{"point": [50, 50], "number": 1}],
        )
        assert result.isError is False
        ann = result.structuredContent["annotations"][0]
        assert ann["type"] == "callout"
        assert ann["number"] == 1

    def test_callout_with_label(self, test_image: str) -> None:
        """Given callout with label, when called, then label in manifest."""
        result = handle_draw_numbered_callouts(
            test_image,
            [{"point": [50, 50], "number": 1, "label": "Item 1"}],
        )
        assert result.isError is False
        ann = result.structuredContent["annotations"][0]
        assert ann["label"] == "Item 1"

    def test_bad_point_type_returns_error(self, test_image: str) -> None:
        """Given callout with bad point type, when called, then isError=True."""
        result = handle_draw_numbered_callouts(
            test_image,
            [{"point": "bad", "number": 1}],
        )
        assert result.isError is True
        assert "Each callout must have" in str(result.content)

    def test_bad_number_type_returns_error(self, test_image: str) -> None:
        """Given callout with bad number type, when called, then isError=True."""
        result = handle_draw_numbered_callouts(
            test_image,
            [{"point": [50, 50], "number": "bad"}],
        )
        assert result.isError is True
        assert "Each callout must have" in str(result.content)
