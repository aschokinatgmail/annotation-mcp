"""Functional tests for the rendering engine."""

from __future__ import annotations

import os
import tempfile

import pytest
from PIL import Image

from annotation_mcp.image_io import encode_png, is_png_data, load_image
from annotation_mcp.models import (
    Annotation,
    BboxAnnotation,
    CalloutAnnotation,
    CircleAnnotation,
    HighlightAnnotation,
    TextAnnotation,
)
from annotation_mcp.render import render_annotations


@pytest.fixture
def test_image() -> str:
    """Create a simple test image and return its path."""
    img = Image.new("RGB", (200, 100), color=(255, 255, 255))
    # Draw a simple shape to make it non-trivial
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 50, 50], fill=(100, 100, 100))

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp.name, format="PNG")
    tmp.close()
    return tmp.name


@pytest.fixture
def test_image_rgba() -> str:
    """Create a simple RGBA test image."""
    img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp.name, format="PNG")
    tmp.close()
    return tmp.name


class TestRenderAnnotations:
    """Tests for the render_annotations function."""

    def test_empty_annotations(self, test_image: str) -> None:
        """Given empty annotations list, when rendered, then returns original image."""
        png_bytes, output_path, manifest = render_annotations(
            path=test_image,
            annotations=[],
        )

        assert is_png_data(png_bytes)
        assert os.path.isfile(output_path)
        assert manifest.version == "1.0"
        assert manifest.annotations == []
        assert manifest.warnings == []

        # Clean up
        os.unlink(output_path)

    def test_single_bbox(self, test_image: str) -> None:
        """Given a single bbox annotation, when rendered, then returns annotated image."""
        ann = BboxAnnotation(bbox=(10, 10, 100, 80), label="test", color="#FF0000")
        png_bytes, output_path, manifest = render_annotations(
            path=test_image,
            annotations=[ann],
        )

        assert is_png_data(png_bytes)
        assert os.path.isfile(output_path)
        assert len(manifest.annotations) == 1
        assert manifest.annotations[0].type == "bbox"
        assert manifest.annotations[0].label == "test"
        assert manifest.annotations[0].bbox_pixels is not None

        # Verify file size changed (annotations were drawn)
        original_size = os.path.getsize(test_image)
        annotated_size = os.path.getsize(output_path)
        assert annotated_size != original_size

        os.unlink(output_path)

    def test_normalized_coords(self, test_image: str) -> None:
        """Given normalized coordinates, when rendered, then correct pixel mapping."""
        ann = BboxAnnotation(bbox=(0.0, 0.0, 0.5, 0.5), label="half")
        png_bytes, output_path, manifest = render_annotations(
            path=test_image,
            annotations=[ann],
            normalized=True,
        )

        assert is_png_data(png_bytes)
        assert len(manifest.annotations) == 1
        bp = manifest.annotations[0].bbox_pixels
        assert bp is not None
        # 200x100 image, 0.5 = 100x50
        assert bp[0] == 0
        assert bp[1] == 0
        assert bp[2] == 100
        assert bp[3] == 50

        os.unlink(output_path)

    def test_highlight_annotation(self, test_image: str) -> None:
        """Given a highlight annotation, when rendered, then returns annotated image."""
        ann = HighlightAnnotation(
            bbox=(20, 20, 80, 80), color="#FFFF00", opacity=0.5
        )
        png_bytes, output_path, manifest = render_annotations(
            path=test_image,
            annotations=[ann],
        )

        assert is_png_data(png_bytes)
        assert len(manifest.annotations) == 1
        assert manifest.annotations[0].type == "highlight"

        os.unlink(output_path)

    def test_callout_annotation(self, test_image: str) -> None:
        """Given a callout annotation, when rendered, then returns annotated image."""
        ann = CalloutAnnotation(
            point=(50, 50), number=1, label="Item 1", color="#00FF00"
        )
        png_bytes, output_path, manifest = render_annotations(
            path=test_image,
            annotations=[ann],
        )

        assert is_png_data(png_bytes)
        assert len(manifest.annotations) == 1
        assert manifest.annotations[0].type == "callout"
        assert manifest.annotations[0].number == 1

        os.unlink(output_path)

    def test_text_annotation(self, test_image: str) -> None:
        """Given a text annotation, when rendered, then returns annotated image."""
        ann = TextAnnotation(
            position=(10, 10), text="Hello World", color="#FFFFFF", font_size=20
        )
        png_bytes, output_path, manifest = render_annotations(
            path=test_image,
            annotations=[ann],
        )

        assert is_png_data(png_bytes)
        assert len(manifest.annotations) == 1
        assert manifest.annotations[0].type == "text"
        assert manifest.annotations[0].text == "Hello World"

        os.unlink(output_path)

    def test_circle_annotation(self, test_image: str) -> None:
        """Given a circle annotation, when rendered, then returns annotated image."""
        ann = CircleAnnotation(
            center=(50, 50), radius=20, color="#0000FF", label="Center"
        )
        png_bytes, output_path, manifest = render_annotations(
            path=test_image,
            annotations=[ann],
        )

        assert is_png_data(png_bytes)
        assert len(manifest.annotations) == 1
        assert manifest.annotations[0].type == "circle"

        os.unlink(output_path)

    def test_multiple_annotations(self, test_image: str) -> None:
        """Given multiple annotations, when rendered, then all are drawn."""
        annotations: list[Annotation] = [
            BboxAnnotation(bbox=(0, 0, 50, 50), color="#FF0000"),
            HighlightAnnotation(bbox=(25, 25, 75, 75), color="#FFFF00", opacity=0.3),
            CalloutAnnotation(point=(100, 50), number=1),
        ]
        png_bytes, output_path, manifest = render_annotations(
            path=test_image,
            annotations=annotations,
        )

        assert is_png_data(png_bytes)
        assert len(manifest.annotations) == 3
        assert manifest.annotations[0].type == "bbox"
        assert manifest.annotations[1].type == "highlight"
        assert manifest.annotations[2].type == "callout"

        os.unlink(output_path)

    def test_swapped_corners_warning(self, test_image: str) -> None:
        """Given swapped corners, when rendered, then warning is added."""
        ann = BboxAnnotation(bbox=(100, 80, 10, 10))
        _, _, manifest = render_annotations(
            path=test_image,
            annotations=[ann],
        )

        assert len(manifest.warnings) >= 1
        assert any("swapped" in w for w in manifest.warnings)

    def test_output_path_respected(self, test_image: str) -> None:
        """Given explicit output_path, when rendered, then file written there."""
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        out_path = tmp.name

        ann = BboxAnnotation(bbox=(0, 0, 10, 10))
        _, output_path, _ = render_annotations(
            path=test_image,
            annotations=[ann],
            output_path=out_path,
        )

        assert output_path == out_path
        assert os.path.isfile(out_path)
        os.unlink(out_path)

    def test_auto_output_path(self, test_image: str) -> None:
        """Given no output_path, when rendered, then auto-generated path used.

        The auto path is created via tempfile.NamedTemporaryFile, so it
        lives in the platform's temp dir (e.g. /tmp on Linux/macOS,
        C:\\Users\\<user>\\AppData\\Local\\Temp on Windows). We just
        verify the prefix, suffix, and that the file exists.
        """
        import tempfile

        ann = BboxAnnotation(bbox=(0, 0, 10, 10))
        _, output_path, _ = render_annotations(
            path=test_image,
            annotations=[ann],
        )

        # The auto path should start with the platform temp dir and have
        # the expected prefix/suffix.
        tmp_root = tempfile.gettempdir()
        assert output_path.startswith(tmp_root), (
            f"Expected output_path {output_path!r} to start with {tmp_root!r}"
        )
        assert "annotation-mcp-" in output_path
        assert output_path.endswith(".png")
        assert os.path.isfile(output_path)
        os.unlink(output_path)

    def test_manifest_structure(self, test_image: str) -> None:
        """Given annotations, when rendered, then manifest has correct structure."""
        ann = BboxAnnotation(bbox=(10, 10, 50, 50), label="box")
        _, _, manifest = render_annotations(
            path=test_image,
            annotations=[ann],
        )

        assert manifest.version == "1.0"
        assert manifest.image.path == test_image
        assert manifest.image.width == 200
        assert manifest.image.height == 100
        assert manifest.created_at is not None
        assert isinstance(manifest.warnings, list)

    def test_rgba_image(self, test_image_rgba: str) -> None:
        """Given RGBA image, when rendered, then no errors."""
        ann = BboxAnnotation(bbox=(10, 10, 50, 50))
        png_bytes, output_path, _ = render_annotations(
            path=test_image_rgba,
            annotations=[ann],
        )

        assert is_png_data(png_bytes)
        os.unlink(output_path)


class TestImageIO:
    """Tests for image I/O functions."""

    def test_is_png_data(self) -> None:
        """Given PNG bytes, when checked, then returns True."""
        img = Image.new("RGB", (10, 10))
        data = encode_png(img)
        assert is_png_data(data)

    def test_is_png_data_non_png(self) -> None:
        """Given non-PNG bytes, when checked, then returns False."""
        assert not is_png_data(b"not a png file")

    def test_load_image(self, test_image: str) -> None:
        """Given valid image path, when loaded, then returns PIL Image."""
        img = load_image(test_image)
        assert img.width == 200
        assert img.height == 100
