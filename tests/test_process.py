"""Unit tests for process.py — _process_annotation and _coerce_annotation."""

from __future__ import annotations

import pytest

from annotation_mcp.models import (
    ArrowAnnotation,
    BboxAnnotation,
    CalloutAnnotation,
    CircleAnnotation,
    HighlightAnnotation,
    TextAnnotation,
)
from annotation_mcp.process import _coerce_annotation, _process_annotation


class TestCoerceAnnotation:
    """Tests for _coerce_annotation."""

    def test_passes_through_model_instance(self) -> None:
        """Given a model instance, when coerced, then returned unchanged."""
        ann = BboxAnnotation(bbox=(0, 0, 10, 10))
        result = _coerce_annotation(ann)
        assert result is ann

    def test_converts_bbox_dict(self) -> None:
        """Given a bbox dict, when coerced, then returns BboxAnnotation."""
        result = _coerce_annotation({"type": "bbox", "bbox": [0, 0, 10, 10]})
        assert isinstance(result, BboxAnnotation)
        assert result.bbox == (0, 0, 10, 10)

    def test_converts_arrow_dict(self) -> None:
        """Given an arrow dict, when coerced, then returns ArrowAnnotation."""
        result = _coerce_annotation(
            {"type": "arrow", "from": [0, 0], "to": [10, 10]}
        )
        assert isinstance(result, ArrowAnnotation)

    def test_converts_highlight_dict(self) -> None:
        """Given a highlight dict, when coerced, then returns HighlightAnnotation."""
        result = _coerce_annotation(
            {"type": "highlight", "bbox": [0, 0, 10, 10]}
        )
        assert isinstance(result, HighlightAnnotation)

    def test_converts_callout_dict(self) -> None:
        """Given a callout dict, when coerced, then returns CalloutAnnotation."""
        result = _coerce_annotation(
            {"type": "callout", "point": [5, 5], "number": 1}
        )
        assert isinstance(result, CalloutAnnotation)

    def test_converts_text_dict(self) -> None:
        """Given a text dict, when coerced, then returns TextAnnotation."""
        result = _coerce_annotation(
            {"type": "text", "position": [5, 5], "text": "hi"}
        )
        assert isinstance(result, TextAnnotation)

    def test_converts_circle_dict(self) -> None:
        """Given a circle dict, when coerced, then returns CircleAnnotation."""
        result = _coerce_annotation(
            {"type": "circle", "center": [5, 5], "radius": 10}
        )
        assert isinstance(result, CircleAnnotation)

    def test_unknown_type_raises_value_error(self) -> None:
        """Given unknown type, when coerced, then raises ValueError."""
        with pytest.raises(ValueError, match="Unknown annotation type"):
            _coerce_annotation({"type": "unknown"})


class TestProcessAnnotation:
    """Tests for _process_annotation."""

    WIDTH = 200
    HEIGHT = 100

    def test_bbox_absolute(self) -> None:
        """Given bbox with absolute coords, when processed, then returns pixel bounds."""
        ann = BboxAnnotation(bbox=(10, 10, 100, 80))
        pixel_bounds, manifest = _process_annotation(
            ann, self.WIDTH, self.HEIGHT, normalized=False, warnings=[]
        )
        assert pixel_bounds is not None
        assert pixel_bounds.x1 == 10
        assert pixel_bounds.x2 == 100
        assert manifest.type == "bbox"
        assert manifest.bbox_pixels == (10, 10, 100, 80)

    def test_bbox_normalized(self) -> None:
        """Given bbox with normalized coords, when processed, then correct pixel bounds."""
        ann = BboxAnnotation(bbox=(0.0, 0.0, 0.5, 0.5))
        pixel_bounds, manifest = _process_annotation(
            ann, self.WIDTH, self.HEIGHT, normalized=True, warnings=[]
        )
        assert pixel_bounds is not None
        assert pixel_bounds.x2 == 100
        assert pixel_bounds.y2 == 50
        assert manifest.bbox_normalized == (0.0, 0.0, 0.5, 0.5)

    def test_arrow_absolute_has_pixel_coords(self) -> None:
        """Given arrow with absolute coords, when processed, then manifest has pixel coords."""
        ann = ArrowAnnotation(**{"from": (10, 20), "to": (100, 80)})
        _, manifest = _process_annotation(
            ann, self.WIDTH, self.HEIGHT, normalized=False, warnings=[]
        )
        assert manifest.type == "arrow"
        assert manifest.from_pixels == (10, 20)
        assert manifest.to_pixels == (100, 80)
        assert manifest.from_normalized is not None
        assert manifest.to_normalized is not None

    def test_arrow_normalized_has_normalized_coords(self) -> None:
        """Given arrow with normalized coords, when processed, then manifest has normalized coords."""
        ann = ArrowAnnotation(**{"from": (0.5, 0.5), "to": (0.8, 0.8)})
        _, manifest = _process_annotation(
            ann, self.WIDTH, self.HEIGHT, normalized=True, warnings=[]
        )
        assert manifest.type == "arrow"
        assert manifest.from_normalized == (0.5, 0.5)
        assert manifest.to_normalized == (0.8, 0.8)
        assert manifest.from_pixels is not None
        assert manifest.to_pixels is not None

    def test_callout_absolute_has_point_coords(self) -> None:
        """Given callout with absolute coords, when processed, then manifest has point coords."""
        ann = CalloutAnnotation(point=(50, 50), number=1)
        _, manifest = _process_annotation(
            ann, self.WIDTH, self.HEIGHT, normalized=False, warnings=[]
        )
        assert manifest.type == "callout"
        assert manifest.point_pixels == (50, 50)
        assert manifest.point_normalized is not None
        assert manifest.radius == 18

    def test_callout_normalized_has_normalized_point(self) -> None:
        """Given callout with normalized coords, when processed, then manifest has normalized point."""
        ann = CalloutAnnotation(point=(0.5, 0.5), number=1)
        _, manifest = _process_annotation(
            ann, self.WIDTH, self.HEIGHT, normalized=True, warnings=[]
        )
        assert manifest.type == "callout"
        assert manifest.point_normalized == (0.5, 0.5)
        assert manifest.point_pixels is not None

    def test_text_absolute_has_point_coords(self) -> None:
        """Given text with absolute coords, when processed, then manifest has point coords."""
        ann = TextAnnotation(position=(10, 10), text="hello")
        _, manifest = _process_annotation(
            ann, self.WIDTH, self.HEIGHT, normalized=False, warnings=[]
        )
        assert manifest.type == "text"
        assert manifest.point_pixels == (10, 10)
        assert manifest.point_normalized is not None
        assert manifest.text == "hello"

    def test_text_normalized_has_normalized_point(self) -> None:
        """Given text with normalized coords, when processed, then manifest has normalized point."""
        ann = TextAnnotation(position=(0.1, 0.1), text="hello")
        _, manifest = _process_annotation(
            ann, self.WIDTH, self.HEIGHT, normalized=True, warnings=[]
        )
        assert manifest.type == "text"
        assert manifest.point_normalized == (0.1, 0.1)
        assert manifest.point_pixels is not None

    def test_circle_absolute_has_center_coords(self) -> None:
        """Given circle with absolute coords, when processed, then manifest has center coords."""
        ann = CircleAnnotation(center=(50, 50), radius=20)
        _, manifest = _process_annotation(
            ann, self.WIDTH, self.HEIGHT, normalized=False, warnings=[]
        )
        assert manifest.type == "circle"
        assert manifest.center_pixels == (50, 50)
        assert manifest.center_normalized is not None
        assert manifest.radius == 20

    def test_circle_normalized_has_normalized_center(self) -> None:
        """Given circle with normalized coords, when processed, then manifest has normalized center."""
        ann = CircleAnnotation(center=(0.5, 0.5), radius=20)
        _, manifest = _process_annotation(
            ann, self.WIDTH, self.HEIGHT, normalized=True, warnings=[]
        )
        assert manifest.type == "circle"
        assert manifest.center_normalized == (0.5, 0.5)
        assert manifest.center_pixels is not None

    def test_highlight_absolute(self) -> None:
        """Given highlight with absolute coords, when processed, then returns pixel bounds."""
        ann = HighlightAnnotation(bbox=(10, 10, 100, 80))
        pixel_bounds, manifest = _process_annotation(
            ann, self.WIDTH, self.HEIGHT, normalized=False, warnings=[]
        )
        assert pixel_bounds is not None
        assert manifest.type == "highlight"
        assert manifest.bbox_pixels == (10, 10, 100, 80)

    def test_highlight_swapped_corners_adds_warning(self) -> None:
        """Given highlight with swapped corners, when processed, then warning added."""
        ann = HighlightAnnotation(bbox=(100, 80, 10, 10))
        warnings: list[str] = []
        _process_annotation(ann, self.WIDTH, self.HEIGHT, normalized=False, warnings=warnings)
        assert any("swapped" in w for w in warnings)

    def test_swapped_corners_adds_warning(self) -> None:
        """Given bbox with swapped corners, when processed, then warning added."""
        ann = BboxAnnotation(bbox=(100, 80, 10, 10))
        warnings: list[str] = []
        _process_annotation(ann, self.WIDTH, self.HEIGHT, normalized=False, warnings=warnings)
        assert any("swapped" in w for w in warnings)

    def test_clamped_coords_adds_warning(self) -> None:
        """Given bbox outside image, when processed, then warning added."""
        ann = BboxAnnotation(bbox=(0, 0, 300, 200))
        warnings: list[str] = []
        _process_annotation(ann, self.WIDTH, self.HEIGHT, normalized=False, warnings=warnings)
        assert any("clamped" in w for w in warnings)

    def test_unknown_annotation_type_raises(self) -> None:
        """Given unknown annotation type, when processed, then raises ValueError."""
        from annotation_mcp.models import Annotation
        # Create a mock-like object that's not a valid annotation type
        class FakeAnnotation:
            def __init__(self) -> None:
                self.type = "fake"
        with pytest.raises(ValueError, match="Unknown annotation type"):
            _process_annotation(
                FakeAnnotation(),  # type: ignore[arg-type]
                self.WIDTH, self.HEIGHT, normalized=False, warnings=[],
            )
