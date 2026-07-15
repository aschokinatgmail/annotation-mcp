"""Regression tests for specific bugs that were fixed."""
from __future__ import annotations
import os
import tempfile
from PIL import Image
from annotation_mcp.models import (
    ArrowAnnotation, BboxAnnotation, CalloutAnnotation,
    CircleAnnotation, HighlightAnnotation, TextAnnotation,
)
from annotation_mcp.render import render_annotations

def _make_white(width: int = 400, height: int = 300) -> str:
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    Image.new("RGB", (width, height), (255, 255, 255)).save(path)
    return path


class TestProcessAnnotationNoCrash:
    """Regression: _process_annotation used to crash with UnboundLocalError."""

    def test_all_six_types_render_without_crash(self) -> None:
        """Given all 6 annotation types with absolute coords,
        when render_annotations is called, then no exception and manifest has 6 entries."""
        path = _make_white()
        anns = [
            BboxAnnotation(bbox=(10, 10, 100, 100)),
            HighlightAnnotation(bbox=(110, 10, 200, 100)),
            ArrowAnnotation(**{"from": (50, 200), "to": (150, 250)}),
            CalloutAnnotation(point=(200, 200), number=1),
            TextAnnotation(position=(10, 280), text="regression"),
            CircleAnnotation(center=(300, 150), radius=20),
        ]
        try:
            _, output_path, manifest = render_annotations(path, anns)
            assert len(manifest.annotations) == 6, f"Expected 6 manifest entries, got {len(manifest.annotations)}"
            assert os.path.exists(output_path), "Output file should exist"
        finally:
            os.unlink(path)
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_all_six_types_normalized_without_crash(self) -> None:
        """Given all 6 types with normalized coords,
        when rendered, then no exception and all 6 manifest entries present."""
        path = _make_white()
        anns = [
            BboxAnnotation(bbox=(0.1, 0.1, 0.3, 0.3)),
            HighlightAnnotation(bbox=(0.4, 0.1, 0.6, 0.3)),
            ArrowAnnotation(**{"from": (0.8, 0.8), "to": (0.5, 0.5)}),
            CalloutAnnotation(point=(0.3, 0.5), number=1),
            TextAnnotation(position=(0.1, 0.9), text="norm"),
            CircleAnnotation(center=(0.7, 0.5), radius=20),
        ]
        try:
            _, output_path, manifest = render_annotations(path, anns, normalized=True)
            assert len(manifest.annotations) == 6
        finally:
            os.unlink(path)
            if 'output_path' in dir() and os.path.exists(output_path):
                os.unlink(output_path)


class TestNormalizedDenormalization:
    """Regression: point-based annotations were not denormalized when normalized=True."""

    def test_normalized_arrow_drawn_at_correct_position(self) -> None:
        """Given 400x300 image, when normalized arrow from [0.8,0.8] to [0.5,0.5],
        then arrow is drawn at pixels (320,240)→(200,150), NOT at (1,1)→(1,1)."""
        path = _make_white()
        ann = ArrowAnnotation(**{"from": (0.8, 0.8), "to": (0.5, 0.5), "color": "#FF0000", "thickness": 3})
        try:
            _, output_path, _ = render_annotations(path, [ann], normalized=True)
            img = Image.open(output_path).convert("RGB")
            # Midpoint of (320,240)→(200,150) = (260,195)
            px = img.getpixel((260, 195))
            assert px[0] > 200, f"Expected red at denormalized midpoint (260,195), got {px}"
        finally:
            os.unlink(path)
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_normalized_callout_drawn_at_correct_position(self) -> None:
        """Given 400x300 image, when normalized callout at [0.5,0.5],
        then callout is drawn at pixel (200,150), NOT at (1,1)."""
        path = _make_white()
        ann = CalloutAnnotation(point=(0.5, 0.5), number=1, color="#FF0000")
        try:
            _, output_path, _ = render_annotations(path, [ann], normalized=True)
            img = Image.open(output_path).convert("RGB")
            px = img.getpixel((200, 150))
            assert px[0] > 200, f"Expected red at denormalized center (200,150), got {px}"
        finally:
            os.unlink(path)
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_normalized_circle_drawn_at_correct_position(self) -> None:
        """Given 400x300 image, when normalized circle at [0.5,0.5] radius 30,
        then circle edge is at pixel (230,150), NOT at (31,1)."""
        path = _make_white()
        ann = CircleAnnotation(center=(0.5, 0.5), radius=30, color="#FF0000", thickness=3)
        try:
            _, output_path, _ = render_annotations(path, [ann], normalized=True)
            img = Image.open(output_path).convert("RGB")
            # Right edge: 200+30=230, y=150
            px = img.getpixel((229, 150))
            assert px[0] > 200, f"Expected red at denormalized circle edge (229,150), got {px}"
        finally:
            os.unlink(path)
            if os.path.exists(output_path):
                os.unlink(output_path)
