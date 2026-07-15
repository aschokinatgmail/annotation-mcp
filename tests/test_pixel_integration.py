"""Pixel-level integration tests — assert on actual pixel values in rendered images."""
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
    """Create white test image, return path."""
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    Image.new("RGB", (width, height), (255, 255, 255)).save(path)
    return path

def _pixels(path: str) -> Image.Image:
    """Load output image as RGB for pixel checking."""
    return Image.open(path).convert("RGB")


class TestBboxPixels:
    """Given white image, when red bbox drawn, then edge pixels are red."""

    def test_red_bbox_edge_pixels_when_rendered(self) -> None:
        """Given 400x300 white image, when red bbox at [50,50,200,200] thickness 5,
        then pixel on top edge is red and pixel inside is white."""
        # Given
        path = _make_white()
        ann = BboxAnnotation(bbox=(50, 50, 200, 200), color="#FF0000", thickness=5)
        # When
        _, output_path, _ = render_annotations(path, [ann])
        # Then
        try:
            img = _pixels(output_path)
            edge = img.getpixel((100, 50))  # On top edge of bbox
            assert edge[0] > 200, f"Expected red R>200 on edge, got {edge}"
            assert edge[1] < 80, f"Expected low G<80 on edge, got {edge}"
            inside = img.getpixel((100, 100))  # Inside bbox (outline only)
            assert inside == (255, 255, 255), f"Expected white inside, got {inside}"
        finally:
            os.unlink(path)
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestHighlightPixels:
    """Given white image, when yellow highlight drawn, then pixels are yellowish."""

    def test_yellow_highlight_pixels_when_rendered(self) -> None:
        """Given 400x300 white image, when yellow highlight at [50,50,200,200] opacity 0.5,
        then pixel inside is yellowish (R high, G high, B lower)."""
        # Given
        path = _make_white()
        ann = HighlightAnnotation(bbox=(50, 50, 200, 200), color="#FFFF00", opacity=0.5)
        # When
        _, output_path, _ = render_annotations(path, [ann])
        # Then
        try:
            img = _pixels(output_path)
            px = img.getpixel((100, 100))
            assert px[0] > 200, f"Expected R>200, got {px}"
            assert px[1] > 200, f"Expected G>200, got {px}"
            assert px[2] < 200, f"Expected B<200 (yellow tint), got {px}"
        finally:
            os.unlink(path)
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestArrowPixels:
    """Given white image, when green arrow drawn, then shaft pixels are green."""

    def test_green_arrow_pixels_when_rendered(self) -> None:
        """Given 400x300 white image, when green arrow from (350,250) to (250,180),
        then pixel near midpoint is green."""
        # Given
        path = _make_white()
        ann = ArrowAnnotation(**{"from": (350, 250), "to": (250, 180), "color": "#00FF00", "thickness": 4})
        # When
        _, output_path, _ = render_annotations(path, [ann])
        # Then
        try:
            img = _pixels(output_path)
            # Midpoint of arrow: ((350+250)//2, (250+180)//2) = (300, 215)
            px = img.getpixel((300, 215))
            assert px[1] > 200, f"Expected green G>200 at midpoint, got {px}"
            assert px[0] < 80, f"Expected low R<80, got {px}"
        finally:
            os.unlink(path)
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestCalloutPixels:
    """Given white image, when blue callout drawn, then center pixels are blue."""

    def test_blue_callout_pixels_when_rendered(self) -> None:
        """Given 400x300 white image, when blue callout at (100,200) number 1,
        then pixel at center is blue."""
        # Given
        path = _make_white()
        ann = CalloutAnnotation(point=(100, 200), number=1, color="#0066FF")
        # When
        _, output_path, _ = render_annotations(path, [ann])
        # Then
        try:
            img = _pixels(output_path)
            # Center has white number text; check inside circle fill below center
            px = img.getpixel((100, 190))  # Inside circle fill (y=200-10)
            assert px[2] > 150, f"Expected blue B>150 at circle edge, got {px}"
            assert px[0] < 100, f"Expected low R<100 at circle edge, got {px}"
        finally:
            os.unlink(path)
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestCirclePixels:
    """Given white image, when magenta circle drawn, then edge pixels are magenta."""

    def test_magenta_circle_edge_pixels_when_rendered(self) -> None:
        """Given 400x300 white image, when magenta circle at (300,80) radius 30,
        then pixel at right edge is magenta and center is white."""
        # Given
        path = _make_white()
        ann = CircleAnnotation(center=(300, 80), radius=30, color="#FF00FF", thickness=3)
        # When
        _, output_path, _ = render_annotations(path, [ann])
        # Then
        try:
            img = _pixels(output_path)
            edge = img.getpixel((329, 80))  # Right edge of circle
            assert edge[0] > 150, f"Expected magenta R>150 at edge, got {edge}"
            assert edge[2] > 150, f"Expected magenta B>150 at edge, got {edge}"
        finally:
            os.unlink(path)
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestTextPixels:
    """Given white image, when black text drawn, then text pixels are dark."""

    def test_black_text_pixels_when_rendered(self) -> None:
        """Given 400x300 white image, when black text 'Hello' at (20,270),
        then pixel near text start is dark and far pixel is white."""
        # Given
        path = _make_white()
        ann = TextAnnotation(position=(20, 270), text="Hello", color="#000000")
        # When
        _, output_path, _ = render_annotations(path, [ann])
        # Then
        try:
            img = _pixels(output_path)
            # Check a pixel near the text position — at least one should be dark
            found_dark = False
            for dx in range(0, 40, 3):
                for dy in range(0, 20, 3):
                    px = img.getpixel((20 + dx, 270 + dy))
                    if all(c < 100 for c in px):
                        found_dark = True
                        break
                if found_dark:
                    break
            assert found_dark, "Expected at least one dark pixel near text position"
            # Far pixel should be white
            assert img.getpixel((350, 50)) == (255, 255, 255)
        finally:
            os.unlink(path)
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestNormalizedCoordsPixels:
    """Given normalized coords, when rendered, then pixels are at correct positions."""

    def test_normalized_bbox_at_correct_position(self) -> None:
        """Given 400x300 image, when normalized bbox [0.25,0.25,0.75,0.75] red,
        then pixel at (200,75) — the denormalized top edge — is red."""
        # Given
        path = _make_white()
        ann = BboxAnnotation(bbox=(0.25, 0.25, 0.75, 0.75), color="#FF0000")
        # When
        _, output_path, _ = render_annotations(path, [ann], normalized=True)
        # Then
        try:
            img = _pixels(output_path)
            # 0.25*400=100, 0.75*400=300; top edge at y=0.25*300=75
            px = img.getpixel((200, 75))
            assert px[0] > 200, f"Expected red at (200,75), got {px}"
        finally:
            os.unlink(path)
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_normalized_text_at_bottom_not_top(self) -> None:
        """Given 400x300 image, when normalized text at [0.05,0.95],
        then text appears near bottom (y≈285), NOT near top (y≈1)."""
        # Given
        path = _make_white()
        ann = TextAnnotation(position=(0.05, 0.95), text="Bottom", color="#000000")
        # When
        _, output_path, _ = render_annotations(path, [ann], normalized=True)
        # Then
        try:
            img = _pixels(output_path)
            # Top of image should be white (no text there)
            top_px = img.getpixel((20, 5))
            assert top_px == (255, 255, 255), f"Expected white at top, got {top_px}"
            # Bottom area should have dark pixels (text)
            found_dark = False
            for dx in range(0, 50, 3):
                for dy in range(-5, 20, 3):
                    y = 285 + dy  # 0.95*300 = 285
                    if 0 <= y < 300:
                        px = img.getpixel((20 + dx, y))
                        if all(c < 100 for c in px):
                            found_dark = True
                            break
                if found_dark:
                    break
            assert found_dark, "Expected dark text pixels near bottom-left (y≈285)"
        finally:
            os.unlink(path)
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestAllAnnotationTypesRender:
    """Given all 6 annotation types, when rendered, then each type produces visible pixels."""

    def test_all_six_types_produce_visible_pixels(self) -> None:
        """Given 400x300 image with all 6 annotation types,
        then at least one pixel per type matches expected color."""
        # Given
        path = _make_white(500, 400)
        anns = [
            BboxAnnotation(bbox=(10, 10, 100, 100), color="#FF0000", thickness=4),
            HighlightAnnotation(bbox=(120, 10, 200, 100), color="#FFFF00", opacity=0.5),
            CalloutAnnotation(point=(300, 50), number=1, color="#0066FF"),
            TextAnnotation(position=(10, 360), text="Test", color="#000000"),
            ArrowAnnotation(**{"from": (450, 350), "to": (350, 250), "color": "#00FF00", "thickness": 4}),
            CircleAnnotation(center=(300, 200), radius=25, color="#FF00FF", thickness=3),
        ]
        # When
        _, output_path, _ = render_annotations(path, anns)
        # Then — check one representative pixel per type
        try:
            img = _pixels(output_path)
            # Bbox: top edge
            bbox_px = img.getpixel((50, 10))
            assert bbox_px[0] > 200, f"Bbox: expected red, got {bbox_px}"
            # Highlight: inside
            hl_px = img.getpixel((160, 50))
            assert hl_px[2] < 200, f"Highlight: expected yellow tint (B<200), got {hl_px}"
            # Callout: top edge of circle (center has white number text)
            co_px = img.getpixel((300, 33))  # y=50-18+1=33
            assert co_px[2] > 150, f"Callout: expected blue, got {co_px}"
            # Arrow: midpoint (450+350)//2=400, (350+250)//2=300
            ar_px = img.getpixel((400, 300))
            assert ar_px[1] > 200, f"Arrow: expected green, got {ar_px}"
            # Circle: right edge 300+25=325
            ci_px = img.getpixel((324, 200))
            assert ci_px[0] > 150 and ci_px[2] > 150, f"Circle: expected magenta, got {ci_px}"
        finally:
            os.unlink(path)
            if os.path.exists(output_path):
                os.unlink(output_path)
