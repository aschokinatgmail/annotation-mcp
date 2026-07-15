"""Functional tests for MCP server tool calls (subprocess-based)."""
from __future__ import annotations
import base64
import os
import tempfile
import pytest
from PIL import Image
from annotation_mcp.image_io import is_png_data
from test_server import MCPServerClient


@pytest.fixture
def test_jpeg() -> str:
    """Create a test JPEG image and return its path."""
    img = Image.new("RGB", (320, 240), color=(200, 200, 200))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 150, 150], fill=(100, 100, 100))
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(tmp.name, format="JPEG")
    tmp.close()
    return tmp.name


class TestMCPTools:
    """Tests for individual MCP tool calls."""

    def test_get_image_info(self, test_jpeg: str) -> None:
        """Given image path, when get_image_info called, then returns correct metadata."""
        client = MCPServerClient()
        try:
            client.initialize()
            client.send_initialized()
            response = client.call_tool("get_image_info", {"path": test_jpeg})
            assert response.get("id") == 3
            result = response.get("result")
            assert isinstance(result, dict)
            sc = result.get("structuredContent")
            assert isinstance(sc, dict)
            assert sc.get("width") == 320
            assert sc.get("height") == 240
            assert sc.get("format") == "JPEG"
            assert isinstance(sc.get("size_bytes"), int)
            assert sc.get("size_bytes", 0) > 0
            assert sc.get("has_alpha") is False
        finally:
            client.close()

    def test_annotate_returns_image_and_manifest(self, test_jpeg: str) -> None:
        """Given annotations, when annotate called, then returns ImageContent + manifest."""
        client = MCPServerClient()
        try:
            client.initialize()
            client.send_initialized()
            response = client.call_tool("annotate", {
                "path": test_jpeg,
                "annotations": [
                    {"type": "bbox", "bbox": [0.1, 0.1, 0.5, 0.5], "label": "test box", "color": "#FF0000"},
                    {"type": "highlight", "bbox": [0.2, 0.2, 0.6, 0.6], "color": "#FFFF00", "opacity": 0.3},
                ],
                "normalized": True,
            })
            assert response.get("id") == 3
            result = response.get("result")
            assert isinstance(result, dict)
            content = result.get("content", [])
            assert isinstance(content, list)
            image_content = [c for c in content if isinstance(c, dict) and c.get("type") == "image"]
            assert len(image_content) == 1
            assert image_content[0].get("mimeType") == "image/png"
            data = image_content[0].get("data")
            assert isinstance(data, str)
            png_data = base64.b64decode(data)
            assert is_png_data(png_data)
            sc = result.get("structuredContent")
            assert isinstance(sc, dict)
            assert sc.get("version") == "1.0"
            annotations_list = sc.get("annotations", [])
            assert isinstance(annotations_list, list)
            assert len(annotations_list) == 2
            assert annotations_list[0].get("type") == "bbox"
            assert annotations_list[1].get("type") == "highlight"
        finally:
            client.close()

    def test_draw_bounding_box(self, test_jpeg: str) -> None:
        """Given bbox params, when draw_bounding_box called, then returns annotated image."""
        client = MCPServerClient()
        try:
            client.initialize()
            client.send_initialized()
            response = client.call_tool("draw_bounding_box", {
                "path": test_jpeg, "bbox": [10, 10, 100, 100], "label": "box", "color": "#00FF00",
            })
            result = response.get("result")
            assert isinstance(result, dict)
            sc = result.get("structuredContent")
            assert isinstance(sc, dict)
            annotations_list = sc.get("annotations", [])
            assert isinstance(annotations_list, list)
            assert len(annotations_list) == 1
            assert annotations_list[0].get("type") == "bbox"
        finally:
            client.close()

    def test_highlight_region(self, test_jpeg: str) -> None:
        """Given region params, when highlight_region called, then returns highlighted image."""
        client = MCPServerClient()
        try:
            client.initialize()
            client.send_initialized()
            response = client.call_tool("highlight_region", {
                "path": test_jpeg, "bbox": [20, 20, 200, 200], "color": "#00FF00", "opacity": 0.5,
            })
            result = response.get("result")
            assert isinstance(result, dict)
            sc = result.get("structuredContent")
            assert isinstance(sc, dict)
            annotations_list = sc.get("annotations", [])
            assert isinstance(annotations_list, list)
            assert len(annotations_list) == 1
            assert annotations_list[0].get("type") == "highlight"
        finally:
            client.close()

    def test_draw_numbered_callouts(self, test_jpeg: str) -> None:
        """Given callouts, when draw_numbered_callouts called, then returns annotated image."""
        client = MCPServerClient()
        try:
            client.initialize()
            client.send_initialized()
            response = client.call_tool("draw_numbered_callouts", {
                "path": test_jpeg,
                "callouts": [
                    {"point": [50, 50], "number": 1, "label": "First"},
                    {"point": [200, 100], "number": 2},
                ],
                "color": "#0000FF", "radius": 20,
            })
            result = response.get("result")
            assert isinstance(result, dict)
            sc = result.get("structuredContent")
            assert isinstance(sc, dict)
            annotations_list = sc.get("annotations", [])
            assert isinstance(annotations_list, list)
            assert len(annotations_list) == 2
            assert annotations_list[0].get("type") == "callout"
            assert annotations_list[0].get("number") == 1
            assert annotations_list[1].get("number") == 2
        finally:
            client.close()
