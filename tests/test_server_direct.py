"""Direct unit tests for server.py — import module and call tool functions directly."""

from __future__ import annotations

import os
import tempfile

import pytest
from PIL import Image

# Import the server module to cover module-level code
import annotation_mcp.server as server_mod  # noqa: F811


@pytest.fixture
def test_image() -> str:
    """Create a simple test image and return its path."""
    img = Image.new("RGB", (200, 100), color=(255, 255, 255))
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp.name, format="PNG")
    tmp.close()
    return tmp.name


class TestServerModule:
    """Tests that import the server module and call tool functions directly."""

    def test_server_is_fastmcp_instance(self) -> None:
        """Given server module imported, when checked, then server is FastMCP instance."""
        assert server_mod.server is not None
        assert server_mod.server.name == "annotation-mcp"

    def test_tool_get_image_info(self, test_image: str) -> None:
        """Given valid path, when tool_get_image_info called, then returns result."""
        result = server_mod.tool_get_image_info(test_image)
        assert result is not None
        assert result.isError is False
        sc = result.structuredContent
        assert sc["width"] == 200
        assert sc["height"] == 100

    def test_tool_get_image_info_nonexistent(self) -> None:
        """Given nonexistent path, when called, then isError=True."""
        result = server_mod.tool_get_image_info("/nonexistent/path.png")
        assert result.isError is True

    def test_tool_annotate(self, test_image: str) -> None:
        """Given annotations, when tool_annotate called, then returns result."""
        result = server_mod.tool_annotate(
            test_image,
            [{"type": "bbox", "bbox": [10, 10, 100, 80]}],
        )
        assert result.isError is False
        assert len(result.structuredContent["annotations"]) == 1

    def test_tool_annotate_normalized(self, test_image: str) -> None:
        """Given normalized annotations, when called, then returns result with normalized coords."""
        result = server_mod.tool_annotate(
            test_image,
            [{"type": "bbox", "bbox": [0.0, 0.0, 0.5, 0.5]}],
            normalized=True,
        )
        assert result.isError is False
        ann = result.structuredContent["annotations"][0]
        assert ann["bbox_normalized"] == (0.0, 0.0, 0.5, 0.5)

    def test_tool_annotate_nonexistent(self) -> None:
        """Given nonexistent path, when called, then isError=True."""
        result = server_mod.tool_annotate("/nonexistent/path.png", [])
        assert result.isError is True

    def test_tool_draw_bounding_box(self, test_image: str) -> None:
        """Given bbox params, when tool_draw_bounding_box called, then returns result."""
        result = server_mod.tool_draw_bounding_box(
            test_image, (10, 10, 100, 80), label="test"
        )
        assert result.isError is False
        ann = result.structuredContent["annotations"][0]
        assert ann["type"] == "bbox"
        assert ann["label"] == "test"

    def test_tool_highlight_region(self, test_image: str) -> None:
        """Given highlight params, when tool_highlight_region called, then returns result."""
        result = server_mod.tool_highlight_region(
            test_image, (10, 10, 100, 80)
        )
        assert result.isError is False
        assert result.structuredContent["annotations"][0]["type"] == "highlight"

    def test_tool_draw_numbered_callouts(self, test_image: str) -> None:
        """Given callouts, when tool_draw_numbered_callouts called, then returns result."""
        result = server_mod.tool_draw_numbered_callouts(
            test_image,
            [{"point": [50, 50], "number": 1}],
        )
        assert result.isError is False
        assert result.structuredContent["annotations"][0]["type"] == "callout"
