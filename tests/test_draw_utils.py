"""Unit tests for draw_utils.py — parse_color, get_font, make_result."""

from __future__ import annotations

import base64
from pathlib import Path

from PIL import Image

from annotation_mcp.draw_utils import get_font, make_result, parse_color
from annotation_mcp.image_io import encode_png


class TestParseColor:
    """Tests for parse_color."""

    def test_hex_6_chars(self) -> None:
        """Given 6-char hex, when parsed, then returns RGBA with alpha=255."""
        result = parse_color("#FF0000")
        assert result == (255, 0, 0, 255)

    def test_hex_8_chars_with_alpha(self) -> None:
        """Given 8-char hex with alpha, when parsed, then returns RGBA with correct alpha."""
        result = parse_color("#FF000080")
        assert result == (255, 0, 0, 128)

    def test_named_color(self) -> None:
        """Given named color, when parsed, then returns RGBA."""
        result = parse_color("red")
        assert result[0] == 255
        assert result[1] == 0
        assert result[2] == 0
        assert result[3] == 255

    def test_rgba_string(self) -> None:
        """Given rgba() string, when parsed, then returns RGBA."""
        result = parse_color("rgba(255, 0, 0, 0.5)")
        # 0.5 * 255 = 127.5, truncated to 127 by int()
        assert result == (255, 0, 0, 127)

    def test_rgba_string_opaque(self) -> None:
        """Given rgba() string with alpha=1.0, when parsed, then returns RGBA with 255."""
        result = parse_color("rgba(0, 255, 0, 1.0)")
        assert result == (0, 255, 0, 255)


class TestGetFont:
    """Tests for get_font."""

    def test_returns_font(self) -> None:
        """Given font size, when called, then returns a font object."""
        font = get_font(16)
        assert font is not None


class TestMakeResult:
    """Tests for make_result."""

    def test_returns_call_tool_result(self, tmp_path: Path) -> None:
        """Given png bytes and manifest, when called, then returns CallToolResult."""
        img = Image.new("RGB", (10, 10))
        png_bytes = encode_png(img)
        manifest = {"version": "1.0", "annotations": []}
        test_path = str(tmp_path / "test.png")
        result = make_result(png_bytes, manifest, test_path)
        assert result.isError is False
        assert result.structuredContent == manifest
        assert len(result.content) == 2
        # First content is ImageContent
        assert result.content[0].type == "image"
        assert result.content[0].mimeType == "image/png"
        # Verify the data is valid base64
        data = result.content[0].data
        decoded = base64.b64decode(data)
        assert decoded == png_bytes
        # Second content is TextContent with the path
        assert result.content[1].type == "text"
        assert test_path in result.content[1].text
