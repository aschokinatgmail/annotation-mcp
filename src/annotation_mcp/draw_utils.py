"""Color, font, and result-building utilities for drawing."""

from __future__ import annotations

import base64

from mcp.types import CallToolResult, ImageContent, TextContent
from PIL import Image, ImageColor, ImageDraw, ImageFont


def parse_color(color: str) -> tuple[int, int, int, int]:
    """Parse a color string to RGBA tuple.

    Accepts hex strings (#FF0000, #FF0000FF), named colors (red),
    and rgba(r,g,b,a) strings.

    Args:
        color: Color string in any supported format.

    Returns:
        (R, G, B, A) tuple with values 0-255.
    """
    if color.startswith("rgba("):
        inner = color[5:-1]
        parts = [p.strip() for p in inner.split(",")]
        r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
        a_raw = float(parts[3])
        a = int(a_raw * 255) if a_raw <= 1.0 else int(a_raw)
        return (r, g, b, a)
    result = ImageColor.getrgb(color)
    if len(result) == 3:  # noqa: PLR2004
        return (result[0], result[1], result[2], 255)
    return result


def get_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Get a font at the requested size.

    Tries Helvetica first, falls back to default.

    Args:
        font_size: Desired font size in points.

    Returns:
        A PIL ImageFont.
    """
    try:
        return ImageFont.truetype("Helvetica.ttf", font_size)
    except OSError:
        return ImageFont.load_default(size=font_size)


def make_result(
    png_bytes: bytes,
    manifest_dict: dict[str, object],
    output_path: str,
) -> CallToolResult:
    """Build a CallToolResult with image content and structured manifest.

    Args:
        png_bytes: PNG-encoded image bytes.
        manifest_dict: Manifest as a dict.
        output_path: Path to the output file on disk.

    Returns:
        CallToolResult with ImageContent + TextContent for the path.
    """
    b64_data = base64.b64encode(png_bytes).decode("ascii")
    return CallToolResult(
        content=[
            ImageContent(
                type="image",
                data=b64_data,
                mimeType="image/png",
            ),
            TextContent(
                type="text",
                text=f"Output saved to: {output_path}",
            ),
        ],
        structuredContent=manifest_dict,
        isError=False,
    )


def composite_rect(
    img: Image.Image,
    xy: list[int | float],
    fill: tuple[int, int, int, int],
) -> None:
    """Draw a semi-transparent rectangle via alpha compositing.

    ImageDraw.rectangle with an RGBA fill replaces pixels instead of
    blending. This helper creates an overlay and alpha-composites it,
    producing correct semi-transparency.
    """
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(xy, fill=fill)
    composited = Image.alpha_composite(img.convert("RGBA"), overlay)
    img.paste(composited, (0, 0))

