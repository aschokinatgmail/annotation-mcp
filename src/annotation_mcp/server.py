"""MCP server for image annotation.

Provides tools for annotating images with bounding boxes, arrows,
highlights, callouts, text, and circles, plus detection tools for
discovering object coordinates (barcodes, text regions) and cropping
for iterative refinement.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from annotation_mcp.handlers import (
    handle_annotate,
    handle_draw_bounding_box,
    handle_draw_numbered_callouts,
    handle_get_image_info,
    handle_highlight_region,
)
from annotation_mcp.handlers_detection import (
    handle_crop_for_inspection,
    handle_detect_barcodes,
    handle_detect_text_regions,
)

server = FastMCP(
    "annotation-mcp",
    instructions=(
        "MCP server for image annotation. "
        "Draws bounding boxes, arrows, highlights, callouts, text, and circles on images. "
        "Also detects barcodes and text regions, and crops image regions for inspection."
    ),
)


@server.tool(
    name="get_image_info",
    description="Get image metadata (dimensions, format, orientation, size, alpha, density).",
)
def tool_get_image_info(path: str) -> object:
    """Get metadata about an image file.

    Args:
        path: Absolute path to the image file.

    Returns:
        CallToolResult with structuredContent containing image metadata.
    """
    return handle_get_image_info(path)


@server.tool(
    name="annotate",
    description=(
        "Draw multiple annotations (bbox, arrow, highlight, callout, text, circle)"
        " on an image. Returns annotated image + manifest JSON."
    ),
)
def tool_annotate(
    path: str,
    annotations: list[dict[str, object]],
    output_path: str | None = None,
    normalized: bool = False,
) -> object:
    """Draw multiple annotations on an image.

    Args:
        path: Input image path.
        annotations: List of annotation dicts with a 'type' discriminator.
        output_path: Optional output path. Auto-generated if None.
        normalized: If True, all coordinates are 0-1 fractions of image dimensions.

    Returns:
        CallToolResult with ImageContent + structured manifest.
    """
    return handle_annotate(
        path=path,
        annotations=annotations,
        output_path=output_path,
        normalized=normalized,
    )


@server.tool(
    name="draw_bounding_box",
    description="Draw a single bounding box on an image. Convenience wrapper around annotate.",
)
def tool_draw_bounding_box(  # noqa: PLR0913
    path: str,
    bbox: tuple[float, float, float, float],
    normalized: bool = False,
    label: str | None = None,
    color: str = "#FF0000",
    thickness: int = 3,
    font_size: int = 16,
    output_path: str | None = None,
) -> object:
    """Draw a single bounding box on an image.

    Args:
        path: Input image path.
        bbox: [x1, y1, x2, y2] bounding box coordinates.
        normalized: If True, coordinates are 0-1 fractions.
        label: Optional label text.
        color: Box color (hex, named, or rgba).
        thickness: Line thickness in pixels.
        font_size: Label font size.
        output_path: Optional output path.

    Returns:
        CallToolResult with ImageContent + structured manifest.
    """
    return handle_draw_bounding_box(
        path=path,
        bbox=bbox,
        normalized=normalized,
        label=label,
        color=color,
        thickness=thickness,
        font_size=font_size,
        output_path=output_path,
    )


@server.tool(
    name="highlight_region",
    description="Highlight a rectangular region on an image. Convenience wrapper around annotate.",
)
def tool_highlight_region(  # noqa: PLR0913
    path: str,
    bbox: tuple[float, float, float, float],
    normalized: bool = False,
    color: str = "#FFFF00",
    opacity: float = 0.3,
    output_path: str | None = None,
) -> object:
    """Highlight a rectangular region on an image.

    Args:
        path: Input image path.
        bbox: [x1, y1, x2, y2] region coordinates.
        normalized: If True, coordinates are 0-1 fractions.
        color: Highlight color.
        opacity: Highlight opacity (0-1).
        output_path: Optional output path.

    Returns:
        CallToolResult with ImageContent + structured manifest.
    """
    return handle_highlight_region(
        path=path,
        bbox=bbox,
        normalized=normalized,
        color=color,
        opacity=opacity,
        output_path=output_path,
    )


@server.tool(
    name="draw_numbered_callouts",
    description="Draw numbered callout circles on an image. Convenience wrapper around annotate.",
)
def tool_draw_numbered_callouts(  # noqa: PLR0913
    path: str,
    callouts: list[dict[str, object]],
    normalized: bool = False,
    color: str = "#FF0000",
    radius: int = 18,
    font_size: int = 16,
    output_path: str | None = None,
) -> object:
    """Draw numbered callout circles on an image.

    Args:
        path: Input image path.
        callouts: List of callout dicts with 'point' ([x,y]), 'number' (int),
            and optional 'label' (str).
        normalized: If True, point coordinates are 0-1 fractions.
        color: Callout circle color.
        radius: Callout circle radius in pixels.
        font_size: Label font size.
        output_path: Optional output path.

    Returns:
        CallToolResult with ImageContent + structured manifest.
    """
    return handle_draw_numbered_callouts(
        path=path,
        callouts=callouts,
        normalized=normalized,
        color=color,
        radius=radius,
        font_size=font_size,
        output_path=output_path,
    )


@server.tool(
    name="detect_barcodes",
    description=(
        "Detect 1D barcodes (EAN, UPC, Code 128/39, etc.) and 2D codes "
        "(QR, DataMatrix, PDF417) in an image. Returns deterministic pixel "
        "bounding boxes for each detected code. Requires pyzbar + zbar library."
    ),
)
def tool_detect_barcodes(
    path: str,
    min_confidence: float = 0.0,
    allowed_types: list[str] | None = None,
) -> object:
    """Detect barcodes and QR codes in an image.

    Args:
        path: Absolute path to image file.
        min_confidence: Discard matches with confidence below this threshold (0-1).
        allowed_types: If provided, only return barcodes of these types
            (e.g. ``["EAN13", "QRCODE"]``). Case-insensitive.

    Returns:
        CallToolResult with structuredContent containing detected barcodes.
    """
    return handle_detect_barcodes(
        path=path,
        min_confidence=min_confidence,
        allowed_types=allowed_types,
    )


@server.tool(
    name="detect_text_regions",
    description=(
        "Detect text regions in an image using Tesseract OCR. Returns each region "
        "with its text content, pixel bounding box, and confidence score. "
        "Use this to find the coordinates of brand titles, dosage tables, "
        "ingredients lists, etc. Supports multiple languages (e.g. ``eng+rus``) "
        "and various preprocessing modes for photos vs. clean scans. "
        "Set ``filter_garbage=false`` to keep OCR-noise regions, or "
        "``crop_regions=true`` to also save per-region image crops that a "
        "vision model can re-recognize (useful when tesseract quality is low)."
    ),
)
def tool_detect_text_regions(  # noqa: PLR0913
    path: str,
    lang: str = "eng",
    detail: str = "word",
    preprocess: str = "clahe",
    min_confidence: float = 0.0,
    psm: int | None = None,
    filter_garbage: bool = True,
    crop_regions: bool = False,
    crop_padding: float = 0.1,
    crop_output_dir: str | None = None,
) -> object:
    """Detect text regions in an image using Tesseract OCR.

    Args:
        path: Absolute path to image file.
        lang: Tesseract language code(s), e.g. ``"eng"``, ``"rus"``, ``"eng+rus"``.
        detail: Granularity: ``"word"`` | ``"line"`` | ``"block"`` | ``"paragraph"``.
        preprocess: Preprocessing: ``"none"`` | ``"clahe"`` | ``"adaptive"`` | ``"otsu"``.
        min_confidence: Discard regions with confidence below this (0-100).
        psm: Tesseract Page Segmentation Mode (1-13), or None for default.
        filter_garbage: If True (default), drop OCR-noise regions.
        crop_regions: If True, also save per-region crops for vision-model re-recognition.
        crop_padding: Padding fraction around each crop (0.1 = 10%).
        crop_output_dir: Where to save crops. If None, a temp dir is used.

    Returns:
        CallToolResult with structuredContent containing detected text regions.
        If crop_regions=True, also includes ``region_crops``: a list of
        {region_index, text, confidence, bbox_pixels, crop_path} entries.
    """
    return handle_detect_text_regions(
        path=path,
        lang=lang,
        detail=detail,
        preprocess=preprocess,
        min_confidence=min_confidence,
        psm=psm,
        filter_garbage=filter_garbage,
        crop_regions=crop_regions,
        crop_padding=crop_padding,
        crop_output_dir=crop_output_dir,
    )


@server.tool(
    name="crop_for_inspection",
    description=(
        "Crop a region of an image and save it to a new file. Useful for "
        "iteratively zooming into a region so a vision model can give more "
        "precise coordinates. Bounding box can be in pixel or normalized "
        "[0, 1] coordinates; optional padding expands the crop on each side."
    ),
)
def tool_crop_for_inspection(  # noqa: PLR0913
    path: str,
    bbox: tuple[float, float, float, float],
    output_path: str | None = None,
    normalized: bool = False,
    padding: float = 0.0,
) -> object:
    """Crop a region of an image and save it.

    Args:
        path: Absolute path to source image.
        bbox: ``[x1, y1, x2, y2]`` region to crop.
        output_path: Where to save the crop (auto-generated if None).
        normalized: If True, bbox is in [0, 1] coords.
        padding: Fractional padding around bbox (0.1 = 10% on each side).

    Returns:
        CallToolResult with structuredContent containing crop geometry.
    """
    return handle_crop_for_inspection(
        path=path,
        bbox=bbox,
        output_path=output_path,
        normalized=normalized,
        padding=padding,
    )


def main() -> None:  # pragma: no cover — blocking stdio server, not testable
    """Run the MCP server on stdio transport."""
    server.run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover — entry point, not testable
    main()