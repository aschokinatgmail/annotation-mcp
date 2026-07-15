"""MCP tool handler functions for image annotation."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _unwrap_mcp_tuple_fields(ann_dict: dict[str, Any]) -> None:
    """Unwrap FastMCP's {"item": [...]} serialization for tuple-typed fields.

    FastMCP wraps tuple values in an object with key "item" when they appear
    inside dicts. Pydantic tuple validators expect a plain array, so we unwrap.

    Also renames "from_" (Python field name) to "from" (Pydantic alias) for
    ArrowAnnotation — Pydantic won't match the alias unless we use the alias key.
    """
    for key in ("bbox", "point", "position", "center", "to"):
        val = ann_dict.get(key)
        if isinstance(val, dict) and "item" in val and len(val) == 1:
            ann_dict[key] = val["item"]
    # Handle "from_" -> "from" alias for ArrowAnnotation
    if "from_" in ann_dict and "from" not in ann_dict:
        ann_dict["from"] = ann_dict.pop("from_")
    # Also unwrap tuple-form for "from" after renaming
    val = ann_dict.get("from")
    if isinstance(val, dict) and "item" in val and len(val) == 1:
        ann_dict["from"] = val["item"]

from mcp.types import CallToolResult, TextContent

from annotation_mcp.draw_utils import make_result
from annotation_mcp.image_io import get_image_info
from annotation_mcp.models import (
    Annotation,
    ArrowAnnotation,
    BboxAnnotation,
    CalloutAnnotation,
    CircleAnnotation,
    HighlightAnnotation,
    TextAnnotation,
)
from annotation_mcp.render import render_annotations


def handle_get_image_info(path: str) -> CallToolResult:
    """Handle get_image_info tool call.

    Args:
        path: Absolute path to the image file.

    Returns:
        CallToolResult with structuredContent containing image metadata.
    """
    if not Path(path).is_file():
        return CallToolResult(
            content=[TextContent(type="text", text=f"File not found: {path}")],
            isError=True,
        )

    try:
        info = get_image_info(path)
        info_dict: dict[str, object] = {
            "width": info.width,
            "height": info.height,
            "format": info.format,
            "orientation": info.orientation,
            "size_bytes": info.size_bytes,
            "has_alpha": info.has_alpha,
            "density": info.density,
        }
        return CallToolResult(
            content=[TextContent(type="text", text=f"Image info for: {path}")],
            structuredContent=info_dict,
            isError=False,
        )
    except OSError as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error reading image: {e}")],
            isError=True,
        )


def handle_annotate(  # noqa: C901
    path: str,
    annotations: list[dict[str, object]],
    output_path: str | None = None,
    normalized: bool = False,
) -> CallToolResult:
    """Handle annotate tool call.

    Args:
        path: Input image path.
        annotations: List of annotation dicts with a 'type' discriminator.
        output_path: Optional output path. Auto-generated if None.
        normalized: If True, all coordinates are 0-1 fractions.

    Returns:
        CallToolResult with ImageContent + structured manifest.
    """
    if not Path(path).is_file():
        return CallToolResult(
            content=[TextContent(type="text", text=f"File not found: {path}")],
            isError=True,
        )

    try:
        parsed_annotations: list[Annotation] = []
        for ann_dict in annotations:
            ann_type = str(ann_dict.get("type", ""))
            _unwrap_mcp_tuple_fields(ann_dict)
            match ann_type:
                case "bbox":
                    parsed_annotations.append(BboxAnnotation.model_validate(ann_dict))
                case "arrow":
                    parsed_annotations.append(
                        ArrowAnnotation.model_validate(ann_dict)
                    )
                case "highlight":
                    parsed_annotations.append(
                        HighlightAnnotation.model_validate(ann_dict)
                    )
                case "callout":
                    parsed_annotations.append(
                        CalloutAnnotation.model_validate(ann_dict)
                    )
                case "text":
                    parsed_annotations.append(
                        TextAnnotation.model_validate(ann_dict)
                    )
                case "circle":
                    parsed_annotations.append(
                        CircleAnnotation.model_validate(ann_dict)
                    )
                case _:
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"Unknown annotation type: {ann_type}",
                            )
                        ],
                        isError=True,
                    )

        png_bytes, out_path, manifest = render_annotations(
            path=path,
            annotations=parsed_annotations,
            normalized=normalized,
            output_path=output_path,
        )

        return make_result(png_bytes, manifest.model_dump(), out_path)

    except ValueError as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Validation error: {e}")],
            isError=True,
        )
    except OSError as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {e}")],
            isError=True,
        )


def handle_draw_bounding_box(  # noqa: PLR0913
    path: str,
    bbox: tuple[float, float, float, float],
    normalized: bool = False,
    label: str | None = None,
    color: str = "#FF0000",
    thickness: int = 3,
    font_size: int = 16,
    output_path: str | None = None,
) -> CallToolResult:
    """Handle draw_bounding_box tool call.

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
    ann_dict: dict[str, object] = {
        "type": "bbox",
        "bbox": bbox,
        "label": label,
        "color": color,
        "thickness": thickness,
        "font_size": font_size,
    }
    return handle_annotate(
        path=path,
        annotations=[ann_dict],
        output_path=output_path,
        normalized=normalized,
    )


def handle_highlight_region(  # noqa: PLR0913
    path: str,
    bbox: tuple[float, float, float, float],
    normalized: bool = False,
    color: str = "#FFFF00",
    opacity: float = 0.3,
    output_path: str | None = None,
) -> CallToolResult:
    """Handle highlight_region tool call.

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
    ann_dict: dict[str, object] = {
        "type": "highlight",
        "bbox": bbox,
        "color": color,
        "opacity": opacity,
    }
    return handle_annotate(
        path=path,
        annotations=[ann_dict],
        output_path=output_path,
        normalized=normalized,
    )


def handle_draw_numbered_callouts(  # noqa: PLR0913
    path: str,
    callouts: list[dict[str, object]],
    normalized: bool = False,
    color: str = "#FF0000",
    radius: int = 18,
    font_size: int = 16,
    output_path: str | None = None,
) -> CallToolResult:
    """Handle draw_numbered_callouts tool call.

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
    annotations: list[dict[str, object]] = []
    for callout in callouts:
        point = callout.get("point")
        number = callout.get("number")
        if not isinstance(point, (list, tuple)) or not isinstance(number, int):
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text="Each callout must have 'point' (list) and 'number' (int)",
                    )
                ],
                isError=True,
            )
        ann: dict[str, object] = {
            "type": "callout",
            "point": point,
            "number": number,
            "color": color,
            "radius": radius,
            "font_size": font_size,
        }
        label = callout.get("label")
        if label is not None:
            ann["label"] = label
        annotations.append(ann)

    return handle_annotate(
        path=path,
        annotations=annotations,
        output_path=output_path,
        normalized=normalized,
    )
