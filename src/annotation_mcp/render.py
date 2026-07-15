"""Pillow rendering engine for annotations — orchestration layer."""

from __future__ import annotations

import atexit
import io
import os
from typing import assert_never

from PIL import ImageDraw

from annotation_mcp.coords import (
    PixelBounds,
    denormalize_point,
)
from annotation_mcp.draw_primitives import (
    draw_arrow,
    draw_bbox,
    draw_callout,
    draw_circle,
    draw_highlight,
    draw_text,
)
from annotation_mcp.image_io import load_image, validate_input_path, validate_output_path
from annotation_mcp.models import (
    Annotation,
    ArrowAnnotation,
    BboxAnnotation,
    CalloutAnnotation,
    CircleAnnotation,
    HighlightAnnotation,
    Manifest,
    ManifestAnnotation,
    ManifestImage,
    TextAnnotation,
    make_timestamp,
)
from annotation_mcp.process import _coerce_annotation, _process_annotation


def _safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def render_annotations(  # noqa: C901
    path: str,
    annotations: list[Annotation],
    normalized: bool = False,
    output_path: str | None = None,
) -> tuple[bytes, str, Manifest]:
    """Render annotations onto an image and return the result.

    Args:
        path: Input image path.
        annotations: List of annotations to draw.
        normalized: Whether coordinates are normalized 0-1.
        output_path: Optional output path. Auto-generated if None.

    Returns:
        (png_bytes, output_path, manifest) tuple.

    Raises:
        FileNotFoundError: If the input image does not exist.
        CoordinateError: If any coordinate is invalid.
    """
    validate_input_path(path)
    if output_path is not None:
        validate_output_path(output_path)
    img = load_image(path)
    # Convert to RGBA so alpha fills (label backgrounds, etc.) render correctly
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    width, height = img.size
    draw = ImageDraw.Draw(img)
    warnings: list[str] = []

    # Coerce dicts to models
    coerced: list[Annotation] = [_coerce_annotation(a) for a in annotations]

    # Phase 1: Validate all annotations (atomic validation)
    processed: list[tuple[PixelBounds | None, ManifestAnnotation]] = []
    for annotation in coerced:
        result = _process_annotation(annotation, width, height, normalized, warnings)
        processed.append(result)

    # Phase 2: Draw all annotations
    for i, (pixel_bounds, _manifest_entry) in enumerate(processed):
        ann = coerced[i]
        match ann:
            case BboxAnnotation():
                if pixel_bounds is not None:  # pragma: no branch — always set for bbox
                    draw_bbox(draw, img, pixel_bounds, ann)
            case ArrowAnnotation():
                if normalized:
                    from_x, from_y = denormalize_point(
                        ann.from_[0], ann.from_[1], width, height
                    )
                    to_x, to_y = denormalize_point(
                        ann.to[0], ann.to[1], width, height
                    )
                    ann = ann.model_copy(
                        update={"from_": (from_x, from_y), "to": (to_x, to_y)}
                    )
                draw_arrow(draw, ann, width, height)
            case HighlightAnnotation():
                if pixel_bounds is not None:  # pragma: no branch — always set for highlight
                    draw_highlight(img, pixel_bounds, ann)
            case CalloutAnnotation():
                if normalized:
                    px, py = denormalize_point(
                        ann.point[0], ann.point[1], width, height
                    )
                    ann = ann.model_copy(update={"point": (px, py)})
                draw_callout(draw, img, ann, width, height)
            case TextAnnotation():
                if normalized:
                    px, py = denormalize_point(
                        ann.position[0], ann.position[1], width, height
                    )
                    ann = ann.model_copy(update={"position": (px, py)})
                draw_text(draw, img, ann, width, height)
            case CircleAnnotation():
                if normalized:
                    cx, cy = denormalize_point(
                        ann.center[0], ann.center[1], width, height
                    )
                    ann = ann.model_copy(update={"center": (cx, cy)})
                draw_circle(draw, img, ann, width, height)
            case _:  # pragma: no cover
                assert_never(ann)

    # Encode to PNG
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    # Determine output path (cross-platform: use tempdir, not /tmp)
    if output_path is None:
        import tempfile

        with tempfile.NamedTemporaryFile(
            prefix="annotation-mcp-", suffix=".png", delete=False
        ) as tmp:
            output_path = tmp.name
        atexit.register(_safe_unlink, output_path)

    # Write to disk
    with open(output_path, "wb") as f:  # noqa: PTH123
        f.write(png_bytes)

    # Build manifest
    manifest = Manifest(
        version="1.0",
        image=ManifestImage(
            path=path,
            width=width,
            height=height,
            format=img.format or "PNG",
        ),
        annotations=[m for _, m in processed],
        created_at=make_timestamp(),
        warnings=warnings,
    )

    return (png_bytes, output_path, manifest)
