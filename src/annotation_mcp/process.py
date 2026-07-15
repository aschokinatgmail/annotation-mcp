"""Annotation processing — validation, normalization, manifest entry creation."""

from __future__ import annotations

from annotation_mcp.coords import (
    PixelBounds,
    clamp_bbox,
    denormalize_bbox,
    denormalize_point,
    normalize_bbox,
    validate_normalized_bbox,
    validate_normalized_point,
)
from annotation_mcp.models import (
    Annotation,
    ArrowAnnotation,
    BboxAnnotation,
    CalloutAnnotation,
    CircleAnnotation,
    HighlightAnnotation,
    ManifestAnnotation,
    TextAnnotation,
    make_annotation_id,
)


def _process_annotation(  # noqa: C901, PLR0912
    annotation: Annotation,
    width: int,
    height: int,
    normalized: bool,
    warnings: list[str],
) -> tuple[PixelBounds | None, ManifestAnnotation]:
    """Process a single annotation: validate, normalize, create manifest entry."""
    ann_id = make_annotation_id()
    pixel_bounds: PixelBounds | None = None
    manifest_entry: ManifestAnnotation | None = None

    match annotation:
        case BboxAnnotation() as ann:
            if normalized:
                bounds = validate_normalized_bbox(*ann.bbox)
                pixel_bounds = denormalize_bbox(bounds, width, height)
            else:
                x1, y1, x2, y2 = ann.bbox
                # Check pre-clamp values for clamping warning
                if (
                    x1 < 0 or y1 < 0
                    or x2 > width or y2 > height
                ):
                    warnings.append(
                        f"Annotation {ann_id}: coordinates clamped to image bounds"
                    )
                pixel_bounds = clamp_bbox(
                    round(x1), round(y1), round(x2), round(y2), width, height
                )

            if ann.bbox[0] > ann.bbox[2] or ann.bbox[1] > ann.bbox[3]:
                warnings.append(
                    f"Annotation {ann_id}: swapped corners auto-corrected"
                )

            manifest_entry = ManifestAnnotation(
                id=ann_id, type="bbox", label=ann.label,
                bbox_normalized=normalize_bbox(
                    pixel_bounds.x1, pixel_bounds.y1,
                    pixel_bounds.x2, pixel_bounds.y2, width, height,
                ),
                bbox_pixels=(
                    pixel_bounds.x1, pixel_bounds.y1,
                    pixel_bounds.x2, pixel_bounds.y2,
                ),
                color=ann.color, thickness=ann.thickness, font_size=ann.font_size,
            )

        case HighlightAnnotation() as ann:
            if normalized:
                bounds = validate_normalized_bbox(*ann.bbox)
                pixel_bounds = denormalize_bbox(bounds, width, height)
            else:
                x1, y1, x2, y2 = ann.bbox
                pixel_bounds = clamp_bbox(
                    round(x1), round(y1), round(x2), round(y2), width, height
                )

            if ann.bbox[0] > ann.bbox[2] or ann.bbox[1] > ann.bbox[3]:
                warnings.append(
                    f"Annotation {ann_id}: swapped corners auto-corrected"
                )

            manifest_entry = ManifestAnnotation(
                id=ann_id, type="highlight",
                bbox_normalized=normalize_bbox(
                    pixel_bounds.x1, pixel_bounds.y1,
                    pixel_bounds.x2, pixel_bounds.y2, width, height,
                ),
                bbox_pixels=(
                    pixel_bounds.x1, pixel_bounds.y1,
                    pixel_bounds.x2, pixel_bounds.y2,
                ),
                color=ann.color, opacity=ann.opacity,
            )

        case ArrowAnnotation() as ann:
            if normalized:
                validate_normalized_point(ann.from_[0], ann.from_[1])
                validate_normalized_point(ann.to[0], ann.to[1])
                from_px = denormalize_point(ann.from_[0], ann.from_[1], width, height)
                to_px = denormalize_point(ann.to[0], ann.to[1], width, height)
                from_norm = (ann.from_[0], ann.from_[1])
                to_norm = (ann.to[0], ann.to[1])
            else:
                from_px = (round(ann.from_[0]), round(ann.from_[1]))
                to_px = (round(ann.to[0]), round(ann.to[1]))
                from_norm = (
                    ann.from_[0] / width if width > 0 else 0.0,
                    ann.from_[1] / height if height > 0 else 0.0,
                )
                to_norm = (
                    ann.to[0] / width if width > 0 else 0.0,
                    ann.to[1] / height if height > 0 else 0.0,
                )
            manifest_entry = ManifestAnnotation(
                id=ann_id, type="arrow",
                from_normalized=from_norm, from_pixels=from_px,
                to_normalized=to_norm, to_pixels=to_px,
                color=ann.color, thickness=ann.thickness,
            )

        case CalloutAnnotation() as ann:
            if normalized:
                validate_normalized_point(ann.point[0], ann.point[1])
                point_px = denormalize_point(ann.point[0], ann.point[1], width, height)
                point_norm = (ann.point[0], ann.point[1])
            else:
                point_px = (round(ann.point[0]), round(ann.point[1]))
                point_norm = (
                    ann.point[0] / width if width > 0 else 0.0,
                    ann.point[1] / height if height > 0 else 0.0,
                )
            manifest_entry = ManifestAnnotation(
                id=ann_id, type="callout",
                point_normalized=point_norm, point_pixels=point_px,
                radius=ann.radius, color=ann.color,
                number=ann.number, label=ann.label, font_size=ann.font_size,
            )

        case TextAnnotation() as ann:
            if normalized:
                validate_normalized_point(ann.position[0], ann.position[1])
                pos_px = denormalize_point(
                    ann.position[0], ann.position[1], width, height
                )
                pos_norm = (ann.position[0], ann.position[1])
            else:
                pos_px = (round(ann.position[0]), round(ann.position[1]))
                pos_norm = (
                    ann.position[0] / width if width > 0 else 0.0,
                    ann.position[1] / height if height > 0 else 0.0,
                )
            manifest_entry = ManifestAnnotation(
                id=ann_id, type="text",
                point_normalized=pos_norm, point_pixels=pos_px,
                color=ann.color, font_size=ann.font_size, text=ann.text,
            )

        case CircleAnnotation() as ann:
            if normalized:
                validate_normalized_point(ann.center[0], ann.center[1])
                center_px = denormalize_point(
                    ann.center[0], ann.center[1], width, height
                )
                center_norm = (ann.center[0], ann.center[1])
            else:
                center_px = (round(ann.center[0]), round(ann.center[1]))
                center_norm = (
                    ann.center[0] / width if width > 0 else 0.0,
                    ann.center[1] / height if height > 0 else 0.0,
                )
            manifest_entry = ManifestAnnotation(
                id=ann_id, type="circle",
                center_normalized=center_norm, center_pixels=center_px,
                radius=ann.radius, color=ann.color,
                thickness=ann.thickness, label=ann.label, font_size=ann.font_size,
            )

        case _:
            msg = f"Unknown annotation type: {type(annotation).__name__}"
            raise ValueError(msg)

    return (pixel_bounds, manifest_entry)


def _coerce_annotation(annotation: Annotation) -> Annotation:
    """Coerce a dict to a Pydantic model if needed."""
    if not isinstance(annotation, dict):
        return annotation
    ann_type = annotation.get("type", "")
    match ann_type:
        case "bbox":
            return BboxAnnotation.model_validate(annotation)
        case "arrow":
            return ArrowAnnotation.model_validate(annotation)
        case "highlight":
            return HighlightAnnotation.model_validate(annotation)
        case "callout":
            return CalloutAnnotation.model_validate(annotation)
        case "text":
            return TextAnnotation.model_validate(annotation)
        case "circle":
            return CircleAnnotation.model_validate(annotation)
        case _:
            msg = f"Unknown annotation type: {ann_type}"
            raise ValueError(msg)
