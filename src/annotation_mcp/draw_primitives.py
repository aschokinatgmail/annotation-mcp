"""Individual drawing primitives for each annotation type.

allow: SIZE_OK — 6 cohesive drawing primitives sharing the same imports;
splitting would create artificial boundaries between peer functions.
"""

from __future__ import annotations

import math

from PIL import Image, ImageDraw

from annotation_mcp.coords import PixelBounds, clamp_point
from annotation_mcp.draw_utils import composite_rect, get_font, parse_color
from annotation_mcp.models import (  # noqa: TC001
    ArrowAnnotation,
    BboxAnnotation,
    CalloutAnnotation,
    CircleAnnotation,
    HighlightAnnotation,
    TextAnnotation,
)


def draw_bbox(
    draw: ImageDraw.ImageDraw,
    img: Image.Image,
    bounds: PixelBounds,
    annotation: BboxAnnotation,
) -> None:
    """Draw a bounding box with optional label.

    Args:
        draw: ImageDraw instance.
        bounds: Pixel bounds of the box.
        annotation: The bbox annotation spec.
    """
    color_rgba = parse_color(annotation.color)
    rgb = color_rgba[:3]
    draw.rectangle(
        [bounds.x1, bounds.y1, bounds.x2, bounds.y2],
        outline=rgb,
        width=annotation.thickness,
    )

    if annotation.label:
        font = get_font(annotation.font_size)
        bbox_text = draw.textbbox((0, 0), annotation.label, font=font)
        text_w = bbox_text[2] - bbox_text[0]
        text_h = bbox_text[3] - bbox_text[1]
        label_bg_x1 = bounds.x1
        label_bg_y1 = bounds.y1 - text_h - 4
        label_bg_x2 = bounds.x1 + text_w + 8
        label_bg_y2 = bounds.y1

        composite_rect(
            img,
            [label_bg_x1, label_bg_y1, label_bg_x2, label_bg_y2],
            (0, 0, 0, 180),
        )
        draw.text(
            (bounds.x1 + 4, bounds.y1 - text_h - 2),
            annotation.label,
            fill=(255, 255, 255),
            font=font,
        )


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    annotation: ArrowAnnotation,
    width: int,
    height: int,
) -> None:
    """Draw an arrow with arrowhead.

    Args:
        draw: ImageDraw instance.
        annotation: The arrow annotation spec.
        width: Image width.
        height: Image height.
    """
    from_x, from_y = clamp_point(
        round(annotation.from_[0]),
        round(annotation.from_[1]),
        width,
        height,
    )
    to_x, to_y = clamp_point(
        round(annotation.to[0]),
        round(annotation.to[1]),
        width,
        height,
    )

    color_rgba = parse_color(annotation.color)
    rgb = color_rgba[:3]

    draw.line([(from_x, from_y), (to_x, to_y)], fill=rgb, width=annotation.thickness)

    dx = to_x - from_x
    dy = to_y - from_y
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1:
        return

    ux = dx / length
    uy = dy / length

    head_len = annotation.thickness * 5
    angle = math.radians(25)

    left_x = to_x - head_len * (ux * math.cos(angle) - uy * math.sin(angle))
    left_y = to_y - head_len * (uy * math.cos(angle) + ux * math.sin(angle))
    right_x = to_x - head_len * (ux * math.cos(angle) + uy * math.sin(angle))
    right_y = to_y - head_len * (uy * math.cos(angle) - ux * math.sin(angle))

    draw.polygon(
        [(to_x, to_y), (left_x, left_y), (right_x, right_y)],
        fill=rgb,
    )


def draw_highlight(
    img: Image.Image,
    bounds: PixelBounds,
    annotation: HighlightAnnotation,
) -> None:
    """Draw a semi-transparent highlight overlay.

    Args:
        img: The image (modified in place via alpha composite).
        bounds: Pixel bounds of the highlight region.
        annotation: The highlight annotation spec.
    """
    color_rgba = parse_color(annotation.color)
    r, g, b = color_rgba[:3]
    x1, y1, x2, y2 = bounds.x1, bounds.y1, bounds.x2, bounds.y2
    rect_w = x2 - x1
    rect_h = y2 - y1
    if rect_w <= 0 or rect_h <= 0:
        return
    # Create overlay at rectangle size to avoid full-image allocation
    overlay = Image.new(
        "RGBA",
        (rect_w, rect_h),
        (r, g, b, int(255 * annotation.opacity)),
    )
    region = img.crop((x1, y1, x2, y2)).convert("RGBA")
    composited = Image.alpha_composite(region, overlay)
    img.paste(composited, (x1, y1))


def draw_callout(
    draw: ImageDraw.ImageDraw,
    img: Image.Image,
    annotation: CalloutAnnotation,
    width: int,
    height: int,
) -> None:
    """Draw a numbered callout circle with optional label.

    Args:
        draw: ImageDraw instance.
        annotation: The callout annotation spec.
        width: Image width.
        height: Image height.
    """
    cx, cy = clamp_point(
        round(annotation.point[0]),
        round(annotation.point[1]),
        width,
        height,
    )
    r = annotation.radius
    color_rgba = parse_color(annotation.color)
    rgb = color_rgba[:3]

    draw.ellipse(
        [cx - r, cy - r, cx + r, cy + r],
        fill=rgb,
        outline=(255, 255, 255),
        width=2,
    )

    font = get_font(r)
    num_text = str(annotation.number)
    bbox_text = draw.textbbox((0, 0), num_text, font=font)
    text_w = bbox_text[2] - bbox_text[0]
    text_h = bbox_text[3] - bbox_text[1]
    draw.text(
        (cx - text_w // 2, cy - text_h // 2),
        num_text,
        fill=(255, 255, 255),
        font=font,
    )

    if annotation.label:
        label_font = get_font(annotation.font_size)
        bbox_label = draw.textbbox((0, 0), annotation.label, font=label_font)
        label_w = bbox_label[2] - bbox_label[0]
        label_h = bbox_label[3] - bbox_label[1]
        label_x = cx + r + 8
        label_y = cy - label_h // 2

        composite_rect(
            img,
            [label_x - 2, label_y - 2, label_x + label_w + 2, label_y + label_h + 2],
            (0, 0, 0, 180),
        )
        draw.text(
            (label_x, label_y),
            annotation.label,
            fill=(255, 255, 255),
            font=label_font,
        )


def draw_text(
    draw: ImageDraw.ImageDraw,
    img: Image.Image,
    annotation: TextAnnotation,
    width: int,
    height: int,
) -> None:
    """Draw text with optional background.

    Args:
        draw: ImageDraw instance.
        img: The image (modified in place via alpha composite).
        annotation: The text annotation spec.
        width: Image width.
        height: Image height.
    """
    px, py = clamp_point(
        round(annotation.position[0]),
        round(annotation.position[1]),
        width,
        height,
    )

    font = get_font(annotation.font_size)
    color_rgba = parse_color(annotation.color)
    rgb = color_rgba[:3]

    if annotation.background:
        bbox_text = draw.textbbox((0, 0), annotation.text, font=font)
        text_w = bbox_text[2] - bbox_text[0]
        text_h = bbox_text[3] - bbox_text[1]
        bg_rgba = parse_color(annotation.background)
        composite_rect(
            img,
            [px - 2, py - 2, px + text_w + 2, py + text_h + 2],
            bg_rgba,
        )

    draw.text((px, py), annotation.text, fill=rgb, font=font)


def draw_circle(
    draw: ImageDraw.ImageDraw,
    img: Image.Image,
    annotation: CircleAnnotation,
    width: int,
    height: int,
) -> None:
    """Draw a circle with optional label.

    Args:
        draw: ImageDraw instance.
        img: The image (modified in place via alpha composite).
        annotation: The circle annotation spec.
        width: Image width.
        height: Image height.
    """
    cx, cy = clamp_point(
        round(annotation.center[0]),
        round(annotation.center[1]),
        width,
        height,
    )
    r = annotation.radius
    color_rgba = parse_color(annotation.color)
    rgb = color_rgba[:3]

    draw.ellipse(
        [cx - r, cy - r, cx + r, cy + r],
        outline=rgb,
        width=annotation.thickness,
    )

    if annotation.label:
        font = get_font(annotation.font_size)
        bbox_text = draw.textbbox((0, 0), annotation.label, font=font)
        text_w = bbox_text[2] - bbox_text[0]
        text_h = bbox_text[3] - bbox_text[1]
        label_x = cx + r + 6
        label_y = cy - text_h // 2

        composite_rect(
            img,
            [label_x - 2, label_y - 2, label_x + text_w + 2, label_y + text_h + 2],
            (0, 0, 0, 180),
        )
        draw.text(
            (label_x, label_y),
            annotation.label,
            fill=(255, 255, 255),
            font=font,
        )
