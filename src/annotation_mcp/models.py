"""Pydantic models for annotation types and manifest."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field


class BboxAnnotation(BaseModel):
    """A bounding box annotation."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    type: Literal["bbox"] = "bbox"
    bbox: tuple[float, float, float, float] = Field(
        ..., description="[x1, y1, x2, y2] in absolute or normalized coords"
    )
    label: str | None = None
    color: str = "#FF0000"
    thickness: int = 3
    font_size: int = 16


class ArrowAnnotation(BaseModel):
    """An arrow annotation from one point to another."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    type: Literal["arrow"] = "arrow"
    from_: tuple[float, float] = Field(
        ..., alias="from", description="[x, y] start point"
    )
    to: tuple[float, float] = Field(..., description="[x, y] end point")
    color: str = "#FF0000"
    thickness: int = 2


class HighlightAnnotation(BaseModel):
    """A highlighted region annotation."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    type: Literal["highlight"] = "highlight"
    bbox: tuple[float, float, float, float] = Field(
        ..., description="[x1, y1, x2, y2] region to highlight"
    )
    color: str = "#FFFF00"
    opacity: float = 0.3


class CalloutAnnotation(BaseModel):
    """A numbered callout annotation."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    type: Literal["callout"] = "callout"
    point: tuple[float, float] = Field(..., description="[x, y] center point")
    number: int = Field(..., ge=0, description="Callout number")
    label: str | None = None
    color: str = "#FF0000"
    radius: int = 18
    font_size: int = 16


class TextAnnotation(BaseModel):
    """A text annotation at a position."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    type: Literal["text"] = "text"
    position: tuple[float, float] = Field(..., description="[x, y] text position")
    text: str
    font_size: int = 16
    color: str = "#FFFFFF"
    background: str = "rgba(0,0,0,0.6)"


class CircleAnnotation(BaseModel):
    """A circle annotation."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    type: Literal["circle"] = "circle"
    center: tuple[float, float] = Field(..., description="[x, y] center point")
    radius: int
    color: str = "#FF0000"
    thickness: int = 2
    label: str | None = None
    font_size: int = 16


Annotation = Annotated[
    BboxAnnotation
    | ArrowAnnotation
    | HighlightAnnotation
    | CalloutAnnotation
    | TextAnnotation
    | CircleAnnotation,
    Field(discriminator="type"),
]
"""Discriminated union of all annotation types."""


class ManifestAnnotation(BaseModel):
    """Serialized annotation entry in the manifest."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    id: str
    type: str
    label: str | None = None
    bbox_normalized: tuple[float, float, float, float] | None = None
    bbox_pixels: tuple[int, int, int, int] | None = None
    point_normalized: tuple[float, float] | None = None
    point_pixels: tuple[int, int] | None = None
    from_normalized: tuple[float, float] | None = None
    from_pixels: tuple[int, int] | None = None
    to_normalized: tuple[float, float] | None = None
    to_pixels: tuple[int, int] | None = None
    center_normalized: tuple[float, float] | None = None
    center_pixels: tuple[int, int] | None = None
    radius: int | None = None
    color: str | None = None
    opacity: float | None = None
    thickness: int | None = None
    font_size: int | None = None
    number: int | None = None
    text: str | None = None


class ManifestImage(BaseModel):
    """Image metadata in the manifest."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    path: str
    width: int
    height: int
    format: str


class Manifest(BaseModel):
    """The structured output manifest for an annotation operation."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    version: str = "1.0"
    image: ManifestImage
    annotations: list[ManifestAnnotation]
    created_at: str
    warnings: list[str]


def make_annotation_id() -> str:
    """Generate a unique annotation ID."""
    return str(uuid.uuid4())


def make_timestamp() -> str:
    """Generate an ISO-8601 timestamp string."""
    return datetime.now(UTC).isoformat()
