"""Geometry schemas shared across grounding, segmentation, crop, and future composition."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, PositiveInt, model_validator

from .base import Identifier, StrictSchema


class BoundingBox(StrictSchema):
    """Axis-aligned bounding box using half-open interval semantics.

    Coordinates follow [left, right) and [top, bottom), matching PIL/NumPy slicing.
    """

    left: int = Field(description="Inclusive left x coordinate.")
    top: int = Field(description="Inclusive top y coordinate.")
    right: int = Field(description="Exclusive right x coordinate.")
    bottom: int = Field(description="Exclusive bottom y coordinate.")

    @model_validator(mode="after")
    def validate_box(self) -> "BoundingBox":
        if self.right <= self.left:
            raise ValueError("right must be greater than left")
        if self.bottom <= self.top:
            raise ValueError("bottom must be greater than top")
        return self


class Point2D(StrictSchema):
    """A point in image coordinates."""

    x: int = Field(description="Horizontal coordinate in pixels.")
    y: int = Field(description="Vertical coordinate in pixels.")


class Polygon2D(StrictSchema):
    """Polygon defined by an ordered list of points."""

    points: list[Point2D] = Field(default_factory=list, description="Polygon vertices.")

    @model_validator(mode="after")
    def validate_polygon(self) -> "Polygon2D":
        if self.points and len(self.points) < 3:
            raise ValueError("polygon must contain at least 3 points when provided")
        return self


class CoordinateInfo(StrictSchema):
    """Describe how local coordinates map into a root coordinate space."""

    root_artifact_id: Identifier = Field(description="Artifact id defining the root coordinate space.")
    width: PositiveInt = Field(description="Width of the current artifact/image space.")
    height: PositiveInt = Field(description="Height of the current artifact/image space.")
    transform_kind: Literal["translation_only", "affine"] = Field(
        default="translation_only",
        description="How local coordinates map into the root coordinate space.",
    )
    offset_x: int | None = Field(
        default=0,
        description="Translation offset in x when transform_kind == translation_only.",
    )
    offset_y: int | None = Field(
        default=0,
        description="Translation offset in y when transform_kind == translation_only.",
    )
    affine_matrix: list[float] | None = Field(
        default=None,
        description="Affine transform [a, b, c, d, e, f] mapping local coordinates into root coordinates.",
    )
    parent_artifact_id: Identifier | None = Field(
        default=None,
        description="Direct parent artifact from which this coordinate space was derived.",
    )

    @model_validator(mode="after")
    def validate_transform(self) -> "CoordinateInfo":
        if self.transform_kind == "translation_only":
            if self.offset_x is None or self.offset_y is None:
                raise ValueError("translation_only coordinates require offset_x and offset_y")
        else:
            if self.affine_matrix is None or len(self.affine_matrix) != 6:
                raise ValueError("affine coordinates require a 6-value affine_matrix")
        return self


class PlacementRecord(StrictSchema):
    """Placement metadata for a source artifact inside a collage canvas."""

    source_artifact_id: Identifier = Field(description="Source artifact placed onto the canvas.")
    transform_to_canvas: list[float] = Field(
        min_length=6,
        max_length=6,
        description="Affine transform [a, b, c, d, e, f] mapping source local coordinates into canvas coordinates.",
    )
    z_index: int = Field(default=0, description="Layer order.")
    opacity: float = Field(default=1.0, description="Layer opacity in [0,1].")

    @model_validator(mode="after")
    def validate_opacity(self) -> "PlacementRecord":
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError("opacity must be within [0, 1]")
        return self


__all__ = ["BoundingBox", "CoordinateInfo", "PlacementRecord", "Point2D", "Polygon2D"]
