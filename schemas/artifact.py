"""Artifact registry schemas.

All persistent business outputs are modeled as artifacts, including images,
masks, prompts, scores, and analysis payloads.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field, model_validator

from .base import Identifier, StrictSchema, UriStr, utc_now
from .enums import ArtifactType
from .geometry import CoordinateInfo


class Artifact(StrictSchema):
    """Single source-of-truth record for a persisted or inline artifact."""

    artifact_id: Identifier = Field(description="Globally unique identifier of the artifact.")
    workflow_id: Identifier = Field(description="Workflow that owns this artifact.")
    artifact_type: ArtifactType = Field(description="Normalized artifact category.")
    value: Any = Field(description="Artifact payload: URI for file artifacts, inline value for lightweight artifacts.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional semantic metadata such as description, labels, dimensions, and attributes.",
    )
    producer_node_id: Identifier | None = Field(
        default=None,
        description="Node that produced this artifact, if any.",
    )
    producer_attempt_id: Identifier | None = Field(
        default=None,
        description="Attempt that produced this artifact, if any.",
    )
    created_at: datetime = Field(default_factory=utc_now, description="UTC timestamp when the artifact was recorded.")

    @model_validator(mode="after")
    def validate_value_type(self) -> "Artifact":
        if self.artifact_type in {ArtifactType.IMAGE, ArtifactType.MASK}:
            if not isinstance(self.value, str) or not self.value.strip():
                raise ValueError("IMAGE and MASK artifacts require a non-empty URI/path string value")
        elif self.artifact_type in {ArtifactType.TEXT, ArtifactType.PROMPT}:
            if not isinstance(self.value, str):
                raise ValueError("TEXT and PROMPT artifacts require a string value")
        elif self.artifact_type == ArtifactType.SCORE:
            if not isinstance(self.value, (int, float, dict)):
                raise ValueError("SCORE artifacts require a numeric value or a score dictionary")
        elif self.artifact_type in {ArtifactType.POINTS, ArtifactType.ANALYSIS}:
            if not isinstance(self.value, dict):
                raise ValueError("POINTS and ANALYSIS artifacts require a dictionary value")
        if self.artifact_type in {ArtifactType.IMAGE, ArtifactType.MASK, ArtifactType.POINTS}:
            coord = self.metadata.get("coordinate_info")
            if coord is None:
                raise ValueError(f"{self.artifact_type.value} artifacts require metadata.coordinate_info")
            CoordinateInfo.model_validate(coord)
        return self


__all__ = ["Artifact"]
