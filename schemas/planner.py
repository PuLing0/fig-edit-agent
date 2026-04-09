"""Planner-facing schemas for semantic task decomposition."""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from .artifact import Artifact
from .base import Identifier, NonEmptyStr, StrictSchema
from .enums import ArtifactType, InputArtifactRole, NodeKind


def _default_available_node_kinds() -> list[NodeKind]:
    return [
        NodeKind.UNDERSTAND,
        NodeKind.EXTRACT_SUBJECT,
        NodeKind.SELECT_BACKGROUND,
        NodeKind.COMPOSE_SCENE,
        NodeKind.POLISH_IMAGE,
        NodeKind.EDIT,
        NodeKind.SCORE,
    ]


class ArtifactSummary(StrictSchema):
    """Compact semantic summary consumed by the planner."""

    artifact_id: Identifier = Field(description="Semantic summary artifact id.")
    description: NonEmptyStr = Field(description="Compact natural-language description of the image content.")
    labels: list[NonEmptyStr] = Field(
        default_factory=list,
        description="Short semantic labels extracted from the image.",
    )
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured image attributes such as style, lighting, or composition.",
    )
    source_artifact_id: Identifier | None = Field(
        default=None,
        description="Source image artifact that this summary describes.",
    )

    @classmethod
    def from_analysis_artifact(cls, artifact: Artifact) -> "ArtifactSummary":
        """Create an ArtifactSummary from an ANALYSIS artifact."""

        if artifact.artifact_type != ArtifactType.ANALYSIS:
            raise TypeError(
                f"ArtifactSummary requires an analysis artifact, got {artifact.artifact_type.value}"
            )

        payload = artifact.value if isinstance(artifact.value, dict) else {}
        metadata = artifact.metadata or {}
        description = payload.get("description") or metadata.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("Analysis artifact must contain a non-empty description")

        raw_labels = payload.get("labels") or metadata.get("labels") or []
        labels = [str(item).strip() for item in raw_labels if str(item).strip()]
        raw_attributes = payload.get("attributes") or metadata.get("attributes") or {}
        attributes = dict(raw_attributes) if isinstance(raw_attributes, dict) else {}

        source_artifact_id = metadata.get("source_artifact_id")
        if source_artifact_id is not None and not str(source_artifact_id).strip():
            source_artifact_id = None

        return cls(
            artifact_id=artifact.artifact_id,
            description=description,
            labels=labels,
            attributes=attributes,
            source_artifact_id=source_artifact_id,
        )


class ArtifactManifest(StrictSchema):
    """Planner-facing manifest entry for one input or intermediate artifact."""

    artifact_id: Identifier = Field(description="Concrete artifact id stored in the workflow.")
    artifact_type: ArtifactType = Field(description="Normalized artifact type.")
    uri_or_value: Any = Field(description="URI/path or inline value carried by the artifact.")
    input_role: InputArtifactRole = Field(
        default=InputArtifactRole.PRIMARY_INPUT,
        description="High-level planner role of this artifact.",
    )
    description: str | None = Field(
        default=None,
        description="Optional planner-facing description.",
    )
    labels: list[NonEmptyStr] = Field(
        default_factory=list,
        description="Optional semantic labels.",
    )
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional semantic attributes.",
    )
    is_user_selected_background: bool = Field(
        default=False,
        description="Whether the user explicitly chose this artifact as the background source.",
    )
    producer_node_id: Identifier | None = Field(default=None, description="Node that produced this artifact, if any.")
    producer_attempt_id: Identifier | None = Field(
        default=None,
        description="Attempt that produced this artifact, if any.",
    )

    @classmethod
    def from_artifact(
        cls,
        artifact: Artifact,
        *,
        input_role: InputArtifactRole = InputArtifactRole.PRIMARY_INPUT,
        is_user_selected_background: bool = False,
    ) -> "ArtifactManifest":
        """Create a planner manifest entry from a concrete artifact."""

        metadata = artifact.metadata or {}
        raw_labels = metadata.get("labels") or []
        labels = [str(item).strip() for item in raw_labels if str(item).strip()]
        raw_attributes = metadata.get("attributes") or {}
        attributes = dict(raw_attributes) if isinstance(raw_attributes, dict) else {}

        return cls(
            artifact_id=artifact.artifact_id,
            artifact_type=artifact.artifact_type,
            uri_or_value=artifact.value,
            input_role=input_role,
            description=metadata.get("description"),
            labels=labels,
            attributes=attributes,
            is_user_selected_background=is_user_selected_background,
            producer_node_id=artifact.producer_node_id,
            producer_attempt_id=artifact.producer_attempt_id,
        )


class PlanAgentRequest(StrictSchema):
    """Structured planner input for semantic task decomposition."""

    workflow_id: Identifier = Field(description="Workflow receiving the new plan.")
    goal: NonEmptyStr = Field(description="Short top-level goal summary.")
    user_prompt: NonEmptyStr = Field(description="Original user instruction.")
    input_artifacts: list[ArtifactManifest] = Field(
        min_length=1,
        description="Planner-facing input artifacts, each with a stable id and semantic role.",
    )
    artifact_summaries: list[ArtifactSummary] = Field(
        default_factory=list,
        description="Semantic summaries available to the planner.",
    )
    available_node_kinds: list[NodeKind] = Field(
        default_factory=_default_available_node_kinds,
        description="Node kinds that the planner may use in the DAG.",
    )
    plan_version: int = Field(default=1, ge=1, description="Monotonic target plan version.")
    planner_hints: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional caller-provided hints for plan generation.",
    )

    @model_validator(mode="after")
    def validate_request(self) -> "PlanAgentRequest":
        input_ids = [artifact.artifact_id for artifact in self.input_artifacts]
        if len(input_ids) != len(set(input_ids)):
            raise ValueError("input_artifact ids must be unique")

        summary_ids = [summary.artifact_id for summary in self.artifact_summaries]
        if len(summary_ids) != len(set(summary_ids)):
            raise ValueError("artifact summary ids must be unique")

        selected_backgrounds = [
            artifact for artifact in self.input_artifacts if artifact.is_user_selected_background
        ]
        if len(selected_backgrounds) > 1:
            raise ValueError("At most one input artifact may be marked as the user-selected background")

        if len(self.available_node_kinds) != len(set(self.available_node_kinds)):
            raise ValueError("available_node_kinds must be unique")
        return self


__all__ = ["ArtifactManifest", "ArtifactSummary", "PlanAgentRequest"]
