"""Shared tool-layer primitives.

This module defines the minimal runtime contract that every tool should obey:
- a shared execution context (`ToolContext`)
- a unified artifact registration helper (`ArtifactRegistry`)
- a normalized return object (`ToolResult`)
- an abstract tool interface (`BaseTool`)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import base64
import mimetypes
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar
from urllib.parse import urlparse, unquote
from uuid import uuid4

from ..core import LLMClient
from ..schemas import Artifact, ArtifactBinding, ArtifactType, CoordinateInfo, StrictSchema


ArgsT = TypeVar("ArgsT", bound=StrictSchema)


class ArtifactRegistry:
    """Thin helper around the workflow artifact store.

    Tools should create artifacts only through this registry so that IDs, lineage,
    and producer metadata remain consistent across the system.
    """

    def __init__(
        self,
        artifacts: MutableMapping[str, Artifact],
        *,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._artifacts = artifacts
        self._id_factory = id_factory or (lambda: f"artifact_{uuid4().hex}")

    def exists(self, artifact_id: str) -> bool:
        return artifact_id in self._artifacts

    def get(self, artifact_id: str) -> Artifact:
        try:
            return self._artifacts[artifact_id]
        except KeyError as exc:
            raise KeyError(f"Unknown artifact_id: {artifact_id}") from exc

    def maybe_get(self, artifact_id: str) -> Artifact | None:
        return self._artifacts.get(artifact_id)

    def require_type(self, artifact_id: str, expected_type: ArtifactType) -> Artifact:
        artifact = self.get(artifact_id)
        if artifact.artifact_type != expected_type:
            raise TypeError(
                f"Artifact '{artifact_id}' must be of type {expected_type.value}, "
                f"got {artifact.artifact_type.value}"
            )
        return artifact

    def require_coordinate_info(self, artifact_id: str) -> CoordinateInfo:
        artifact = self.get(artifact_id)
        coord_info = artifact.metadata.get("coordinate_info")
        if coord_info is None:
            raise KeyError(f"Artifact '{artifact_id}' is missing metadata.coordinate_info")
        return CoordinateInfo.model_validate(coord_info)

    def register(
        self,
        *,
        workflow_id: str,
        artifact_type: ArtifactType,
        value: Any,
        metadata: dict[str, Any] | None = None,
        producer_node_id: str | None = None,
        producer_attempt_id: str | None = None,
        artifact_id: str | None = None,
    ) -> Artifact:
        new_artifact = Artifact(
            artifact_id=artifact_id or self._id_factory(),
            workflow_id=workflow_id,
            artifact_type=artifact_type,
            value=value,
            metadata=metadata or {},
            producer_node_id=producer_node_id,
            producer_attempt_id=producer_attempt_id,
        )
        self._artifacts[new_artifact.artifact_id] = new_artifact
        return new_artifact

    def bind_output(self, *, spec_name: str, role: str, artifact: Artifact | str) -> ArtifactBinding:
        artifact_id = artifact.artifact_id if isinstance(artifact, Artifact) else artifact
        if artifact_id not in self._artifacts:
            raise KeyError(f"Cannot bind unknown artifact_id: {artifact_id}")
        return ArtifactBinding(spec_name=spec_name, role=role, artifact_id=artifact_id)

    def list_all(self) -> list[Artifact]:
        return list(self._artifacts.values())


@dataclass(slots=True)
class ToolContext:
    """Runtime context passed into every tool execution."""

    llm: LLMClient
    artifact_registry: ArtifactRegistry
    workflow_id: str
    node_id: str | None = None
    attempt_id: str | None = None
    storage: Any | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def register_artifact(
        self,
        *,
        artifact_type: ArtifactType,
        value: Any,
        metadata: dict[str, Any] | None = None,
        artifact_id: str | None = None,
    ) -> Artifact:
        """Register a new artifact using the current workflow/node/attempt context."""

        return self.artifact_registry.register(
            workflow_id=self.workflow_id,
            artifact_type=artifact_type,
            value=value,
            metadata=metadata,
            producer_node_id=self.node_id,
            producer_attempt_id=self.attempt_id,
            artifact_id=artifact_id,
        )

    def bind_output(self, *, spec_name: str, role: str, artifact: Artifact | str) -> ArtifactBinding:
        """Create an output binding for a newly or previously registered artifact."""

        return self.artifact_registry.bind_output(spec_name=spec_name, role=role, artifact=artifact)


class ToolResult(StrictSchema):
    """Normalized return object produced by every tool."""

    outputs: list[ArtifactBinding]
    summary: str


class BaseTool(ABC, Generic[ArgsT]):
    """Abstract base class for all tools."""

    name: str
    description: str
    args_model: type[ArgsT]

    async def __call__(self, ctx: ToolContext, args: ArgsT | Mapping[str, Any]) -> ToolResult:
        parsed_args = self._coerce_args(args)
        return await self.run(ctx, parsed_args)

    def _coerce_args(self, args: ArgsT | Mapping[str, Any]) -> ArgsT:
        if isinstance(args, self.args_model):
            return args
        if isinstance(args, Mapping):
            return self.args_model.model_validate(dict(args))
        raise TypeError(
            f"Tool '{self.name}' expected {self.args_model.__name__} or mapping, got {type(args).__name__}"
        )

    @abstractmethod
    async def run(self, ctx: ToolContext, args: ArgsT) -> ToolResult:
        """Execute the tool with validated arguments."""


def to_model_image_url(value: str) -> str:
    """Convert an artifact image value into a model-compatible image URL.

    Supports:
    - https/http URLs (returned unchanged)
    - file:// URIs (converted to data URLs)
    - local filesystem paths (converted to data URLs)
    - already-embedded data URLs (returned unchanged)
    """

    if value.startswith("data:"):
        return value

    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"}:
        return value

    if parsed.scheme == "file":
        path = Path(unquote(parsed.path)).resolve()
    else:
        path = Path(value).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Image path not found: {path}")

    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


__all__ = [
    "ArtifactRegistry",
    "BaseTool",
    "ToolContext",
    "ToolResult",
    "to_model_image_url",
]
