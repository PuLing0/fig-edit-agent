"""Tool for semantic grounding: text + image -> anchor points."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import Field, model_validator

from ..core import CoordinateManager, LLMClient
from ..schemas import Artifact, ArtifactType, CoordinateInfo, Identifier, NonEmptyStr, Point2D, StrictSchema
from .base import ArtifactRegistry, BaseTool, ToolContext, ToolResult, to_model_image_url
from .registry import tool_registry


class ImageGroundingArgs(StrictSchema):
    """Arguments for semantic grounding."""

    base_image_artifact_id: Identifier = Field(description="Artifact id of the source image.")
    prompt: NonEmptyStr = Field(description="Natural-language description of the target to locate.")
    output_spec_name: NonEmptyStr = Field(default="points", description="Output slot name for the grounding result.")
    output_role: NonEmptyStr = Field(default="grounding_points", description="Semantic role for the POINTS artifact.")


class ImageGroundingResult(StrictSchema):
    """Structured output returned by the grounding model."""

    label: NonEmptyStr = Field(description="Short label for the grounded object.")
    positive_points: list[Point2D] = Field(
        description="One to three anchor points that are clearly inside the target object.",
        min_length=1,
    )
    negative_points: list[Point2D] = Field(
        default_factory=list,
        description="Optional background or distractor points that should stay outside the target mask.",
    )
    confidence: float | None = Field(default=None, description="Optional confidence score in [0,1].")
    rationale: str | None = Field(default=None, description="Optional grounding rationale.")

    @model_validator(mode="after")
    def validate_confidence(self) -> "ImageGroundingResult":
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")
        if len(self.positive_points) > 3:
            raise ValueError("positive_points may contain at most 3 anchors")
        if len(self.negative_points) > 2:
            raise ValueError("negative_points may contain at most 2 anchors")
        return self


class ImageGroundingTool(BaseTool[ImageGroundingArgs]):
    name = "image_grounding"
    description = "Ground a semantic target in an image and emit anchor points."
    args_model = ImageGroundingArgs

    async def run(self, ctx: ToolContext, args: ImageGroundingArgs) -> ToolResult:
        image_artifact = ctx.artifact_registry.require_type(args.base_image_artifact_id, ArtifactType.IMAGE)
        image_coord = ctx.artifact_registry.require_coordinate_info(args.base_image_artifact_id)
        image_path = Path(image_artifact.value).expanduser().resolve()
        with Image.open(image_path) as image:
            width, height = image.size

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are an image grounding assistant. Given an image and a target description, "
                    "return one to three positive anchor points that are clearly inside the target object. "
                    "Optionally return up to two negative points on nearby distractors or background. "
                    "Do not return bounding boxes. Positive points must lie well inside the target, not on its border."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": args.prompt},
                    {"type": "image_url", "image_url": {"url": to_model_image_url(image_artifact.value)}},
                ],
            },
        ]
        result = await ctx.llm.generate_structured(
            messages=messages,
            response_model=ImageGroundingResult,
            task_name=self.name,
        )
        positive_points = [
            CoordinateManager.clamp_point(point, width=width, height=height) for point in result.positive_points
        ]
        negative_points = [
            CoordinateManager.clamp_point(point, width=width, height=height) for point in result.negative_points
        ]
        points_artifact = ctx.register_artifact(
            artifact_type=ArtifactType.POINTS,
            value={
                "label": result.label,
                "positive_points": [point.model_dump() for point in positive_points],
                "negative_points": [point.model_dump() for point in negative_points],
                "confidence": result.confidence,
                "rationale": result.rationale,
            },
            metadata={
                "label": result.label,
                "confidence": result.confidence,
                "rationale": result.rationale,
                "coordinate_info": image_coord.model_dump(),
            },
        )
        output = ctx.bind_output(spec_name=args.output_spec_name, role=args.output_role, artifact=points_artifact)
        return ToolResult(
            outputs=[output],
            summary=f"Grounded '{result.label}' in image artifact '{image_artifact.artifact_id}'.",
        )


image_grounding_tool = ImageGroundingTool()
tool_registry.register(image_grounding_tool, replace=True)


async def _demo_main() -> None:
    example_image = Path(__file__).resolve().parent.parent / "examples" / "edit_example.png"
    with Image.open(example_image) as image:
        width, height = image.size

    artifacts: dict[str, Artifact] = {}
    registry = ArtifactRegistry(artifacts)
    root_artifact = registry.register(
        workflow_id="wf_demo",
        artifact_type=ArtifactType.IMAGE,
        value=str(example_image),
        metadata={
            "description": "Local example image for grounding testing",
            "coordinate_info": CoordinateInfo(
                root_artifact_id="artifact_demo_image",
                width=width,
                height=height,
                transform_kind="translation_only",
                offset_x=0,
                offset_y=0,
                parent_artifact_id=None,
            ).model_dump(),
        },
        artifact_id="artifact_demo_image",
    )
    ctx = ToolContext(
        llm=LLMClient(),
        artifact_registry=registry,
        workflow_id="wf_demo",
        node_id="node_grounding",
        attempt_id="attempt_1",
    )
    result = await image_grounding_tool(
        ctx,
        {
            "base_image_artifact_id": root_artifact.artifact_id,
            "prompt": (
                "From the viewer's perspective, find the watch-wearing hand on the left side of the image "
                "that is holding the Python book. Return 1-3 positive anchor points clearly inside that hand, "
                "and optionally one negative point on the book or forearm if helpful."
            ),
        },
    )
    print("=== image_grounding demo ===")
    print(result.model_dump_json(indent=2))
    print("=== registered artifacts ===")
    for artifact in registry.list_all():
        print(artifact.model_dump_json(indent=2))


def main() -> None:
    asyncio.run(_demo_main())


if __name__ == "__main__":
    main()


__all__ = [
    "ImageGroundingArgs",
    "ImageGroundingResult",
    "ImageGroundingTool",
    "image_grounding_tool",
]
