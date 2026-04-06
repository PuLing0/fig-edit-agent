"""Tool for scoring an image against an expected task objective."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import Field, model_validator

from ..core import LLMClient
from ..schemas import ArtifactType, CoordinateInfo, Identifier, NonEmptyStr, StrictSchema
from .base import ArtifactRegistry, BaseTool, ToolContext, ToolResult, to_model_image_url
from .registry import tool_registry


class ImageScoreArgs(StrictSchema):
    """Arguments for scoring an image."""

    image_artifact_id: Identifier = Field(description="Artifact id of the image to evaluate.")
    expected_prompt: NonEmptyStr = Field(description="Expected task objective or target prompt.")
    output_spec_name: NonEmptyStr = Field(
        default="score",
        description="Output slot name that should receive the score artifact.",
    )
    output_role: NonEmptyStr = Field(
        default="score",
        description="Semantic output role for the score artifact.",
    )


class ScoreDimension(StrictSchema):
    """One named scoring dimension."""

    name: NonEmptyStr = Field(description="Short score dimension name, e.g. prompt_alignment.")
    value: float = Field(description="Dimension score in [0, 1].")

    @model_validator(mode="after")
    def validate_value(self) -> "ScoreDimension":
        if not 0.0 <= self.value <= 1.0:
            raise ValueError("ScoreDimension.value must be within [0, 1]")
        return self


class ImageScoreResult(StrictSchema):
    """Structured score result returned by the model."""

    overall_score: float = Field(description="Overall image quality / goal completion score in [0, 1].")
    dimension_scores: list[ScoreDimension] = Field(
        default_factory=list,
        description="Optional named sub-scores chosen by the model.",
    )
    summary: NonEmptyStr = Field(description="Short evaluation summary.")

    @model_validator(mode="after")
    def validate_scores(self) -> "ImageScoreResult":
        if not 0.0 <= self.overall_score <= 1.0:
            raise ValueError("overall_score must be within [0, 1]")
        return self


class ImageScoreTool(BaseTool[ImageScoreArgs]):
    """LLM-driven scoring tool for image outputs."""

    name = "image_score"
    description = "Score an image against an expected task objective."
    args_model = ImageScoreArgs

    async def run(self, ctx: ToolContext, args: ImageScoreArgs) -> ToolResult:
        image_artifact = ctx.artifact_registry.require_type(args.image_artifact_id, ArtifactType.IMAGE)
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a strict image evaluation assistant. "
                    "Score how well the provided image satisfies the expected task objective. "
                    "All scores must be within [0, 1]. "
                    "overall_score should summarize overall completion quality. "
                    "dimension_scores may include any relevant dimensions you judge useful, "
                    "using short names such as prompt_alignment, visual_quality, text_accuracy, "
                    "composition, or subject_consistency."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Expected task objective:\n{args.expected_prompt}"},
                    {"type": "image_url", "image_url": {"url": to_model_image_url(image_artifact.value)}},
                ],
            },
        ]
        result = await ctx.llm.generate_structured(
            messages=messages,
            response_model=ImageScoreResult,
            task_name=self.name,
        )
        score_artifact = ctx.register_artifact(
            artifact_type=ArtifactType.SCORE,
            value={
                "overall_score": result.overall_score,
                "dimension_scores": [dimension.model_dump() for dimension in result.dimension_scores],
                "summary": result.summary,
            },
            metadata={
                "description": "Structured image evaluation score",
                "source_artifact_id": image_artifact.artifact_id,
                "expected_prompt": args.expected_prompt,
                "overall_score": result.overall_score,
                "dimension_names": [dimension.name for dimension in result.dimension_scores],
            },
        )
        output = ctx.bind_output(
            spec_name=args.output_spec_name,
            role=args.output_role,
            artifact=score_artifact,
        )
        return ToolResult(
            outputs=[output],
            summary=f"Scored image artifact '{image_artifact.artifact_id}' against the expected objective.",
        )


image_score_tool = ImageScoreTool()
tool_registry.register(image_score_tool, replace=True)


__all__ = [
    "ImageScoreArgs",
    "ScoreDimension",
    "ImageScoreResult",
    "ImageScoreTool",
    "image_score_tool",
]


async def _demo_main() -> None:
    """Real LLM smoke test for image scoring using the local example image."""

    example_image = Path(__file__).resolve().parent.parent / "examples" / "edit_example.png"
    with Image.open(example_image) as image:
        width, height = image.size

    artifacts: dict[str, Any] = {}
    registry = ArtifactRegistry(artifacts)
    input_artifact = registry.register(
        workflow_id="wf_demo",
        artifact_type=ArtifactType.IMAGE,
        value=str(example_image),
        metadata={
            "description": "Local example image for image-score testing",
            "coordinate_info": CoordinateInfo(
                root_artifact_id="artifact_demo_score_image",
                width=width,
                height=height,
                transform_kind="translation_only",
                offset_x=0,
                offset_y=0,
                parent_artifact_id=None,
            ).model_dump(),
        },
        artifact_id="artifact_demo_score_image",
    )
    ctx = ToolContext(
        llm=LLMClient(),
        artifact_registry=registry,
        workflow_id="wf_demo",
        node_id="node_score",
        attempt_id="attempt_1",
    )
    result = await image_score_tool(
        ctx,
        {
            "image_artifact_id": input_artifact.artifact_id,
            "expected_prompt": (
                "The image should clearly show a person holding a Python-themed book, "
                "with good visual clarity and strong semantic alignment."
            ),
        },
    )
    print("=== image_score demo ===")
    print(result.model_dump_json(indent=2))
    print("=== registered artifacts ===")
    for artifact in registry.list_all():
        print(artifact.model_dump_json(indent=2))


def main() -> None:
    asyncio.run(_demo_main())


if __name__ == "__main__":
    main()
