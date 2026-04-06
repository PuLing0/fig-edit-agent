"""Tool for rubric-based scoring of an image against an expected task objective."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Literal

from PIL import Image
from pydantic import Field, computed_field

from ..core import LLMClient
from ..schemas import ArtifactType, CoordinateInfo, Identifier, NonEmptyStr, StrictSchema
from .base import ArtifactRegistry, BaseTool, ToolContext, ToolResult, to_model_image_url
from .registry import tool_registry


ScoreValue = Literal[0, 1, 2]


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


class _RubricScoreResponse(StrictSchema):
    """LLM-facing rubric score response without computed overall_score."""

    prompt_alignment: ScoreValue = Field(description="0, 1, or 2 based on prompt alignment.")
    visual_quality: ScoreValue = Field(description="0, 1, or 2 based on visual quality.")
    text_accuracy: ScoreValue = Field(description="0, 1, or 2 based on text accuracy.")
    composition_aesthetics: ScoreValue = Field(description="0, 1, or 2 based on composition and aesthetics.")
    subject_consistency: ScoreValue = Field(description="0, 1, or 2 based on subject consistency.")
    summary: NonEmptyStr = Field(description="Short explanation of deductions and reasoning.")


class ImageScoreResult(StrictSchema):
    """Final structured image score with a computed overall score."""

    prompt_alignment: ScoreValue = Field(description="0, 1, or 2 based on prompt alignment.")
    visual_quality: ScoreValue = Field(description="0, 1, or 2 based on visual quality.")
    text_accuracy: ScoreValue = Field(description="0, 1, or 2 based on text accuracy.")
    composition_aesthetics: ScoreValue = Field(description="0, 1, or 2 based on composition and aesthetics.")
    subject_consistency: ScoreValue = Field(description="0, 1, or 2 based on subject consistency.")
    summary: NonEmptyStr = Field(description="Short explanation of deductions and reasoning.")

    @computed_field(return_type=float)
    @property
    def overall_score(self) -> float:
        total = (
            self.prompt_alignment
            + self.visual_quality
            + self.text_accuracy
            + self.composition_aesthetics
            + self.subject_consistency
        )
        return total / 10.0


class ImageScoreTool(BaseTool[ImageScoreArgs]):
    """LLM-driven rubric scoring tool for image outputs."""

    name = "image_score"
    description = "Score an image against an expected task objective using a fixed five-dimension rubric."
    args_model = ImageScoreArgs

    async def run(self, ctx: ToolContext, args: ImageScoreArgs) -> ToolResult:
        image_artifact = ctx.artifact_registry.require_type(args.image_artifact_id, ArtifactType.IMAGE)
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a strict image evaluation assistant. "
                    "You must score the image using exactly five fixed rubric dimensions, and each dimension must be an integer: 0, 1, or 2. "
                    "Do not invent new dimensions. Do not output floating-point sub-scores. "
                    "Do not output overall_score; the program will compute it.\n\n"
                    "Rubric:\n"
                    "1) prompt_alignment\n"
                    "- 0: The image clearly fails to satisfy the expected task objective.\n"
                    "- 1: The image partially satisfies the task objective but has important deviations.\n"
                    "- 2: The image strongly matches the expected task objective.\n\n"
                    "2) visual_quality\n"
                    "- 0: Severe blur, artifacts, distortion, or obvious visual failure.\n"
                    "- 1: Usable but with noticeable visual issues.\n"
                    "- 2: Clear, natural, and visually strong.\n\n"
                    "3) text_accuracy\n"
                    "- 0: The task involves text and the text is missing, unreadable, or seriously incorrect.\n"
                    "- 1: The task involves text and the text is mostly right but has minor mistakes.\n"
                    "- 2: The text is fully correct, or the task does not involve any text.\n"
                    "IMPORTANT: If the expected prompt does NOT mention any text, immediately award 2 points for text_accuracy to avoid unfair penalization.\n\n"
                    "4) composition_aesthetics\n"
                    "- 0: Composition is awkward, unbalanced, or visually broken.\n"
                    "- 1: Composition is acceptable but not strong or fully natural.\n"
                    "- 2: Composition is balanced, natural, and aesthetically good.\n\n"
                    "5) subject_consistency\n"
                    "- 0: The main subject identity, structure, or key attributes drift badly.\n"
                    "- 1: The subject is mostly consistent but has noticeable deviations.\n"
                    "- 2: The subject remains highly consistent.\n\n"
                    "In summary, briefly explain which dimensions lost points and why."
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
        llm_result = await ctx.llm.generate_structured(
            messages=messages,
            response_model=_RubricScoreResponse,
            task_name=self.name,
        )
        result = ImageScoreResult.model_validate(llm_result.model_dump())
        score_artifact = ctx.register_artifact(
            artifact_type=ArtifactType.SCORE,
            value={
                "overall_score": result.overall_score,
                "prompt_alignment": result.prompt_alignment,
                "visual_quality": result.visual_quality,
                "text_accuracy": result.text_accuracy,
                "composition_aesthetics": result.composition_aesthetics,
                "subject_consistency": result.subject_consistency,
                "summary": result.summary,
            },
            metadata={
                "description": "Structured image evaluation score",
                "source_artifact_id": image_artifact.artifact_id,
                "expected_prompt": args.expected_prompt,
                "scoring_template": "five_dimension_ten_point_rubric_v1",
                "overall_score": result.overall_score,
                "prompt_alignment": result.prompt_alignment,
                "visual_quality": result.visual_quality,
                "text_accuracy": result.text_accuracy,
                "composition_aesthetics": result.composition_aesthetics,
                "subject_consistency": result.subject_consistency,
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
