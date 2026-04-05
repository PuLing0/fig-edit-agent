"""Tool for multimodal image understanding."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from pydantic import Field

from ..core import LLMClient
from ..schemas import ArtifactType, Identifier, NonEmptyStr, StrictSchema
from .base import ArtifactRegistry, BaseTool, ToolContext, ToolResult, to_model_image_url
from .registry import tool_registry


class ImageUnderstandArgs(StrictSchema):
    """Arguments for the image understanding tool."""

    image_artifact_id: Identifier = Field(description="Artifact id of the image to understand.")
    prompt: str | None = Field(
        default=None,
        description="Optional focused question or instruction for the understanding model.",
    )
    output_spec_name: NonEmptyStr = Field(
        default="analysis",
        description="Output slot name that should receive the produced analysis artifact.",
    )
    output_role: NonEmptyStr = Field(
        default="analysis",
        description="Semantic output role for the analysis artifact.",
    )


class ImageUnderstandResult(StrictSchema):
    """Structured output returned by the understanding model."""

    description: NonEmptyStr = Field(description="Natural-language description of the image content.")
    labels: list[NonEmptyStr] = Field(
        default_factory=list,
        description="Short semantic labels extracted from the image.",
    )
    attributes: list["ImageAttribute"] = Field(
        default_factory=list,
        description="Additional visual attributes such as style, lighting, or composition.",
    )


class ImageAttribute(StrictSchema):
    """One structured image attribute."""

    name: NonEmptyStr = Field(description="Attribute name such as lighting or style.")
    value: NonEmptyStr = Field(description="Attribute value expressed as short text.")


class ImageUnderstandTool(BaseTool[ImageUnderstandArgs]):
    name = "image_understand"
    description = "Understand an image and produce structured semantic analysis."
    args_model = ImageUnderstandArgs

    async def run(self, ctx: ToolContext, args: ImageUnderstandArgs) -> ToolResult:
        image_artifact = ctx.artifact_registry.require_type(args.image_artifact_id, ArtifactType.IMAGE)
        prompt = args.prompt or "Describe the image in detail and return concise semantic labels and attributes."
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are an image understanding assistant. Analyze the provided image and "
                    "return a precise, compact semantic summary."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": to_model_image_url(image_artifact.value)}},
                ],
            },
        ]
        result = await ctx.llm.generate_structured(
            messages=messages,
            response_model=ImageUnderstandResult,
            task_name=self.name,
        )
        analysis_artifact = ctx.register_artifact(
            artifact_type=ArtifactType.ANALYSIS,
            value={
                "description": result.description,
                "labels": list(result.labels),
                "attributes": {item.name: item.value for item in result.attributes},
            },
            metadata={
                "description": result.description,
                "labels": list(result.labels),
                "attributes": {item.name: item.value for item in result.attributes},
                "source_artifact_id": image_artifact.artifact_id,
            },
        )
        output = ctx.bind_output(
            spec_name=args.output_spec_name,
            role=args.output_role,
            artifact=analysis_artifact,
        )
        return ToolResult(
            outputs=[output],
            summary=f"Generated semantic analysis for image artifact '{image_artifact.artifact_id}'.",
        )


image_understand_tool = ImageUnderstandTool()
tool_registry.register(image_understand_tool, replace=True)


__all__ = [
    "ImageUnderstandArgs",
    "ImageAttribute",
    "ImageUnderstandResult",
    "ImageUnderstandTool",
    "image_understand_tool",
]


async def _demo_main() -> None:
    """Real LLM smoke test for this tool using the local example image."""

    example_image = Path(__file__).resolve().parent.parent / "examples" / "edit_example.png"
    artifacts: dict[str, Any] = {}
    registry = ArtifactRegistry(artifacts)
    input_artifact = registry.register(
        workflow_id="wf_demo",
        artifact_type=ArtifactType.IMAGE,
        value=str(example_image),
        metadata={"description": "Local example image for real image-understand testing"},
        producer_node_id="input_loader",
        producer_attempt_id="attempt_0",
        artifact_id="artifact_demo_image",
    )
    ctx = ToolContext(
        llm=LLMClient(),
        artifact_registry=registry,
        workflow_id="wf_demo",
        node_id="node_understand",
        attempt_id="attempt_1",
    )
    result = await image_understand_tool(
        ctx,
        {
            "image_artifact_id": input_artifact.artifact_id,
            "prompt": "Describe the main content of the image.",
        },
    )
    print("=== image_understand demo ===")
    print(result.model_dump_json(indent=2))
    print("=== registered artifacts ===")
    for artifact in registry.list_all():
        print(artifact.model_dump_json(indent=2))


def main() -> None:
    asyncio.run(_demo_main())


if __name__ == "__main__":
    main()
