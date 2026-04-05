"""Tool for rewriting and disambiguating prompts based on current artifact context."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from pydantic import Field

from ..core import LLMClient
from ..schemas import Artifact, ArtifactType, Identifier, NonEmptyStr, StrictSchema
from .base import ArtifactRegistry, BaseTool, ToolContext, ToolResult, to_model_image_url
from .registry import tool_registry


class PromptReconstructArgs(StrictSchema):
    """Arguments for prompt reconstruction."""

    original_prompt: NonEmptyStr = Field(description="Original prompt that needs refinement.")
    related_artifact_ids: list[Identifier] = Field(
        default_factory=list,
        description="Artifacts whose context should be considered during prompt reconstruction.",
    )
    task_goal: str | None = Field(
        default=None,
        description="Optional high-level task objective to steer the rewrite.",
    )
    output_spec_name: NonEmptyStr = Field(
        default="prompt",
        description="Output slot name that should receive the refined prompt artifact.",
    )
    output_role: NonEmptyStr = Field(
        default="prompt",
        description="Semantic output role for the refined prompt artifact.",
    )


class PromptReconstructResult(StrictSchema):
    """Structured output returned by the prompt reconstruction model."""

    refined_prompt: NonEmptyStr = Field(description="Refined prompt suitable for downstream image tools or models.")
    notes: list[NonEmptyStr] = Field(
        default_factory=list,
        description="Optional notes about rewrite choices and trade-offs.",
    )
    resolved_references: list[NonEmptyStr] = Field(
        default_factory=list,
        description="Explicitly resolved reference mappings or disambiguations.",
    )


def _artifact_summary(artifact: Artifact) -> str:
    metadata = artifact.metadata or {}
    description = metadata.get("description")
    labels = metadata.get("labels")
    pieces = [f"id={artifact.artifact_id}", f"type={artifact.artifact_type.value}"]
    if description:
        pieces.append(f"description={description}")
    if labels:
        pieces.append(f"labels={labels}")
    if artifact.artifact_type in {ArtifactType.TEXT, ArtifactType.PROMPT}:
        pieces.append(f"value={artifact.value}")
    elif artifact.artifact_type == ArtifactType.ANALYSIS and isinstance(artifact.value, dict):
        pieces.append(f"analysis_keys={list(artifact.value.keys())}")
    return "; ".join(pieces)


class PromptReconstructTool(BaseTool[PromptReconstructArgs]):
    name = "prompt_reconstruct"
    description = "Rewrite and disambiguate prompts using current artifact context."
    args_model = PromptReconstructArgs

    async def run(self, ctx: ToolContext, args: PromptReconstructArgs) -> ToolResult:
        related_artifacts = [ctx.artifact_registry.get(artifact_id) for artifact_id in args.related_artifact_ids]
        summaries = [_artifact_summary(artifact) for artifact in related_artifacts]

        user_text = [f"Original prompt:\n{args.original_prompt}"]
        if args.task_goal:
            user_text.append(f"Task goal:\n{args.task_goal}")
        if summaries:
            user_text.append("Related artifacts:\n- " + "\n- ".join(summaries))
        else:
            user_text.append("Related artifacts:\n- none")

        content: list[dict[str, Any]] = [{"type": "text", "text": "\n\n".join(user_text)}]
        for artifact in related_artifacts:
            if artifact.artifact_type == ArtifactType.IMAGE:
                content.append({"type": "image_url", "image_url": {"url": to_model_image_url(artifact.value)}})

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You rewrite prompts for multimodal and image-editing workflows. Resolve ambiguous references, "
                    "preserve user intent, and produce a concise prompt that downstream tools can follow."
                ),
            },
            {"role": "user", "content": content},
        ]
        result = await ctx.llm.generate_structured(
            messages=messages,
            response_model=PromptReconstructResult,
            task_name=self.name,
        )
        prompt_artifact = ctx.register_artifact(
            artifact_type=ArtifactType.PROMPT,
            value=result.refined_prompt,
            metadata={
                "notes": list(result.notes),
                "resolved_references": list(result.resolved_references),
                "source_artifact_ids": [artifact.artifact_id for artifact in related_artifacts],
                "original_prompt": args.original_prompt,
                "task_goal": args.task_goal,
            },
        )
        output = ctx.bind_output(
            spec_name=args.output_spec_name,
            role=args.output_role,
            artifact=prompt_artifact,
        )
        return ToolResult(
            outputs=[output],
            summary="Reconstructed a refined prompt with contextual disambiguation.",
        )


prompt_reconstruct_tool = PromptReconstructTool()
tool_registry.register(prompt_reconstruct_tool, replace=True)


__all__ = [
    "PromptReconstructArgs",
    "PromptReconstructResult",
    "PromptReconstructTool",
    "prompt_reconstruct_tool",
]


async def _demo_main() -> None:
    """Real LLM smoke test for this tool using the local example image and its analysis."""

    example_image = Path(__file__).resolve().parent.parent / "examples" / "edit_example.png"
    artifacts: dict[str, Any] = {}
    registry = ArtifactRegistry(artifacts)
    image_artifact = registry.register(
        workflow_id="wf_demo",
        artifact_type=ArtifactType.IMAGE,
        value=str(example_image),
        metadata={"description": "A person holding a Python book", "labels": ["person", "book", "python"]},
        artifact_id="artifact_book_image",
    )
    analysis_artifact = registry.register(
        workflow_id="wf_demo",
        artifact_type=ArtifactType.ANALYSIS,
        value={
            "description": "A person is holding a white Python programming book with a pink title area on the cover.",
            "labels": ["person", "hands", "book", "python", "cover"],
            "attributes": {"camera_angle": "top-down", "context": "indoor"},
        },
        metadata={"description": "Semantic analysis of the example image", "labels": ["person", "book", "python"]},
        artifact_id="artifact_book_analysis",
    )
    ctx = ToolContext(
        llm=LLMClient(),
        artifact_registry=registry,
        workflow_id="wf_demo",
        node_id="node_prompt_reconstruct",
        attempt_id="attempt_1",
    )
    result = await prompt_reconstruct_tool(
        ctx,
        {
            "original_prompt": (
                "把图片里的书换成一本封面写着 Advanced Python 的书，保持双手、姿势、拍摄角度和整体光线不变。"
            ),
            "related_artifact_ids": [image_artifact.artifact_id, analysis_artifact.artifact_id],
            "task_goal": "生成一个清晰、无歧义、适合图像编辑模型使用的英文编辑提示词。",
        },
    )
    print("=== prompt_reconstruct demo ===")
    print(result.model_dump_json(indent=2))
    print("=== registered artifacts ===")
    for artifact in registry.list_all():
        print(artifact.model_dump_json(indent=2))


def main() -> None:
    asyncio.run(_demo_main())


if __name__ == "__main__":
    main()
