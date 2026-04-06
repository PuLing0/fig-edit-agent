"""Tool for rewriting and disambiguating prompts based on current artifact context."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
import re
from typing import Any

from pydantic import Field

from ..core import LLMClient
from ..schemas import Artifact, ArtifactType, Identifier, NonEmptyStr, StrictSchema
from .base import ArtifactRegistry, BaseTool, ToolContext, ToolResult, to_model_image_url
from .registry import tool_registry

logger = logging.getLogger(__name__)


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
    reference_rewrites: list["ReferenceRewrite"] = Field(
        default_factory=list,
        description="Optional deterministic source-to-target reference rewrites applied before LLM reconstruction.",
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


class ReferenceRewrite(StrictSchema):
    """Deterministic source-to-target rewrite applied before the LLM step."""

    source: NonEmptyStr = Field(description="Original reference token or phrase.")
    target: NonEmptyStr = Field(description="Replacement token or phrase.")


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


def _infer_language(text: str) -> str:
    """Infer the dominant language in a lightweight, best-effort way."""

    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin_count = len(re.findall(r"[A-Za-z]", text))
    if cjk_count > 0 and latin_count == 0:
        return "zh"
    if latin_count > 0 and cjk_count == 0:
        return "en"
    if cjk_count > 0 and latin_count > 0:
        return "mixed"
    return "unknown"


def _translation_to_english_allowed(task_goal: str | None) -> bool:
    """Whether task_goal explicitly requests English output."""

    if not task_goal:
        return False
    lowered = task_goal.lower()
    return "english" in lowered or "英文" in task_goal or "英语" in task_goal


def _detect_prompt_mode(original_prompt: str, task_goal: str | None) -> str:
    """Roughly classify the rewrite scenario to specialize the system prompt."""

    text = f"{original_prompt}\n{task_goal or ''}".lower()
    if any(token in text for token in ("text", "caption", "label", "subtitle", "文字", "文本", "字样", "字体")):
        return "text_edit"
    if any(
        token in text
        for token in (
            "portrait",
            "face",
            "person",
            "woman",
            "man",
            "girl",
            "boy",
            "人物",
            "人像",
            "人脸",
            "脸",
            "头发",
        )
    ):
        return "portrait_edit"
    if any(token in text for token in ("style", "restore", "enhance", "colorize", "风格", "修复", "增强", "上色", "复古")):
        return "style_transfer"
    if any(token in text for token in ("replace", "swap", "change", "add", "remove", "替换", "换成", "添加", "删除")):
        return "replace"
    return "generic_edit"


def _reference_patterns(source: str) -> list[re.Pattern[str]]:
    """Create safe regex patterns for known image-reference formats.

    Uses explicit boundaries / lookaheads to avoid accidentally rewriting
    references such as 图1 inside 图10.
    """

    source = source.strip()
    patterns: list[re.Pattern[str]] = []

    ordinal_match = re.fullmatch(r"第\s*(\d+)\s*张图", source)
    simple_cn_match = re.fullmatch(r"图\s*(\d+)", source)
    en_match = re.fullmatch(r"(?i)(image|img)\s*(\d+)", source)

    if ordinal_match:
        index = ordinal_match.group(1)
        patterns.append(re.compile(rf"第\s*{index}\s*张图(?!\d)"))
        patterns.append(re.compile(rf"图\s*{index}(?!\d)"))
        return patterns

    if simple_cn_match:
        index = simple_cn_match.group(1)
        patterns.append(re.compile(rf"第\s*{index}\s*张图(?!\d)"))
        patterns.append(re.compile(rf"图\s*{index}(?!\d)"))
        return patterns

    if en_match:
        index = en_match.group(2)
        patterns.append(re.compile(rf"\b(?:image|img)\s*{index}\b", re.IGNORECASE))
        return patterns

    escaped = re.escape(source)
    patterns.append(re.compile(rf"(?<![\w\u4e00-\u9fff]){escaped}(?![\w\u4e00-\u9fff])"))
    return patterns


def _apply_reference_rewrites(
    text: str,
    rewrites: list[ReferenceRewrite],
) -> tuple[str, list[ReferenceRewrite]]:
    """Apply deterministic, boundary-safe reference rewrites."""

    updated = text
    applied: list[ReferenceRewrite] = []
    for index, rewrite in enumerate(rewrites):
        changed = False
        placeholder = f"__REF_REWRITE_PLACEHOLDER_{index}__"
        for pattern in _reference_patterns(rewrite.source):
            updated, count = pattern.subn(placeholder, updated)
            if count > 0:
                changed = True
        if changed:
            updated = updated.replace(placeholder, rewrite.target)
            applied.append(rewrite)
    return updated, applied


def _build_system_prompt(*, prompt_mode: str, allow_translation: bool) -> str:
    """Build a stronger, task-aware system prompt for reconstruction."""

    language_rule = (
        "Default to preserving the user's original language. "
        "You may translate into English only if the task goal explicitly requests English for a downstream model. "
        "If you translate, preserve the user's original intent exactly and keep traceability by mentioning the translation decision in notes."
        if allow_translation
        else "Preserve the user's original language. Do not translate."
    )

    mode_rules = {
        "replace": (
            "For object addition, removal, or replacement, preserve the requested edit type and fill in only the minimum "
            "necessary details such as category, color, approximate size, orientation, and placement."
        ),
        "text_edit": (
            'For text editing tasks, wrap explicit text content in English double quotes and make the text target, '
            "replacement, placement, and style unambiguous."
        ),
        "portrait_edit": (
            "For portrait or person editing tasks, preserve identity-critical traits unless the user explicitly requests changes. "
            "Keep pose, camera angle, lighting, hairstyle, and facial characteristics stable when they are not part of the edit."
        ),
        "style_transfer": (
            "For style transfer, enhancement, or restoration tasks, rewrite the target style using 3 to 5 concrete visual cues "
            "such as palette, lighting, texture, era, or medium rather than vague adjectives."
        ),
        "generic_edit": (
            "Rewrite the prompt to be clear, executable, and minimally sufficient for downstream image editing."
        ),
    }

    return (
        "You rewrite prompts for multimodal and image-editing workflows.\n\n"
        "Core rules:\n"
        "1. Preserve the user's intent. Do not invent new edits, objects, or constraints.\n"
        f"2. {language_rule}\n"
        "3. Resolve ambiguous image references and use the already-rewritten references if they are provided.\n"
        "4. Produce one clear, executable prompt for the downstream image-editing model.\n"
        "5. Keep the output concise but sufficiently specific; do not add explanations outside the structured fields.\n\n"
        f"Task-specific rewrite rule:\n{mode_rules[prompt_mode]}"
    )


class PromptReconstructTool(BaseTool[PromptReconstructArgs]):
    name = "prompt_reconstruct"
    description = "Rewrite and disambiguate prompts using current artifact context."
    args_model = PromptReconstructArgs

    async def run(self, ctx: ToolContext, args: PromptReconstructArgs) -> ToolResult:
        related_artifacts = [ctx.artifact_registry.get(artifact_id) for artifact_id in args.related_artifact_ids]
        summaries = [_artifact_summary(artifact) for artifact in related_artifacts]
        preprocessed_prompt, applied_rewrites = _apply_reference_rewrites(args.original_prompt, args.reference_rewrites)
        prompt_mode = _detect_prompt_mode(preprocessed_prompt, args.task_goal)
        source_language = _infer_language(args.original_prompt)
        allow_translation = _translation_to_english_allowed(args.task_goal)

        user_text = [
            f"Original prompt:\n{args.original_prompt}",
            f"Preprocessed prompt:\n{preprocessed_prompt}",
            f"Detected source language:\n{source_language}",
            f"Rewrite mode:\n{prompt_mode}",
        ]
        if args.task_goal:
            user_text.append(f"Task goal:\n{args.task_goal}")
        if applied_rewrites:
            user_text.append(
                "Applied deterministic reference rewrites:\n- "
                + "\n- ".join(f"{rewrite.source} -> {rewrite.target}" for rewrite in applied_rewrites)
            )
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
                "content": _build_system_prompt(prompt_mode=prompt_mode, allow_translation=allow_translation),
            },
            {"role": "user", "content": content},
        ]
        try:
            result = await ctx.llm.generate_structured(
                messages=messages,
                response_model=PromptReconstructResult,
                task_name=self.name,
            )
        except Exception:
            fallback_note = "fallback_to_preprocessed_prompt_due_to_llm_failure"
            logger.exception(
                "Prompt reconstruction failed; falling back to preprocessed prompt.",
                extra={
                    "workflow_id": ctx.workflow_id,
                    "node_id": ctx.node_id,
                    "attempt_id": ctx.attempt_id,
                    "prompt_mode": prompt_mode,
                },
            )
            result = PromptReconstructResult(
                refined_prompt=preprocessed_prompt,
                notes=[fallback_note],
                resolved_references=[f"{rewrite.source} -> {rewrite.target}" for rewrite in applied_rewrites],
            )

        notes = list(result.notes)
        resolved_references = list(result.resolved_references)
        for rewrite in applied_rewrites:
            rewrite_note = f"{rewrite.source} -> {rewrite.target}"
            if rewrite_note not in resolved_references:
                resolved_references.append(rewrite_note)
        if applied_rewrites:
            deterministic_note = "applied_deterministic_reference_rewrites_before_llm"
            if deterministic_note not in notes:
                notes.append(deterministic_note)

        output_language = _infer_language(result.refined_prompt)
        translation_applied = (
            source_language not in {"unknown"}
            and output_language not in {"unknown"}
            and source_language != output_language
        )
        if translation_applied:
            translation_note = f"translation_applied:{source_language}_to_{output_language}"
            if translation_note not in notes:
                notes.append(translation_note)

        prompt_artifact = ctx.register_artifact(
            artifact_type=ArtifactType.PROMPT,
            value=result.refined_prompt,
            metadata={
                "notes": notes,
                "resolved_references": resolved_references,
                "source_artifact_ids": [artifact.artifact_id for artifact in related_artifacts],
                "original_prompt": args.original_prompt,
                "preprocessed_prompt": preprocessed_prompt,
                "applied_reference_rewrites": [rewrite.model_dump() for rewrite in applied_rewrites],
                "task_goal": args.task_goal,
                "rewrite_mode": prompt_mode,
                "source_language": source_language,
                "output_language": output_language,
                "translation_applied": translation_applied,
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
    "ReferenceRewrite",
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
