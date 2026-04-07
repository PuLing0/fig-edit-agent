"""Tool for running real generative image editing through the FireRed backend."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image
from pydantic import Field, model_validator

from ..core import FireRedBackendError, LLMClient, backend_config_snapshot, edit_images
from ..schemas import Artifact, ArtifactType, CoordinateInfo, Identifier, NonEmptyStr, StrictSchema
from .base import ArtifactRegistry, BaseTool, ToolContext, ToolResult
from .registry import tool_registry


class ImageEditArgs(StrictSchema):
    """Arguments for the generative image edit tool."""

    base_image_artifact_id: Identifier = Field(description="Primary base image to edit.")
    prompt_artifact_id: Identifier | None = Field(
        default=None,
        description="Optional PROMPT/TEXT artifact providing the main edit instruction.",
    )
    prompt_text: str | None = Field(
        default=None,
        description="Optional direct prompt text or additional guidance.",
    )
    mask_artifact_id: Identifier | None = Field(
        default=None,
        description="Optional mask artifact. Kept in the interface for future support but not consumed by FireRed v1.",
    )
    reference_image_ids: list[Identifier] = Field(
        default_factory=list,
        description="Optional reference images provided after the base image.",
    )
    layout_reference_id: Identifier | None = Field(
        default=None,
        description="Optional layout reference image appended after reference images.",
    )
    output_spec_name: NonEmptyStr = Field(
        default="edited_image",
        description="Output slot name for the edited image artifact.",
    )
    output_role: NonEmptyStr = Field(
        default="edited_image",
        description="Semantic output role for the edited image artifact.",
    )

    @model_validator(mode="after")
    def validate_inputs(self) -> "ImageEditArgs":
        prompt_text_present = self.prompt_text is not None and self.prompt_text.strip() != ""
        if self.prompt_artifact_id is None and not prompt_text_present:
            raise ValueError("prompt_artifact_id and prompt_text cannot both be empty")
        if len(set(self.reference_image_ids)) != len(self.reference_image_ids):
            raise ValueError("reference_image_ids must not contain duplicates")
        if self.base_image_artifact_id in self.reference_image_ids:
            raise ValueError("base_image_artifact_id must not also appear in reference_image_ids")
        if self.layout_reference_id is not None:
            if self.layout_reference_id == self.base_image_artifact_id:
                raise ValueError("layout_reference_id must differ from base_image_artifact_id")
            if self.layout_reference_id in self.reference_image_ids:
                raise ValueError("layout_reference_id must not also appear in reference_image_ids")
        return self


class ImageEditTool(BaseTool[ImageEditArgs]):
    """FireRed-backed generative image edit tool."""

    name = "image_edit"
    description = "Run generative image editing with the local FireRed backend."
    args_model = ImageEditArgs

    async def run(self, ctx: ToolContext, args: ImageEditArgs) -> ToolResult:
        base_artifact = ctx.artifact_registry.require_type(args.base_image_artifact_id, ArtifactType.IMAGE)
        base_coord = ctx.artifact_registry.require_coordinate_info(args.base_image_artifact_id)

        prompt_artifact = self._resolve_prompt_artifact(ctx, args.prompt_artifact_id) if args.prompt_artifact_id else None
        mask_artifact = (
            ctx.artifact_registry.require_type(args.mask_artifact_id, ArtifactType.MASK)
            if args.mask_artifact_id is not None
            else None
        )
        reference_artifacts = [
            ctx.artifact_registry.require_type(artifact_id, ArtifactType.IMAGE)
            for artifact_id in args.reference_image_ids
        ]
        layout_artifact = (
            ctx.artifact_registry.require_type(args.layout_reference_id, ArtifactType.IMAGE)
            if args.layout_reference_id is not None
            else None
        )

        effective_prompt = self._build_effective_prompt(
            prompt_artifact=prompt_artifact,
            prompt_text=args.prompt_text,
            mask_provided=mask_artifact is not None,
        )
        image_inputs = [base_artifact, *reference_artifacts]
        if layout_artifact is not None:
            image_inputs.append(layout_artifact)
        pil_images = self._load_images(image_inputs)

        try:
            edited_image = await asyncio.to_thread(edit_images, images=pil_images, prompt=effective_prompt)
        except FireRedBackendError as exc:
            raise RuntimeError(
                "image_edit could not run the FireRed backend. "
                "Check FireRed inference mode, model configuration, and runtime dependencies."
            ) from exc

        output_path = self._write_output(edited_image, attempt_id=ctx.attempt_id)
        output_coord = CoordinateInfo(
            root_artifact_id=base_coord.root_artifact_id,
            width=edited_image.width,
            height=edited_image.height,
            transform_kind="translation_only",
            offset_x=0,
            offset_y=0,
            parent_artifact_id=base_artifact.artifact_id,
        )
        edited_artifact = ctx.register_artifact(
            artifact_type=ArtifactType.IMAGE,
            value=str(output_path),
            metadata={
                "description": "Edited image generated by FireRed backend",
                "source_base_image_id": base_artifact.artifact_id,
                "source_prompt_artifact_id": prompt_artifact.artifact_id if prompt_artifact else None,
                "source_reference_image_ids": [artifact.artifact_id for artifact in reference_artifacts],
                "source_layout_reference_id": layout_artifact.artifact_id if layout_artifact else None,
                "source_mask_artifact_id": mask_artifact.artifact_id if mask_artifact else None,
                "mask_provided_but_not_consumed": mask_artifact is not None,
                "effective_prompt": effective_prompt,
                "backend_name": "firered_qwen_image_edit_plus",
                "backend_config_snapshot": backend_config_snapshot(),
                "coordinate_info": output_coord.model_dump(),
            },
        )
        output = ctx.bind_output(
            spec_name=args.output_spec_name,
            role=args.output_role,
            artifact=edited_artifact,
        )
        return ToolResult(
            outputs=[output],
            summary=(
                f"Generated one edited image from base artifact '{base_artifact.artifact_id}' "
                f"using FireRed with {len(reference_artifacts)} reference image(s)"
                f"{' and one layout reference' if layout_artifact else ''}."
            ),
        )

    @staticmethod
    def _resolve_prompt_artifact(ctx: ToolContext, prompt_artifact_id: str) -> Artifact:
        artifact = ctx.artifact_registry.get(prompt_artifact_id)
        if artifact.artifact_type not in {ArtifactType.PROMPT, ArtifactType.TEXT}:
            raise TypeError(
                f"Prompt artifact '{prompt_artifact_id}' must be of type prompt or text, "
                f"got {artifact.artifact_type.value}"
            )
        return artifact

    @staticmethod
    def _build_effective_prompt(
        *,
        prompt_artifact: Artifact | None,
        prompt_text: str | None,
        mask_provided: bool,
    ) -> str:
        parts: list[str] = []
        if prompt_artifact is not None:
            parts.append(str(prompt_artifact.value).strip())
        if prompt_text is not None and prompt_text.strip():
            if prompt_artifact is not None:
                parts.append(f"Additional guidance: {prompt_text.strip()}")
            else:
                parts.append(prompt_text.strip())
        if mask_provided:
            parts.append(
                "Primary edit focus is constrained by an external mask provided by the orchestration system."
            )
        effective_prompt = "\n\n".join(part for part in parts if part)
        if not effective_prompt:
            raise ValueError("Failed to construct a non-empty effective prompt for image_edit")
        return effective_prompt

    @staticmethod
    def _load_images(artifacts: list[Artifact]) -> list[Image.Image]:
        images: list[Image.Image] = []
        for artifact in artifacts:
            path = Path(str(artifact.value)).expanduser().resolve()
            with Image.open(path) as image:
                images.append(image.convert("RGB").copy())
        return images

    @staticmethod
    def _write_output(image: Image.Image, *, attempt_id: str | None) -> Path:
        generated_dir = Path(__file__).resolve().parent.parent / "generated" / "image_edit"
        generated_dir.mkdir(parents=True, exist_ok=True)
        output_path = generated_dir / f"{attempt_id or uuid4().hex}_edit.png"
        image.save(output_path)
        return output_path


image_edit_tool = ImageEditTool()
tool_registry.register(image_edit_tool, replace=True)


__all__ = [
    "ImageEditArgs",
    "ImageEditTool",
    "image_edit_tool",
]


async def _demo_main() -> None:
    """Real FireRed smoke test wiring using the local example image and a prompt artifact."""

    example_image = Path(__file__).resolve().parent.parent / "examples" / "edit_example.png"
    with Image.open(example_image) as image:
        width, height = image.size

    artifacts: dict[str, Any] = {}
    registry = ArtifactRegistry(artifacts)
    base_artifact = registry.register(
        workflow_id="wf_demo",
        artifact_type=ArtifactType.IMAGE,
        value=str(example_image),
        metadata={
            "description": "Base image for FireRed edit demo",
            "coordinate_info": CoordinateInfo(
                root_artifact_id="artifact_demo_edit_image",
                width=width,
                height=height,
                transform_kind="translation_only",
                offset_x=0,
                offset_y=0,
                parent_artifact_id=None,
            ).model_dump(),
        },
        artifact_id="artifact_demo_edit_image",
    )
    prompt_artifact = registry.register(
        workflow_id="wf_demo",
        artifact_type=ArtifactType.PROMPT,
        value="在书本封面 Python 的下方，添加一行英文文字 \"2nd Edition\"，并保持双手与视角不变。",
        metadata={"description": "Prompt artifact for FireRed edit demo"},
        artifact_id="artifact_demo_edit_prompt",
    )
    ctx = ToolContext(
        llm=LLMClient(),
        artifact_registry=registry,
        workflow_id="wf_demo",
        node_id="node_edit",
        attempt_id="attempt_1",
    )
    result = await image_edit_tool(
        ctx,
        {
            "base_image_artifact_id": base_artifact.artifact_id,
            "prompt_artifact_id": prompt_artifact.artifact_id,
        },
    )
    print("=== image_edit demo ===")
    print(result.model_dump_json(indent=2))
    print("=== registered artifacts ===")
    for artifact in registry.list_all():
        print(artifact.model_dump_json(indent=2))


def main() -> None:
    asyncio.run(_demo_main())


if __name__ == "__main__":
    main()
