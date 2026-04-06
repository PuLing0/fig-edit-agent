"""Tool for extracting readable text from an image."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
from pydantic import Field, model_validator

from ..core import LLMClient
from ..schemas import ArtifactType, CoordinateInfo, Identifier, NonEmptyStr, StrictSchema
from .base import ArtifactRegistry, BaseTool, ToolContext, ToolResult, to_model_image_url
from .registry import tool_registry


class ImageOCRArgs(StrictSchema):
    """Arguments for OCR extraction."""

    image_artifact_id: Identifier = Field(description="Artifact id of the image to run OCR on.")
    output_spec_name: NonEmptyStr = Field(
        default="ocr_result",
        description="Output slot name that should receive the OCR analysis artifact.",
    )
    output_role: NonEmptyStr = Field(
        default="ocr_text",
        description="Semantic output role for the OCR analysis artifact.",
    )


class OCRBlock(StrictSchema):
    """One readable text block with its image-space bounding box."""

    text: NonEmptyStr = Field(description="Readable text content for this block.")
    bbox: list[int] = Field(
        min_length=4,
        max_length=4,
        description="Bounding box in [left, top, right, bottom] pixel coordinates.",
    )

    @model_validator(mode="after")
    def validate_bbox(self) -> "OCRBlock":
        left, top, right, bottom = self.bbox
        if right <= left:
            raise ValueError("bbox right must be greater than left")
        if bottom <= top:
            raise ValueError("bbox bottom must be greater than top")
        return self


class OCRResult(StrictSchema):
    """Structured OCR result returned by the model."""

    full_text: str = Field(description="Full readable text in natural reading order.")
    blocks: list[OCRBlock] = Field(
        default_factory=list,
        description="Per-block OCR results with localized bounding boxes.",
    )


class ImageOCRTool(BaseTool[ImageOCRArgs]):
    """OCR tool powered by the current multimodal LLM client."""

    name = "image_ocr"
    description = "Extract readable text from an image and preserve block bounding boxes."
    args_model = ImageOCRArgs

    async def run(self, ctx: ToolContext, args: ImageOCRArgs) -> ToolResult:
        image_artifact = ctx.artifact_registry.require_type(args.image_artifact_id, ArtifactType.IMAGE)
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are an OCR assistant. Extract all readable text from the image. "
                    "Return full_text in natural reading order and return blocks with bounding boxes in "
                    "[left, top, right, bottom] absolute integer pixel coordinates based on the original image dimensions. "
                    "Do not return normalized coordinates, percentages, or floating-point values for bounding boxes. "
                    "If no readable text exists, return "
                    "full_text as an empty string and blocks as an empty list."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Extract all readable text from this image and localize each text block. "
                            "Use absolute integer pixel coordinates based on the original image dimensions."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": to_model_image_url(image_artifact.value)}},
                ],
            },
        ]
        result = await ctx.llm.generate_structured(
            messages=messages,
            response_model=OCRResult,
            task_name=self.name,
        )
        analysis_artifact = ctx.register_artifact(
            artifact_type=ArtifactType.ANALYSIS,
            value={
                "full_text": result.full_text,
                "blocks": [block.model_dump() for block in result.blocks],
            },
            metadata={
                "description": "OCR result extracted from image",
                "source_artifact_id": image_artifact.artifact_id,
                "block_count": len(result.blocks),
                "full_text": result.full_text,
            },
        )
        output = ctx.bind_output(
            spec_name=args.output_spec_name,
            role=args.output_role,
            artifact=analysis_artifact,
        )
        return ToolResult(
            outputs=[output],
            summary=f"Extracted OCR result for image artifact '{image_artifact.artifact_id}'.",
        )


image_ocr_tool = ImageOCRTool()
tool_registry.register(image_ocr_tool, replace=True)


__all__ = [
    "ImageOCRArgs",
    "OCRBlock",
    "OCRResult",
    "ImageOCRTool",
    "image_ocr_tool",
]


def _make_demo_image() -> Path:
    generated_dir = Path(__file__).resolve().parent.parent / "generated" / "image_ocr"
    generated_dir.mkdir(parents=True, exist_ok=True)
    output_path = generated_dir / "ocr_demo_input.png"
    image = Image.new("RGB", (720, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 50), "Advanced Python", fill="black")
    draw.text((40, 110), "Version 3.1", fill="black")
    draw.text((40, 170), "Open the book", fill="black")
    image.save(output_path)
    return output_path


async def _demo_main() -> None:
    """Real LLM smoke test for OCR using a generated local text image."""

    example_image = _make_demo_image()
    with Image.open(example_image) as image:
        width, height = image.size

    artifacts: dict[str, Any] = {}
    registry = ArtifactRegistry(artifacts)
    input_artifact = registry.register(
        workflow_id="wf_demo",
        artifact_type=ArtifactType.IMAGE,
        value=str(example_image),
        metadata={
            "description": "Generated OCR demo image with several short text lines",
            "coordinate_info": CoordinateInfo(
                root_artifact_id="artifact_demo_ocr_image",
                width=width,
                height=height,
                transform_kind="translation_only",
                offset_x=0,
                offset_y=0,
                parent_artifact_id=None,
            ).model_dump(),
        },
        artifact_id="artifact_demo_ocr_image",
    )
    ctx = ToolContext(
        llm=LLMClient(),
        artifact_registry=registry,
        workflow_id="wf_demo",
        node_id="node_ocr",
        attempt_id="attempt_1",
    )
    result = await image_ocr_tool(
        ctx,
        {
            "image_artifact_id": input_artifact.artifact_id,
        },
    )
    print("=== image_ocr demo ===")
    print(result.model_dump_json(indent=2))
    print("=== registered artifacts ===")
    for artifact in registry.list_all():
        print(artifact.model_dump_json(indent=2))


def main() -> None:
    asyncio.run(_demo_main())


if __name__ == "__main__":
    main()
