"""Tool for deterministic crop and alpha-cutout composition."""

from __future__ import annotations

import asyncio
from enum import Enum
from pathlib import Path
from uuid import uuid4

from PIL import Image
from pydantic import Field, model_validator

from ..core import CoordinateManager, LLMClient
from ..schemas import Artifact, ArtifactType, BoundingBox, CoordinateInfo, Identifier, NonEmptyStr, Point2D, StrictSchema
from .base import ArtifactRegistry, BaseTool, ToolContext, ToolResult
from .registry import tool_registry


class CropMode(str, Enum):
    RECT = "rect"
    ALPHA_CUTOUT = "alpha_cutout"


class ImageCropArgs(StrictSchema):
    base_image_artifact_id: Identifier = Field(description="Artifact id of the source image to crop.")
    points_artifact_id: Identifier | None = Field(
        default=None,
        description="Optional POINTS artifact id containing bbox information.",
    )
    mask_artifact_id: Identifier | None = Field(
        default=None,
        description="Optional MASK artifact id used for alpha cutout or bbox inference.",
    )
    padding: int = Field(default=0, ge=0, description="Optional pixel padding added around the crop.")
    crop_mode: CropMode = Field(default=CropMode.RECT, description="Rectangular crop or alpha cutout.")
    output_spec_name: NonEmptyStr = Field(default="cropped_image", description="Output slot name for the cropped image artifact.")
    output_role: NonEmptyStr = Field(default="cropped_image", description="Semantic output role for the cropped image artifact.")

    @model_validator(mode="after")
    def validate_sources(self) -> "ImageCropArgs":
        if self.points_artifact_id is None and self.mask_artifact_id is None:
            raise ValueError("Either points_artifact_id or mask_artifact_id must be provided")
        if self.crop_mode == CropMode.ALPHA_CUTOUT and self.mask_artifact_id is None:
            raise ValueError("mask_artifact_id is required when crop_mode == ALPHA_CUTOUT")
        return self


class ImageCropTool(BaseTool[ImageCropArgs]):
    name = "image_crop"
    description = "Crop an image using bbox or mask, and optionally apply alpha cutout."
    args_model = ImageCropArgs

    async def run(self, ctx: ToolContext, args: ImageCropArgs) -> ToolResult:
        image_artifact = ctx.artifact_registry.require_type(args.base_image_artifact_id, ArtifactType.IMAGE)
        image_coord = ctx.artifact_registry.require_coordinate_info(args.base_image_artifact_id)
        image_path = Path(image_artifact.value).expanduser().resolve()
        image = Image.open(image_path).convert("RGBA")

        crop_bbox, mask = self._resolve_geometry(ctx, args, image_coord=image_coord)
        left = max(0, crop_bbox.left - args.padding)
        top = max(0, crop_bbox.top - args.padding)
        right = min(image.width, crop_bbox.right + args.padding)
        bottom = min(image.height, crop_bbox.bottom + args.padding)
        final_bbox = BoundingBox(left=left, top=top, right=right, bottom=bottom)

        cropped = image.crop((left, top, right, bottom))
        if args.crop_mode == CropMode.ALPHA_CUTOUT and mask is not None:
            local_mask = mask.crop((left, top, right, bottom)).convert("L")
            cropped.putalpha(local_mask)

        output_path = self._write_crop(cropped, attempt_id=ctx.attempt_id)
        coord_info = CoordinateManager.derive_crop_coordinate_info(
            parent_coord=image_coord,
            crop_bbox_in_parent=final_bbox,
            parent_artifact_id=image_artifact.artifact_id,
        )
        cropped_artifact = ctx.register_artifact(
            artifact_type=ArtifactType.IMAGE,
            value=str(output_path),
            metadata={
                "description": "Cropped image region",
                "source_artifact_id": image_artifact.artifact_id,
                "crop_bbox_in_parent": final_bbox.model_dump(),
                "crop_mode": args.crop_mode.value,
                "coordinate_info": coord_info.model_dump(),
            },
        )
        output = ctx.bind_output(spec_name=args.output_spec_name, role=args.output_role, artifact=cropped_artifact)
        return ToolResult(
            outputs=[output],
            summary=f"Cropped image artifact '{image_artifact.artifact_id}' to bbox ({left}, {top}, {right}, {bottom}).",
        )

    @staticmethod
    def _resolve_geometry(ctx: ToolContext, args: ImageCropArgs, *, image_coord: CoordinateInfo) -> tuple[BoundingBox, Image.Image | None]:
        if args.points_artifact_id is not None:
            points_artifact = ctx.artifact_registry.require_type(args.points_artifact_id, ArtifactType.POINTS)
            points_coord = ctx.artifact_registry.require_coordinate_info(args.points_artifact_id)
            payload = points_artifact.value
            if not isinstance(payload, dict):
                raise ValueError("POINTS artifact must contain a dictionary payload")
            if isinstance(payload.get("bbox"), dict):
                source_bbox = BoundingBox.model_validate(payload["bbox"])
                target_bbox = CoordinateManager.sync_bbox_between_spaces(source_bbox, points_coord, image_coord)
                clamped_bbox = CoordinateManager.clamp_bbox(target_bbox, width=image_coord.width, height=image_coord.height)
            elif isinstance(payload.get("positive_points"), list) and payload["positive_points"]:
                source_points = [Point2D.model_validate(item) for item in payload["positive_points"]]
                target_points = CoordinateManager.sync_points_between_spaces(source_points, points_coord, image_coord)
                xs = [point.x for point in target_points]
                ys = [point.y for point in target_points]
                clamped_bbox = CoordinateManager.clamp_bbox(
                    BoundingBox(
                        left=min(xs),
                        top=min(ys),
                        right=max(xs) + 1,
                        bottom=max(ys) + 1,
                    ),
                    width=image_coord.width,
                    height=image_coord.height,
                )
            else:
                raise ValueError("POINTS artifact must contain either a bbox or positive_points")
        else:
            clamped_bbox = None

        if args.mask_artifact_id is not None:
            mask_artifact = ctx.artifact_registry.require_type(args.mask_artifact_id, ArtifactType.MASK)
            mask_coord = ctx.artifact_registry.require_coordinate_info(args.mask_artifact_id)
            if mask_coord.root_artifact_id != image_coord.root_artifact_id:
                raise ValueError("Mask and image must share the same root coordinate space")
            mask = Image.open(Path(mask_artifact.value).expanduser().resolve()).convert("L")
        else:
            mask = None

        if clamped_bbox is None:
            if mask is None:
                raise ValueError("Unable to resolve crop bbox")
            bbox_tuple = mask.getbbox()
            if bbox_tuple is None:
                raise ValueError("Mask artifact produced an empty bounding box")
            clamped_bbox = CoordinateManager.clamp_bbox(
                BoundingBox(left=bbox_tuple[0], top=bbox_tuple[1], right=bbox_tuple[2], bottom=bbox_tuple[3]),
                width=image_coord.width,
                height=image_coord.height,
            )

        return clamped_bbox, mask

    @staticmethod
    def _write_crop(image: Image.Image, *, attempt_id: str | None) -> Path:
        generated_dir = Path(__file__).resolve().parent.parent / "generated" / "image_crop"
        generated_dir.mkdir(parents=True, exist_ok=True)
        output_path = generated_dir / f"{attempt_id or uuid4().hex}_crop.png"
        image.save(output_path)
        return output_path


image_crop_tool = ImageCropTool()
tool_registry.register(image_crop_tool, replace=True)


async def _demo_main() -> None:
    from .image_grounding import image_grounding_tool
    from .image_segment import image_segment_tool

    example_image = Path(__file__).resolve().parent.parent / "examples" / "edit_example.png"
    with Image.open(example_image) as image:
        width, height = image.size

    artifacts: dict[str, Artifact] = {}
    registry = ArtifactRegistry(artifacts)
    image_artifact = registry.register(
        workflow_id="wf_demo",
        artifact_type=ArtifactType.IMAGE,
        value=str(example_image),
        metadata={
            "description": "Local example image for crop testing",
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
        node_id="node_crop",
        attempt_id="attempt_1",
    )
    grounding = await image_grounding_tool(
        ctx,
        {
            "base_image_artifact_id": image_artifact.artifact_id,
            "prompt": (
                "From the viewer's perspective, find the watch-wearing hand on the left side of the image "
                "that is holding the Python book. Return 1-3 positive anchor points clearly inside that hand, "
                "and optionally one negative point on the book or forearm if helpful."
            ),
        },
    )
    segment = await image_segment_tool(
        ctx,
        {
            "base_image_artifact_id": image_artifact.artifact_id,
            "points_artifact_id": grounding.outputs[0].artifact_id,
        },
    )
    result = await image_crop_tool(
        ctx,
        {
            "base_image_artifact_id": image_artifact.artifact_id,
            "mask_artifact_id": segment.outputs[0].artifact_id,
            "padding": 6,
            "crop_mode": "alpha_cutout",
        },
    )
    print("=== image_crop demo ===")
    print(result.model_dump_json(indent=2))
    print("=== registered artifacts ===")
    for artifact in registry.list_all():
        print(artifact.model_dump_json(indent=2))


def main() -> None:
    asyncio.run(_demo_main())


if __name__ == "__main__":
    main()


__all__ = ["CropMode", "ImageCropArgs", "ImageCropTool", "image_crop_tool"]
