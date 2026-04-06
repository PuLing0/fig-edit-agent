"""Tool for creating a hard collage layout reference image."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image
from pydantic import Field, model_validator

from ..core import CoordinateManager, LLMClient
from ..schemas import (
    Artifact,
    ArtifactType,
    CoordinateInfo,
    Identifier,
    NonEmptyStr,
    PlacementRecord,
    StrictSchema,
)
from .base import ArtifactRegistry, BaseTool, ToolContext, ToolResult, to_model_image_url
from .registry import tool_registry

MAX_CANVAS_WIDTH = 4096
MAX_CANVAS_HEIGHT = 4096
MAX_CANVAS_PIXELS = 16_777_216  # 4096 * 4096


class ImageCollageArgs(StrictSchema):
    """Arguments for collage layout generation."""

    image_artifact_ids: list[Identifier] = Field(
        min_length=1,
        description="Input image artifact ids to arrange into a collage.",
    )
    prompt: NonEmptyStr = Field(description="Natural-language description of the desired layout.")
    output_spec_name: NonEmptyStr = Field(
        default="collage_image",
        description="Output slot name for the collage image artifact.",
    )
    output_role: NonEmptyStr = Field(
        default="layout_reference",
        description="Semantic output role for the collage image artifact.",
    )

    @model_validator(mode="after")
    def validate_unique_inputs(self) -> "ImageCollageArgs":
        if len(set(self.image_artifact_ids)) != len(self.image_artifact_ids):
            raise ValueError("image_artifact_ids must not contain duplicates")
        return self


class CollageLayoutItem(StrictSchema):
    """One placed image inside the collage canvas."""

    artifact_id: Identifier
    x: int
    y: int
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    rotation: float = 0.0
    z_index: int = 0
    opacity: float = 1.0

    @model_validator(mode="after")
    def validate_item(self) -> "CollageLayoutItem":
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError("opacity must be within [0, 1]")
        return self


class CollageLayoutResult(StrictSchema):
    """Structured layout plan produced by the LLM."""

    canvas_width: int = Field(gt=0)
    canvas_height: int = Field(gt=0)
    items: list[CollageLayoutItem] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_items(self) -> "CollageLayoutResult":
        artifact_ids = [item.artifact_id for item in self.items]
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError("Each artifact_id may appear at most once in the collage layout")
        if self.canvas_width > MAX_CANVAS_WIDTH:
            raise ValueError(f"canvas_width must be <= {MAX_CANVAS_WIDTH}")
        if self.canvas_height > MAX_CANVAS_HEIGHT:
            raise ValueError(f"canvas_height must be <= {MAX_CANVAS_HEIGHT}")
        if self.canvas_width * self.canvas_height > MAX_CANVAS_PIXELS:
            raise ValueError(f"canvas pixel count must be <= {MAX_CANVAS_PIXELS}")
        return self


@dataclass(slots=True)
class PreparedLayer:
    """One rendered layer ready to be pasted on the canvas."""

    item: CollageLayoutItem
    image: Image.Image
    transform_to_canvas: list[float]


def _artifact_summary(artifact: Artifact) -> str:
    metadata = artifact.metadata or {}
    description = metadata.get("description")
    labels = metadata.get("labels")
    coord_info = metadata.get("coordinate_info") or {}
    pieces = [
        f"id={artifact.artifact_id}",
        f"type={artifact.artifact_type.value}",
        f"size={coord_info.get('width', '?')}x{coord_info.get('height', '?')}",
    ]
    if description:
        pieces.append(f"description={description}")
    if labels:
        pieces.append(f"labels={labels}")
    return "; ".join(pieces)


class ImageCollageTool(BaseTool[ImageCollageArgs]):
    name = "image_collage"
    description = "Create a hard collage reference image from multiple source images."
    args_model = ImageCollageArgs

    async def run(self, ctx: ToolContext, args: ImageCollageArgs) -> ToolResult:
        image_artifacts = [
            ctx.artifact_registry.require_type(artifact_id, ArtifactType.IMAGE)
            for artifact_id in args.image_artifact_ids
        ]
        layout = await self._plan_layout(ctx=ctx, args=args, image_artifacts=image_artifacts)
        self._validate_layout(layout=layout, input_ids=args.image_artifact_ids)

        canvas, placements = await self._render_collage(layout=layout, image_artifacts=image_artifacts)
        output_path = self._write_collage(canvas, attempt_id=ctx.attempt_id)

        collage_artifact_id = f"artifact_{uuid4().hex}"
        collage_coord = CoordinateInfo(
            root_artifact_id=collage_artifact_id,
            width=layout.canvas_width,
            height=layout.canvas_height,
            transform_kind="translation_only",
            offset_x=0,
            offset_y=0,
            parent_artifact_id=None,
        )
        collage_artifact = ctx.register_artifact(
            artifact_type=ArtifactType.IMAGE,
            value=str(output_path),
            metadata={
                "description": "Hard collage layout reference",
                "source_artifact_ids": [artifact.artifact_id for artifact in image_artifacts],
                "layout_prompt": args.prompt,
                "canvas_width": layout.canvas_width,
                "canvas_height": layout.canvas_height,
                "background": "transparent",
                "placements": [placement.model_dump() for placement in placements],
                "layout_items": [item.model_dump() for item in layout.items],
                "coordinate_info": collage_coord.model_dump(),
            },
            artifact_id=collage_artifact_id,
        )
        output = ctx.bind_output(
            spec_name=args.output_spec_name,
            role=args.output_role,
            artifact=collage_artifact,
        )
        return ToolResult(
            outputs=[output],
            summary=(
                f"Created collage reference image from {len(image_artifacts)} input artifact(s) "
                f"on a {layout.canvas_width}x{layout.canvas_height} transparent canvas."
            ),
        )

    async def _plan_layout(
        self,
        *,
        ctx: ToolContext,
        args: ImageCollageArgs,
        image_artifacts: list[Artifact],
    ) -> CollageLayoutResult:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Create a rough visual layout plan for a hard collage reference image.\n\n"
                    f"User layout prompt:\n{args.prompt}\n\n"
                    "Requirements:\n"
                    "- Use every provided artifact exactly once.\n"
                    "- Output only a CollageLayoutResult.\n"
                    "- canvas_width and canvas_height must be just large enough to contain the arranged items.\n"
                    f"- canvas_width must not exceed {MAX_CANVAS_WIDTH}.\n"
                    f"- canvas_height must not exceed {MAX_CANVAS_HEIGHT}.\n"
                    f"- canvas_width * canvas_height must not exceed {MAX_CANVAS_PIXELS}.\n"
                    "- x and y are the top-left coordinates of each transformed image on the final canvas.\n"
                    "- width and height are the resized dimensions before rotation.\n"
                    "- rotation is in degrees counterclockwise.\n"
                    "- z_index controls layer order; larger values appear above smaller ones.\n"
                    "- opacity must stay within [0,1].\n"
                    "- This is only a coarse reference layout, not a generative edit."
                ),
            }
        ]
        for artifact in image_artifacts:
            content.append({"type": "text", "text": _artifact_summary(artifact)})
            content.append({"type": "image_url", "image_url": {"url": to_model_image_url(artifact.value)}})

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a layout planner for image collages. "
                    "Given several source images and a layout instruction, produce a compact, valid layout plan. "
                    "Use every input image exactly once and choose canvas dimensions that fully contain the final composition "
                    f"without exceeding {MAX_CANVAS_WIDTH}x{MAX_CANVAS_HEIGHT} or {MAX_CANVAS_PIXELS} total pixels."
                ),
            },
            {"role": "user", "content": content},
        ]
        return await ctx.llm.generate_structured(
            messages=messages,
            response_model=CollageLayoutResult,
            task_name=self.name,
        )

    @staticmethod
    def _validate_layout(*, layout: CollageLayoutResult, input_ids: list[str]) -> None:
        input_id_set = set(input_ids)
        layout_ids = [item.artifact_id for item in layout.items]
        unknown_ids = [artifact_id for artifact_id in layout_ids if artifact_id not in input_id_set]
        if unknown_ids:
            raise ValueError(f"Collage layout referenced unknown artifact ids: {unknown_ids}")
        missing_ids = [artifact_id for artifact_id in input_ids if artifact_id not in set(layout_ids)]
        if missing_ids:
            raise ValueError(f"Collage layout omitted required artifact ids: {missing_ids}")
        if layout.canvas_width > MAX_CANVAS_WIDTH or layout.canvas_height > MAX_CANVAS_HEIGHT:
            raise ValueError(
                f"Collage canvas exceeds max size: {layout.canvas_width}x{layout.canvas_height} "
                f"(limit {MAX_CANVAS_WIDTH}x{MAX_CANVAS_HEIGHT})"
            )
        if layout.canvas_width * layout.canvas_height > MAX_CANVAS_PIXELS:
            raise ValueError(
                f"Collage canvas pixel count exceeds limit: "
                f"{layout.canvas_width * layout.canvas_height} > {MAX_CANVAS_PIXELS}"
            )

    async def _render_collage(
        self,
        *,
        layout: CollageLayoutResult,
        image_artifacts: list[Artifact],
    ) -> tuple[Image.Image, list[PlacementRecord]]:
        artifact_map = {artifact.artifact_id: artifact for artifact in image_artifacts}
        canvas = Image.new("RGBA", (layout.canvas_width, layout.canvas_height), (0, 0, 0, 0))

        indexed_items = list(enumerate(layout.items))
        indexed_items.sort(key=lambda pair: (pair[1].z_index, pair[0]))

        layers = await asyncio.gather(
            *[
                asyncio.to_thread(self._prepare_layer, item=item, artifact=artifact_map[item.artifact_id])
                for _, item in indexed_items
            ]
        )

        placements: list[PlacementRecord] = []
        for (_, item), layer in zip(indexed_items, layers, strict=True):
            self._assert_within_canvas(layer=layer, canvas_width=layout.canvas_width, canvas_height=layout.canvas_height)
            canvas.alpha_composite(layer.image, dest=(item.x, item.y))
            placements.append(
                PlacementRecord(
                    source_artifact_id=item.artifact_id,
                    transform_to_canvas=layer.transform_to_canvas,
                    z_index=item.z_index,
                    opacity=item.opacity,
                )
            )
        return canvas, placements

    def _prepare_layer(self, *, item: CollageLayoutItem, artifact: Artifact) -> PreparedLayer:
        source_path = Path(artifact.value).expanduser().resolve()
        with Image.open(source_path) as source_image:
            source_rgba = source_image.convert("RGBA")
            original_width, original_height = source_rgba.size
            # Keep the transform pipeline to exactly one resize and one rotate to
            # minimize cumulative resampling loss.
            resized = source_rgba.resize((item.width, item.height), Image.Resampling.LANCZOS)
            rotated = resized.rotate(item.rotation, expand=True, resample=Image.Resampling.BICUBIC)

        if item.opacity < 1.0:
            alpha = np.asarray(rotated.getchannel("A"), dtype=np.float32)
            alpha = np.clip(alpha * item.opacity, 0, 255).astype(np.uint8)
            rotated.putalpha(Image.fromarray(alpha, mode="L"))

        transform_to_canvas = self._build_transform_to_canvas(
            source_width=original_width,
            source_height=original_height,
            target_width=item.width,
            target_height=item.height,
            rotation_degrees=item.rotation,
            canvas_x=item.x,
            canvas_y=item.y,
        )
        return PreparedLayer(item=item, image=rotated, transform_to_canvas=transform_to_canvas)

    @staticmethod
    def _build_transform_to_canvas(
        *,
        source_width: int,
        source_height: int,
        target_width: int,
        target_height: int,
        rotation_degrees: float,
        canvas_x: int,
        canvas_y: int,
    ) -> list[float]:
        scale_x = target_width / float(source_width)
        scale_y = target_height / float(source_height)
        scale_matrix = np.array(
            [[scale_x, 0.0, 0.0], [0.0, scale_y, 0.0], [0.0, 0.0, 1.0]],
            dtype=float,
        )

        center_x = target_width / 2.0
        center_y = target_height / 2.0
        radians = np.deg2rad(rotation_degrees)
        cos_theta = float(np.cos(radians))
        sin_theta = float(np.sin(radians))
        rotate_about_origin = np.array(
            [[cos_theta, -sin_theta, 0.0], [sin_theta, cos_theta, 0.0], [0.0, 0.0, 1.0]],
            dtype=float,
        )
        translate_to_center = np.array(
            [[1.0, 0.0, center_x], [0.0, 1.0, center_y], [0.0, 0.0, 1.0]],
            dtype=float,
        )
        translate_from_center = np.array(
            [[1.0, 0.0, -center_x], [0.0, 1.0, -center_y], [0.0, 0.0, 1.0]],
            dtype=float,
        )
        rotate_center_matrix = translate_to_center @ rotate_about_origin @ translate_from_center

        corners = np.array(
            [[0.0, 0.0], [target_width, 0.0], [target_width, target_height], [0.0, target_height]],
            dtype=float,
        )
        rotated_corners = CoordinateManager.apply_matrix_to_points(corners, rotate_center_matrix)
        min_x = float(rotated_corners[:, 0].min())
        min_y = float(rotated_corners[:, 1].min())
        expand_shift = np.array(
            [[1.0, 0.0, -min_x], [0.0, 1.0, -min_y], [0.0, 0.0, 1.0]],
            dtype=float,
        )
        paste_shift = np.array(
            [[1.0, 0.0, float(canvas_x)], [0.0, 1.0, float(canvas_y)], [0.0, 0.0, 1.0]],
            dtype=float,
        )
        full_matrix = paste_shift @ expand_shift @ rotate_center_matrix @ scale_matrix
        return CoordinateManager.matrix_to_affine6(full_matrix)

    @staticmethod
    def _assert_within_canvas(*, layer: PreparedLayer, canvas_width: int, canvas_height: int) -> None:
        item = layer.item
        if item.x < 0 or item.y < 0:
            raise ValueError(f"Collage item '{item.artifact_id}' starts outside the canvas at ({item.x}, {item.y})")
        if item.x + layer.image.width > canvas_width or item.y + layer.image.height > canvas_height:
            raise ValueError(
                f"Collage item '{item.artifact_id}' exceeds the canvas bounds after rotation/resize: "
                f"item_box=({item.x}, {item.y}, {item.x + layer.image.width}, {item.y + layer.image.height}), "
                f"canvas=({canvas_width}, {canvas_height})"
            )

    @staticmethod
    def _write_collage(image: Image.Image, *, attempt_id: str | None) -> Path:
        generated_dir = Path(__file__).resolve().parent.parent / "generated" / "image_collage"
        generated_dir.mkdir(parents=True, exist_ok=True)
        output_path = generated_dir / f"{attempt_id or uuid4().hex}_collage.png"
        image.save(output_path)
        return output_path


image_collage_tool = ImageCollageTool()
tool_registry.register(image_collage_tool, replace=True)


__all__ = [
    "CollageLayoutItem",
    "CollageLayoutResult",
    "ImageCollageArgs",
    "ImageCollageTool",
    "image_collage_tool",
]


async def _demo_main() -> None:
    """Real LLM smoke test for the collage tool using locally registered image artifacts."""

    example_image = Path(__file__).resolve().parent.parent / "examples" / "edit_example.png"
    with Image.open(example_image) as image:
        width, height = image.size

    artifacts: dict[str, Any] = {}
    registry = ArtifactRegistry(artifacts)
    artifact_a = registry.register(
        workflow_id="wf_demo",
        artifact_type=ArtifactType.IMAGE,
        value=str(example_image),
        metadata={
            "description": "Top-down photo of hands holding a Python book.",
            "labels": ["hands", "book", "python"],
            "coordinate_info": CoordinateInfo(
                root_artifact_id="artifact_demo_a",
                width=width,
                height=height,
                transform_kind="translation_only",
                offset_x=0,
                offset_y=0,
                parent_artifact_id=None,
            ).model_dump(),
        },
        artifact_id="artifact_demo_a",
    )
    artifact_b = registry.register(
        workflow_id="wf_demo",
        artifact_type=ArtifactType.IMAGE,
        value=str(example_image),
        metadata={
            "description": "Another copy of the same book photo, intended to be placed smaller and rotated.",
            "labels": ["hands", "book", "reference"],
            "coordinate_info": CoordinateInfo(
                root_artifact_id="artifact_demo_b",
                width=width,
                height=height,
                transform_kind="translation_only",
                offset_x=0,
                offset_y=0,
                parent_artifact_id=None,
            ).model_dump(),
        },
        artifact_id="artifact_demo_b",
    )
    ctx = ToolContext(
        llm=LLMClient(),
        artifact_registry=registry,
        workflow_id="wf_demo",
        node_id="node_collage",
        attempt_id="attempt_1",
    )
    result = await image_collage_tool(
        ctx,
        {
            "image_artifact_ids": [artifact_a.artifact_id, artifact_b.artifact_id],
            "prompt": (
                "Create a rough reference collage where the first image is the main large background, "
                "and the second image appears smaller, slightly rotated, and overlapping near the top-right area."
            ),
        },
    )
    print("=== image_collage demo ===")
    print(result.model_dump_json(indent=2))
    print("=== registered artifacts ===")
    for artifact in registry.list_all():
        print(artifact.model_dump_json(indent=2))


def main() -> None:
    asyncio.run(_demo_main())


if __name__ == "__main__":
    main()
