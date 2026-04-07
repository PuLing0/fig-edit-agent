"""Tool for single-pass point-guided segmentation on the full image."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image
from pydantic import Field

from ..core import CoordinateManager, LLMClient
from ..core.sam3_point_backend import Sam3BackendError, predict_candidates as sam3_predict_candidates
from ..schemas import (
    Artifact,
    ArtifactType,
    BoundingBox,
    CoordinateInfo,
    Identifier,
    NonEmptyStr,
    Point2D,
    StrictSchema,
)
from .base import ArtifactRegistry, BaseTool, ToolContext, ToolResult
from .registry import tool_registry


@dataclass(slots=True)
class SegmentCandidate:
    """Single candidate mask returned by the segmentation backend."""

    name: str
    mask: np.ndarray
    score: float


@dataclass(slots=True)
class CandidateSelection:
    """Selected candidate with normalized mask-level statistics."""

    candidate_name: str
    candidate_index: int
    mask: np.ndarray
    score: float
    area: int
    component_count: int


class ImageSegmentArgs(StrictSchema):
    """Arguments for precise segmentation."""

    base_image_artifact_id: Identifier = Field(description="Artifact id of the image to segment.")
    points_artifact_id: Identifier = Field(description="POINTS artifact providing anchor points.")
    output_spec_name: NonEmptyStr = Field(
        default="mask",
        description="Output slot name for the generated mask artifact.",
    )
    output_role: NonEmptyStr = Field(
        default="subject_mask",
        description="Semantic output role for the generated mask artifact.",
    )


class ImageSegmentTool(BaseTool[ImageSegmentArgs]):
    name = "image_segment"
    description = "Generate a mask from a full image using positive/negative point anchors."
    args_model = ImageSegmentArgs

    async def run(self, ctx: ToolContext, args: ImageSegmentArgs) -> ToolResult:
        image_artifact = ctx.artifact_registry.require_type(args.base_image_artifact_id, ArtifactType.IMAGE)
        points_artifact = ctx.artifact_registry.require_type(args.points_artifact_id, ArtifactType.POINTS)
        image_coord = ctx.artifact_registry.require_coordinate_info(args.base_image_artifact_id)
        points_coord = ctx.artifact_registry.require_coordinate_info(args.points_artifact_id)

        payload = points_artifact.value
        if not isinstance(payload, dict):
            raise TypeError("POINTS artifact must store a dictionary payload")

        positive_points = self._load_points(
            payload=payload,
            key="positive_points",
            source_coord=points_coord,
            target_coord=image_coord,
        )
        negative_points = self._load_points(
            payload=payload,
            key="negative_points",
            source_coord=points_coord,
            target_coord=image_coord,
            required=False,
        )
        if not positive_points:
            raise ValueError("POINTS artifact must contain at least one positive point")

        image_path = Path(image_artifact.value).expanduser().resolve()
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            image_array = np.asarray(rgb_image, dtype=np.uint8)

        candidates = await self._predict_candidates(
            ctx=ctx,
            image_array=image_array,
            positive_points=positive_points,
            negative_points=negative_points,
        )
        selected = self._select_best_candidate(
            candidates=candidates,
            positive_points=positive_points,
            negative_points=negative_points,
        )
        mask_path, bbox = self._write_mask(
            mask=selected.mask,
            attempt_id=ctx.attempt_id,
        )

        mask_artifact = ctx.register_artifact(
            artifact_type=ArtifactType.MASK,
            value=str(mask_path),
            metadata={
                "description": f"Mask for {payload.get('label', 'target')}",
                "source_image_artifact_id": image_artifact.artifact_id,
                "source_points_artifact_id": points_artifact.artifact_id,
                "bbox_from_mask": bbox.model_dump(),
                "selected_candidate_name": selected.candidate_name,
                "selected_candidate_index": selected.candidate_index,
                "model_score": selected.score,
                "mask_area": selected.area,
                "component_count": selected.component_count,
                "positive_points": [point.model_dump() for point in positive_points],
                "negative_points": [point.model_dump() for point in negative_points],
                "coordinate_info": image_coord.model_dump(),
            },
        )
        output = ctx.bind_output(spec_name=args.output_spec_name, role=args.output_role, artifact=mask_artifact)
        return ToolResult(
            outputs=[output],
            summary=(
                f"Segmented '{payload.get('label', 'target')}' from image artifact "
                f"'{image_artifact.artifact_id}' using {len(positive_points)} positive anchor(s)."
            ),
        )

    @staticmethod
    def _load_points(
        *,
        payload: dict[str, Any],
        key: str,
        source_coord: CoordinateInfo,
        target_coord: CoordinateInfo,
        required: bool = True,
    ) -> list[Point2D]:
        raw_points = payload.get(key)
        if raw_points is None:
            if required:
                raise ValueError(f"POINTS artifact must contain '{key}'")
            return []
        if not isinstance(raw_points, list):
            raise ValueError(f"POINTS artifact field '{key}' must be a list")
        points = [Point2D.model_validate(item) for item in raw_points]
        return CoordinateManager.sync_points_between_spaces(points, source_coord, target_coord)

    async def _predict_candidates(
        self,
        *,
        ctx: ToolContext,
        image_array: np.ndarray,
        positive_points: list[Point2D],
        negative_points: list[Point2D],
    ) -> list[SegmentCandidate]:
        backend = ctx.extras.get("segment_backend")
        if backend is not None:
            backend_result = backend(
                image_array=image_array,
                positive_points=positive_points,
                negative_points=negative_points,
            )
            if asyncio.iscoroutine(backend_result):
                backend_result = await backend_result
            candidates = self._coerce_backend_candidates(backend_result)
            if candidates:
                return candidates
            raise ValueError("Injected segment_backend returned no candidate masks")

        try:
            return self._coerce_backend_candidates(
                sam3_predict_candidates(
                    image_array=image_array,
                    positive_points=positive_points,
                    negative_points=negative_points,
                )
            )
        except Sam3BackendError as exc:
            raise RuntimeError(
                "Real SAM3 backend is required for image_segment but could not be used. "
                "Check the vendored SAM3 package, checkpoint settings, and Python dependencies."
            ) from exc

    @staticmethod
    def _coerce_backend_candidates(raw_candidates: Any) -> list[SegmentCandidate]:
        if raw_candidates is None:
            return []
        if isinstance(raw_candidates, SegmentCandidate):
            return [raw_candidates]
        if isinstance(raw_candidates, dict):
            raw_candidates = [raw_candidates]

        candidates: list[SegmentCandidate] = []
        if not isinstance(raw_candidates, list):
            raise TypeError("segment_backend must return a candidate, dict, or list of candidates")

        for index, item in enumerate(raw_candidates):
            if isinstance(item, SegmentCandidate):
                candidates.append(item)
                continue
            if not isinstance(item, dict):
                raise TypeError("segment_backend candidate entries must be SegmentCandidate or dict")
            mask = np.asarray(item.get("mask"))
            score = float(item.get("score", 0.0))
            name = str(item.get("name", f"backend_{index}"))
            candidates.append(SegmentCandidate(name=name, mask=mask, score=score))
        return candidates

    @staticmethod
    def _generate_fallback_candidates(
        *,
        image_array: np.ndarray,
        positive_points: list[Point2D],
        negative_points: list[Point2D],
    ) -> list[SegmentCandidate]:
        height, width = image_array.shape[:2]
        seed_colors = np.array([image_array[point.y, point.x] for point in positive_points], dtype=np.float32)
        mean_color = seed_colors.mean(axis=0)
        spread = (
            float(np.linalg.norm(seed_colors - mean_color[None, :], axis=1).mean())
            if len(seed_colors) > 1
            else 0.0
        )
        color_distance = np.linalg.norm(image_array.astype(np.float32) - mean_color[None, None, :], axis=2)
        thresholds = sorted(
            {
                int(max(16.0, spread * 1.2 + 18.0)),
                int(max(28.0, spread * 2.0 + 30.0)),
                int(max(44.0, spread * 3.0 + 44.0)),
            }
        )

        candidates: list[SegmentCandidate] = []
        for index, threshold in enumerate(thresholds):
            mask = color_distance <= float(threshold)
            mask = ImageSegmentTool._erase_negative_neighborhoods(mask, negative_points, radius=4)
            if not mask.any():
                continue
            score = max(0.1, 0.95 - index * 0.1)
            candidates.append(
                SegmentCandidate(
                    name=f"color_threshold_{threshold}",
                    mask=mask,
                    score=score,
                )
            )

        disk_mask = ImageSegmentTool._make_seed_disk_mask(
            width=width,
            height=height,
            positive_points=positive_points,
            negative_points=negative_points,
        )
        if disk_mask.any():
            candidates.append(SegmentCandidate(name="seed_disks", mask=disk_mask, score=0.35))

        return ImageSegmentTool._deduplicate_candidates(candidates)

    @staticmethod
    def _erase_negative_neighborhoods(mask: np.ndarray, negative_points: list[Point2D], radius: int) -> np.ndarray:
        updated = np.asarray(mask, dtype=bool).copy()
        if radius <= 0:
            return updated
        height, width = updated.shape
        y_grid, x_grid = np.ogrid[:height, :width]
        for point in negative_points:
            distance_sq = (x_grid - point.x) ** 2 + (y_grid - point.y) ** 2
            updated[distance_sq <= radius**2] = False
        return updated

    @staticmethod
    def _make_seed_disk_mask(
        *,
        width: int,
        height: int,
        positive_points: list[Point2D],
        negative_points: list[Point2D],
    ) -> np.ndarray:
        mask = np.zeros((height, width), dtype=bool)
        if len(positive_points) > 1:
            xs = [point.x for point in positive_points]
            ys = [point.y for point in positive_points]
            span = max(max(xs) - min(xs), max(ys) - min(ys))
            radius = int(min(64, max(12, span / 2 + 8)))
        else:
            radius = 24

        y_grid, x_grid = np.ogrid[:height, :width]
        for point in positive_points:
            distance_sq = (x_grid - point.x) ** 2 + (y_grid - point.y) ** 2
            mask[distance_sq <= radius**2] = True
        return ImageSegmentTool._erase_negative_neighborhoods(mask, negative_points, radius=max(4, radius // 3))

    @staticmethod
    def _deduplicate_candidates(candidates: list[SegmentCandidate]) -> list[SegmentCandidate]:
        unique: list[SegmentCandidate] = []
        seen_signatures: set[bytes] = set()
        for candidate in candidates:
            mask = np.asarray(candidate.mask, dtype=bool)
            if mask.ndim != 2 or not mask.any():
                continue
            signature = mask.tobytes()
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            unique.append(SegmentCandidate(name=candidate.name, mask=mask, score=float(candidate.score)))
        return unique

    @staticmethod
    def _select_best_candidate(
        *,
        candidates: list[SegmentCandidate],
        positive_points: list[Point2D],
        negative_points: list[Point2D],
    ) -> CandidateSelection:
        if not candidates:
            raise ValueError("Segmentation backend produced no candidate masks")

        valid: list[CandidateSelection] = []
        min_area = max(8, len(positive_points) * 4)
        for index, candidate in enumerate(candidates):
            normalized_mask, component_count = ImageSegmentTool._retain_seed_components(candidate.mask, positive_points)
            if normalized_mask is None:
                continue
            pos_hit_count = sum(ImageSegmentTool._mask_contains_point(normalized_mask, point) for point in positive_points)
            neg_hit_count = sum(ImageSegmentTool._mask_contains_point(normalized_mask, point) for point in negative_points)
            area = int(normalized_mask.sum())
            if pos_hit_count != len(positive_points):
                continue
            if neg_hit_count != 0:
                continue
            if area < min_area:
                continue
            valid.append(
                CandidateSelection(
                    candidate_name=candidate.name,
                    candidate_index=index,
                    mask=normalized_mask,
                    score=float(candidate.score),
                    area=area,
                    component_count=component_count,
                )
            )

        if not valid:
            raise ValueError("No candidate mask satisfied all positive/negative anchor constraints")

        valid.sort(
            key=lambda item: (
                -item.score,
                item.area,
                item.component_count,
                item.candidate_index,
            )
        )
        return valid[0]

    @staticmethod
    def _retain_seed_components(mask: np.ndarray, positive_points: list[Point2D]) -> tuple[np.ndarray | None, int]:
        binary = np.asarray(mask, dtype=bool)
        if binary.ndim != 2:
            raise ValueError("Candidate masks must be 2D arrays")

        height, width = binary.shape
        kept = np.zeros_like(binary, dtype=bool)
        visited = np.zeros_like(binary, dtype=bool)
        component_count = 0

        for point in positive_points:
            if not (0 <= point.x < width and 0 <= point.y < height):
                continue
            if not binary[point.y, point.x] or visited[point.y, point.x]:
                continue

            component_count += 1
            queue: deque[tuple[int, int]] = deque([(point.y, point.x)])
            visited[point.y, point.x] = True

            while queue:
                y, x = queue.popleft()
                kept[y, x] = True
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < height and 0 <= nx < width and binary[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))

        if not kept.any():
            return None, 0
        return kept, component_count

    @staticmethod
    def _mask_contains_point(mask: np.ndarray, point: Point2D) -> bool:
        height, width = mask.shape
        if not (0 <= point.x < width and 0 <= point.y < height):
            return False
        return bool(mask[point.y, point.x])

    @staticmethod
    def _write_mask(*, mask: np.ndarray, attempt_id: str | None) -> tuple[Path, BoundingBox]:
        generated_dir = Path(__file__).resolve().parent.parent / "generated" / "image_segment"
        generated_dir.mkdir(parents=True, exist_ok=True)
        output_path = generated_dir / f"{attempt_id or uuid4().hex}_mask.png"
        mask_uint8 = (np.asarray(mask, dtype=bool).astype(np.uint8)) * 255
        mask_image = Image.fromarray(mask_uint8, mode="L")
        bbox_tuple = mask_image.getbbox()
        if bbox_tuple is None:
            raise ValueError("Selected mask was empty after normalization")
        mask_image.save(output_path)
        bbox = BoundingBox(left=bbox_tuple[0], top=bbox_tuple[1], right=bbox_tuple[2], bottom=bbox_tuple[3])
        return output_path, bbox


image_segment_tool = ImageSegmentTool()
tool_registry.register(image_segment_tool, replace=True)


async def _demo_main() -> None:
    from .image_grounding import image_grounding_tool

    example_image = Path(__file__).resolve().parent.parent / "examples" / "edit_example.png"
    with Image.open(example_image) as image:
        width, height = image.size

    artifacts: dict[str, Artifact] = {}
    registry = ArtifactRegistry(artifacts)
    input_artifact = registry.register(
        workflow_id="wf_demo",
        artifact_type=ArtifactType.IMAGE,
        value=str(example_image),
        metadata={
            "description": "Local example image for segmentation testing",
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
        node_id="node_segment",
        attempt_id="attempt_1",
    )
    grounding = await image_grounding_tool(
        ctx,
        {
            "base_image_artifact_id": input_artifact.artifact_id,
            "prompt": (
                "From the viewer's perspective, find the watch-wearing hand on the left side of the image "
                "that is holding the Python book. Return 1-3 positive anchor points clearly inside that hand, "
                "and optionally one negative point on the book or forearm if helpful."
            ),
        },
    )
    result = await image_segment_tool(
        ctx,
        {
            "base_image_artifact_id": input_artifact.artifact_id,
            "points_artifact_id": grounding.outputs[0].artifact_id,
        },
    )
    print("=== image_segment demo ===")
    print(result.model_dump_json(indent=2))
    print("=== registered artifacts ===")
    for artifact in registry.list_all():
        print(artifact.model_dump_json(indent=2))


def main() -> None:
    asyncio.run(_demo_main())


if __name__ == "__main__":
    main()


__all__ = ["ImageSegmentArgs", "ImageSegmentTool", "image_segment_tool"]
