"""Deterministic demo/test for left-hand cutout using the project tool chain.

This script avoids the LLM grounding step and uses stable hand-picked points for
the bundled `examples/edit_example.png` image. It runs:

    image_segment(refinement_mode="grabcut") -> image_crop(alpha_cutout)

and writes a few easy-to-inspect outputs under `generated/left_hand_tool_chain/`.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _bootstrap_local_package() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    if "fig_edit_agent" in sys.modules:
        return repo_root

    spec = importlib.util.spec_from_file_location(
        "fig_edit_agent",
        repo_root / "__init__.py",
        submodule_search_locations=[str(repo_root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to bootstrap local fig_edit_agent package.")
    module = importlib.util.module_from_spec(spec)
    sys.modules["fig_edit_agent"] = module
    spec.loader.exec_module(module)
    return repo_root


REPO_ROOT = _bootstrap_local_package()

from fig_edit_agent.core import LLMClient  # noqa: E402
from fig_edit_agent.schemas import ArtifactType, CoordinateInfo  # noqa: E402
from fig_edit_agent.tools.base import ArtifactRegistry, ToolContext  # noqa: E402
from fig_edit_agent.tools.image_crop import image_crop_tool  # noqa: E402
from fig_edit_agent.tools.image_segment import image_segment_tool  # noqa: E402


EXAMPLE_IMAGE = REPO_ROOT / "examples" / "edit_example.png"
OUTPUT_DIR = REPO_ROOT / "generated" / "left_hand_tool_chain"

POSITIVE_POINTS = [
    {"x": 286, "y": 187},
    {"x": 312, "y": 171},
    {"x": 243, "y": 215},
]
NEGATIVE_POINTS = [
    {"x": 371, "y": 171},
    {"x": 318, "y": 292},
]


async def run_demo() -> dict[str, object]:
    with Image.open(EXAMPLE_IMAGE) as image:
        width, height = image.size

    artifacts: dict[str, object] = {}
    registry = ArtifactRegistry(artifacts)
    image_artifact = registry.register(
        workflow_id="wf_left_hand_demo",
        artifact_type=ArtifactType.IMAGE,
        value=str(EXAMPLE_IMAGE),
        artifact_id="artifact_demo_image",
        metadata={
            "description": "Example image for deterministic left-hand cutout demo",
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
    )
    points_artifact = registry.register(
        workflow_id="wf_left_hand_demo",
        artifact_type=ArtifactType.POINTS,
        value={
            "label": "left hand",
            "positive_points": POSITIVE_POINTS,
            "negative_points": NEGATIVE_POINTS,
        },
        artifact_id="artifact_demo_points",
        metadata={
            "description": "Stable prompt points for the left hand in edit_example.png",
            "coordinate_info": CoordinateInfo(
                root_artifact_id="artifact_demo_image",
                width=width,
                height=height,
                transform_kind="translation_only",
                offset_x=0,
                offset_y=0,
                parent_artifact_id="artifact_demo_image",
            ).model_dump(),
        },
    )

    ctx = ToolContext(
        llm=LLMClient(),
        artifact_registry=registry,
        workflow_id="wf_left_hand_demo",
        node_id="node_left_hand_demo",
        attempt_id="left_hand_tool_chain",
    )

    segment_result = await image_segment_tool(
        ctx,
        {
            "base_image_artifact_id": image_artifact.artifact_id,
            "points_artifact_id": points_artifact.artifact_id,
            "refinement_mode": "grabcut",
        },
    )
    mask_artifact = registry.get(segment_result.outputs[0].artifact_id)

    crop_result = await image_crop_tool(
        ctx,
        {
            "base_image_artifact_id": image_artifact.artifact_id,
            "mask_artifact_id": mask_artifact.artifact_id,
            "padding": 8,
            "crop_mode": "alpha_cutout",
        },
    )
    crop_artifact = registry.get(crop_result.outputs[0].artifact_id)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mask_path = Path(mask_artifact.value)
    cutout_path = Path(crop_artifact.value)
    stable_mask_path = OUTPUT_DIR / "left_hand_mask.png"
    stable_cutout_path = OUTPUT_DIR / "left_hand_cutout.png"
    stable_white_bg_path = OUTPUT_DIR / "left_hand_cutout_white.png"
    stable_overlay_path = OUTPUT_DIR / "left_hand_overlay.png"

    shutil.copy2(mask_path, stable_mask_path)
    shutil.copy2(cutout_path, stable_cutout_path)

    mask_array = np.asarray(Image.open(stable_mask_path).convert("L")) > 0
    image_array = np.asarray(Image.open(EXAMPLE_IMAGE).convert("RGB"))
    overlay_array = image_array.copy()
    overlay_array[mask_array] = (
        0.55 * overlay_array[mask_array] + 0.45 * np.array([255, 0, 0], dtype=np.float32)
    ).astype(np.uint8)
    Image.fromarray(overlay_array).save(stable_overlay_path)

    cutout_rgba = Image.open(stable_cutout_path).convert("RGBA")
    white_bg = Image.new("RGBA", cutout_rgba.size, (255, 255, 255, 255))
    white_bg.alpha_composite(cutout_rgba)
    white_bg.convert("RGB").save(stable_white_bg_path)

    metadata = dict(mask_artifact.metadata)
    mask_area = int(metadata.get("mask_area", int(mask_array.sum())))
    if mask_area < 5000:
        raise AssertionError(f"Mask area is unexpectedly small: {mask_area}")

    summary = {
        "example_image": str(EXAMPLE_IMAGE),
        "positive_points": POSITIVE_POINTS,
        "negative_points": NEGATIVE_POINTS,
        "selected_candidate_name": metadata.get("selected_candidate_name"),
        "mask_area": mask_area,
        "bbox_from_mask": metadata.get("bbox_from_mask"),
        "mask_path": str(stable_mask_path),
        "cutout_path": str(stable_cutout_path),
        "cutout_white_path": str(stable_white_bg_path),
        "overlay_path": str(stable_overlay_path),
        "crop_bbox_in_parent": crop_artifact.metadata.get("crop_bbox_in_parent"),
    }
    return summary


def main() -> None:
    summary = asyncio.run(run_demo())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
