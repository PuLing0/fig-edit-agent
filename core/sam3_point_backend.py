"""Lazy SAM3 point-prompt backend for image segmentation."""

from __future__ import annotations

from functools import lru_cache
import importlib
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from ..schemas import Point2D
from .config import PACKAGE_ROOT, settings


class Sam3BackendError(RuntimeError):
    """Raised when the real SAM3 backend cannot be initialized or executed."""


def _maybe_configure_cuda_visible_devices() -> None:
    visible_devices = settings.sam3_cuda_visible_devices
    if visible_devices is None:
        return
    visible_devices = visible_devices.strip()
    if not visible_devices:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices


def _ensure_vendored_sam3_on_syspath() -> None:
    vendored_root = PACKAGE_ROOT / "core" / "_vendor"
    vendored_package_dir = vendored_root / "sam3"
    vendored_model_builder = vendored_package_dir / "model_builder.py"
    if not vendored_model_builder.exists():
        raise Sam3BackendError(
            "Vendored SAM3 package is missing from this project. "
            "Expected to find core/_vendor/sam3/model_builder.py inside the repository."
        )
    vendored_root_str = str(vendored_root)
    if vendored_root_str not in sys.path:
        sys.path.insert(0, vendored_root_str)

    loaded_module = sys.modules.get("sam3")
    if loaded_module is None:
        return
    loaded_from = Path(getattr(loaded_module, "__file__", "")).resolve()
    if vendored_package_dir in loaded_from.parents:
        return
    for module_name in list(sys.modules):
        if module_name == "sam3" or module_name.startswith("sam3."):
            sys.modules.pop(module_name, None)


@lru_cache(maxsize=1)
def _load_predictor():
    _maybe_configure_cuda_visible_devices()
    _ensure_vendored_sam3_on_syspath()
    try:
        model_builder = importlib.import_module("sam3.model_builder")
    except Exception as exc:  # pragma: no cover - import error path
        raise Sam3BackendError(
            "Failed to import the vendored SAM3 package from this project. "
            "Make sure the SAM3 repo dependencies are installed in the active environment."
        ) from exc

    build_sam3_image_model = getattr(model_builder, "build_sam3_image_model")
    download_ckpt_from_hf = getattr(model_builder, "download_ckpt_from_hf")

    checkpoint_path = settings.sam3_checkpoint_path or None
    load_from_hf = settings.sam3_load_from_hf
    if checkpoint_path is None and load_from_hf:
        checkpoint_path = download_ckpt_from_hf(version=settings.sam3_model_version)
        load_from_hf = False

    try:
        model = build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            load_from_HF=load_from_hf,
            device=settings.sam3_device,
            eval_mode=True,
            enable_inst_interactivity=True,
            compile=settings.sam3_compile,
        )
    except Exception as exc:  # pragma: no cover - backend init path
        raise Sam3BackendError(
            "Failed to initialize the SAM3 image model. "
            "Check SAM3 checkpoint settings, device selection, and dependency installation."
        ) from exc

    predictor = getattr(model, "inst_interactive_predictor", None)
    if predictor is None:
        raise Sam3BackendError(
            "SAM3 image model was created without inst_interactive_predictor. "
            "The backend requires enable_inst_interactivity=True."
        )
    return predictor


def predict_candidates(
    *,
    image_array: np.ndarray,
    positive_points: list[Point2D],
    negative_points: list[Point2D],
) -> list[dict[str, object]]:
    """Run real SAM3 point-prompt segmentation and return raw candidate masks."""

    predictor = _load_predictor()
    pil_image = Image.fromarray(np.asarray(image_array, dtype=np.uint8), mode="RGB")
    try:
        predictor.set_image(pil_image)
        point_coords = np.array(
            [[point.x, point.y] for point in [*positive_points, *negative_points]],
            dtype=np.float32,
        )
        point_labels = np.array(
            [1] * len(positive_points) + [0] * len(negative_points),
            dtype=np.int32,
        )
        masks, iou_predictions, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
            return_logits=False,
            normalize_coords=False,
        )
    except Exception as exc:  # pragma: no cover - backend runtime path
        raise Sam3BackendError(
            "SAM3 point-prompt prediction failed while processing the input image."
        ) from exc

    masks = np.asarray(masks)
    iou_predictions = np.asarray(iou_predictions)
    if masks.ndim == 2:
        masks = masks[None, ...]
    if iou_predictions.ndim == 0:
        iou_predictions = iou_predictions[None]

    candidates: list[dict[str, object]] = []
    for index in range(min(len(masks), len(iou_predictions))):
        candidates.append(
            {
                "name": f"sam3_mask_{index}",
                "mask": np.asarray(masks[index] > 0, dtype=bool),
                "score": float(iou_predictions[index]),
            }
        )
    return candidates


__all__ = ["Sam3BackendError", "predict_candidates"]
