"""Lazy FireRed image-edit backend."""

from __future__ import annotations

from functools import lru_cache
import importlib
import os
import sys
from pathlib import Path
from typing import Any

from PIL import Image

from .config import settings


class FireRedBackendError(RuntimeError):
    """Raised when the FireRed backend cannot be loaded or executed."""


def _maybe_configure_cuda_visible_devices() -> None:
    visible_devices = settings.firered_cuda_visible_devices
    if visible_devices is None:
        return
    visible_devices = visible_devices.strip()
    if not visible_devices:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices


def _ensure_repo_on_syspath(repo_path: str) -> None:
    root = Path(repo_path).expanduser().resolve()
    if not root.exists():
        raise FireRedBackendError(
            f"FireRed repository path does not exist: {root}. "
            "Set FIRERED_REPO_PATH in fig_edit_agent/.env to your local FireRed-Image-Edit checkout."
        )
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _build_max_memory(torch_module: Any, per_gpu_max_memory: str | None, cpu_max_memory: str | None) -> dict[Any, str] | None:
    if per_gpu_max_memory is None or not str(per_gpu_max_memory).strip():
        return None
    if not torch_module.cuda.is_available():
        raise FireRedBackendError("firered_per_gpu_max_memory requires CUDA to be available.")
    max_memory: dict[Any, str] = {
        idx: per_gpu_max_memory for idx in range(torch_module.cuda.device_count())
    }
    if cpu_max_memory:
        max_memory["cpu"] = cpu_max_memory
    return max_memory


@lru_cache(maxsize=1)
def load_pipeline():
    """Load and cache the FireRed image edit pipeline."""

    _maybe_configure_cuda_visible_devices()
    _ensure_repo_on_syspath(settings.firered_repo_path)
    try:
        torch = importlib.import_module("torch")
        diffusers = importlib.import_module("diffusers")
    except Exception as exc:  # pragma: no cover - import path
        raise FireRedBackendError(
            "FireRed backend requires torch and diffusers to be installed in the active environment."
        ) from exc

    try:
        QwenImageEditPlusPipeline = getattr(diffusers, "QwenImageEditPlusPipeline")
    except AttributeError as exc:  # pragma: no cover - version mismatch path
        raise FireRedBackendError(
            "Installed diffusers package does not expose QwenImageEditPlusPipeline."
        ) from exc

    if settings.firered_optimized:
        if any(
            [
                settings.firered_lora_path and settings.firered_lora_path.strip(),
                settings.firered_device_map and settings.firered_device_map.strip(),
                settings.firered_per_gpu_max_memory and settings.firered_per_gpu_max_memory.strip(),
                settings.firered_local_files_only,
            ]
        ):
            raise FireRedBackendError(
                "firered_optimized only supports direct single-GPU loading without LoRA or device_map sharding."
            )
        try:
            load_fast_pipeline = importlib.import_module("utils.fast_pipeline").load_fast_pipeline
        except Exception as exc:  # pragma: no cover - optimized path
            raise FireRedBackendError(
                "Failed to import FireRed's optimized fast pipeline helper."
            ) from exc
        pipe = load_fast_pipeline(settings.firered_model_path)
        pipe.set_progress_bar_config(disable=None)
        return pipe

    device_map = settings.firered_device_map or None
    per_gpu_max_memory = settings.firered_per_gpu_max_memory or None
    lora_path = settings.firered_lora_path or None
    lora_weight_name = settings.firered_lora_weight_name or None
    if device_map is None and per_gpu_max_memory:
        device_map = "balanced"

    load_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
    }
    if settings.firered_local_files_only:
        load_kwargs["local_files_only"] = True
    if device_map:
        load_kwargs["device_map"] = device_map

    max_memory = _build_max_memory(
        torch,
        per_gpu_max_memory,
        settings.firered_cpu_max_memory,
    )
    if max_memory is not None:
        load_kwargs["max_memory"] = max_memory

    try:
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            settings.firered_model_path,
            **load_kwargs,
        )
    except Exception as exc:  # pragma: no cover - model load path
        raise FireRedBackendError(
            "Failed to load FireRed model. Check FIRERED_MODEL_PATH, local file availability, and backend dependencies."
        ) from exc

    if not device_map:
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(target_device)

    if lora_path:
        lora_kwargs: dict[str, Any] = {"adapter_name": settings.firered_lora_adapter_name}
        if lora_weight_name:
            lora_kwargs["weight_name"] = lora_weight_name
        if settings.firered_local_files_only:
            lora_kwargs["local_files_only"] = True
        pipe.load_lora_weights(lora_path, **lora_kwargs)
        if settings.firered_fuse_lora:
            pipe.fuse_lora()

    if settings.firered_enable_attention_slicing or device_map:
        pipe.enable_attention_slicing()

    pipe.set_progress_bar_config(disable=None)
    return pipe


def _resolve_generator_device(torch_module: Any) -> str:
    generator_device = settings.firered_generator_device or "auto"
    if generator_device == "auto":
        return "cuda:0" if torch_module.cuda.is_available() else "cpu"
    return generator_device


def edit_images(*, images: list[Image.Image], prompt: str) -> Image.Image:
    """Run one FireRed edit call and return a single edited image."""

    if not images:
        raise FireRedBackendError("FireRed backend requires at least one input image.")

    try:
        torch = importlib.import_module("torch")
    except Exception as exc:  # pragma: no cover - import path
        raise FireRedBackendError(
            "FireRed backend requires torch to be installed in the active environment."
        ) from exc

    pipeline = load_pipeline()
    generator_device = _resolve_generator_device(torch)
    try:
        inputs = {
            "image": images,
            "prompt": prompt,
            "generator": torch.Generator(device=generator_device).manual_seed(settings.firered_seed),
            "true_cfg_scale": settings.firered_true_cfg_scale,
            "guidance_scale": settings.firered_guidance_scale,
            "negative_prompt": settings.firered_negative_prompt,
            "num_inference_steps": settings.firered_num_inference_steps,
            "num_images_per_prompt": 1,
        }
        with torch.inference_mode():
            result = pipeline(**inputs)
    except Exception as exc:  # pragma: no cover - runtime path
        raise FireRedBackendError(
            "FireRed inference failed while editing the provided images."
        ) from exc

    if not getattr(result, "images", None):
        raise FireRedBackendError("FireRed pipeline returned no output images.")
    output = result.images[0]
    if not isinstance(output, Image.Image):
        raise FireRedBackendError("FireRed pipeline returned a non-PIL output image.")
    return output


def backend_config_snapshot() -> dict[str, Any]:
    """Return a non-sensitive snapshot of relevant FireRed runtime settings."""

    return {
        "model_path": settings.firered_model_path,
        "local_files_only": settings.firered_local_files_only,
        "cuda_visible_devices": settings.firered_cuda_visible_devices or None,
        "device_map": settings.firered_device_map or None,
        "per_gpu_max_memory": settings.firered_per_gpu_max_memory or None,
        "cpu_max_memory": settings.firered_cpu_max_memory,
        "generator_device": settings.firered_generator_device,
        "enable_attention_slicing": settings.firered_enable_attention_slicing,
        "lora_path": settings.firered_lora_path or None,
        "lora_weight_name": settings.firered_lora_weight_name or None,
        "lora_adapter_name": settings.firered_lora_adapter_name,
        "fuse_lora": settings.firered_fuse_lora,
        "optimized": settings.firered_optimized,
        "num_inference_steps": settings.firered_num_inference_steps,
        "true_cfg_scale": settings.firered_true_cfg_scale,
        "guidance_scale": settings.firered_guidance_scale,
        "negative_prompt": settings.firered_negative_prompt,
        "seed": settings.firered_seed,
    }


__all__ = ["FireRedBackendError", "backend_config_snapshot", "edit_images", "load_pipeline"]
