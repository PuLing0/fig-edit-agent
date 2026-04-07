"""Manual multi-GPU sharding helpers for FireRed normal-mode loading."""

from __future__ import annotations

from functools import lru_cache
import importlib
from types import MethodType
from typing import Any


def _chunk_ranges(total: int, parts: int) -> list[tuple[int, int]]:
    if total <= 0:
        return []
    if parts <= 0:
        raise ValueError("parts must be positive")
    width, remainder = divmod(total, parts)
    ranges: list[tuple[int, int]] = []
    start = 0
    for index in range(parts):
        stop = start + width + (1 if index < remainder else 0)
        ranges.append((start, stop))
        start = stop
    return [item for item in ranges if item[0] < item[1]]


def _apply_block_ranges(prefix: str, ranges: list[tuple[int, int]], devices: list[int]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for device, (start, stop) in zip(devices, ranges, strict=False):
        for index in range(start, stop):
            mapping[f"{prefix}.{index}"] = device
    return mapping


@lru_cache(maxsize=8)
def _inspect_transformer_structure(model_path: str, local_files_only: bool) -> dict[str, int]:
    diffusers = importlib.import_module("diffusers")
    accelerate = importlib.import_module("accelerate")
    QwenImageTransformer2DModel = getattr(diffusers, "QwenImageTransformer2DModel")
    init_empty_weights = getattr(accelerate, "init_empty_weights")

    config = QwenImageTransformer2DModel.load_config(
        model_path,
        subfolder="transformer",
        local_files_only=local_files_only,
    )
    with init_empty_weights():
        model = QwenImageTransformer2DModel.from_config(config)
    return {"block_count": len(model.transformer_blocks)}


@lru_cache(maxsize=8)
def _inspect_text_encoder_structure(model_path: str, local_files_only: bool) -> dict[str, int]:
    accelerate = importlib.import_module("accelerate")
    transformers = importlib.import_module("transformers")
    init_empty_weights = getattr(accelerate, "init_empty_weights")
    Qwen2_5_VLForConditionalGeneration = getattr(transformers, "Qwen2_5_VLForConditionalGeneration")

    config = Qwen2_5_VLForConditionalGeneration.config_class.from_pretrained(
        model_path,
        subfolder="text_encoder",
        local_files_only=local_files_only,
    )
    with init_empty_weights():
        model = Qwen2_5_VLForConditionalGeneration(config)
    return {
        "visual_block_count": len(model.model.visual.blocks),
        "language_layer_count": len(model.model.language_model.layers),
    }


def build_manual_shard_plan(
    *,
    model_path: str,
    visible_gpu_ids: list[int],
    local_files_only: bool,
) -> dict[str, Any]:
    if len(visible_gpu_ids) < 2:
        raise ValueError("manual FireRed sharding requires at least 2 visible CUDA devices")

    transformer_info = _inspect_transformer_structure(model_path, local_files_only)
    text_encoder_info = _inspect_text_encoder_structure(model_path, local_files_only)

    forward_devices = list(visible_gpu_ids)
    reverse_devices = list(reversed(visible_gpu_ids))

    transformer_ranges = _chunk_ranges(transformer_info["block_count"], len(forward_devices))
    visual_ranges = _chunk_ranges(text_encoder_info["visual_block_count"], len(forward_devices))
    language_ranges = _chunk_ranges(text_encoder_info["language_layer_count"], len(reverse_devices))

    transformer_device_map: dict[str, int] = {
        "pos_embed": forward_devices[0],
        "time_text_embed": forward_devices[0],
        "txt_norm": forward_devices[0],
        "img_in": forward_devices[0],
        "txt_in": forward_devices[0],
        "norm_out": forward_devices[-1],
        "proj_out": forward_devices[-1],
    }
    transformer_device_map.update(
        _apply_block_ranges("transformer_blocks", transformer_ranges, forward_devices)
    )

    text_encoder_device_map: dict[str, int] = {
        "model.visual.patch_embed": forward_devices[0],
        "model.visual.rotary_pos_emb": forward_devices[0],
        "model.visual.merger": reverse_devices[0],
        "model.language_model.embed_tokens": reverse_devices[0],
        "model.language_model.rotary_emb": reverse_devices[-1],
        "model.language_model.norm": reverse_devices[-1],
        "lm_head": reverse_devices[-1],
    }
    text_encoder_device_map.update(
        _apply_block_ranges("model.visual.blocks", visual_ranges, forward_devices)
    )
    text_encoder_device_map.update(
        _apply_block_ranges("model.language_model.layers", language_ranges, reverse_devices)
    )

    return {
        "strategy": "manual_visible_gpu_shard",
        "visible_gpu_ids": forward_devices,
        "transformer_block_ranges": [
            {"device": device, "start": start, "stop": stop}
            for device, (start, stop) in zip(forward_devices, transformer_ranges, strict=False)
        ],
        "text_visual_block_ranges": [
            {"device": device, "start": start, "stop": stop}
            for device, (start, stop) in zip(forward_devices, visual_ranges, strict=False)
        ],
        "text_language_layer_ranges": [
            {"device": device, "start": start, "stop": stop}
            for device, (start, stop) in zip(reverse_devices, language_ranges, strict=False)
        ],
        "transformer_device_map": transformer_device_map,
        "text_encoder_device_map": text_encoder_device_map,
        "vae_device": forward_devices[-1],
    }


def load_manual_sharded_pipeline(
    model_path: str,
    *,
    local_files_only: bool = False,
):
    """Load FireRed with explicit layer-wise sharding across all visible GPUs."""

    torch = importlib.import_module("torch")
    diffusers = importlib.import_module("diffusers")
    transformers = importlib.import_module("transformers")

    if not torch.cuda.is_available():
        raise RuntimeError("manual FireRed sharding requires CUDA to be available")

    visible_gpu_ids = list(range(torch.cuda.device_count()))
    shard_plan = build_manual_shard_plan(
        model_path=model_path,
        visible_gpu_ids=visible_gpu_ids,
        local_files_only=local_files_only,
    )

    QwenImageEditPlusPipeline = getattr(diffusers, "QwenImageEditPlusPipeline")
    QwenImageTransformer2DModel = getattr(diffusers, "QwenImageTransformer2DModel")
    AutoencoderKLQwenImage = getattr(diffusers, "AutoencoderKLQwenImage")
    Qwen2_5_VLForConditionalGeneration = getattr(
        transformers,
        "Qwen2_5_VLForConditionalGeneration",
    )

    weight_dtype = torch.bfloat16
    pretrained_kwargs = {
        "torch_dtype": weight_dtype,
        "local_files_only": local_files_only,
    }

    transformer = QwenImageTransformer2DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        device_map=shard_plan["transformer_device_map"],
        **pretrained_kwargs,
    )
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        subfolder="text_encoder",
        device_map=shard_plan["text_encoder_device_map"],
        **pretrained_kwargs,
    )
    vae = AutoencoderKLQwenImage.from_pretrained(
        model_path,
        subfolder="vae",
        device_map={"": shard_plan["vae_device"]},
        **pretrained_kwargs,
    )

    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        model_path,
        transformer=transformer,
        text_encoder=text_encoder,
        vae=vae,
        **pretrained_kwargs,
    )

    latent_device = torch.device(f"cuda:{visible_gpu_ids[0]}")
    vae_device = torch.device(f"cuda:{shard_plan['vae_device']}")
    original_encode_vae_image = pipeline._encode_vae_image
    original_vae_decode = pipeline.vae.decode

    def _manual_encode_vae_image(self, image, generator):
        image = image.to(device=vae_device, dtype=self.vae.dtype)
        image_latents = original_encode_vae_image(image=image, generator=generator)
        return image_latents.to(device=latent_device)

    def _manual_vae_decode(latents, *args, **kwargs):
        latents = latents.to(device=vae_device, dtype=pipeline.vae.dtype)
        return original_vae_decode(latents, *args, **kwargs)

    pipeline._encode_vae_image = MethodType(_manual_encode_vae_image, pipeline)
    pipeline.vae.decode = _manual_vae_decode
    pipeline.hf_device_map = {
        "transformer": "manual",
        "text_encoder": "manual",
        "vae": shard_plan["vae_device"],
    }
    pipeline._manual_devices = {
        "latent_device": str(latent_device),
        "vae_device": str(vae_device),
    }
    pipeline._manual_shard_summary = shard_plan
    return pipeline


__all__ = [
    "build_manual_shard_plan",
    "load_manual_sharded_pipeline",
]
