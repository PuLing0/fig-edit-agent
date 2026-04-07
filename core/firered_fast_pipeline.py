"""Vendored fast FireRed pipeline loader."""

from __future__ import annotations

import importlib
from typing import Any

from PIL import Image


def _linear_forward_hook(self, x, *args, **kwargs):
    """Custom forward path compatible with LoRA layers under torch.compile."""

    result = self.base_layer(x, *args, **kwargs)
    if not hasattr(self, "active_adapters"):
        return result
    for active_adapter in self.active_adapters:
        if active_adapter not in self.lora_A:
            continue
        lora_A = self.lora_A[active_adapter]
        lora_B = self.lora_B[active_adapter]
        dropout = self.lora_dropout[active_adapter]
        scaling = self.scaling[active_adapter]
        x_input = x.to(lora_A.weight.dtype)
        output = lora_B(lora_A(dropout(x_input))) * scaling
        result = result + output.to(result.dtype)
    return result


def _apply_compile(*, pipeline: Any, torch_module: Any) -> None:
    """Enable local compile optimizations for the fast mode."""

    try:
        peft_layer_module = importlib.import_module("peft.tuners.lora.layer")
        Linear = getattr(peft_layer_module, "Linear")
    except Exception as exc:  # pragma: no cover - dependency path
        raise RuntimeError(
            "FireRed fast mode requires the 'peft' package for compile-compatible LoRA layers."
        ) from exc

    for module in pipeline.transformer.modules():
        if isinstance(module, Linear):
            module.forward = _linear_forward_hook.__get__(module, Linear)

    torch_module._dynamo.config.recompile_limit = 1024
    pipeline.transformer.compile_repeated_blocks(mode="default", dynamic=True)
    pipeline.vae = torch_module.compile(pipeline.vae, mode="reduce-overhead")


def _apply_cache(*, pipeline: Any, cache_dit_module: Any, DBCacheConfig: Any, TaylorSeerCalibratorConfig: Any) -> None:
    """Enable DiT caching for the fast mode."""

    cache_dit_module.enable_cache(
        pipeline,
        cache_config=DBCacheConfig(
            Fn_compute_blocks=8,
            Bn_compute_blocks=0,
            residual_diff_threshold=0.15,
            max_warmup_steps=3,
        ),
        calibrator_config=TaylorSeerCalibratorConfig(taylorseer_order=1),
    )


def load_fast_pipeline(
    model_path: str,
    *,
    device: str = "cuda:0",
    local_files_only: bool = False,
):
    """Initialize the accelerated FireRed pipeline using only local vendored code."""

    try:
        torch = importlib.import_module("torch")
        diffusers = importlib.import_module("diffusers")
        transformers = importlib.import_module("transformers")
        cache_dit = importlib.import_module("cache_dit")
        optimum_quanto = importlib.import_module("optimum.quanto")
    except Exception as exc:  # pragma: no cover - dependency path
        raise RuntimeError(
            "FireRed fast mode requires torch, diffusers, transformers, cache_dit, and optimum-quanto."
        ) from exc

    try:
        QwenImageEditPlusPipeline = getattr(diffusers, "QwenImageEditPlusPipeline")
        QwenImageTransformer2DModel = getattr(diffusers, "QwenImageTransformer2DModel")
        Qwen2_5_VLForConditionalGeneration = getattr(
            transformers,
            "Qwen2_5_VLForConditionalGeneration",
        )
        quantize = getattr(optimum_quanto, "quantize")
        qint8 = getattr(optimum_quanto, "qint8")
        freeze = getattr(optimum_quanto, "freeze")
        DBCacheConfig = getattr(cache_dit, "DBCacheConfig")
        TaylorSeerCalibratorConfig = getattr(cache_dit, "TaylorSeerCalibratorConfig")
    except AttributeError as exc:  # pragma: no cover - dependency version path
        raise RuntimeError(
            "Installed fast-mode dependencies are missing required FireRed runtime classes."
        ) from exc

    weight_dtype = torch.bfloat16
    pretrained_kwargs: dict[str, Any] = {"torch_dtype": weight_dtype}
    if local_files_only:
        pretrained_kwargs["local_files_only"] = True

    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        subfolder="text_encoder",
        **pretrained_kwargs,
    ).to(device)
    quantize(text_encoder, weights=qint8)
    freeze(text_encoder)

    transformer = QwenImageTransformer2DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        **pretrained_kwargs,
    )
    quantize(transformer, weights=qint8, exclude=["proj_out"])
    freeze(transformer)

    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        model_path,
        transformer=transformer,
        text_encoder=text_encoder,
        **pretrained_kwargs,
    )

    _apply_cache(
        pipeline=pipeline,
        cache_dit_module=cache_dit,
        DBCacheConfig=DBCacheConfig,
        TaylorSeerCalibratorConfig=TaylorSeerCalibratorConfig,
    )
    _apply_compile(pipeline=pipeline, torch_module=torch)

    pipeline.vae.enable_tiling()
    pipeline.vae.enable_slicing()
    pipeline.to(device)

    fake_pil = Image.new("RGB", (896, 896), (128, 128, 128))
    with torch.no_grad():
        pipeline(
            image=[fake_pil],
            prompt="warmup session",
            num_inference_steps=4,
            negative_prompt=" ",
        )

    return pipeline


__all__ = ["load_fast_pipeline"]
