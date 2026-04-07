"""Load and validate environment configurations."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


PACKAGE_ROOT = Path(__file__).resolve().parent.parent


class AgentSettings(BaseSettings):
    """Global settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=str(PACKAGE_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for the OpenAI-compatible LLM API.",
    )
    llm_model_name: str = Field(
        default="gpt-4o-mini",
        description="Default model name used for structured and text generation.",
    )
    llm_api_key: SecretStr = Field(description="API key used to access the LLM provider.")
    sam3_checkpoint_path: str | None = Field(
        default=None,
        description="Optional explicit path to a SAM3/SAM3.1 checkpoint file.",
    )
    sam3_model_version: str = Field(
        default="sam3.1",
        description="Checkpoint version to download from Hugging Face when sam3_checkpoint_path is not provided.",
    )
    sam3_load_from_hf: bool = Field(
        default=True,
        description="Whether to download SAM3 checkpoints from Hugging Face when no explicit checkpoint path is provided.",
    )
    sam3_device: str = Field(
        default="cpu",
        description="Device string used to initialize SAM3, e.g. 'cpu' or 'cuda'.",
    )
    sam3_cuda_visible_devices: str | None = Field(
        default=None,
        description="Optional CUDA_VISIBLE_DEVICES value used to restrict which physical GPU(s) SAM3 can see.",
    )
    sam3_compile: bool = Field(
        default=False,
        description="Whether to enable torch.compile when constructing the SAM3 image model.",
    )
    firered_model_path: str = Field(
        default="FireRedTeam/FireRed-Image-Edit-1.0",
        description="Path to the FireRed model or Hugging Face model id.",
    )
    firered_inference_mode: Literal["normal", "fast"] = Field(
        default="normal",
        description="FireRed inference pipeline variant. Use 'normal' for standard diffusers loading or 'fast' for vendored accelerated loading.",
    )
    firered_local_files_only: bool = Field(
        default=False,
        description="Whether FireRed model loading should only use local files.",
    )
    firered_cuda_visible_devices: str | None = Field(
        default=None,
        description="Optional CUDA_VISIBLE_DEVICES value used to restrict which physical GPU(s) FireRed can see.",
    )
    firered_device_map: str | None = Field(
        default=None,
        description="Optional FireRed device map. Use 'balanced' for pipeline-level component placement or 'manual' for project-local multi-GPU layer sharding.",
    )
    firered_per_gpu_max_memory: str | None = Field(
        default=None,
        description="Optional per-visible-GPU memory budget such as '22GiB'.",
    )
    firered_cpu_max_memory: str = Field(
        default="128GiB",
        description="CPU RAM budget used together with firered_per_gpu_max_memory.",
    )
    firered_generator_device: str = Field(
        default="auto",
        description="Torch generator device for FireRed, e.g. 'auto' or 'cuda:0'.",
    )
    firered_enable_attention_slicing: bool = Field(
        default=False,
        description="Whether to enable attention slicing for FireRed inference.",
    )
    firered_lora_path: str | None = Field(
        default=None,
        description="Optional LoRA path for FireRed inference.",
    )
    firered_lora_weight_name: str | None = Field(
        default=None,
        description="Optional LoRA weight file name inside firered_lora_path.",
    )
    firered_lora_adapter_name: str = Field(
        default="demo",
        description="Adapter name used when loading FireRed LoRA weights.",
    )
    firered_fuse_lora: bool = Field(
        default=False,
        description="Whether to fuse FireRed LoRA weights after loading.",
    )
    firered_num_inference_steps: int = Field(
        default=40,
        description="Number of FireRed diffusion inference steps.",
    )
    firered_true_cfg_scale: float = Field(
        default=4.0,
        description="FireRed true CFG scale.",
    )
    firered_guidance_scale: float = Field(
        default=1.0,
        description="FireRed guidance scale.",
    )
    firered_negative_prompt: str = Field(
        default=" ",
        description="Negative prompt passed into FireRed.",
    )
    firered_seed: int = Field(
        default=49,
        description="Random seed used for FireRed generation.",
    )


settings = AgentSettings()


__all__ = ["AgentSettings", "settings"]
