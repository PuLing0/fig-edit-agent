"""Load and validate environment configurations."""

from __future__ import annotations

from pathlib import Path

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
    sam3_repo_path: str = Field(
        default="/Users/sijuzheng/project/sam3_260405/sam3",
        description="Local filesystem path to the SAM3 repository root.",
    )
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


settings = AgentSettings()


__all__ = ["AgentSettings", "settings"]
