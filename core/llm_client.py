"""Robust async LLM client with pydantic-ai structured output."""

from __future__ import annotations

import json
import logging
from typing import Any, Type, TypeVar, cast

from openai import AsyncOpenAI
from pydantic_ai import Agent, ImageUrl
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from .config import settings
from ..schemas.base import StrictSchema


T = TypeVar("T", bound=StrictSchema)


logger = logging.getLogger(__name__)


class LLMClient:
    """Thin async wrapper around OpenAI-compatible text and structured generation."""

    def __init__(self) -> None:
        self.client = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key.get_secret_value(),
        )
        self.provider = OpenAIProvider(openai_client=self.client)
        self.default_model = settings.llm_model_name
        self._structured_models: dict[str, OpenAIChatModel] = {}

    async def generate_structured(
        self,
        messages: list[dict[str, Any]],
        response_model: Type[T],
        max_retries: int = 3,
        task_name: str = "default_task",
        model: str | None = None,
        temperature: float = 0.0,
    ) -> T:
        """Generate a structured object that must validate against `response_model`."""

        log_prefix = f"[{task_name}]"
        model_name = model or self.default_model
        logger.info(
            "%s Requesting %s via pydantic-ai with output schema %s",
            log_prefix,
            model_name,
            response_model.__name__,
        )

        system_prompts = self._extract_system_prompts(messages)
        user_prompt = self._build_user_prompt(messages)
        agent: Agent[None, T] = Agent(
            self._get_structured_model(model_name),
            output_type=response_model,
            retries=max_retries,
            instructions=(
                "Return a complete object matching the declared output schema exactly. "
                "Do not omit required fields, nested arrays, or constrained values."
            ),
            system_prompt=tuple(system_prompts),
            defer_model_check=True,
        )

        result = await agent.run(
            user_prompt,
            model_settings={"temperature": temperature},
        )
        logger.info("%s Structured generation successful.", log_prefix)
        return cast(T, result.output)

    async def generate_text(
        self,
        messages: list[dict[str, Any]],
        task_name: str = "default_text_task",
        model: str | None = None,
        temperature: float = 0.2,
        max_retries: int = 1,
    ) -> str:
        """Generate plain text for non-structured helper tasks."""

        log_prefix = f"[{task_name}]"
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(
                    "%s Attempt %s/%s - Requesting %s",
                    log_prefix,
                    attempt,
                    max_retries,
                    model or self.default_model,
                )
                response = await self.client.chat.completions.create(
                    model=model or self.default_model,
                    messages=messages,
                    temperature=temperature,
                )
                message = response.choices[0].message
                content = message.content
                if not content:
                    raise ValueError("Model returned empty text content.")
                if isinstance(content, list):
                    text_parts = [part.text for part in content if getattr(part, "type", None) == "text"]
                    if not text_parts:
                        raise ValueError("Model returned no text parts in message content.")
                    return "\n".join(text_parts)
                return content
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "%s Text generation error on attempt %s/%s: %s",
                    log_prefix,
                    attempt,
                    max_retries,
                    str(exc),
                )
        assert last_error is not None
        raise last_error

    def _get_structured_model(self, model_name: str) -> OpenAIChatModel:
        """Reuse pydantic-ai OpenAI chat model instances per model name."""

        structured_model = self._structured_models.get(model_name)
        if structured_model is None:
            structured_model = OpenAIChatModel(model_name, provider=self.provider)
            self._structured_models[model_name] = structured_model
        return structured_model

    @staticmethod
    def _extract_system_prompts(messages: list[dict[str, Any]]) -> list[str]:
        """Collect system-role messages as pydantic-ai system prompts."""

        prompts: list[str] = []
        for message in messages:
            if message.get("role") != "system":
                continue
            content = LLMClient._stringify_content(message.get("content"))
            if content:
                prompts.append(content)
        return prompts

    @staticmethod
    def _build_user_prompt(messages: list[dict[str, Any]]) -> str | list[str | ImageUrl]:
        """Convert OpenAI-style message payloads into pydantic-ai user prompt parts."""

        prompt_parts: list[str | ImageUrl] = []
        for message in messages:
            role = str(message.get("role", "user"))
            if role == "system":
                continue
            prompt_parts.extend(LLMClient._convert_content_parts(message.get("content"), role=role))

        if not prompt_parts:
            return ""
        if len(prompt_parts) == 1 and isinstance(prompt_parts[0], str):
            return prompt_parts[0]
        return prompt_parts

    @staticmethod
    def _convert_content_parts(content: Any, *, role: str) -> list[str | ImageUrl]:
        """Convert one message content payload into pydantic-ai user content parts."""

        role_prefix = "" if role == "user" else f"{role.title()} message:\n"
        if isinstance(content, str):
            if not content and not role_prefix:
                return []
            return [f"{role_prefix}{content}" if content else role_prefix.rstrip()]

        if isinstance(content, list):
            parts: list[str | ImageUrl] = []
            text_prefix = role_prefix
            for item in content:
                if not isinstance(item, dict):
                    parts.append(f"{text_prefix}{json.dumps(item, ensure_ascii=False, default=str)}")
                    text_prefix = ""
                    continue

                item_type = item.get("type")
                if item_type == "text":
                    text = str(item.get("text", ""))
                    if text:
                        parts.append(f"{text_prefix}{text}")
                        text_prefix = ""
                    continue

                if item_type == "image_url":
                    image_url = item.get("image_url")
                    if not isinstance(image_url, dict) or not image_url.get("url"):
                        raise ValueError("image_url content must include a non-empty url")
                    if text_prefix:
                        parts.append(text_prefix.rstrip())
                        text_prefix = ""
                    vendor_metadata = {}
                    if image_url.get("detail") is not None:
                        vendor_metadata["detail"] = image_url["detail"]
                    parts.append(ImageUrl(url=str(image_url["url"]), vendor_metadata=vendor_metadata or None))
                    continue

                parts.append(f"{text_prefix}{json.dumps(item, ensure_ascii=False, default=str)}")
                text_prefix = ""

            if not parts and text_prefix:
                return [text_prefix.rstrip()]
            return parts

        serialized = LLMClient._stringify_content(content)
        if not serialized:
            return []
        return [f"{role_prefix}{serialized}"]

    @staticmethod
    def _stringify_content(content: Any) -> str:
        """Serialize arbitrary message content into a readable text block."""

        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            serialized: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = str(item.get("text", ""))
                    if text:
                        serialized.append(text)
                    continue
                if isinstance(item, dict) and item.get("type") == "image_url":
                    image_url = item.get("image_url")
                    if isinstance(image_url, dict) and image_url.get("url"):
                        serialized.append(f"[image: {image_url['url']}]")
                    continue
                serialized.append(json.dumps(item, ensure_ascii=False, default=str))
            return "\n".join(part for part in serialized if part)
        return json.dumps(content, ensure_ascii=False, default=str)


__all__ = ["LLMClient"]
