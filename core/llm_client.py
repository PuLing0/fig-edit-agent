"""Robust async LLM client with structured output and self-correction."""

from __future__ import annotations

import json
import logging
from typing import Any, Type, TypeVar

from openai import AsyncOpenAI
from pydantic import ValidationError

from .config import settings
from ..schemas.base import StrictSchema


T = TypeVar("T", bound=StrictSchema)


logger = logging.getLogger(__name__)


class LLMClient:
    """Thin async wrapper around OpenAI's structured output interface.

    This client is intentionally small: it handles configuration, request sending,
    refusal/empty-response checks, and a Pydantic-driven self-correction loop.
    """

    def __init__(self) -> None:
        self.client = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key.get_secret_value(),
        )
        self.default_model = settings.llm_model_name

    async def generate_structured(
        self,
        messages: list[dict[str, Any]],
        response_model: Type[T],
        max_retries: int = 3,
        task_name: str = "default_task",
        model: str | None = None,
        temperature: float = 0.0,
    ) -> T:
        """Generate a structured object that must validate against `response_model`.

        If the model returns invalid content, the validation error is appended back
        into the conversation to give the model a chance to self-correct.
        """

        log_prefix = f"[{task_name}]"
        current_messages = list(messages)

        for attempt in range(1, max_retries + 1):
            response = None
            try:
                logger.info(
                    "%s Attempt %s/%s - Requesting %s",
                    log_prefix,
                    attempt,
                    max_retries,
                    model or self.default_model,
                )

                response = await self.client.beta.chat.completions.parse(
                    model=model or self.default_model,
                    messages=current_messages,
                    response_format=response_model,
                    temperature=temperature,
                )

                message = response.choices[0].message

                refusal = getattr(message, "refusal", None)
                if refusal:
                    raise ValueError(f"Model refused to answer: {refusal}")

                parsed_object = message.parsed
                if parsed_object is None:
                    raise ValueError("Model returned success but parsed object is None.")

                logger.info("%s Attempt %s successful. Validation passed.", log_prefix, attempt)
                return parsed_object

            except (ValidationError, ValueError) as exc:
                logger.warning(
                    "%s Validation/Logic Error on attempt %s/%s: %s",
                    log_prefix,
                    attempt,
                    max_retries,
                    str(exc),
                )
                if attempt == max_retries:
                    logger.error("%s Max retries reached. Failing task.", log_prefix)
                    raise

                raw_failed_content = self._extract_raw_content(response)
                current_messages.append({"role": "assistant", "content": raw_failed_content})
                current_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response failed structural or logical validation.\n"
                            f"Error Details:\n{str(exc)}\n"
                            "Please fix the specific issues and output the complete valid JSON again."
                        ),
                    }
                )

            except Exception as exc:
                logger.error("%s Unexpected API error: %s", log_prefix, str(exc))
                raise

        raise RuntimeError("Unreachable state in generate_structured")

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

    @staticmethod
    def _extract_raw_content(response) -> str:
        """Best-effort extraction of the model's raw failed response."""

        if response is None or not getattr(response, "choices", None):
            return "{}"

        message = response.choices[0].message
        content = getattr(message, "content", None)
        if content is None:
            return "{}"
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            serialized: list[str] = []
            for item in content:
                if hasattr(item, "text") and item.text:
                    serialized.append(item.text)
                elif isinstance(item, dict) and "text" in item:
                    serialized.append(str(item["text"]))
                else:
                    try:
                        serialized.append(json.dumps(item, ensure_ascii=False, default=str))
                    except TypeError:
                        serialized.append(str(item))
            return "\n".join(serialized) if serialized else "{}"
        return str(content)


__all__ = ["LLMClient"]
