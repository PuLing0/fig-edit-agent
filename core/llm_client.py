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
        use_parse_api = True
        json_mode_instructions_added = False

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
                if use_parse_api:
                    response = await self.client.beta.chat.completions.parse(
                        model=model or self.default_model,
                        messages=current_messages,
                        response_format=response_model,
                        temperature=temperature,
                    )
                    parsed_object = self._extract_parsed_object(response)
                else:
                    response = await self.client.chat.completions.create(
                        model=model or self.default_model,
                        messages=current_messages,
                        temperature=temperature,
                    )
                    raw_content = self._extract_raw_content(response)
                    parsed_object = self._parse_json_response(
                        raw_content=raw_content,
                        response_model=response_model,
                    )

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
                if use_parse_api and self._should_fallback_from_parse(exc=exc, response=response):
                    logger.warning(
                        "%s Parse API is incompatible with this endpoint. Falling back to JSON text mode. Error: %s",
                        log_prefix,
                        str(exc),
                    )
                    use_parse_api = False
                    if not json_mode_instructions_added:
                        current_messages = self._augment_messages_for_json_mode(
                            current_messages,
                            response_model=response_model,
                        )
                        json_mode_instructions_added = True
                    continue
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

        if response is None:
            return "{}"
        if isinstance(response, str):
            return response
        if not getattr(response, "choices", None):
            return str(response)

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

    @staticmethod
    def _extract_parsed_object(response) -> T:
        """Extract a parsed structured object from SDK parse responses."""

        if isinstance(response, str):
            raise TypeError("parse endpoint returned a plain string instead of an SDK response object")
        if response is None or not getattr(response, "choices", None):
            raise TypeError("parse endpoint returned an object without choices")

        message = response.choices[0].message
        refusal = getattr(message, "refusal", None)
        if refusal:
            raise ValueError(f"Model refused to answer: {refusal}")

        parsed_object = getattr(message, "parsed", None)
        if parsed_object is None:
            raise ValueError("Model returned success but parsed object is None.")
        return parsed_object

    @staticmethod
    def _parse_json_response(*, raw_content: str, response_model: Type[T]) -> T:
        """Parse a plain-text JSON response into the target schema."""

        json_payload = LLMClient._extract_json_payload(raw_content)
        data = json.loads(json_payload)
        return response_model.model_validate(data)

    @staticmethod
    def _extract_json_payload(raw_content: str) -> str:
        """Extract the JSON object payload from a text response."""

        content = raw_content.strip()
        if not content:
            raise ValueError("Model returned empty text content.")

        if content.startswith("```"):
            lines = content.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        try:
            json.loads(content)
            return content
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1 or end < start:
                raise
            candidate = content[start : end + 1]
            json.loads(candidate)
            return candidate

    @staticmethod
    def _augment_messages_for_json_mode(
        messages: list[dict[str, Any]],
        *,
        response_model: Type[T],
    ) -> list[dict[str, Any]]:
        """Append explicit JSON-only instructions for endpoints without parse support."""

        schema_json = json.dumps(response_model.model_json_schema(), ensure_ascii=False, indent=2)
        return [
            *messages,
            {
                "role": "user",
                "content": (
                    "Return only one valid JSON object and nothing else. "
                    "Do not use markdown fences or commentary. "
                    "The JSON must satisfy this schema exactly:\n"
                    f"{schema_json}"
                ),
            },
        ]

    @staticmethod
    def _should_fallback_from_parse(*, exc: Exception, response: Any) -> bool:
        """Whether a parse failure looks like endpoint incompatibility rather than model output failure."""

        if isinstance(response, str):
            return True
        if response is not None and not getattr(response, "choices", None):
            return True

        lowered = str(exc).lower()
        incompatible_markers = (
            "invalid_json_schema",
            "response_format",
            "additionalproperties",
            "object has no attribute 'choices'",
            "plain string",
            "without choices",
        )
        if any(marker in lowered for marker in incompatible_markers):
            return True
        return isinstance(exc, (AttributeError, TypeError))


__all__ = ["LLMClient"]
