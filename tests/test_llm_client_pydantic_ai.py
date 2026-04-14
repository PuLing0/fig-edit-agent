import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def _bootstrap_package() -> None:
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("LLM_API_KEY", "test-key")

    if "fig_edit_agent" not in sys.modules:
        package = types.ModuleType("fig_edit_agent")
        package.__path__ = [str(root)]
        sys.modules["fig_edit_agent"] = package


_bootstrap_package()

from pydantic_ai import ImageUrl

from fig_edit_agent.core.llm_client import LLMClient
from fig_edit_agent.schemas.base import StrictSchema


class DemoResponse(StrictSchema):
    answer: str


class _FakeRunResult:
    def __init__(self, output: DemoResponse) -> None:
        self.output = output


class _FakeAgent:
    init_calls: list[dict] = []
    run_calls: list[dict] = []
    next_output = DemoResponse(answer="ok")

    def __init__(self, model, **kwargs) -> None:
        self.model = model
        self.kwargs = kwargs
        type(self).init_calls.append({"model": model, **kwargs})

    async def run(self, user_prompt, **kwargs):
        type(self).run_calls.append({"user_prompt": user_prompt, **kwargs})
        return _FakeRunResult(type(self).next_output)

    @classmethod
    def reset(cls) -> None:
        cls.init_calls = []
        cls.run_calls = []
        cls.next_output = DemoResponse(answer="ok")


class LLMClientPydanticAITests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        _FakeAgent.reset()

    async def test_generate_structured_uses_pydantic_ai_agent_with_output_type(self) -> None:
        client = LLMClient()
        fake_model = object()

        with patch("fig_edit_agent.core.llm_client.Agent", _FakeAgent), patch.object(
            client,
            "_get_structured_model",
            return_value=fake_model,
        ):
            result = await client.generate_structured(
                messages=[
                    {"role": "system", "content": "Follow the schema exactly."},
                    {"role": "user", "content": "say ok"},
                ],
                response_model=DemoResponse,
                max_retries=4,
                task_name="demo",
                model="demo-model",
                temperature=0.0,
            )

        self.assertEqual(result.answer, "ok")
        self.assertEqual(len(_FakeAgent.init_calls), 1)
        self.assertEqual(len(_FakeAgent.run_calls), 1)

        init_call = _FakeAgent.init_calls[0]
        self.assertIs(init_call["model"], fake_model)
        self.assertIs(init_call["output_type"], DemoResponse)
        self.assertEqual(init_call["retries"], 4)
        self.assertEqual(init_call["system_prompt"], ("Follow the schema exactly.",))
        self.assertTrue(init_call["defer_model_check"])

        run_call = _FakeAgent.run_calls[0]
        self.assertEqual(run_call["user_prompt"], "say ok")
        self.assertEqual(run_call["model_settings"], {"temperature": 0.0})

    def test_build_user_prompt_converts_text_and_image_parts(self) -> None:
        prompt = LLMClient._build_user_prompt(
            [
                {"role": "system", "content": "system instructions"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image."},
                        {"type": "image_url", "image_url": {"url": "https://example.com/a.png", "detail": "high"}},
                    ],
                },
                {"role": "assistant", "content": "Previous tool summary."},
            ]
        )

        self.assertIsInstance(prompt, list)
        assert isinstance(prompt, list)
        self.assertEqual(prompt[0], "Describe this image.")
        self.assertIsInstance(prompt[1], ImageUrl)
        self.assertEqual(str(prompt[1].url), "https://example.com/a.png")
        self.assertEqual(prompt[1].vendor_metadata, {"detail": "high"})
        self.assertEqual(prompt[2], "Assistant message:\nPrevious tool summary.")

    def test_extract_system_prompts_ignores_non_system_messages(self) -> None:
        prompts = LLMClient._extract_system_prompts(
            [
                {"role": "system", "content": "first"},
                {"role": "user", "content": "hello"},
                {"role": "system", "content": [{"type": "text", "text": "second"}]},
            ]
        )

        self.assertEqual(prompts, ["first", "second"])


if __name__ == "__main__":
    unittest.main()
