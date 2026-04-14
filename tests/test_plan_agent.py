import json
import importlib
import os
import sys
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ValidationError


def _bootstrap_package() -> None:
    root = Path(__file__).resolve().parents[1]
    use_real_llm = os.environ.get("PLAN_AGENT_REAL", "").lower() in {"1", "true", "yes"}

    if not use_real_llm:
        os.environ.setdefault("LLM_API_KEY", "test-key")

    if use_real_llm:
        for module_name in ("openai", "pydantic_settings"):
            existing = sys.modules.get(module_name)
            if existing is not None and getattr(existing, "__file__", None) is None:
                del sys.modules[module_name]
            importlib.import_module(module_name)

    if not use_real_llm and "pydantic_settings" not in sys.modules:
        module = types.ModuleType("pydantic_settings")

        class BaseSettings(BaseModel):
            model_config = {}

            def __init__(self, **data):
                for name in self.__class__.model_fields:
                    env_name = name.upper()
                    if name not in data and env_name in os.environ:
                        data[name] = os.environ[env_name]
                super().__init__(**data)

        module.BaseSettings = BaseSettings
        module.SettingsConfigDict = dict
        sys.modules["pydantic_settings"] = module

    if not use_real_llm and "openai" not in sys.modules:
        module = types.ModuleType("openai")

        class AsyncOpenAI:
            def __init__(self, *args, **kwargs):
                self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=None))
                self.beta = types.SimpleNamespace(
                    chat=types.SimpleNamespace(completions=types.SimpleNamespace(parse=None))
                )

        module.AsyncOpenAI = AsyncOpenAI
        sys.modules["openai"] = module

    if "fig_edit_agent" not in sys.modules:
        package = types.ModuleType("fig_edit_agent")
        package.__path__ = [str(root)]
        sys.modules["fig_edit_agent"] = package


_bootstrap_package()

from fig_edit_agent.core.plan_agent import PlanAgent
from fig_edit_agent.core.config import settings
from fig_edit_agent.schemas import (
    ArtifactManifest,
    ArtifactSummary,
    ArtifactType,
    DAGPlan,
    InputArtifactRole,
    NodeKind,
    PlanAgentRequest,
    SlotSpec,
    SuccessCriteria,
    TaskNode,
)


TESTS_DIR = Path(__file__).resolve().parent


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_group_photo_request() -> PlanAgentRequest:
    return PlanAgentRequest(
        workflow_id="wf_group_photo",
        goal="Create one polished group photo from multiple source images.",
        user_prompt=(
            "Use bg_1 as the background image. Extract the main person from person_1, person_2, "
            "person_3, person_4, and person_5. Put all five people together into one natural group photo. "
            "Keep each person recognizable, avoid awkward overlaps, make the composition balanced, "
            "and polish the final image so the lighting and edges feel coherent."
        ),
        input_artifacts=[
            ArtifactManifest(
                artifact_id="bg_1",
                artifact_type=ArtifactType.IMAGE,
                uri_or_value="/tmp/bg_1.png",
                input_role=InputArtifactRole.BACKGROUND_CANDIDATE,
                is_user_selected_background=True,
                description="Wide outdoor park scene with enough empty space for a group photo.",
                labels=["park", "outdoor", "background"],
                attributes={"lighting": "soft daylight", "camera_angle": "eye level"},
            ),
            ArtifactManifest(
                artifact_id="person_1",
                artifact_type=ArtifactType.IMAGE,
                uri_or_value="/tmp/person_1.png",
                input_role=InputArtifactRole.SUBJECT_SOURCE,
                description="Portrait of person 1 standing and facing slightly left.",
                labels=["person", "portrait"],
                attributes={"pose": "standing", "framing": "full body"},
            ),
            ArtifactManifest(
                artifact_id="person_2",
                artifact_type=ArtifactType.IMAGE,
                uri_or_value="/tmp/person_2.png",
                input_role=InputArtifactRole.SUBJECT_SOURCE,
                description="Portrait of person 2 standing with arms relaxed.",
                labels=["person", "portrait"],
                attributes={"pose": "standing", "framing": "three quarter"},
            ),
            ArtifactManifest(
                artifact_id="person_3",
                artifact_type=ArtifactType.IMAGE,
                uri_or_value="/tmp/person_3.png",
                input_role=InputArtifactRole.SUBJECT_SOURCE,
                description="Portrait of person 3 looking at the camera.",
                labels=["person", "portrait"],
                attributes={"pose": "standing", "framing": "full body"},
            ),
            ArtifactManifest(
                artifact_id="person_4",
                artifact_type=ArtifactType.IMAGE,
                uri_or_value="/tmp/person_4.png",
                input_role=InputArtifactRole.SUBJECT_SOURCE,
                description="Portrait of person 4 in a casual upright pose.",
                labels=["person", "portrait"],
                attributes={"pose": "upright", "framing": "full body"},
            ),
            ArtifactManifest(
                artifact_id="person_5",
                artifact_type=ArtifactType.IMAGE,
                uri_or_value="/tmp/person_5.png",
                input_role=InputArtifactRole.SUBJECT_SOURCE,
                description="Portrait of person 5 standing naturally.",
                labels=["person", "portrait"],
                attributes={"pose": "standing", "framing": "full body"},
            ),
        ],
        artifact_summaries=[
            ArtifactSummary(
                artifact_id="summary_bg_1",
                source_artifact_id="bg_1",
                description="A wide park background with open grass area that can fit a small group of people.",
                labels=["park", "open space", "daylight"],
                attributes={"lighting": "soft daylight", "depth": "moderate"},
            ),
            ArtifactSummary(
                artifact_id="summary_person_group",
                description="Five separate portrait-style source images, each containing one person intended to appear in the final group photo.",
                labels=["multi-person composition target", "group photo"],
                attributes={"subject_count": 5, "task_bias": "extract subjects then composite"},
            ),
        ],
    )


def _build_valid_plan() -> DAGPlan:
    return DAGPlan(
        workflow_id="wf_group_photo",
        plan_id="draft",
        version=1,
        goal="draft",
        nodes=[
            TaskNode(
                node_id="extract_person_1",
                kind=NodeKind.EXTRACT_SUBJECT,
                goal="Extract the main person from artifact person_1 for later composition.",
                inputs=[SlotSpec(name="source_image", role="subject_source")],
                outputs=[SlotSpec(name="subject", role="extracted_subject")],
                success=SuccessCriteria(
                    hard_rules=["The extracted person must remain recognizable."],
                    required_outputs=["subject"],
                ),
            ),
            TaskNode(
                node_id="extract_person_2",
                kind=NodeKind.EXTRACT_SUBJECT,
                goal="Extract the main person from artifact person_2 for later composition.",
                inputs=[SlotSpec(name="source_image", role="subject_source")],
                outputs=[SlotSpec(name="subject", role="extracted_subject")],
                success=SuccessCriteria(
                    hard_rules=["The extracted person must remain recognizable."],
                    required_outputs=["subject"],
                ),
            ),
            TaskNode(
                node_id="extract_person_3",
                kind=NodeKind.EXTRACT_SUBJECT,
                goal="Extract the main person from artifact person_3 for later composition.",
                inputs=[SlotSpec(name="source_image", role="subject_source")],
                outputs=[SlotSpec(name="subject", role="extracted_subject")],
                success=SuccessCriteria(
                    hard_rules=["The extracted person must remain recognizable."],
                    required_outputs=["subject"],
                ),
            ),
            TaskNode(
                node_id="extract_person_4",
                kind=NodeKind.EXTRACT_SUBJECT,
                goal="Extract the main person from artifact person_4 for later composition.",
                inputs=[SlotSpec(name="source_image", role="subject_source")],
                outputs=[SlotSpec(name="subject", role="extracted_subject")],
                success=SuccessCriteria(
                    hard_rules=["The extracted person must remain recognizable."],
                    required_outputs=["subject"],
                ),
            ),
            TaskNode(
                node_id="extract_person_5",
                kind=NodeKind.EXTRACT_SUBJECT,
                goal="Extract the main person from artifact person_5 for later composition.",
                inputs=[SlotSpec(name="source_image", role="subject_source")],
                outputs=[SlotSpec(name="subject", role="extracted_subject")],
                success=SuccessCriteria(
                    hard_rules=["The extracted person must remain recognizable."],
                    required_outputs=["subject"],
                ),
            ),
            TaskNode(
                node_id="select_background",
                kind=NodeKind.SELECT_BACKGROUND,
                goal="Select bg_1 as the required background image for the final group photo.",
                inputs=[SlotSpec(name="source_image", role="background_candidate")],
                outputs=[SlotSpec(name="background", role="background_image")],
                success=SuccessCriteria(
                    hard_rules=["The selected background must come from artifact bg_1."],
                    required_outputs=["background"],
                ),
            ),
            TaskNode(
                node_id="compose_group_scene",
                kind=NodeKind.COMPOSE_SCENE,
                goal=(
                    "Compose a single natural group photo by placing the extracted subjects from "
                    "person_1, person_2, person_3, person_4, and person_5 together onto bg_1."
                ),
                dependencies=[
                    "extract_person_1",
                    "extract_person_2",
                    "extract_person_3",
                    "extract_person_4",
                    "extract_person_5",
                    "select_background",
                ],
                inputs=[
                    SlotSpec(name="background", role="background_image"),
                    SlotSpec(name="subjects", role="extracted_subject", multiple=True),
                ],
                outputs=[SlotSpec(name="composed_image", role="composed_image")],
                success=SuccessCriteria(
                    hard_rules=["All five extracted subjects must appear in the composition."],
                    required_outputs=["composed_image"],
                ),
            ),
            TaskNode(
                node_id="polish_final_image",
                kind=NodeKind.POLISH_IMAGE,
                goal="Polish the composed group photo to make lighting and edges coherent.",
                dependencies=["compose_group_scene"],
                inputs=[SlotSpec(name="composed_image", role="composed_image")],
                outputs=[SlotSpec(name="final_image", role="final_image")],
                success=SuccessCriteria(
                    hard_rules=["The final image must remain natural and coherent."],
                    required_outputs=["final_image"],
                ),
            ),
        ],
    )


class _NeverCalledLLM:
    async def generate_structured(self, *args, **kwargs):
        raise AssertionError("This fake LLM should not be called in local validation tests.")


class PlanAgentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        log_name = f"{self._testMethodName}.log"
        self.log_path = TESTS_DIR / log_name
        self.log_path.write_text("", encoding="utf-8")
        self._log_line(f"Initialized log file for {self._testMethodName}")
        self._log_line(f"Log path: {self.log_path}")

    def _log_line(self, message: str) -> None:
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"[{_utc_timestamp()}] {message}\n")

    def _log_section(self, title: str) -> None:
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write("\n" + "=" * 100 + "\n")
            fh.write(f"[{_utc_timestamp()}] {title}\n")
            fh.write("=" * 100 + "\n")

    def _log_json(self, title: str, payload) -> None:
        self._log_section(title)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, indent=2))
            fh.write("\n")

    async def test_local_semantic_validation_for_group_photo_plan(self) -> None:
        request = _build_group_photo_request()
        plan = _build_valid_plan()
        agent = PlanAgent(llm=_NeverCalledLLM())
        normalize_started_at = datetime.now(timezone.utc)

        self._log_json("PLAN AGENT TEST INPUT REQUEST", request.model_dump(mode="json"))
        self._log_json("PLAN AGENT TEST RAW PLAN BEFORE NORMALIZATION", plan.model_dump(mode="json"))

        normalized = agent._normalize_plan(plan, request)
        normalize_finished_at = datetime.now(timezone.utc)

        self._log_json("PLAN AGENT TEST NORMALIZED PLAN", normalized.model_dump(mode="json"))
        self._log_line("Running semantic validation for normalized plan...")

        agent._validate_plan(normalized, request)

        self._log_line("Validation passed.")
        self._log_line(f"Normalized plan id: {normalized.plan_id}")
        self._log_line(f"Node count: {len(normalized.nodes)}")
        self._log_line(f"Node ids: {[node.node_id for node in normalized.nodes]}")

        self.assertEqual(normalized.plan_id, "wf_group_photo_plan_v1")
        self.assertEqual(len(normalized.nodes), 8)
        self.assertTrue(any(node.kind == NodeKind.COMPOSE_SCENE for node in normalized.nodes))
        self.assertTrue(any(node.kind == NodeKind.POLISH_IMAGE for node in normalized.nodes))
        self.assertGreaterEqual(normalized.created_at, normalize_started_at)
        self.assertLessEqual(normalized.created_at, normalize_finished_at)

        by_id = {node.node_id: node for node in normalized.nodes}
        self.assertEqual(by_id["select_background"].max_retries, 1)
        self.assertEqual(by_id["select_background"].escalate_after, 1)
        self.assertEqual(by_id["extract_person_1"].max_retries, 2)
        self.assertEqual(by_id["extract_person_1"].escalate_after, 1)
        self.assertEqual(by_id["compose_group_scene"].max_retries, 2)
        self.assertEqual(by_id["compose_group_scene"].escalate_after, 1)
        self.assertEqual(by_id["polish_final_image"].max_retries, 2)
        self.assertEqual(by_id["polish_final_image"].escalate_after, 1)

    async def test_local_semantic_validation_rejects_missing_subject_extraction(self) -> None:
        request = _build_group_photo_request()
        invalid_plan = DAGPlan(
            workflow_id="wf_group_photo",
            plan_id="draft_invalid",
            version=1,
            goal="draft invalid",
            nodes=[
                TaskNode(
                    node_id="extract_person_1",
                    kind=NodeKind.EXTRACT_SUBJECT,
                    goal="Extract the main person from artifact person_1 for later composition.",
                    inputs=[SlotSpec(name="source_image", role="subject_source")],
                    outputs=[SlotSpec(name="subject", role="extracted_subject")],
                    success=SuccessCriteria(
                        hard_rules=["The extracted person must remain recognizable."],
                        required_outputs=["subject"],
                    ),
                ),
                TaskNode(
                    node_id="select_background",
                    kind=NodeKind.SELECT_BACKGROUND,
                    goal="Select bg_1 as the required background image for the final group photo.",
                    inputs=[SlotSpec(name="source_image", role="background_candidate")],
                    outputs=[SlotSpec(name="background", role="background_image")],
                    success=SuccessCriteria(
                        hard_rules=["The selected background must come from artifact bg_1."],
                        required_outputs=["background"],
                    ),
                ),
                TaskNode(
                    node_id="compose_group_scene",
                    kind=NodeKind.COMPOSE_SCENE,
                    goal="Compose a single natural group photo by placing extracted subjects onto bg_1.",
                    dependencies=["extract_person_1", "select_background"],
                    inputs=[
                        SlotSpec(name="background", role="background_image"),
                        SlotSpec(name="subjects", role="extracted_subject", multiple=True),
                    ],
                    outputs=[SlotSpec(name="composed_image", role="composed_image")],
                    success=SuccessCriteria(
                        hard_rules=["All five extracted subjects must appear in the composition."],
                        required_outputs=["composed_image"],
                    ),
                ),
            ],
        )
        agent = PlanAgent(llm=_NeverCalledLLM())

        self._log_json("PLAN AGENT INVALID REQUEST", request.model_dump(mode="json"))
        self._log_json("PLAN AGENT INVALID RAW PLAN", invalid_plan.model_dump(mode="json"))

        normalized = agent._normalize_plan(invalid_plan, request)
        self._log_json("PLAN AGENT INVALID NORMALIZED PLAN", normalized.model_dump(mode="json"))

        with self.assertRaisesRegex(ValueError, "extract_subject node per subject_source input artifact"):
            self._log_line("Running semantic validation for intentionally invalid plan...")
            agent._validate_plan(normalized, request)

        self._log_line("Invalid plan was correctly rejected.")

    async def test_slot_spec_rejects_role_outside_controlled_vocabulary(self) -> None:
        self._log_line("Verifying SlotSpec rejects an out-of-vocabulary role...")
        with self.assertRaises(ValidationError):
            SlotSpec(name="source_image", role="base_image")

    async def test_local_semantic_validation_rejects_non_image_edit_input_role(self) -> None:
        request = PlanAgentRequest(
            workflow_id="wf_single_edit",
            goal="Edit one image into a final image.",
            user_prompt="Make the single input image warmer and more polished.",
            input_artifacts=[
                ArtifactManifest(
                    artifact_id="input_image",
                    artifact_type=ArtifactType.IMAGE,
                    uri_or_value="/tmp/input.png",
                    input_role=InputArtifactRole.PRIMARY_INPUT,
                    description="Single input image.",
                )
            ],
            artifact_summaries=[
                ArtifactSummary(
                    artifact_id="summary_input_image",
                    source_artifact_id="input_image",
                    description="A single portrait image ready for editing.",
                )
            ],
        )
        invalid_plan = DAGPlan(
            workflow_id="wf_single_edit",
            plan_id="draft_invalid_edit",
            version=1,
            goal="draft invalid edit",
            nodes=[
                TaskNode(
                    node_id="node_edit_invalid",
                    kind=NodeKind.EDIT,
                    goal="Edit input_image into a final polished image.",
                    inputs=[SlotSpec(name="source_image", role="caption")],
                    outputs=[SlotSpec(name="final_image", role="final_image")],
                    success=SuccessCriteria(required_outputs=["final_image"]),
                )
            ],
        )
        agent = PlanAgent(llm=_NeverCalledLLM())

        self._log_json("PLAN AGENT SINGLE-EDIT INVALID REQUEST", request.model_dump(mode="json"))
        self._log_json("PLAN AGENT SINGLE-EDIT INVALID PLAN", invalid_plan.model_dump(mode="json"))

        normalized = agent._normalize_plan(invalid_plan, request)

        with self.assertRaisesRegex(ValueError, "must consume an image role"):
            agent._validate_plan(normalized, request)

    async def test_real_plan_agent_prompt_group_photo(self) -> None:
        if os.environ.get("PLAN_AGENT_REAL", "").lower() not in {"1", "true", "yes"}:
            self.skipTest("Set PLAN_AGENT_REAL=1 to run the real LLM-backed planner test.")

        request = _build_group_photo_request()

        self._log_json("REAL PLAN AGENT REQUEST", request.model_dump(mode="json"))
        self._log_line("About to call the real PlanAgent with the configured LLM provider...")
        self._log_line(f"Effective LLM base URL: {settings.llm_base_url}")
        self._log_line(f"Effective LLM model name: {settings.llm_model_name}")
        self._log_line(
            f"Effective LLM API key configured: {bool(settings.llm_api_key.get_secret_value())}"
        )

        agent = PlanAgent()
        started_at = datetime.now(timezone.utc)
        plan = await agent.plan(request)
        finished_at = datetime.now(timezone.utc)

        self._log_json("REAL PLAN AGENT OUTPUT DAG", plan.model_dump(mode="json"))
        self._log_line("Inspecting returned DAG...")
        self._log_line(f"Returned node count: {len(plan.nodes)}")
        self._log_line(f"Returned node ids: {[node.node_id for node in plan.nodes]}")
        self._log_line(f"Returned node kinds: {[node.kind.value for node in plan.nodes]}")
        self._log_line(f"Returned created_at: {plan.created_at.isoformat()}")

        self.assertTrue(any(node.kind == NodeKind.SELECT_BACKGROUND for node in plan.nodes))
        self.assertEqual(
            len([node for node in plan.nodes if node.kind == NodeKind.EXTRACT_SUBJECT]),
            5,
        )
        self.assertTrue(any(node.kind == NodeKind.COMPOSE_SCENE for node in plan.nodes))
        self.assertTrue(any(node.kind == NodeKind.POLISH_IMAGE for node in plan.nodes))
        self.assertGreaterEqual(plan.created_at, started_at)
        self.assertLessEqual(plan.created_at, finished_at)

        for node in plan.nodes:
            if node.kind == NodeKind.SELECT_BACKGROUND:
                self.assertEqual((node.max_retries, node.escalate_after), (1, 1))
            elif node.kind in {NodeKind.EXTRACT_SUBJECT, NodeKind.COMPOSE_SCENE, NodeKind.POLISH_IMAGE}:
                self.assertEqual((node.max_retries, node.escalate_after), (2, 1))


if __name__ == "__main__":
    unittest.main(verbosity=2, buffer=False)
