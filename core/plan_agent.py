"""Semantic task-level DAG planner for complex image workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..schemas import (
    ArtifactRole,
    ArtifactManifest,
    ArtifactSummary,
    DAGPlan,
    InputArtifactRole,
    NodeKind,
    PlanAgentRequest,
    TaskNode,
    utc_now,
)
from .llm_client import LLMClient


EXTRACT_OUTPUT_ROLE = ArtifactRole.EXTRACTED_SUBJECT
BACKGROUND_OUTPUT_ROLE = ArtifactRole.BACKGROUND_IMAGE
COMPOSE_OUTPUT_ROLE = ArtifactRole.COMPOSED_IMAGE
FINAL_OUTPUT_ROLE = ArtifactRole.FINAL_IMAGE
EDIT_OUTPUT_ROLE = ArtifactRole.EDITED_IMAGE

IMAGE_SOURCE_ROLES = {
    ArtifactRole.PRIMARY_INPUT,
    ArtifactRole.SUBJECT_SOURCE,
    ArtifactRole.BACKGROUND_CANDIDATE,
    ArtifactRole.BACKGROUND_IMAGE,
    ArtifactRole.COMPOSED_IMAGE,
    ArtifactRole.EDITED_IMAGE,
    ArtifactRole.FINAL_IMAGE,
    ArtifactRole.CROPPED_IMAGE,
}

UNDERSTAND_OUTPUT_ROLES = {
    ArtifactRole.ANALYSIS,
    ArtifactRole.SCENE_UNDERSTANDING,
}

SCORE_OUTPUT_ROLES = {
    ArtifactRole.SCORE,
    ArtifactRole.SCORE_ARTIFACT,
}

DEFAULT_NODE_RETRY_POLICY: dict[NodeKind, tuple[int, int]] = {
    NodeKind.SELECT_BACKGROUND: (1, 1),
    NodeKind.EXTRACT_SUBJECT: (2, 1),
    NodeKind.COMPOSE_SCENE: (2, 1),
    NodeKind.POLISH_IMAGE: (2, 1),
    NodeKind.EDIT: (2, 1),
    NodeKind.SCORE: (1, 1),
    NodeKind.UNDERSTAND: (1, 1),
}


@dataclass(slots=True)
class PlanAgentConfig:
    """Runtime knobs controlling planner behavior."""

    model: str | None = None
    temperature: float = 0.0
    default_max_retries: int = 3
    default_escalate_after: int = 2

    def __post_init__(self) -> None:
        if self.temperature < 0.0:
            raise ValueError("temperature must be >= 0")
        if self.default_max_retries <= 0:
            raise ValueError("default_max_retries must be positive")
        if self.default_escalate_after <= 0:
            raise ValueError("default_escalate_after must be positive")
        if self.default_escalate_after > self.default_max_retries:
            raise ValueError("default_escalate_after must be <= default_max_retries")


class PlanAgent:
    """Generate a semantic task DAG from user intent and artifact context."""

    def __init__(
        self,
        *,
        llm: LLMClient | None = None,
        config: PlanAgentConfig | None = None,
    ) -> None:
        self.llm = llm or LLMClient()
        self.config = config or PlanAgentConfig()

    async def plan(self, request: PlanAgentRequest) -> DAGPlan:
        """Generate, normalize, and validate a task-level DAG plan."""

        messages = self._build_messages(request)
        raw_plan = await self.llm.generate_structured(
            messages=messages,
            response_model=DAGPlan,
            task_name="plan_agent",
            model=self.config.model,
            temperature=self.config.temperature,
        )
        plan = self._normalize_plan(raw_plan, request)
        plan.validate_dag()
        self._validate_plan(plan, request)
        return plan

    def _build_messages(self, request: PlanAgentRequest) -> list[dict[str, Any]]:
        allowed_kinds = ", ".join(kind.value for kind in request.available_node_kinds)
        artifact_lines = [self._format_manifest(item) for item in request.input_artifacts]
        summary_lines = [self._format_summary(item) for item in request.artifact_summaries] or ["- none"]
        hints_text = (
            json.dumps(request.planner_hints, ensure_ascii=False, sort_keys=True, indent=2)
            if request.planner_hints
            else "{}"
        )

        system_prompt = (
            "You are the Plan Agent for a DAG-based image workflow system.\n\n"
            "Your job is to decompose the user's goal into semantic task nodes, not tool calls.\n"
            "Do not choose or reference concrete tools. Leave allowed_tools as an empty list for every node.\n"
            "Each node must describe a business subtask with clear goal, inputs, outputs, dependencies, and success criteria.\n"
            "Before planning, the system has already run image understanding on every user-provided input image. "
            "Treat the provided artifact_summaries as the initial per-image understanding context for planning.\n"
            "Every targeted input artifact id must be mentioned explicitly in the node goal when that node acts on a specific input image.\n"
            "When the task is multi-image composition, prefer task-level nodes such as extract_subject, "
            "select_background, compose_scene, and polish_image.\n"
            "Plan image work in stages so execute agents can carry it out step by step. "
            "Prefer multiple clear edit or composition phases over one oversized node when the request is complex.\n"
            "Understand nodes are optional control points for re-reading an intermediate image after a significant change. "
            "Score nodes are optional control points for judging stage quality, prompt alignment, or readiness to continue. "
            "Insert these nodes when they materially improve robustness, not by default after every edit.\n"
            "The plan must end with a node that produces the final output image for the user.\n"
            "Use the smallest DAG that fully captures the required subtasks.\n"
            "The output must be a valid DAGPlan with no cycles."
        )

        user_prompt = "\n\n".join(
            [
                f"Top-level goal:\n{request.goal}",
                f"Original user instruction:\n{request.user_prompt}",
                f"Available node kinds:\n{allowed_kinds}",
                "Input artifacts:\n" + "\n".join(artifact_lines),
                "Artifact summaries:\n" + "\n".join(summary_lines),
                "Planner hints:\n" + hints_text,
                "Planning workflow:\n"
                "- First use the artifact summaries as the initial understanding of each user-provided input image.\n"
                "- Then build a staged execution plan that transforms the available images toward the user's goal.\n"
                "- Add understand nodes when the workflow needs to re-read an edited intermediate image and refresh semantic understanding.\n"
                "- Add score nodes when the workflow needs to evaluate whether an intermediate or final image is good enough before continuing.\n"
                "- The final stage must produce an output image that represents the finished result for the user.\n",
                "Role conventions:\n"
                "- extract_subject nodes output role=extracted_subject.\n"
                "- select_background nodes output role=background_image.\n"
                "- compose_scene nodes consume background_image + extracted_subject and output composed_image.\n"
                "- polish_image nodes consume composed_image or edited_image and output final_image.\n"
                "- score nodes must output role=score or role=score_artifact.\n"
                "- Use semantic slot names such as source_image, subjects, background, composed_image, final_image.\n"
                "- Choose slot roles from the schema-controlled role vocabulary. For image inputs prefer primary_input, "
                "subject_source, background_candidate, background_image, composed_image, edited_image, or final_image.\n"
                "- Mention artifact ids in node goal text when needed, but do not place artifact ids inside slot objects.\n"
                "- Use allowed_tools: [] for all nodes.\n"
                "- Include success.required_outputs for every node.\n"
                "- For non-trivial editing requests, prefer a sequence of stage-specific edit or compose nodes rather than one large catch-all edit node.\n"
                "- Edit node goals should state the specific stage objective and the image they operate on.\n"
                "- Understand node goals should state which intermediate image is being re-read and what changes matter.\n"
                "- Score node goals should state whether they evaluate a stage objective or the overall goal.\n"
                "- If the user explicitly selected a background artifact, preserve that artifact id in the relevant goal text.\n"
                "- If multiple subject_source artifacts are provided, create one extraction task per artifact unless the request clearly says otherwise.",
            ]
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _normalize_plan(self, plan: DAGPlan, request: PlanAgentRequest) -> DAGPlan:
        normalized_nodes = [
            node.model_copy(
                update={
                    "allowed_tools": [],
                    "max_retries": self._default_max_retries_for_kind(node.kind),
                    "escalate_after": self._default_escalate_after_for_kind(node.kind),
                }
            )
            for node in plan.nodes
        ]
        return plan.model_copy(
            update={
                "workflow_id": request.workflow_id,
                "plan_id": f"{request.workflow_id}_plan_v{request.plan_version}",
                "version": request.plan_version,
                "goal": request.goal,
                "nodes": normalized_nodes,
                "created_at": utc_now(),
            }
        )

    def _default_max_retries_for_kind(self, kind: NodeKind) -> int:
        return DEFAULT_NODE_RETRY_POLICY.get(
            kind,
            (self.config.default_max_retries, self.config.default_escalate_after),
        )[0]

    def _default_escalate_after_for_kind(self, kind: NodeKind) -> int:
        return DEFAULT_NODE_RETRY_POLICY.get(
            kind,
            (self.config.default_max_retries, self.config.default_escalate_after),
        )[1]

    def _validate_plan(self, plan: DAGPlan, request: PlanAgentRequest) -> None:
        available_node_kinds = set(request.available_node_kinds)
        node_by_id = {node.node_id: node for node in plan.nodes}

        for node in plan.nodes:
            if node.kind not in available_node_kinds:
                raise ValueError(
                    f"Planner produced unavailable node kind '{node.kind.value}' for node '{node.node_id}'"
                )
            if not node.outputs:
                raise ValueError(f"Planner node '{node.node_id}' must declare at least one output slot")
            if node.allowed_tools:
                raise ValueError(
                    f"Planner node '{node.node_id}' must not assign concrete tools; allowed_tools must be empty"
                )

        self._validate_common_image_nodes(plan)

        if self._is_multi_image_composition(request):
            self._validate_subject_extraction(plan, request)
            self._validate_background_selection(plan, request)
            self._validate_compose_nodes(plan, node_by_id=node_by_id)
            self._validate_polish_nodes(plan)

    @staticmethod
    def _validate_common_image_nodes(plan: DAGPlan) -> None:
        for node in plan.nodes:
            if node.kind == NodeKind.EDIT:
                if not any(slot.role in IMAGE_SOURCE_ROLES for slot in node.inputs):
                    raise ValueError(
                        f"edit node '{node.node_id}' must consume an image role such as primary_input, "
                        "edited_image, or composed_image"
                    )
                if not any(slot.role in {EDIT_OUTPUT_ROLE, FINAL_OUTPUT_ROLE} for slot in node.outputs):
                    raise ValueError(
                        f"edit node '{node.node_id}' must output role "
                        f"'{EDIT_OUTPUT_ROLE.value}' or '{FINAL_OUTPUT_ROLE.value}'"
                    )
            elif node.kind == NodeKind.UNDERSTAND:
                if not any(slot.role in IMAGE_SOURCE_ROLES for slot in node.inputs):
                    raise ValueError(
                        f"understand node '{node.node_id}' must consume an image role such as primary_input, "
                        "edited_image, or composed_image"
                    )
                if not any(slot.role in UNDERSTAND_OUTPUT_ROLES for slot in node.outputs):
                    raise ValueError(
                        f"understand node '{node.node_id}' must output role "
                        f"'{ArtifactRole.ANALYSIS.value}' or '{ArtifactRole.SCENE_UNDERSTANDING.value}'"
                    )
            elif node.kind == NodeKind.SCORE:
                if not any(slot.role in IMAGE_SOURCE_ROLES for slot in node.inputs):
                    raise ValueError(
                        f"score node '{node.node_id}' must consume an image role such as primary_input, "
                        "edited_image, or composed_image"
                    )
                if not any(slot.role in SCORE_OUTPUT_ROLES for slot in node.outputs):
                    raise ValueError(
                        f"score node '{node.node_id}' must output role "
                        f"'{ArtifactRole.SCORE.value}' or '{ArtifactRole.SCORE_ARTIFACT.value}'"
                    )

    @staticmethod
    def _validate_subject_extraction(plan: DAGPlan, request: PlanAgentRequest) -> None:
        subject_inputs = [
            artifact for artifact in request.input_artifacts if artifact.input_role == InputArtifactRole.SUBJECT_SOURCE
        ]
        if not subject_inputs:
            return

        extract_nodes = [node for node in plan.nodes if node.kind == NodeKind.EXTRACT_SUBJECT]
        if len(extract_nodes) < len(subject_inputs):
            raise ValueError("Planner must create one extract_subject node per subject_source input artifact")

        for artifact in subject_inputs:
            matches = [node for node in extract_nodes if artifact.artifact_id in node.goal]
            if not matches:
                raise ValueError(
                    f"Planner must reference subject_source artifact '{artifact.artifact_id}' in an extract_subject goal"
                )
            if not any(slot.role == EXTRACT_OUTPUT_ROLE for node in matches for slot in node.outputs):
                raise ValueError(
                    f"extract_subject node for '{artifact.artifact_id}' must output role '{EXTRACT_OUTPUT_ROLE.value}'"
                )

    @staticmethod
    def _validate_background_selection(plan: DAGPlan, request: PlanAgentRequest) -> None:
        selected_backgrounds = [
            artifact for artifact in request.input_artifacts if artifact.is_user_selected_background
        ]
        if not selected_backgrounds:
            return

        selected = selected_backgrounds[0]
        relevant_nodes = [
            node
            for node in plan.nodes
            if node.kind in {NodeKind.SELECT_BACKGROUND, NodeKind.COMPOSE_SCENE} and selected.artifact_id in node.goal
        ]
        if not relevant_nodes:
            raise ValueError(
                "Planner must preserve the user-selected background artifact id in a select_background "
                f"or compose_scene goal: '{selected.artifact_id}'"
            )

    @staticmethod
    def _validate_compose_nodes(plan: DAGPlan, *, node_by_id: dict[str, TaskNode]) -> None:
        compose_nodes = [node for node in plan.nodes if node.kind == NodeKind.COMPOSE_SCENE]
        extract_nodes = [node for node in plan.nodes if node.kind == NodeKind.EXTRACT_SUBJECT]

        if extract_nodes and not compose_nodes:
            raise ValueError("Planner created extracted subjects but no compose_scene node")

        for node in compose_nodes:
            dependency_ids = set(node.dependencies)
            extract_ids = {extract_node.node_id for extract_node in extract_nodes}
            if extract_ids and not extract_ids.issubset(dependency_ids):
                missing = sorted(extract_ids.difference(dependency_ids))
                raise ValueError(
                    f"compose_scene node '{node.node_id}' must depend on all extract_subject nodes. Missing: {missing}"
                )
            if not any(slot.role == EXTRACT_OUTPUT_ROLE for slot in node.inputs):
                raise ValueError(
                    f"compose_scene node '{node.node_id}' must consume role '{EXTRACT_OUTPUT_ROLE.value}'"
                )
            if not any(slot.role == BACKGROUND_OUTPUT_ROLE for slot in node.inputs):
                upstream_background = any(
                    any(output.role == BACKGROUND_OUTPUT_ROLE for output in node_by_id[dep].outputs)
                    for dep in node.dependencies
                )
                if not upstream_background:
                    raise ValueError(
                        f"compose_scene node '{node.node_id}' must consume role '{BACKGROUND_OUTPUT_ROLE.value}'"
                    )
            if not any(slot.role == COMPOSE_OUTPUT_ROLE for slot in node.outputs):
                raise ValueError(
                    f"compose_scene node '{node.node_id}' must output role '{COMPOSE_OUTPUT_ROLE.value}'"
                )

    @staticmethod
    def _validate_polish_nodes(plan: DAGPlan) -> None:
        polish_nodes = [node for node in plan.nodes if node.kind == NodeKind.POLISH_IMAGE]
        compose_nodes = [node for node in plan.nodes if node.kind == NodeKind.COMPOSE_SCENE]
        compose_ids = {node.node_id for node in compose_nodes}

        if compose_nodes and not polish_nodes:
            raise ValueError("Planner must include a polish_image node after compose_scene for multi-image composition")

        for node in polish_nodes:
            dependency_ids = set(node.dependencies)
            if compose_ids and not compose_ids.intersection(dependency_ids):
                raise ValueError(
                    f"polish_image node '{node.node_id}' must depend on a compose_scene node"
                )
            if not any(slot.role in {COMPOSE_OUTPUT_ROLE, EDIT_OUTPUT_ROLE} for slot in node.inputs):
                raise ValueError(
                    f"polish_image node '{node.node_id}' must consume role "
                    f"'{COMPOSE_OUTPUT_ROLE.value}' or '{EDIT_OUTPUT_ROLE.value}'"
                )
            if not any(slot.role == FINAL_OUTPUT_ROLE for slot in node.outputs):
                raise ValueError(
                    f"polish_image node '{node.node_id}' must output role '{FINAL_OUTPUT_ROLE.value}'"
                )

    @staticmethod
    def _is_multi_image_composition(request: PlanAgentRequest) -> bool:
        subject_count = sum(
            1 for artifact in request.input_artifacts if artifact.input_role == InputArtifactRole.SUBJECT_SOURCE
        )
        return len(request.input_artifacts) > 1 and (
            subject_count > 0
            or any(artifact.is_user_selected_background for artifact in request.input_artifacts)
        )

    @staticmethod
    def _format_manifest(manifest: ArtifactManifest) -> str:
        parts = [
            f"- artifact_id={manifest.artifact_id}",
            f"type={manifest.artifact_type.value}",
            f"role={manifest.input_role.value}",
            f"user_selected_background={str(manifest.is_user_selected_background).lower()}",
        ]
        if manifest.description:
            parts.append(f"description={manifest.description}")
        if manifest.labels:
            parts.append(f"labels={manifest.labels}")
        if manifest.attributes:
            parts.append(f"attributes={json.dumps(manifest.attributes, ensure_ascii=False, sort_keys=True)}")
        return "; ".join(parts)

    @staticmethod
    def _format_summary(summary: ArtifactSummary) -> str:
        parts = [
            f"- summary_artifact_id={summary.artifact_id}",
            f"description={summary.description}",
        ]
        if summary.source_artifact_id:
            parts.append(f"source_artifact_id={summary.source_artifact_id}")
        if summary.labels:
            parts.append(f"labels={summary.labels}")
        if summary.attributes:
            parts.append(f"attributes={json.dumps(summary.attributes, ensure_ascii=False, sort_keys=True)}")
        return "; ".join(parts)


__all__ = ["PlanAgent", "PlanAgentConfig"]
