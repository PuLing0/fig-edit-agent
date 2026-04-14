"""Minimal orchestrator loop for planning and executing image workflows."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from pydantic import Field, model_validator

from ..schemas import (
    Artifact,
    ArtifactBinding,
    ArtifactManifest,
    ArtifactSummary,
    ArtifactType,
    AttemptStatus,
    CoordinateInfo,
    DAGPlan,
    ExecutionLog,
    ExecutionRequest,
    InputArtifactRole,
    NodeState,
    NodeStatus,
    PlanAgentRequest,
    SlotSpec,
    StrictSchema,
    TaskNode,
    WorkflowState,
    WorkflowStatus,
    utc_now,
)
from ..tools.base import ArtifactRegistry, ToolContext
from ..tools.registry import (
    ToolRegistry,
    ensure_builtin_tools_registered,
    tool_registry as default_tool_registry,
)
from .execute_agent import ExecuteAgent
from .llm_client import LLMClient
from .plan_agent import PlanAgent


class OrchestratorInputArtifact(ArtifactManifest):
    """User-supplied raw artifact to seed a workflow run."""

    coordinate_info: CoordinateInfo | None = Field(
        default=None,
        description="Coordinate information required for image-like artifacts.",
    )

    @model_validator(mode="after")
    def validate_coordinate_info(self) -> "OrchestratorInputArtifact":
        if self.artifact_type in {ArtifactType.IMAGE, ArtifactType.MASK, ArtifactType.POINTS}:
            if self.coordinate_info is None:
                raise ValueError(f"{self.artifact_type.value} inputs require coordinate_info")
        return self


class WorkflowRunResult(StrictSchema):
    """Final result of one orchestrator run."""

    workflow_state: WorkflowState
    plan: DAGPlan
    execution_logs: list[ExecutionLog] = Field(default_factory=list)


@dataclass(slots=True)
class OrchestratorConfig:
    """Runtime knobs for the orchestrator loop."""

    understand_prompt: str | None = None
    fail_fast: bool = True
    concurrent_ready_nodes: bool = True


@dataclass(slots=True)
class _DispatchBundle:
    """Prepared execution payload for one node."""

    node_id: str
    request: ExecutionRequest


class Orchestrator:
    """Run the minimal workflow lifecycle: understand -> plan -> execute."""

    def __init__(
        self,
        *,
        llm: LLMClient | None = None,
        plan_agent: PlanAgent | None = None,
        execute_agent: ExecuteAgent | None = None,
        tool_registry: ToolRegistry | None = None,
        config: OrchestratorConfig | None = None,
        storage: Any | None = None,
        extras: dict[str, Any] | None = None,
    ) -> None:
        self.llm = llm or LLMClient()
        self.tool_registry = tool_registry or default_tool_registry
        if self.tool_registry is default_tool_registry:
            ensure_builtin_tools_registered()
        self.plan_agent = plan_agent or PlanAgent(llm=self.llm)
        self.execute_agent = execute_agent or ExecuteAgent(
            llm=self.llm,
            tool_registry=self.tool_registry,
            storage=storage,
            extras=extras,
        )
        self.config = config or OrchestratorConfig()
        self.storage = storage
        self.extras = dict(extras or {})

    async def run(
        self,
        *,
        workflow_id: str,
        goal: str,
        user_prompt: str,
        input_artifacts: list[OrchestratorInputArtifact],
    ) -> WorkflowRunResult:
        """Execute the minimal orchestrator lifecycle."""

        workflow_state = WorkflowState(
            workflow_id=workflow_id,
            status=WorkflowStatus.PENDING,
            active_plan_id=f"{workflow_id}_plan_pending",
            active_plan_version=1,
        )
        artifact_registry = ArtifactRegistry(workflow_state.artifacts)
        execution_logs: dict[str, ExecutionLog] = {}

        self._register_input_artifacts(
            workflow_state=workflow_state,
            artifact_registry=artifact_registry,
            input_artifacts=input_artifacts,
        )
        await self._run_pre_perception(
            workflow_state=workflow_state,
            artifact_registry=artifact_registry,
        )

        plan = await self.plan_agent.plan(
            PlanAgentRequest(
                workflow_id=workflow_id,
                goal=goal,
                user_prompt=user_prompt,
                input_artifacts=list(workflow_state.artifact_manifests.values()),
                artifact_summaries=list(workflow_state.artifact_summaries.values()),
                plan_version=workflow_state.active_plan_version,
            )
        )
        self._initialize_plan_state(workflow_state=workflow_state, plan=plan)

        while True:
            self._refresh_node_statuses(workflow_state=workflow_state, plan=plan)
            ready_nodes = [
                node for node in plan.nodes if workflow_state.nodes[node.node_id].status == NodeStatus.READY
            ]

            if all(state.status == NodeStatus.SUCCESS for state in workflow_state.nodes.values()):
                workflow_state.status = WorkflowStatus.SUCCESS
                workflow_state.updated_at = utc_now()
                return WorkflowRunResult(
                    workflow_state=workflow_state,
                    plan=plan,
                    execution_logs=list(execution_logs.values()),
                )

            failed_node_ids = [
                node_id for node_id, state in workflow_state.nodes.items() if state.status == NodeStatus.FAILED
            ]
            if failed_node_ids:
                workflow_state.status = WorkflowStatus.FAILED
                workflow_state.updated_at = utc_now()
                return WorkflowRunResult(
                    workflow_state=workflow_state,
                    plan=plan,
                    execution_logs=list(execution_logs.values()),
                )

            if not ready_nodes:
                workflow_state.status = WorkflowStatus.FAILED
                workflow_state.updated_at = utc_now()
                raise RuntimeError("Workflow deadlocked with no READY nodes and no terminal failure recorded.")

            dispatch_bundles: list[_DispatchBundle] = []
            for node in ready_nodes:
                try:
                    bundle = self._build_execution_request(
                        workflow_state=workflow_state,
                        plan=plan,
                        node=node,
                        execution_logs=execution_logs,
                    )
                except Exception:
                    self._mark_node_failed(workflow_state=workflow_state, node_id=node.node_id)
                    if self.config.fail_fast:
                        workflow_state.status = WorkflowStatus.FAILED
                        workflow_state.updated_at = utc_now()
                        raise
                    continue
                dispatch_bundles.append(bundle)

            if not dispatch_bundles:
                workflow_state.status = WorkflowStatus.FAILED
                workflow_state.updated_at = utc_now()
                return WorkflowRunResult(
                    workflow_state=workflow_state,
                    plan=plan,
                    execution_logs=list(execution_logs.values()),
                )

            if self.config.concurrent_ready_nodes:
                results = await asyncio.gather(
                    *[
                        self.execute_agent.execute(bundle.request, workflow_state)
                        for bundle in dispatch_bundles
                    ],
                    return_exceptions=True,
                )
            else:
                results = []
                for bundle in dispatch_bundles:
                    try:
                        results.append(await self.execute_agent.execute(bundle.request, workflow_state))
                    except Exception as exc:
                        results.append(exc)

            for bundle, result in zip(dispatch_bundles, results, strict=True):
                if isinstance(result, Exception):
                    self._mark_node_failed(workflow_state=workflow_state, node_id=bundle.node_id)
                    if self.config.fail_fast:
                        workflow_state.status = WorkflowStatus.FAILED
                        workflow_state.updated_at = utc_now()
                        raise result
                    continue

                execution_logs[result.execution_log_id] = result
                self._record_execution_log(
                    workflow_state=workflow_state,
                    node_id=bundle.node_id,
                    execution_log=result,
                )

                missing_required_outputs = self._missing_required_outputs(bundle.request, result)
                if result.status == AttemptStatus.COMPLETED and not missing_required_outputs:
                    self._backfill_execution_outputs(workflow_state=workflow_state, execution_log=result)
                    self._mark_node_success(workflow_state=workflow_state, node_id=bundle.node_id)
                else:
                    self._mark_node_failed(workflow_state=workflow_state, node_id=bundle.node_id)
                    if self.config.fail_fast:
                        workflow_state.status = WorkflowStatus.FAILED
                        workflow_state.updated_at = utc_now()
                        return WorkflowRunResult(
                            workflow_state=workflow_state,
                            plan=plan,
                            execution_logs=list(execution_logs.values()),
                        )

    def _register_input_artifacts(
        self,
        *,
        workflow_state: WorkflowState,
        artifact_registry: ArtifactRegistry,
        input_artifacts: list[OrchestratorInputArtifact],
    ) -> None:
        for item in input_artifacts:
            metadata = {
                "description": item.description,
                "labels": list(item.labels),
                "attributes": dict(item.attributes),
                "planner_role": item.input_role.value,
                "input_role": item.input_role.value,
                "is_user_selected_background": item.is_user_selected_background,
            }
            if item.coordinate_info is not None:
                metadata["coordinate_info"] = item.coordinate_info.model_dump()
            artifact = artifact_registry.register(
                workflow_id=workflow_state.workflow_id,
                artifact_type=item.artifact_type,
                value=item.uri_or_value,
                metadata=metadata,
                artifact_id=item.artifact_id,
            )
            workflow_state.artifact_manifests[item.artifact_id] = ArtifactManifest(
                artifact_id=artifact.artifact_id,
                artifact_type=artifact.artifact_type,
                uri_or_value=artifact.value,
                input_role=item.input_role,
                description=item.description,
                labels=list(item.labels),
                attributes=dict(item.attributes),
                is_user_selected_background=item.is_user_selected_background,
                producer_node_id=artifact.producer_node_id,
                producer_attempt_id=artifact.producer_attempt_id,
            )
        workflow_state.updated_at = utc_now()

    async def _run_pre_perception(
        self,
        *,
        workflow_state: WorkflowState,
        artifact_registry: ArtifactRegistry,
    ) -> None:
        understand_tool = self.tool_registry.get("image_understand")
        for artifact_id, manifest in workflow_state.artifact_manifests.items():
            if manifest.artifact_type != ArtifactType.IMAGE:
                continue
            ctx = ToolContext(
                llm=self.llm,
                artifact_registry=artifact_registry,
                workflow_id=workflow_state.workflow_id,
                node_id="node_pre_perception",
                attempt_id=f"understand_{artifact_id}",
                storage=self.storage,
                extras=dict(self.extras),
            )
            result = await understand_tool(
                ctx,
                {
                    "image_artifact_id": artifact_id,
                    "prompt": self.config.understand_prompt,
                    "output_spec_name": f"analysis_{artifact_id}",
                    "output_role": "analysis",
                },
            )
            binding = result.outputs[0]
            analysis_artifact = workflow_state.artifacts[binding.artifact_id]
            analysis_artifact.metadata["output_role"] = binding.role
            analysis_artifact.metadata["output_spec_name"] = binding.spec_name
            workflow_state.artifact_summaries[analysis_artifact.artifact_id] = (
                ArtifactSummary.from_analysis_artifact(analysis_artifact)
            )
        workflow_state.updated_at = utc_now()

    @staticmethod
    def _initialize_plan_state(*, workflow_state: WorkflowState, plan: DAGPlan) -> None:
        workflow_state.active_plan_id = plan.plan_id
        workflow_state.active_plan_version = plan.version
        workflow_state.status = WorkflowStatus.RUNNING
        workflow_state.nodes = {
            node.node_id: NodeState(status=NodeStatus.PENDING)
            for node in plan.nodes
        }
        workflow_state.updated_at = utc_now()

    @staticmethod
    def _refresh_node_statuses(*, workflow_state: WorkflowState, plan: DAGPlan) -> None:
        for node in plan.nodes:
            state = workflow_state.nodes[node.node_id]
            if state.status in {NodeStatus.SUCCESS, NodeStatus.FAILED, NodeStatus.RUNNING}:
                continue
            blocked_by = [
                dependency_id
                for dependency_id in node.dependencies
                if workflow_state.nodes[dependency_id].status != NodeStatus.SUCCESS
            ]
            state.blocked_by = blocked_by
            state.status = NodeStatus.BLOCKED if blocked_by else NodeStatus.READY
            state.updated_at = utc_now()
        workflow_state.updated_at = utc_now()

    def _build_execution_request(
        self,
        *,
        workflow_state: WorkflowState,
        plan: DAGPlan,
        node: TaskNode,
        execution_logs: dict[str, ExecutionLog],
    ) -> _DispatchBundle:
        node_state = workflow_state.nodes[node.node_id]
        next_attempt = node_state.attempts + 1
        attempt_id = f"{node.node_id}_attempt_{next_attempt}"
        bindings = self._bind_input_slots(
            workflow_state=workflow_state,
            plan=plan,
            node=node,
            execution_logs=execution_logs,
        )
        request = ExecutionRequest(
            workflow_id=workflow_state.workflow_id,
            plan_id=plan.plan_id,
            plan_version=plan.version,
            node_id=node.node_id,
            node_kind=node.kind,
            attempt_id=attempt_id,
            objective=node.goal,
            inputs=bindings,
            output_slots=node.outputs,
            success=node.success,
            allowed_tools=node.allowed_tools,
        )
        node_state.status = NodeStatus.RUNNING
        node_state.attempts = next_attempt
        node_state.last_attempt_id = attempt_id
        node_state.blocked_by = []
        node_state.updated_at = utc_now()
        workflow_state.updated_at = utc_now()
        return _DispatchBundle(node_id=node.node_id, request=request)

    def _bind_input_slots(
        self,
        *,
        workflow_state: WorkflowState,
        plan: DAGPlan,
        node: TaskNode,
        execution_logs: dict[str, ExecutionLog],
    ) -> list[ArtifactBinding]:
        bindings: list[ArtifactBinding] = []
        for slot in node.inputs:
            matched_artifact_ids = self._match_dependency_artifacts(
                workflow_state=workflow_state,
                node=node,
                slot=slot,
                execution_logs=execution_logs,
            )
            if not matched_artifact_ids:
                matched_artifact_ids = self._match_global_artifacts(
                    workflow_state=workflow_state,
                    slot=slot,
                )

            if slot.multiple:
                unique_ids: list[str] = []
                for artifact_id in matched_artifact_ids:
                    if artifact_id not in unique_ids:
                        unique_ids.append(artifact_id)
                if slot.required and not unique_ids:
                    raise ValueError(
                        f"Could not bind required multi slot '{slot.name}' (role={slot.role}) for node '{node.node_id}'"
                    )
                bindings.extend(
                    ArtifactBinding(spec_name=slot.name, role=slot.role, artifact_id=artifact_id)
                    for artifact_id in unique_ids
                )
            else:
                artifact_id = matched_artifact_ids[0] if matched_artifact_ids else None
                if slot.required and artifact_id is None:
                    raise ValueError(
                        f"Could not bind required slot '{slot.name}' (role={slot.role}) for node '{node.node_id}'"
                    )
                if artifact_id is not None:
                    bindings.append(
                        ArtifactBinding(spec_name=slot.name, role=slot.role, artifact_id=artifact_id)
                    )
        return bindings

    @staticmethod
    def _match_dependency_artifacts(
        *,
        workflow_state: WorkflowState,
        node: TaskNode,
        slot: SlotSpec,
        execution_logs: dict[str, ExecutionLog],
    ) -> list[str]:
        matched: list[str] = []
        for dependency_id in reversed(node.dependencies):
            dependency_state = workflow_state.nodes[dependency_id]
            if not dependency_state.history_logs:
                continue
            execution_log = execution_logs.get(dependency_state.history_logs[-1])
            if execution_log is None:
                continue
            for output in execution_log.outputs:
                if output.role == slot.role and output.artifact_id in workflow_state.artifacts:
                    matched.append(output.artifact_id)
                    if not slot.multiple:
                        return matched
        return matched

    def _match_global_artifacts(
        self,
        *,
        workflow_state: WorkflowState,
        slot: SlotSpec,
    ) -> list[str]:
        matched: list[str] = []
        for artifact in sorted(
            workflow_state.artifacts.values(),
            key=lambda item: item.created_at,
            reverse=True,
        ):
            if self._artifact_matches_role(workflow_state=workflow_state, artifact=artifact, role=slot.role):
                matched.append(artifact.artifact_id)
                if not slot.multiple:
                    break
        return matched

    @staticmethod
    def _artifact_matches_role(
        *,
        workflow_state: WorkflowState,
        artifact: Artifact,
        role: str,
    ) -> bool:
        metadata = artifact.metadata or {}
        if metadata.get("planner_role") == role:
            return True
        if metadata.get("output_role") == role:
            return True
        if artifact.artifact_type == ArtifactType.ANALYSIS and role == "analysis":
            return True
        manifest = workflow_state.artifact_manifests.get(artifact.artifact_id)
        if manifest is not None and manifest.input_role.value == role:
            return True
        if (
            role == "background_image"
            and manifest is not None
            and manifest.is_user_selected_background
            and artifact.artifact_type == ArtifactType.IMAGE
        ):
            return True
        return False

    @staticmethod
    def _record_execution_log(
        *,
        workflow_state: WorkflowState,
        node_id: str,
        execution_log: ExecutionLog,
    ) -> None:
        node_state = workflow_state.nodes[node_id]
        node_state.history_logs.append(execution_log.execution_log_id)
        node_state.updated_at = utc_now()
        workflow_state.updated_at = utc_now()

    @staticmethod
    def _backfill_execution_outputs(
        *,
        workflow_state: WorkflowState,
        execution_log: ExecutionLog,
    ) -> None:
        for binding in execution_log.outputs:
            try:
                artifact = workflow_state.artifacts[binding.artifact_id]
            except KeyError as exc:
                raise KeyError(
                    f"Execution output '{binding.artifact_id}' was not registered in the workflow artifact store"
                ) from exc
            artifact.metadata["output_role"] = binding.role
            artifact.metadata["output_spec_name"] = binding.spec_name
        workflow_state.updated_at = utc_now()

    @staticmethod
    def _missing_required_outputs(request: ExecutionRequest, execution_log: ExecutionLog) -> list[str]:
        produced = {binding.spec_name for binding in execution_log.outputs}
        return [
            spec_name for spec_name in request.success.required_outputs if spec_name not in produced
        ]

    @staticmethod
    def _mark_node_success(*, workflow_state: WorkflowState, node_id: str) -> None:
        node_state = workflow_state.nodes[node_id]
        node_state.status = NodeStatus.SUCCESS
        node_state.blocked_by = []
        node_state.updated_at = utc_now()
        workflow_state.updated_at = utc_now()

    @staticmethod
    def _mark_node_failed(*, workflow_state: WorkflowState, node_id: str) -> None:
        node_state = workflow_state.nodes[node_id]
        node_state.status = NodeStatus.FAILED
        node_state.updated_at = utc_now()
        workflow_state.updated_at = utc_now()


__all__ = [
    "Orchestrator",
    "OrchestratorConfig",
    "OrchestratorInputArtifact",
    "WorkflowRunResult",
]
