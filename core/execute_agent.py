"""LLM-driven ReAct executor for DAG task nodes."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import Field, model_validator

from ..schemas import (
    ActionStep,
    Artifact,
    ArtifactBinding,
    AttemptStatus,
    ExecutionLog,
    ExecutionRequest,
    SlotSpec,
    StrictSchema,
    WorkflowState,
    utc_now,
)
from ..tools.base import ArtifactRegistry, ToolContext, ToolResult
from ..tools.registry import (
    ToolRegistry,
    ensure_builtin_tools_registered,
    tool_registry as default_tool_registry,
)
from .llm_client import LLMClient


FINISH_SENTINEL_TOOL_NAME = "__finish__"


@dataclass(slots=True)
class ExecuteAgentConfig:
    """Runtime knobs controlling executor behavior."""

    max_steps: int = 6
    allow_all_tools_when_empty: bool = True
    record_failed_tool_steps: bool = True
    tool_timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.tool_timeout_seconds is not None and self.tool_timeout_seconds <= 0:
            raise ValueError("tool_timeout_seconds must be positive when provided")


class _ExecutorDecision(StrictSchema):
    """One structured ReAct decision emitted by the executor model."""

    thought: str = Field(description="Short external rationale, not private chain-of-thought.")
    action_type: Literal["tool", "finish"] = Field(description="Whether to call a tool or stop.")
    tool_name: str | None = Field(default=None, description="Tool name when action_type == 'tool'.")
    tool_args_json: str = Field(
        default="{}",
        description="Tool arguments encoded as a JSON object string.",
    )
    target_output_slot: str | None = Field(
        default=None,
        description="Optional final output slot name from the execution request.",
    )
    output_spec_name: str | None = Field(
        default=None,
        description="Explicit output slot name for intermediate artifacts when not targeting a final slot.",
    )
    output_role: str | None = Field(
        default=None,
        description="Explicit output role for intermediate artifacts when not targeting a final slot.",
    )
    finish_summary: str | None = Field(default=None, description="Optional final summary when action_type == 'finish'.")

    @model_validator(mode="after")
    def validate_decision(self) -> "_ExecutorDecision":
        if not self.thought.strip():
            raise ValueError("thought must be non-empty")
        parsed_tool_args = self.parse_tool_args()
        if self.action_type == "tool":
            if self.tool_name is None or not self.tool_name.strip():
                raise ValueError("tool_name is required when action_type == 'tool'")
            if self.finish_summary is not None:
                raise ValueError("finish_summary must be omitted when action_type == 'tool'")
        else:
            if self.tool_name is not None:
                raise ValueError("tool_name must be omitted when action_type == 'finish'")
            if parsed_tool_args:
                raise ValueError("tool_args must be empty when action_type == 'finish'")
            if self.target_output_slot is not None or self.output_spec_name is not None or self.output_role is not None:
                raise ValueError("finish actions cannot set output bindings")
        return self

    def parse_tool_args(self) -> dict[str, Any]:
        """Decode tool_args_json into a Python dictionary."""

        try:
            parsed = json.loads(self.tool_args_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"tool_args_json must be valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("tool_args_json must decode into a JSON object")
        return parsed


@dataclass(slots=True)
class _PreparedToolCall:
    """Validated tool invocation ready for execution."""

    tool_name: str
    tool_args: dict[str, Any]


class ExecuteAgent:
    """Run one task node by repeatedly deciding and acting with tools."""

    def __init__(
        self,
        *,
        llm: LLMClient | None = None,
        tool_registry: ToolRegistry | None = None,
        config: ExecuteAgentConfig | None = None,
        storage: Any | None = None,
        extras: dict[str, Any] | None = None,
    ) -> None:
        self.llm = llm or LLMClient()
        self.tool_registry = tool_registry or default_tool_registry
        if self.tool_registry is default_tool_registry:
            ensure_builtin_tools_registered()
        self.config = config or ExecuteAgentConfig()
        self.storage = storage
        self.extras = dict(extras or {})

    async def execute(self, request: ExecutionRequest, workflow_state: WorkflowState) -> ExecutionLog:
        """Execute one task attempt and return a complete execution log."""

        artifact_registry = ArtifactRegistry(workflow_state.artifacts)
        log = ExecutionLog(
            execution_log_id=f"log_{request.attempt_id}",
            workflow_id=request.workflow_id,
            plan_id=request.plan_id,
            plan_version=request.plan_version,
            node_id=request.node_id,
            attempt_id=request.attempt_id,
            status=AttemptStatus.RUNNING,
            started_at=utc_now(),
        )

        output_slots_by_name = {slot.name: slot for slot in request.output_slots}
        final_outputs: dict[str, ArtifactBinding] = {}

        allowed_tools_or_error = self._resolve_allowed_tools(request)
        if isinstance(allowed_tools_or_error, str):
            return self._finish_log(
                log,
                status=AttemptStatus.FAILED,
                final_outputs=final_outputs,
                summary=allowed_tools_or_error,
            )
        allowed_tool_names = allowed_tools_or_error

        visible_artifact_ids, missing_inputs = self._resolve_initial_visible_artifacts(
            request=request,
            artifact_registry=artifact_registry,
        )
        if missing_inputs:
            return self._finish_log(
                log,
                status=AttemptStatus.FAILED,
                final_outputs=final_outputs,
                summary=(
                    "Execution request referenced missing input artifacts: "
                    + ", ".join(missing_inputs)
                ),
            )

        for step_index in range(1, self.config.max_steps + 1):
            step_started_at = utc_now()
            try:
                decision = await self._decide_next_action(
                    request=request,
                    artifact_registry=artifact_registry,
                    visible_artifact_ids=visible_artifact_ids,
                    allowed_tool_names=allowed_tool_names,
                    final_outputs=final_outputs,
                    steps=log.steps,
                )
            except Exception as exc:
                return self._finish_log(
                    log,
                    status=AttemptStatus.FAILED,
                    final_outputs=final_outputs,
                    summary=(
                        "Executor could not obtain a valid ReAct decision: "
                        f"{exc.__class__.__name__}: {exc}"
                    ),
                )

            if decision.action_type == "finish":
                missing_outputs = self._missing_required_outputs(request, final_outputs)
                observation = self._format_finish_observation(
                    missing_outputs=missing_outputs,
                    final_outputs=final_outputs,
                )
                log.steps.append(
                    ActionStep(
                        step_index=step_index,
                        thought=decision.thought.strip(),
                        tool_name=FINISH_SENTINEL_TOOL_NAME,
                        tool_args={},
                        observation=observation,
                        started_at=step_started_at,
                        finished_at=utc_now(),
                    )
                )
                if not missing_outputs:
                    summary = decision.finish_summary or "Executor declared the node outputs ready for evaluation."
                    return self._finish_log(
                        log,
                        status=AttemptStatus.COMPLETED,
                        final_outputs=final_outputs,
                        summary=summary,
                    )
                continue

            prepared_call_or_error = self._prepare_tool_call(
                decision=decision,
                request=request,
                allowed_tool_names=allowed_tool_names,
                output_slots_by_name=output_slots_by_name,
            )
            if isinstance(prepared_call_or_error, str):
                if self.config.record_failed_tool_steps:
                    log.steps.append(
                        ActionStep(
                            step_index=step_index,
                            thought=decision.thought.strip(),
                            tool_name=decision.tool_name or "<missing_tool_name>",
                            tool_args=decision.parse_tool_args(),
                            observation=prepared_call_or_error,
                            started_at=step_started_at,
                            finished_at=utc_now(),
                        )
                    )
                continue
            prepared_call = prepared_call_or_error

            try:
                result = await self._run_tool(
                    prepared_call=prepared_call,
                    request=request,
                    artifact_registry=artifact_registry,
                    visible_artifact_ids=visible_artifact_ids,
                    step_index=step_index,
                )
            except Exception as exc:
                if self.config.record_failed_tool_steps:
                    log.steps.append(
                        ActionStep(
                            step_index=step_index,
                            thought=decision.thought.strip(),
                            tool_name=prepared_call.tool_name,
                            tool_args=prepared_call.tool_args,
                            observation=(
                                f"Tool '{prepared_call.tool_name}' failed: "
                                f"{exc.__class__.__name__}: {exc}"
                            ),
                            started_at=step_started_at,
                            finished_at=utc_now(),
                        )
                    )
                continue

            self._merge_tool_outputs(
                request=request,
                visible_artifact_ids=visible_artifact_ids,
                final_outputs=final_outputs,
                tool_result=result,
            )
            log.steps.append(
                ActionStep(
                    step_index=step_index,
                    thought=decision.thought.strip(),
                    tool_name=prepared_call.tool_name,
                    tool_args=prepared_call.tool_args,
                    observation=self._format_tool_observation(result, artifact_registry),
                    started_at=step_started_at,
                    finished_at=utc_now(),
                )
            )

            if step_index == self.config.max_steps and not self._missing_required_outputs(request, final_outputs):
                return self._finish_log(
                    log,
                    status=AttemptStatus.COMPLETED,
                    final_outputs=final_outputs,
                    summary="Executor reached the step budget after producing all required outputs.",
                )

        missing_outputs = self._missing_required_outputs(request, final_outputs)
        if not missing_outputs:
            return self._finish_log(
                log,
                status=AttemptStatus.COMPLETED,
                final_outputs=final_outputs,
                summary="Executor produced the required outputs before the loop ended.",
            )
        return self._finish_log(
            log,
            status=AttemptStatus.FAILED,
            final_outputs=final_outputs,
            summary=(
                "Executor exhausted its step budget without producing all required outputs. "
                f"Missing: {', '.join(missing_outputs)}"
            ),
        )

    def _resolve_allowed_tools(self, request: ExecutionRequest) -> list[str] | str:
        if request.allowed_tools:
            unknown_tools = [name for name in request.allowed_tools if not self.tool_registry.has(name)]
            if unknown_tools:
                return "Execution request referenced unknown allowed tools: " + ", ".join(sorted(unknown_tools))
            return list(request.allowed_tools)

        if not self.config.allow_all_tools_when_empty:
            return "Execution request did not declare allowed tools and the executor is configured to require them."

        return self.tool_registry.list_names()

    @staticmethod
    def _resolve_initial_visible_artifacts(
        *,
        request: ExecutionRequest,
        artifact_registry: ArtifactRegistry,
    ) -> tuple[list[str], list[str]]:
        visible_artifact_ids: list[str] = []
        missing_inputs: list[str] = []
        for binding in request.inputs:
            if artifact_registry.exists(binding.artifact_id):
                if binding.artifact_id not in visible_artifact_ids:
                    visible_artifact_ids.append(binding.artifact_id)
            else:
                missing_inputs.append(binding.artifact_id)
        return visible_artifact_ids, missing_inputs

    async def _decide_next_action(
        self,
        *,
        request: ExecutionRequest,
        artifact_registry: ArtifactRegistry,
        visible_artifact_ids: list[str],
        allowed_tool_names: list[str],
        final_outputs: dict[str, ArtifactBinding],
        steps: list[ActionStep],
    ) -> _ExecutorDecision:
        messages = self._build_react_messages(
            request=request,
            artifact_registry=artifact_registry,
            visible_artifact_ids=visible_artifact_ids,
            allowed_tool_names=allowed_tool_names,
            final_outputs=final_outputs,
            steps=steps,
        )
        return await self.llm.generate_structured(
            messages=messages,
            response_model=_ExecutorDecision,
            task_name="execute_agent",
            temperature=0.0,
        )

    def _build_react_messages(
        self,
        *,
        request: ExecutionRequest,
        artifact_registry: ArtifactRegistry,
        visible_artifact_ids: list[str],
        allowed_tool_names: list[str],
        final_outputs: dict[str, ArtifactBinding],
        steps: list[ActionStep],
    ) -> list[dict[str, Any]]:
        required_outputs = request.success.required_outputs
        output_slot_lines = [self._format_output_slot(slot) for slot in request.output_slots] or ["- none declared"]
        final_output_lines = [
            f"- {binding.spec_name}: artifact_id={binding.artifact_id}, role={binding.role}"
            for slot_name in required_outputs
            if (binding := final_outputs.get(slot_name)) is not None
        ]
        if not final_output_lines:
            final_output_lines = ["- none yet"]

        input_lines = [
            f"- {binding.spec_name}: artifact_id={binding.artifact_id}, role={binding.role}"
            for binding in request.inputs
        ] or ["- none"]

        visible_artifact_lines = [
            "- " + self._summarize_artifact(artifact_registry.get(artifact_id))
            for artifact_id in visible_artifact_ids
        ] or ["- none"]

        step_lines = [
            (
                f"- step {step.step_index}: tool={step.tool_name}; "
                f"args={json.dumps(step.tool_args, ensure_ascii=False, sort_keys=True)}; "
                f"observation={step.observation}"
            )
            for step in steps
        ] or ["- none"]

        tool_lines = [self._describe_tool(name) for name in allowed_tool_names] or ["- none"]

        retry_lines = self._format_retry_advice(request)
        success_lines = self._format_success_criteria(request)

        user_prompt = "\n\n".join(
            [
                f"Objective:\n{request.objective}",
                f"Node kind:\n{request.node_kind.value}",
                "Input bindings:\n" + "\n".join(input_lines),
                "Declared final output slots:\n" + "\n".join(output_slot_lines),
                "Required output status:\n" + "\n".join(final_output_lines),
                "Success criteria:\n" + "\n".join(success_lines),
                "Retry context:\n" + "\n".join(retry_lines),
                "Visible artifacts right now:\n" + "\n".join(visible_artifact_lines),
                "Previous steps:\n" + "\n".join(step_lines),
                "Allowed tools:\n" + "\n".join(tool_lines),
            ]
        )

        system_prompt = (
            "You are the Execute Agent in a ReAct-style workflow. "
            "Think briefly, choose exactly one next action, observe the result, and repeat.\n\n"
            "Rules:\n"
            "1. Use only allowed tools.\n"
            "2. Reference only visible artifact ids.\n"
            "3. Keep 'thought' to 1-3 short sentences of external rationale. Do not expose private chain-of-thought.\n"
            "4. Prefer the fewest tool steps necessary.\n"
            "5. Use action_type='finish' only when the required outputs are already produced.\n"
            "6. If you are creating a declared final node output, set target_output_slot to that exact slot name.\n"
            "7. If you need an intermediate artifact, you may leave target_output_slot empty and instead set "
            "output_spec_name/output_role for scratch outputs, or omit them to use tool defaults.\n"
            "8. tool_args_json must be a valid JSON object string. Example: "
            '{"image_artifact_id":"artifact_1","prompt":"describe the image"}.\n'
            "9. When a previous observation shows an error, correct the next action instead of repeating it unchanged.\n"
            "10. Never invent artifacts, tool names, or arguments that are not supported."
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _prepare_tool_call(
        self,
        *,
        decision: _ExecutorDecision,
        request: ExecutionRequest,
        allowed_tool_names: list[str],
        output_slots_by_name: dict[str, SlotSpec],
    ) -> _PreparedToolCall | str:
        assert decision.tool_name is not None
        if decision.tool_name not in allowed_tool_names:
            return f"Tool '{decision.tool_name}' is not in the allowed tool list."
        if not self.tool_registry.has(decision.tool_name):
            return f"Tool '{decision.tool_name}' is not registered."

        tool_args = decision.parse_tool_args()
        if decision.target_output_slot is not None:
            slot = output_slots_by_name.get(decision.target_output_slot)
            if slot is None:
                return (
                    f"target_output_slot '{decision.target_output_slot}' is not a declared final output slot."
                )
            tool_args["output_spec_name"] = slot.name
            tool_args["output_role"] = slot.role
        else:
            if decision.output_spec_name is not None:
                tool_args["output_spec_name"] = decision.output_spec_name
            if decision.output_role is not None:
                tool_args["output_role"] = decision.output_role

        tool = self.tool_registry.get(decision.tool_name)
        try:
            tool.args_model.model_validate(tool_args)
        except Exception as exc:
            return (
                f"Arguments for tool '{decision.tool_name}' failed validation: "
                f"{exc.__class__.__name__}: {exc}"
            )

        return _PreparedToolCall(tool_name=decision.tool_name, tool_args=tool_args)

    async def _run_tool(
        self,
        *,
        prepared_call: _PreparedToolCall,
        request: ExecutionRequest,
        artifact_registry: ArtifactRegistry,
        visible_artifact_ids: list[str],
        step_index: int,
    ) -> ToolResult:
        ctx = ToolContext(
            llm=self.llm,
            artifact_registry=artifact_registry,
            workflow_id=request.workflow_id,
            node_id=request.node_id,
            attempt_id=request.attempt_id,
            storage=self.storage,
            extras={
                **self.extras,
                "execution_request": request.model_dump(mode="json"),
                "visible_artifact_ids": list(visible_artifact_ids),
                "step_index": step_index,
            },
        )
        coro = self.tool_registry.run(prepared_call.tool_name, ctx, prepared_call.tool_args)
        if self.config.tool_timeout_seconds is None:
            return await coro
        return await asyncio.wait_for(coro, timeout=self.config.tool_timeout_seconds)

    @staticmethod
    def _merge_tool_outputs(
        *,
        request: ExecutionRequest,
        visible_artifact_ids: list[str],
        final_outputs: dict[str, ArtifactBinding],
        tool_result: ToolResult,
    ) -> None:
        final_slot_names = {slot.name for slot in request.output_slots}
        for binding in tool_result.outputs:
            if binding.artifact_id not in visible_artifact_ids:
                visible_artifact_ids.append(binding.artifact_id)
            if binding.spec_name in final_slot_names:
                final_outputs[binding.spec_name] = binding

    @staticmethod
    def _format_tool_observation(tool_result: ToolResult, artifact_registry: ArtifactRegistry) -> str:
        output_lines: list[str] = []
        for binding in tool_result.outputs:
            artifact = artifact_registry.get(binding.artifact_id)
            output_lines.append(
                f"{binding.spec_name} -> {binding.artifact_id} "
                f"(role={binding.role}, type={artifact.artifact_type.value})"
            )
        outputs_text = "; ".join(output_lines) if output_lines else "no outputs"
        return f"{tool_result.summary} Outputs: {outputs_text}."

    @staticmethod
    def _format_finish_observation(
        *,
        missing_outputs: list[str],
        final_outputs: dict[str, ArtifactBinding],
    ) -> str:
        if missing_outputs:
            return "Finish rejected because required outputs are still missing: " + ", ".join(missing_outputs)
        produced = ", ".join(
            f"{binding.spec_name}={binding.artifact_id}" for binding in final_outputs.values()
        ) or "none"
        return "Finish accepted. Required outputs are available: " + produced

    @staticmethod
    def _missing_required_outputs(
        request: ExecutionRequest,
        final_outputs: dict[str, ArtifactBinding],
    ) -> list[str]:
        return [slot_name for slot_name in request.success.required_outputs if slot_name not in final_outputs]

    @staticmethod
    def _finish_log(
        log: ExecutionLog,
        *,
        status: AttemptStatus,
        final_outputs: dict[str, ArtifactBinding],
        summary: str,
    ) -> ExecutionLog:
        log.status = status
        log.outputs = list(final_outputs.values())
        log.final_summary = summary
        log.finished_at = utc_now()
        return log

    @staticmethod
    def _format_output_slot(slot: SlotSpec) -> str:
        suffix = []
        if slot.required:
            suffix.append("required")
        if slot.multiple:
            suffix.append("multiple")
        suffix_text = f" ({', '.join(suffix)})" if suffix else ""
        return f"- {slot.name}: role={slot.role}{suffix_text}"

    @staticmethod
    def _format_success_criteria(request: ExecutionRequest) -> list[str]:
        lines: list[str] = []
        if request.success.required_outputs:
            lines.append("- required outputs: " + ", ".join(request.success.required_outputs))
        else:
            lines.append("- required outputs: none")
        if request.success.hard_rules:
            lines.extend(f"- hard rule: {rule}" for rule in request.success.hard_rules)
        else:
            lines.append("- hard rules: none")
        if request.success.score_thresholds:
            lines.extend(
                f"- score threshold: {threshold.name} >= {threshold.threshold:.2f}"
                for threshold in request.success.score_thresholds
            )
        else:
            lines.append("- score thresholds: none")
        return lines

    @staticmethod
    def _format_retry_advice(request: ExecutionRequest) -> list[str]:
        if request.retry_advice is None:
            return ["- none"]
        advice = request.retry_advice
        lines = [f"- diagnostic: {advice.error_diagnostic}"]
        if advice.suggested_prompts:
            lines.extend(f"- suggested prompt: {item}" for item in advice.suggested_prompts)
        if advice.avoid_tools:
            lines.extend(f"- avoid tool: {item}" for item in advice.avoid_tools)
        if advice.extra_constraints:
            lines.extend(f"- extra constraint: {item}" for item in advice.extra_constraints)
        if advice.parameter_adjustments:
            lines.append(
                "- parameter adjustments: "
                + json.dumps(advice.parameter_adjustments, ensure_ascii=False, sort_keys=True)
            )
        return lines

    def _describe_tool(self, name: str) -> str:
        tool = self.tool_registry.get(name)
        field_descriptions: list[str] = []
        for field_name, field_info in tool.args_model.model_fields.items():
            description = field_info.description or ""
            default = "" if field_info.is_required() else f", default={field_info.default!r}"
            field_descriptions.append(f"{field_name} ({description}{default})".strip())
        args_text = "; ".join(field_descriptions) if field_descriptions else "no args"
        return f"- {tool.name}: {tool.description} Args: {args_text}"

    @staticmethod
    def _summarize_artifact(artifact: Artifact) -> str:
        metadata = artifact.metadata or {}
        description = metadata.get("description")
        coord = metadata.get("coordinate_info")
        pieces = [
            f"id={artifact.artifact_id}",
            f"type={artifact.artifact_type.value}",
        ]
        if description:
            pieces.append(f"description={description}")
        if artifact.artifact_type.value in {"text", "prompt"}:
            pieces.append(f"value={artifact.value}")
        if isinstance(artifact.value, dict):
            pieces.append(f"value_keys={sorted(artifact.value.keys())}")
        if isinstance(coord, dict):
            pieces.append(f"size={coord.get('width', '?')}x{coord.get('height', '?')}")
        return "; ".join(pieces)


__all__ = ["ExecuteAgent", "ExecuteAgentConfig"]
