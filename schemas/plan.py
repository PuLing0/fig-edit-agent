"""Planning-time schemas: slot contracts, nodes, and DAG plans."""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime

from pydantic import Field, PositiveInt, model_validator

from .base import Identifier, NonEmptyStr, StrictSchema, utc_now
from .enums import NodeKind


class SlotSpec(StrictSchema):
    """Logical input or output slot inside a task node."""

    name: NonEmptyStr = Field(description="Stable slot name used for exact runtime binding.")
    role: NonEmptyStr = Field(description="Semantic role of the slot, such as base_image or subject_mask.")
    required: bool = Field(default=True, description="Whether this slot must be bound for successful execution.")
    multiple: bool = Field(default=False, description="Whether this slot accepts multiple bound artifacts.")


class ScoreThreshold(StrictSchema):
    """Named score threshold used inside SuccessCriteria."""

    name: NonEmptyStr = Field(description="Score dimension name, such as fidelity or aesthetics.")
    threshold: float = Field(description="Minimum acceptable score in the inclusive range [0, 1].")

    @model_validator(mode="after")
    def validate_threshold(self) -> "ScoreThreshold":
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be within [0, 1]")
        return self


class SuccessCriteria(StrictSchema):
    """Evaluator-facing acceptance contract for a task node."""

    hard_rules: list[NonEmptyStr] = Field(default_factory=list, description="Hard rules that must all pass.")
    score_thresholds: list[ScoreThreshold] = Field(
        default_factory=list,
        description="Named score thresholds represented as a list for better LLM structured-output compatibility.",
    )
    required_outputs: list[NonEmptyStr] = Field(
        default_factory=list,
        description="Output slot names that must be produced by the node.",
    )

    @model_validator(mode="after")
    def validate_criteria(self) -> "SuccessCriteria":
        if not (self.hard_rules or self.score_thresholds or self.required_outputs):
            raise ValueError("success criteria must define at least one acceptance condition")
        score_names = [threshold.name for threshold in self.score_thresholds]
        if len(score_names) != len(set(score_names)):
            raise ValueError("score threshold names must be unique")
        return self


class TaskNode(StrictSchema):
    """Immutable planning-time node in the DAG."""

    node_id: Identifier = Field(description="Unique identifier of the node within the plan.")
    kind: NodeKind = Field(default=NodeKind.CUSTOM, description="Canonical task kind for this node.")
    goal: NonEmptyStr = Field(description="Objective the node should accomplish.")
    dependencies: list[Identifier] = Field(
        default_factory=list,
        description="Upstream node ids that must succeed before this node may run.",
    )
    inputs: list[SlotSpec] = Field(default_factory=list, description="Logical input slots.")
    outputs: list[SlotSpec] = Field(default_factory=list, description="Logical output slots.")
    allowed_tools: list[NonEmptyStr] = Field(
        default_factory=list,
        description="Static tool whitelist for this node.",
    )
    success: SuccessCriteria = Field(description="Acceptance contract for this node.")
    max_retries: PositiveInt = Field(default=3, description="Maximum local retry budget for this node.")
    escalate_after: PositiveInt = Field(
        default=2,
        description="Attempt count after which failure should escalate to replanning.",
    )

    @model_validator(mode="after")
    def validate_node(self) -> "TaskNode":
        if self.node_id in self.dependencies:
            raise ValueError("node cannot depend on itself")
        if len(self.dependencies) != len(set(self.dependencies)):
            raise ValueError("dependencies must be unique")
        output_names = [slot.name for slot in self.outputs]
        if len(output_names) != len(set(output_names)):
            raise ValueError("output slot names must be unique")
        missing_required_outputs = sorted(
            set(self.success.required_outputs).difference(output_names)
        )
        if missing_required_outputs:
            raise ValueError(
                "success.required_outputs contains undefined output slot names: "
                + ", ".join(missing_required_outputs)
            )
        if self.escalate_after > self.max_retries:
            raise ValueError("escalate_after must be <= max_retries")
        return self


class DAGPlan(StrictSchema):
    """Versioned immutable DAG produced by the planner."""

    workflow_id: Identifier = Field(description="Workflow that owns this plan.")
    plan_id: Identifier = Field(description="Unique identifier of this plan version.")
    version: PositiveInt = Field(default=1, description="Monotonic plan version number.")
    goal: NonEmptyStr = Field(description="Top-level goal summary for the plan.")
    nodes: list[TaskNode] = Field(min_length=1, description="All nodes contained in this DAG plan.")
    created_at: datetime = Field(default_factory=utc_now, description="UTC timestamp when the plan was created.")

    @model_validator(mode="after")
    def validate_dag(self) -> "DAGPlan":
        node_map = {node.node_id: node for node in self.nodes}
        if len(node_map) != len(self.nodes):
            raise ValueError("node_id values must be unique within a plan")

        indegree: dict[str, int] = {node_id: 0 for node_id in node_map}
        adjacency: dict[str, list[str]] = defaultdict(list)

        for node in self.nodes:
            for dep in node.dependencies:
                if dep not in node_map:
                    raise ValueError(f"dependency '{dep}' not found for node '{node.node_id}'")
                adjacency[dep].append(node.node_id)
                indegree[node.node_id] += 1

        queue = deque(node_id for node_id, degree in indegree.items() if degree == 0)
        visited = 0
        while queue:
            current = queue.popleft()
            visited += 1
            for child in adjacency[current]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)

        if visited != len(node_map):
            raise ValueError("DAG contains cycles")
        return self


__all__ = ["DAGPlan", "ScoreThreshold", "SlotSpec", "SuccessCriteria", "TaskNode"]
