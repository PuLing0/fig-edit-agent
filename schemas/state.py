"""Workflow runtime state maintained by the orchestrator."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from .artifact import Artifact
from .base import Identifier, StrictSchema, utc_now
from .enums import NodeStatus, WorkflowStatus


class NodeState(StrictSchema):
    """Mutable runtime snapshot for a single node."""

    status: NodeStatus = Field(default=NodeStatus.PENDING, description="Current runtime status of this node.")
    attempts: int = Field(default=0, ge=0, description="How many attempts have been made for this node.")
    blocked_by: list[Identifier] = Field(
        default_factory=list,
        description="Dependency node ids currently blocking this node.",
    )
    history_logs: list[Identifier] = Field(
        default_factory=list,
        description="Execution log ids produced by this node across attempts.",
    )
    last_attempt_id: Identifier | None = Field(default=None, description="Most recent attempt id for this node.")
    last_evaluation_id: Identifier | None = Field(default=None, description="Most recent evaluation result id.")
    last_advice_id: Identifier | None = Field(default=None, description="Most recent retry advice id, if any.")
    updated_at: datetime = Field(default_factory=utc_now, description="UTC timestamp of the latest node-state update.")


class WorkflowState(StrictSchema):
    """Global runtime container maintained by the orchestrator."""

    workflow_id: Identifier = Field(description="Unique identifier of the workflow.")
    status: WorkflowStatus = Field(default=WorkflowStatus.PENDING, description="Current high-level workflow status.")
    active_plan_id: Identifier = Field(description="Currently active plan id.")
    active_plan_version: int = Field(ge=1, description="Currently active plan version.")
    nodes: dict[Identifier, NodeState] = Field(default_factory=dict, description="Runtime node states keyed by node id.")
    artifacts: dict[Identifier, Artifact] = Field(default_factory=dict, description="Artifact registry keyed by artifact id.")
    replan_count: int = Field(default=0, ge=0, description="How many replans have been performed so far.")
    created_at: datetime = Field(default_factory=utc_now, description="UTC timestamp when the workflow state was created.")
    updated_at: datetime = Field(default_factory=utc_now, description="UTC timestamp of the latest workflow update.")


__all__ = ["NodeState", "WorkflowState"]
