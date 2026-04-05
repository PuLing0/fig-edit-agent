"""Global enums used across planning, execution, feedback, and workflow state."""

from __future__ import annotations

from enum import Enum


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    PLANNING = "planning"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    ABORTED = "aborted"


class NodeStatus(str, Enum):
    PENDING = "pending"
    BLOCKED = "blocked"
    READY = "ready"
    RUNNING = "running"
    EVALUATING = "evaluating"
    RETRYING = "retrying"
    SUCCESS = "success"
    FAILED = "failed"
    ESCALATED = "escalated"


class NodeKind(str, Enum):
    EDIT = "edit"
    SEGMENT = "segment"
    CROP = "crop"
    COMPOSE = "compose"
    UNDERSTAND = "understand"
    REWRITE_PROMPT = "rewrite_prompt"
    OCR = "ocr"
    SCORE = "score"
    CUSTOM = "custom"


class AttemptStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class EvaluationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"


class EvaluationDecision(str, Enum):
    ACCEPT = "accept"
    RETRY = "retry"
    REPLAN = "replan"
    ABORT = "abort"


class FailureType(str, Enum):
    TOOL_ERROR = "tool_error"
    QUALITY_ISSUE = "quality_issue"
    LOGIC_ERROR = "logic_error"
    TIMEOUT = "timeout"
    DEPENDENCY_FAIL = "dependency_fail"


class ReplanScope(str, Enum):
    NODE = "node"
    SUBGRAPH = "subgraph"
    WORKFLOW = "workflow"


class ArtifactType(str, Enum):
    IMAGE = "image"
    MASK = "mask"
    POINTS = "points"
    TEXT = "text"
    PROMPT = "prompt"
    SCORE = "score"
    ANALYSIS = "analysis"


__all__ = [
    "ArtifactType",
    "AttemptStatus",
    "EvaluationDecision",
    "EvaluationStatus",
    "FailureType",
    "NodeKind",
    "NodeStatus",
    "ReplanScope",
    "WorkflowStatus",
]
