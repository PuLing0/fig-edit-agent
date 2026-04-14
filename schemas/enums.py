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
    EXTRACT_SUBJECT = "extract_subject"
    SELECT_BACKGROUND = "select_background"
    COMPOSE_SCENE = "compose_scene"
    POLISH_IMAGE = "polish_image"
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


class InputArtifactRole(str, Enum):
    PRIMARY_INPUT = "primary_input"
    SUBJECT_SOURCE = "subject_source"
    BACKGROUND_CANDIDATE = "background_candidate"
    REFERENCE_ONLY = "reference_only"


class ArtifactRole(str, Enum):
    PRIMARY_INPUT = "primary_input"
    SUBJECT_SOURCE = "subject_source"
    BACKGROUND_CANDIDATE = "background_candidate"
    REFERENCE_ONLY = "reference_only"
    ANALYSIS = "analysis"
    IMAGE_SUMMARY = "image_summary"
    SCENE_UNDERSTANDING = "scene_understanding"
    SCORE = "score"
    SCORE_ARTIFACT = "score_artifact"
    EXTRACTED_SUBJECT = "extracted_subject"
    BACKGROUND_IMAGE = "background_image"
    COMPOSED_IMAGE = "composed_image"
    EDITED_IMAGE = "edited_image"
    FINAL_IMAGE = "final_image"
    GROUNDING_POINTS = "grounding_points"
    SUBJECT_MASK = "subject_mask"
    OCR_TEXT = "ocr_text"
    CROPPED_IMAGE = "cropped_image"
    CAPTION = "caption"
    TEXT = "text"


__all__ = [
    "ArtifactRole",
    "ArtifactType",
    "AttemptStatus",
    "EvaluationDecision",
    "EvaluationStatus",
    "FailureType",
    "InputArtifactRole",
    "NodeKind",
    "NodeStatus",
    "ReplanScope",
    "WorkflowStatus",
]
