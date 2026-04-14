"""Core infrastructure exports."""

from .config import AgentSettings, settings
from .firered_edit_backend import FireRedBackendError, backend_config_snapshot, edit_images, load_pipeline
from .geometry_utils import CoordinateManager
from .llm_client import LLMClient


def __getattr__(name: str):
    if name in {"ExecuteAgent", "ExecuteAgentConfig"}:
        from .execute_agent import ExecuteAgent, ExecuteAgentConfig

        exports = {
            "ExecuteAgent": ExecuteAgent,
            "ExecuteAgentConfig": ExecuteAgentConfig,
        }
        return exports[name]
    if name in {"PlanAgent", "PlanAgentConfig"}:
        from .plan_agent import PlanAgent, PlanAgentConfig

        exports = {
            "PlanAgent": PlanAgent,
            "PlanAgentConfig": PlanAgentConfig,
        }
        return exports[name]
    if name in {"Orchestrator", "OrchestratorConfig", "OrchestratorInputArtifact", "WorkflowRunResult"}:
        from .orchestrator import (
            Orchestrator,
            OrchestratorConfig,
            OrchestratorInputArtifact,
            WorkflowRunResult,
        )

        exports = {
            "Orchestrator": Orchestrator,
            "OrchestratorConfig": OrchestratorConfig,
            "OrchestratorInputArtifact": OrchestratorInputArtifact,
            "WorkflowRunResult": WorkflowRunResult,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AgentSettings",
    "CoordinateManager",
    "ExecuteAgent",
    "ExecuteAgentConfig",
    "FireRedBackendError",
    "LLMClient",
    "Orchestrator",
    "OrchestratorConfig",
    "OrchestratorInputArtifact",
    "PlanAgent",
    "PlanAgentConfig",
    "WorkflowRunResult",
    "backend_config_snapshot",
    "edit_images",
    "load_pipeline",
    "settings",
]
