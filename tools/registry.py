"""Registry for discovering and invoking tools by name."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .base import BaseTool, ToolContext, ToolResult


_BUILTIN_TOOL_MODULES = (
    ".image_collage",
    ".image_crop",
    ".image_edit",
    ".image_grounding",
    ".image_ocr",
    ".image_score",
    ".image_segment",
    ".image_understand",
    ".prompt_reconstruct",
)

_builtin_tools_loaded = False


class ToolRegistry:
    """Mutable in-memory registry of tool instances."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool, *, replace: bool = False) -> None:
        if not replace and tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> BaseTool:
        try:
            return self._tools.pop(name)
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def get(self, name: str) -> BaseTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def has(self, name: str) -> bool:
        return name in self._tools

    def list_names(self) -> list[str]:
        return sorted(self._tools)

    def list_tools(self) -> list[BaseTool]:
        return [self._tools[name] for name in self.list_names()]

    async def run(self, name: str, ctx: ToolContext, args: Any) -> ToolResult:
        tool = self.get(name)
        return await tool(ctx, args)

    def clear(self) -> None:
        self._tools.clear()


tool_registry = ToolRegistry()


def ensure_builtin_tools_registered() -> None:
    """Import all built-in tool modules exactly once so they self-register."""

    global _builtin_tools_loaded
    if _builtin_tools_loaded:
        return
    for module_name in _BUILTIN_TOOL_MODULES:
        import_module(module_name, package=__package__)
    _builtin_tools_loaded = True


__all__ = ["ToolRegistry", "ensure_builtin_tools_registered", "tool_registry"]
