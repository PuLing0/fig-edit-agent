"""Top-level package exports with lazy forwarding."""

from __future__ import annotations

from . import core as _core
from . import schemas as _schemas
from . import tools as _tools


_EXPORT_MODULES = (_core, _schemas, _tools)
__all__ = list(dict.fromkeys([*_core.__all__, *_schemas.__all__, *_tools.__all__]))


def __getattr__(name: str):
    for module in _EXPORT_MODULES:
        if name in getattr(module, "__all__", ()):
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
