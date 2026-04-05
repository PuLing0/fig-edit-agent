"""Shared base types for the schema package."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints


NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
UriStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class StrictSchema(BaseModel):
    """Strict base model used by all orchestrator-facing contracts."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=False,
        populate_by_name=True,
    )


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


__all__ = ["StrictSchema", "Identifier", "NonEmptyStr", "UriStr", "utc_now"]
