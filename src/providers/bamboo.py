"""Bamboo CI provider stub."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from .base import BaseProvider, require_keys
from .registry import provider


@provider
class BambooProvider(BaseProvider):
    """Provider stub for Atlassian Bamboo plans."""

    type_name = "bamboo"
    display_name = "Bamboo CI"
    description = "Integrate with Atlassian Bamboo build plans."

    @classmethod
    def validate_static_config(cls, config: Optional[Dict[str, Any]]) -> Iterable[str]:
        config = config or {}
        yield from require_keys(config, ["base_url", "user", "token"])

    def sync_runs(
        self, *, limit: int = 50
    ) -> list[Dict[str, Any]]:  # pragma: no cover - stub
        _ = limit
        return []
