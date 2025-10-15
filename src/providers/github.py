"""GitHub Actions provider stub."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from .base import BaseProvider, require_keys
from .registry import provider


@provider
class GitHubActionsProvider(BaseProvider):
    """Provider stub for GitHub Actions integration."""

    type_name = "github"
    display_name = "GitHub Actions"
    description = "Synchronize GitHub Actions workflow results into Full Auto CI."

    @classmethod
    def validate_static_config(cls, config: Optional[Dict[str, Any]]) -> Iterable[str]:
        config = config or {}
        yield from require_keys(config, ["token", "owner", "repository"])

    def sync_runs(
        self, *, limit: int = 50
    ) -> list[Dict[str, Any]]:  # pragma: no cover - stub
        _ = limit
        return []
