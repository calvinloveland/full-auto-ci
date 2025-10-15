"""Jenkins provider stub."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from .base import BaseProvider, require_keys
from .registry import provider


@provider
class JenkinsProvider(BaseProvider):
    """Provider stub for Jenkins automation servers."""

    type_name = "jenkins"
    display_name = "Jenkins"
    description = "Integrate with Jenkins jobs and pipelines."

    @classmethod
    def validate_static_config(cls, config: Optional[Dict[str, Any]]) -> Iterable[str]:
        config = config or {}
        yield from require_keys(config, ["base_url", "user", "token"])

    def sync_runs(
        self, *, limit: int = 50
    ) -> list[Dict[str, Any]]:  # pragma: no cover - stub
        _ = limit
        return []
