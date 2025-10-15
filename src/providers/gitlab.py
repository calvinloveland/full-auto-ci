"""GitLab CI provider stub."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from .base import BaseProvider, require_keys
from .registry import provider


@provider
class GitLabCIProvider(BaseProvider):
    """Provider stub for GitLab pipelines."""

    type_name = "gitlab"
    display_name = "GitLab CI"
    description = "Synchronize GitLab pipelines and jobs."

    @classmethod
    def validate_static_config(cls, config: Optional[Dict[str, Any]]) -> Iterable[str]:
        config = config or {}
        yield from require_keys(config, ["token", "base_url", "project_id"])
