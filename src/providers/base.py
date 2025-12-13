"""Base classes for CI providers."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional

if TYPE_CHECKING:  # pragma: no cover
    from ..service import CIService


class ProviderConfigError(ValueError):
    """Raised when a provider configuration fails validation."""


class BaseProvider(ABC):
    """Abstract base class for external CI providers."""

    type_name: str = "base"
    display_name: str = "Base Provider"
    description: str = ""

    def __init__(self, service: "CIService", record: Dict[str, Any]):
        self.service = service
        self.record = record
        self.config: Dict[str, Any] = dict(record.get("config") or {})

    # Public API -----------------------------------------------------------------
    @property
    def provider_id(self) -> int:
        """Return the database identifier for the provider."""
        return int(self.record["id"])

    @property
    def name(self) -> str:
        """Return the user-facing name for the provider."""
        return str(self.record["name"])

    @classmethod
    def validate_static_config(cls, config: Optional[Dict[str, Any]]) -> Iterable[str]:
        """Return validation errors for ``config`` (empty iterable when valid)."""
        _ = config  # Placeholder for subclasses overriding the method.
        return ()

    def validate_runtime(self) -> Iterable[str]:
        """Return validation errors for the currently loaded configuration."""
        return self.validate_static_config(self.config)  # pragma: no cover - override

    def sync_runs(self, *, limit: int = 50) -> list[Dict[str, Any]]:
        """Fetch recent job metadata and map to internal test runs."""
        _ = limit
        return []

    def enqueue_from_webhook(self, payload: Dict[str, Any]) -> Optional[int]:
        """Handle webhook payloads. Return queued test run ID when applicable."""
        raise NotImplementedError("Webhook handling not implemented for this provider")

    def trigger_run(
        self,
        repository_id: int,
        *,
        commit_hash: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Trigger a new remote job and return its external identifier."""
        raise NotImplementedError(
            "Triggering runs is not implemented for this provider"
        )

    # Utility helpers ------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable representation of the provider."""
        return {
            "id": self.provider_id,
            "name": self.name,
            "type": self.type_name,
            "display_name": self.display_name,
            "description": self.description,
            "config": dict(self.config),
        }

    def update_config(self, config: Dict[str, Any]) -> None:
        """Persist an updated configuration back to storage."""
        errors = list(self.validate_static_config(config))
        if errors:
            raise ProviderConfigError("; ".join(errors))
        self.config = dict(config)
        self.service.data.update_external_provider(self.provider_id, config=config)


def require_keys(config: Dict[str, Any], required: Iterable[str]) -> Iterable[str]:
    """Yield validation errors for each missing required key."""
    for key in required:
        if key not in config or config[key] in (None, ""):
            yield f"Missing required configuration key '{key}'"
