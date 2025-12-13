"""Provider registry utilities."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Type

from .base import BaseProvider, ProviderConfigError


class ProviderRegistrationError(RuntimeError):
    """Raised when a provider class fails to register correctly."""


class ProviderRegistry:
    """Central registry for provider classes."""

    def __init__(self):
        self._providers: Dict[str, Type[BaseProvider]] = {}

    # Registration -------------------------------------------------------------
    def register(self, provider_cls: Type[BaseProvider]) -> Type[BaseProvider]:
        """Register ``provider_cls`` and return it for decorator usage."""
        type_name = getattr(provider_cls, "type_name", None)
        if not type_name or not isinstance(type_name, str):
            raise ProviderRegistrationError(
                f"Provider {provider_cls!r} missing valid 'type_name' attribute"
            )
        normalized = type_name.strip().lower()
        if not normalized:
            raise ProviderRegistrationError(
                f"Provider {provider_cls!r} has empty 'type_name'"
            )
        if normalized in self._providers:
            raise ProviderRegistrationError(
                f"Provider type '{normalized}' already registered"
            )
        self._providers[normalized] = provider_cls
        return provider_cls

    def get(self, type_name: str) -> Type[BaseProvider]:
        """Lookup a provider class by type name."""
        normalized = type_name.strip().lower()
        provider_cls = self._providers.get(normalized)
        if provider_cls is None:
            raise KeyError(f"Provider type '{type_name}' is not registered")
        return provider_cls

    def available_types(self) -> Iterable[Dict[str, Any]]:
        """Yield metadata describing all registered provider types."""
        for type_name, provider_cls in sorted(self._providers.items()):
            yield {
                "type": type_name,
                "display_name": getattr(provider_cls, "display_name", type_name),
                "description": getattr(provider_cls, "description", ""),
            }

    def create(self, type_name: str, service, record: Dict[str, Any]) -> BaseProvider:
        """Instantiate and validate a provider of ``type_name``."""
        provider_cls = self.get(type_name)
        instance = provider_cls(service, record)
        errors = list(instance.validate_runtime())
        if errors:
            raise ProviderConfigError("; ".join(errors))
        return instance


registry = ProviderRegistry()


def provider(provider_cls: Type[BaseProvider]) -> Type[BaseProvider]:
    """Class decorator to register provider classes."""

    return registry.register(provider_cls)
