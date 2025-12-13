"""Integration provider registry for Full Auto CI."""

# Import built-in providers to trigger registration side effects.
from . import (  # noqa: F401  pylint: disable=unused-import
    bamboo,
    github,
    gitlab,
    jenkins,
)
from .base import BaseProvider, ProviderConfigError
from .registry import ProviderRegistrationError, ProviderRegistry, registry

__all__ = [
    "BaseProvider",
    "ProviderConfigError",
    "ProviderRegistry",
    "ProviderRegistrationError",
    "registry",
]
