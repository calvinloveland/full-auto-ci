"""Helper utilities for provider-related CLI commands."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, Optional

from .providers import ProviderConfigError

if TYPE_CHECKING:  # pragma: no cover - typing only
    import argparse

    from .cli import CLI


def handle_provider_command(cli: "CLI", args: "argparse.Namespace") -> int:
    """Dispatch provider sub-commands for the CLI."""

    command = getattr(args, "provider_command", None)
    handler_map = {
        "list": _provider_list,
        "types": _provider_types,
        "add": _provider_add,
        "remove": _provider_remove,
        "sync": _provider_sync,
    }

    handler = handler_map.get(command)
    if handler is None:
        print(f"Error: Unknown provider command {command}")
        return 1
    return handler(cli, args)


def _provider_list(cli: "CLI", _args: "argparse.Namespace") -> int:
    providers = cli.service.list_providers()
    if not providers:
        print("No external providers configured.")
        return 0

    print("Configured providers:")
    for provider in providers:
        descriptor = provider.get("display_name") or provider["type"]
        print(
            f"  [{provider['id']}] {provider['name']} ({provider['type']}) -> {descriptor}"
        )
    return 0


def _provider_types(cli: "CLI", _args: "argparse.Namespace") -> int:
    types = list(cli.service.get_provider_types())
    if not types:
        print("No provider types registered.")
        return 0

    print("Available provider types:")
    for entry in types:
        description = entry.get("description") or ""
        suffix = f" - {description}" if description else ""
        print(f"  {entry['type']}: {entry['display_name']}{suffix}")
    return 0


def _provider_add(cli: "CLI", args: "argparse.Namespace") -> int:
    try:
        config = _load_provider_config(
            inline=getattr(args, "config", None),
            file_path=getattr(args, "config_file", None),
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    try:
        provider = cli.service.add_provider(args.type, args.name, config=config)
    except (ProviderConfigError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    print(
        f"Provider '{provider.get('name', args.name)}' registered with id {provider.get('id')}"
    )
    return 0


def _provider_remove(cli: "CLI", args: "argparse.Namespace") -> int:
    removed = cli.service.remove_provider(args.provider_id)
    if removed:
        print(f"Provider {args.provider_id} removed")
        return 0
    print(f"Error: Provider {args.provider_id} not found")
    return 1


def _provider_sync(cli: "CLI", args: "argparse.Namespace") -> int:
    try:
        runs = cli.service.sync_provider(args.provider_id, limit=args.limit)
    except KeyError:
        print(f"Error: Provider {args.provider_id} not found")
        return 1
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1

    count = len(runs) if isinstance(runs, list) else 0
    print(f"Synced provider {args.provider_id}; fetched {count} run(s)")
    return 0


def _load_provider_config(
    *, inline: Optional[str], file_path: Optional[str]
) -> Dict[str, Any]:
    if inline and file_path:
        raise ValueError("Specify either --config or --config-file, not both")

    if file_path:
        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except FileNotFoundError as exc:  # pragma: no cover - pass through for CLI
            raise ValueError(f"Configuration file not found: {file_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in configuration file: {exc}") from exc

    if inline:
        try:
            return json.loads(inline)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON payload: {exc}") from exc

    return {}


__all__ = ["handle_provider_command"]
