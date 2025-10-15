"""Helper utilities for handling MCP-related CLI commands."""

from __future__ import annotations

import asyncio
import errno
import json
import logging
import os
import socket
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence

from . import __version__ as PACKAGE_VERSION
from .mcp import MCPServer

if TYPE_CHECKING:  # pragma: no cover - typing-only imports
    from .service import CIService


logger = logging.getLogger(__name__)


def serve(
    args: Any,
    *,
    service: "CIService",
    resolve_log_level: Callable[[str], int],
    print_fn: Optional[Callable[[str], None]] = None,
    asyncio_run: Optional[Callable[[Any], None]] = None,
) -> int:
    """Entry point for the ``mcp serve`` CLI command."""

    if print_fn is None:
        print_fn = print
    if asyncio_run is None:
        asyncio_run = asyncio.run

    try:
        _configure_logging(resolve_log_level, getattr(args, "log_level", "INFO"))
    except ValueError as exc:
        print_fn(f"Error: {exc}")
        return 1

    token = _resolve_token(args)
    server = MCPServer(service, auth_token=token)
    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 8765)
    token_state = "enabled" if token else "disabled"

    if getattr(args, "stdio", False):
        print_fn(f"Starting MCP server on stdio (token={token_state})")
    else:
        if not _prepare_mcp_tcp_launch(host, port, token, token_state, print_fn):
            return 1

    async def runner() -> None:
        try:
            if getattr(args, "stdio", False):
                await server.serve_stdio()
            else:
                await server.serve_tcp(host=host, port=port)
        except OSError as exc:  # pragma: no cover - environment dependent
            if not getattr(args, "stdio", False) and exc.errno == errno.EADDRINUSE:
                print_fn(f"Error: MCP server port {host}:{port} is already in use.")
                return
            raise
        except asyncio.CancelledError:  # pragma: no cover - cancellation path
            pass

    try:
        asyncio_run(runner())
    except KeyboardInterrupt:  # pragma: no cover - manual interruption
        print_fn("MCP server stopped")
    return 0


def _configure_logging(
    resolve_log_level: Callable[[str], int], log_level_str: str
) -> int:
    log_level = resolve_log_level(log_level_str)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    for handler in root_logger.handlers:
        handler.setLevel(log_level)

    logging.getLogger("src").setLevel(log_level)
    logging.getLogger(__name__).setLevel(log_level)
    logging.getLogger("src.mcp.server").setLevel(log_level)

    logger.info(
        "Launching MCP server version %s with log level %s",
        PACKAGE_VERSION,
        logging.getLevelName(log_level),
    )
    return log_level


def _resolve_token(args: Any) -> Optional[str]:
    if getattr(args, "no_token", False):
        return None
    token = getattr(args, "token", None)
    if token:
        return token
    return os.getenv("FULL_AUTO_CI_MCP_TOKEN")


def _prepare_mcp_tcp_launch(
    host: str,
    port: int,
    token: Optional[str],
    token_state: str,
    print_fn: Callable[[str], None],
) -> bool:
    probe_state = _probe_mcp_server(host, port, token)

    if probe_state == "available":
        print_fn(f"Restarting MCP server on {host}:{port} (token={token_state})")
        shutdown_status = _request_mcp_shutdown(host, port, token)
        if shutdown_status == "unauthorized":
            print_fn(
                "Error: Existing MCP server rejected the provided token; cannot restart."
            )
            return False
        if shutdown_status not in {"success", "unreachable"}:
            print_fn("Error: Failed to shut down existing MCP server.")
            return False
        if shutdown_status == "success":
            print_fn("Existing MCP server shutdown initiated.")
        else:
            print_fn("Existing MCP server was unreachable during restart; continuing.")
        if not _wait_for_mcp_shutdown(host, port, token, timeout=5.0):
            print_fn("Error: Existing MCP server did not stop in time.")
            return False
        print_fn(f"Starting MCP server on {host}:{port} (token={token_state})")
        return True

    if probe_state == "unauthorized":
        print_fn(
            "Error: MCP server is already running but rejected the provided token."
        )
        return False

    print_fn(f"Starting MCP server on {host}:{port} (token={token_state})")
    return True


def _probe_mcp_server(host: str, port: int, token: Optional[str]) -> Optional[str]:
    try:
        with socket.create_connection((host, port), timeout=1.0) as sock:
            params: Dict[str, Any] = {}
            if token:
                params["token"] = token
            message = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "handshake",
                "params": params,
            }
            sock.sendall((json.dumps(message) + "\n").encode("utf-8"))
            sock.settimeout(1.0)
            buffer = b""
            while not buffer.endswith(b"\n"):
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buffer += chunk
    except (OSError, ValueError):
        return None

    if not buffer:
        return None

    try:
        response = json.loads(buffer.decode("utf-8"))
    except json.JSONDecodeError:
        return None

    if isinstance(response, dict) and "result" in response:
        return "available"

    error = response.get("error") if isinstance(response, dict) else None
    if isinstance(error, dict) and error.get("code") == -32604:
        return "unauthorized"

    return None


def _request_mcp_shutdown(host: str, port: int, token: Optional[str]) -> str:
    try:
        with socket.create_connection((host, port), timeout=1.0) as sock:
            params: Dict[str, Any] = {}
            if token:
                params["token"] = token
            message = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "shutdown",
                "params": params,
            }
            sock.sendall((json.dumps(message) + "\n").encode("utf-8"))
            sock.settimeout(2.0)
            buffer = b""
            while not buffer.endswith(b"\n"):
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buffer += chunk
    except (OSError, ValueError):
        return "unreachable"

    if not buffer:
        return "failure"

    try:
        response = json.loads(buffer.decode("utf-8"))
    except json.JSONDecodeError:
        return "failure"

    if isinstance(response, dict):
        if "result" in response:
            return "success"
        error = response.get("error")
        if isinstance(error, dict) and error.get("code") == -32604:
            return "unauthorized"

    return "failure"


def _wait_for_mcp_shutdown(
    host: str,
    port: int,
    token: Optional[str],
    *,
    timeout: float = 5.0,
    poll_interval: float = 0.2,
) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _probe_mcp_server(host, port, token) is None:
            return True
        time.sleep(poll_interval)
    return False


__all__: Sequence[str] = ("serve",)
