"""JSON-RPC server implementing the Model Context Protocol surface for Full Auto CI."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

from ..service import CIService

logger = logging.getLogger(__name__)


@dataclass
class MCPError(Exception):
    """Structured error raised for MCP request failures."""

    code: int
    message: str
    data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            payload["data"] = self.data
        return payload


class MCPServer:
    """Minimal JSON-RPC server exposing CIService over MCP."""

    def __init__(self, service: CIService, *, auth_token: str | None = None):
        self.service = service
        self.auth_token = auth_token
        self._methods: Dict[str, Callable[[Dict[str, Any]], Awaitable[Any]]] = {
            "handshake": self._handle_handshake,
            "listRepositories": self._handle_list_repositories,
            "queueTestRun": self._handle_queue_test_run,
            "getLatestResults": self._handle_get_latest_results,
        }

    async def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single JSON-RPC message and return the response object."""

        logger.debug("Received MCP message: %s", message)

        if message.get("jsonrpc") != "2.0":
            raise MCPError(code=-32600, message="Invalid JSON-RPC version")

        method = message.get("method")
        if not isinstance(method, str):
            raise MCPError(code=-32600, message="Method must be a string")

        handler = self._methods.get(method)
        if handler is None:
            raise MCPError(code=-32601, message=f"Method not found: {method}")

        params = message.get("params") or {}
        if not isinstance(params, dict):
            raise MCPError(code=-32602, message="Params must be an object")

        self._verify_token(params)

        try:
            result = await handler(params)
        except MCPError:
            raise
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Unhandled exception in MCP handler %s", method)
            raise MCPError(code=-32000, message=str(exc)) from exc

        response = {"jsonrpc": "2.0", "id": message.get("id"), "result": result}
        logger.debug("Responding to MCP message %s", response)
        return response

    async def _handle_handshake(self, _params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "name": "full-auto-ci",
            "version": "0.1.0",
            "capabilities": [
                {
                    "name": "listRepositories",
                    "description": "List all repositories tracked by the CI service.",
                },
                {
                    "name": "queueTestRun",
                    "description": "Queue a test run for a repository/commit pair.",
                },
                {
                    "name": "getLatestResults",
                    "description": "Fetch recent test runs with tool results for a repository.",
                },
            ],
        }

    async def _handle_list_repositories(
        self, _params: Dict[str, Any]
    ) -> Dict[str, Any]:
        repositories = self.service.list_repositories()
        return {"repositories": repositories}

    async def _handle_queue_test_run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        repo_id = params.get("repositoryId")
        commit_hash = params.get("commit")
        if not isinstance(repo_id, int):
            raise MCPError(code=-32602, message="repositoryId must be an integer")
        if not isinstance(commit_hash, str) or not commit_hash:
            raise MCPError(code=-32602, message="commit must be a non-empty string")

        success = self.service.add_test_task(repo_id, commit_hash)
        if not success:
            raise MCPError(
                code=-32001,
                message="Failed to queue test run",
                data={"repositoryId": repo_id, "commit": commit_hash},
            )
        return {"queued": True}

    async def _handle_get_latest_results(
        self, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        repo_id = params.get("repositoryId")
        limit = params.get("limit", 5)
        if not isinstance(repo_id, int):
            raise MCPError(code=-32602, message="repositoryId must be an integer")
        if not isinstance(limit, int) or limit <= 0:
            raise MCPError(code=-32602, message="limit must be a positive integer")

        test_runs = self.service.get_test_results(repo_id, limit=limit)
        return {"testRuns": test_runs}

    def _verify_token(self, params: Dict[str, Any]) -> None:
        if not self.auth_token:
            return
        provided = params.get("token")
        if provided != self.auth_token:
            raise MCPError(code=-32604, message="Unauthorized")

    async def serve_tcp(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        *,
        shutdown_event: asyncio.Event | None = None,
    ) -> None:
        """Start a plain TCP JSON-RPC loop bound to ``host:port``."""

        async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
            peer = writer.get_extra_info("peername")
            logger.info("MCP client connected: %s", peer)
            try:
                while True:
                    data = await reader.readline()
                    if not data:
                        break
                    try:
                        message = json.loads(data.decode("utf-8"))
                    except json.JSONDecodeError as exc:
                        error = MCPError(
                            code=-32700,
                            message="Parse error",
                            data={"detail": str(exc)},
                        )
                        writer.write(
                            (
                                json.dumps(self._error_response(None, error)) + "\n"
                            ).encode("utf-8")
                        )
                        await writer.drain()
                        continue

                    try:
                        response = await self.handle_message(message)
                    except MCPError as mcp_error:
                        response = self._error_response(message.get("id"), mcp_error)
                    writer.write((json.dumps(response) + "\n").encode("utf-8"))
                    await writer.drain()
            finally:
                writer.close()
                with contextlib.suppress(Exception):
                    await writer.wait_closed()
                logger.info("MCP client disconnected: %s", peer)

        server = await asyncio.start_server(handler, host, port)
        sockets = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
        logger.info("MCP server listening on %s", sockets)

        try:
            if shutdown_event is None:
                await asyncio.Future()
            else:
                await shutdown_event.wait()
        except asyncio.CancelledError:  # pragma: no cover - triggered on cancellation
            pass
        finally:
            server.close()
            await server.wait_closed()
            logger.info("MCP server stopped")

    @staticmethod
    def _error_response(message_id: Any, error: MCPError) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": message_id, "error": error.to_dict()}
