"""JSON-RPC server implementing the Model Context Protocol surface for Full Auto CI."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sys
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

from .. import __version__ as PACKAGE_VERSION
from ..service import CIService

logger = logging.getLogger(__name__)


@dataclass
class MCPError(Exception):
    """Structured error raised for MCP request failures."""

    code: int
    message: str
    data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the error into a JSON-RPC compliant dictionary."""

        payload: Dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            payload["data"] = self.data
        return payload


class MCPServer:
    """Minimal JSON-RPC server exposing CIService over MCP."""

    _SUPPORTED_PROTOCOL_VERSIONS = {
        "2025-06-18",
        "2024-12-06",
    }

    _DEFAULT_PROTOCOL_VERSION = "2025-06-18"

    def __init__(self, service: CIService, *, auth_token: str | None = None):
        """Initialize the MCP server with the backing CI service and optional token."""

        self.service = service
        self.auth_token = auth_token
        self._shutdown_event: asyncio.Event | None = None
        self._session_id: str | None = None
        self._methods: Dict[str, Callable[[Dict[str, Any]], Awaitable[Any]]] = {
            "initialize": self._handle_initialize,
            "handshake": self._handle_handshake,
            "listRepositories": self._handle_list_repositories,
            "queueTestRun": self._handle_queue_test_run,
            "getLatestResults": self._handle_get_latest_results,
            "addRepository": self._handle_add_repository,
            "removeRepository": self._handle_remove_repository,
            "runTests": self._handle_run_tests,
            "logging/setLevel": self._handle_logging_set_level,
            "tools/list": self._handle_tools_list,
            "prompts/list": self._handle_prompts_list,
            "shutdown": self._handle_shutdown,
        }

    async def handle_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single JSON-RPC message and return the response object.

        Returns ``None`` when the message is a JSON-RPC notification that does not
        require a response.
        """

        logger.info("MCP request %s (id=%s)", message.get("method"), message.get("id"))
        logger.debug("MCP request payload: %s", message)

        if message.get("jsonrpc") != "2.0":
            raise MCPError(code=-32600, message="Invalid JSON-RPC version")

        method = message.get("method")
        if not isinstance(method, str):
            raise MCPError(code=-32600, message="Method must be a string")

        message_id = message.get("id")
        is_notification = message_id is None

        handler = self._methods.get(method)
        if handler is None:
            if is_notification:
                logger.debug("Ignoring unknown notification %s", method)
                return None
            raise MCPError(code=-32601, message=f"Method not found: {method}")

        params = message.get("params") or {}
        if not isinstance(params, dict):
            raise MCPError(code=-32602, message="Params must be an object")

        if not is_notification:
            self._verify_token(method, params)

        try:
            result = await handler(params)
        except MCPError:
            raise
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Unhandled exception in MCP handler %s", method)
            raise MCPError(code=-32000, message=str(exc)) from exc

        if is_notification:
            logger.debug("Notification %s handled without response", method)
            return None

        response = {"jsonrpc": "2.0", "id": message_id, "result": result}
        logger.debug(
            "MCP response payload for %s (id=%s): %s",
            method,
            message.get("id"),
            response,
        )
        logger.info(
            "MCP response for %s (id=%s)",
            method,
            message.get("id"),
        )
        return response

    async def _handle_handshake(self, _params: Dict[str, Any]) -> Dict[str, Any]:
        """Return the server identity and advertised capabilities."""

        return {
            "name": "full-auto-ci",
            "version": PACKAGE_VERSION,
            "capabilities": self._legacy_capabilities(),
        }

    async def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Negotiate protocol support per the MCP initialize handshake."""

        logger.debug("Initialize request received with params: %s", params)

        protocol_candidates: List[str] = []
        requested_version = params.get("protocolVersion")
        if isinstance(requested_version, str) and requested_version.strip():
            protocol_candidates.append(requested_version.strip())

        requested_versions = params.get("protocolVersions")
        if isinstance(requested_versions, list):
            protocol_candidates.extend(
                version.strip()
                for version in requested_versions
                if isinstance(version, str) and version.strip()
            )

        negotiated_version: Optional[str] = None
        for candidate in protocol_candidates:
            if candidate in self._SUPPORTED_PROTOCOL_VERSIONS:
                negotiated_version = candidate
                break

        if negotiated_version is None:
            if protocol_candidates:
                logger.warning(
                    "Unsupported protocol requested: %s", protocol_candidates
                )
                raise MCPError(
                    code=-32602,
                    message="Unsupported protocol version",
                    data={
                        "supportedVersions": sorted(self._SUPPORTED_PROTOCOL_VERSIONS)
                    },
                )
            negotiated_version = self._DEFAULT_PROTOCOL_VERSION

        client_info = params.get("clientInfo")
        if not isinstance(client_info, dict):
            client_info = {}

        capabilities = params.get("capabilities")
        if not isinstance(capabilities, dict):
            capabilities = {}

        self._session_id = uuid.uuid4().hex

        logger.debug(
            "Initialize negotiation complete. candidates=%s, negotiated=%s, client=%s",
            protocol_candidates,
            negotiated_version,
            client_info or "<unknown>",
        )

        logger.debug(
            "Client capabilities provided: top-level keys=%s",
            sorted(capabilities.keys()),
        )

        response = {
            "protocolVersion": negotiated_version,
            "serverInfo": {
                "name": "full-auto-ci",
                "version": PACKAGE_VERSION,
            },
            "sessionId": self._session_id,
            "capabilities": self._server_capabilities(),
            "instructions": (
                "This server exposes Full Auto CI operations via the "
                "listRepositories, addRepository, removeRepository, runTests, "
                "queueTestRun, and getLatestResults methods."
            ),
        }
        logger.info(
            "Completed initialize handshake (client=%s %s, protocol=%s)",
            client_info.get("name"),
            client_info.get("version"),
            response["protocolVersion"],
        )
        return response

    async def _handle_list_repositories(
        self, _params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Expose the repository catalog tracked by the CI service."""

        repositories = self.service.list_repositories()
        return {"repositories": repositories}

    async def _handle_queue_test_run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Enqueue a CI test run for the provided repository and commit."""

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
        """Fetch recent test runs for a repository."""

        repo_id = params.get("repositoryId")
        commit_hash = params.get("commit")
        limit = params.get("limit", 5)
        if not isinstance(repo_id, int):
            raise MCPError(code=-32602, message="repositoryId must be an integer")
        if not isinstance(limit, int) or limit <= 0:
            raise MCPError(code=-32602, message="limit must be a positive integer")
        if commit_hash is not None:
            if not isinstance(commit_hash, str) or not commit_hash.strip():
                raise MCPError(
                    code=-32602,
                    message="commit must be a non-empty string when provided",
                )
            commit_hash = commit_hash.strip()

        test_runs = self.service.get_test_results(
            repo_id,
            limit=limit,
            commit_hash=commit_hash,
        )
        return {"testRuns": test_runs}

    async def _handle_add_repository(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Register a new repository with the CI service."""

        name = params.get("name")
        url = params.get("url")
        branch = params.get("branch", "main")

        if not isinstance(name, str) or not name.strip():
            raise MCPError(code=-32602, message="name must be a non-empty string")
        if not isinstance(url, str) or not url.strip():
            raise MCPError(code=-32602, message="url must be a non-empty string")
        if not isinstance(branch, str) or not branch.strip():
            raise MCPError(code=-32602, message="branch must be a non-empty string")

        repo_id = self.service.add_repository(name.strip(), url.strip(), branch.strip())
        if not repo_id:
            raise MCPError(
                code=-32002,
                message="Failed to add repository",
                data={"name": name, "url": url, "branch": branch},
            )

        return {"repositoryId": repo_id}

    async def _handle_remove_repository(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Remove a repository from the CI service."""

        repo_id = params.get("repositoryId")
        if not isinstance(repo_id, int) or repo_id <= 0:
            raise MCPError(
                code=-32602, message="repositoryId must be a positive integer"
            )

        success = self.service.remove_repository(repo_id)
        if not success:
            raise MCPError(
                code=-32003,
                message="Failed to remove repository",
                data={"repositoryId": repo_id},
            )

        return {"removed": True, "repositoryId": repo_id}

    async def _handle_run_tests(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run tests synchronously for a given repository and commit."""

        repo_id = params.get("repositoryId")
        commit_hash = params.get("commit")
        if not isinstance(repo_id, int) or repo_id <= 0:
            raise MCPError(
                code=-32602, message="repositoryId must be a positive integer"
            )
        if not isinstance(commit_hash, str) or not commit_hash.strip():
            raise MCPError(code=-32602, message="commit must be a non-empty string")

        result = self.service.run_tests(repo_id, commit_hash.strip())
        status = result.get("status")
        if status != "success":
            raise MCPError(
                code=-32004,
                message="Test run failed",
                data=result,
            )

        return {"status": status, "results": result}

    async def _handle_logging_set_level(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust the effective log level for the MCP server."""

        level = params.get("level")
        if not isinstance(level, str) or not level.strip():
            raise MCPError(code=-32602, message="level must be a non-empty string")

        normalized = level.strip().lower()
        aliases = {
            "warn": "warning",
            "trace": "debug",
            "notice": "info",
            "fatal": "critical",
            "critical": "critical",
            "alert": "critical",
            "emergency": "critical",
        }
        normalized = aliases.get(normalized, normalized)

        valid_levels = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL,
        }

        resolved = valid_levels.get(normalized)
        if resolved is None:
            raise MCPError(code=-32602, message=f"Unsupported log level: {level}")

        logging.getLogger().setLevel(resolved)
        logger.setLevel(resolved)

        return {}

    async def _handle_tools_list(self, _params: Dict[str, Any]) -> Dict[str, Any]:
        """Return the catalog of available MCP tools."""

        return {"tools": []}

    async def _handle_prompts_list(self, _params: Dict[str, Any]) -> Dict[str, Any]:
        """Return the catalog of available prompts."""

        return {"prompts": []}

    async def _handle_shutdown(self, _params: Dict[str, Any]) -> Dict[str, Any]:
        """Request the server to stop processing new requests and exit."""

        logger.info("Shutdown requested via MCP protocol")
        triggered = self._trigger_shutdown()
        if triggered:
            logger.debug("Shutdown event signaled successfully")
        else:
            logger.warning("Shutdown requested but no active shutdown event is set")
        return {"shuttingDown": triggered}

    def _trigger_shutdown(self) -> bool:
        """Signal the active shutdown event if present."""

        event = self._shutdown_event
        if event is None:
            return False
        if not event.is_set():
            event.set()
            return True
        return False

    def _verify_token(self, method: str, params: Dict[str, Any]) -> None:
        """Validate the shared secret token if authentication is enabled."""

        if not self.auth_token:
            return
        logger.debug("Authentication required for method %s", method)
        provided = params.get("token")
        if provided is None and method == "initialize":
            client_caps = params.get("capabilities")
            if isinstance(client_caps, dict):
                experimental_caps = client_caps.get("experimental")
                if isinstance(experimental_caps, dict):
                    token_container = experimental_caps.get("fullAutoCI")
                    if isinstance(token_container, dict):
                        provided = token_container.get("token") or token_container.get(
                            "authToken"
                        )
                        if provided is not None:
                            logger.debug("Token received via experimental capabilities")
        elif provided is not None:
            logger.debug("Token received directly in params")
        if provided != self.auth_token:
            logger.warning("MCP authentication failed for method %s", method)
            raise MCPError(code=-32604, message="Unauthorized")
        logger.debug("Authentication succeeded for method %s", method)

    async def serve_tcp(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        *,
        shutdown_event: asyncio.Event | None = None,
    ) -> None:
        """Start a plain TCP JSON-RPC loop bound to ``host:port``."""

        logger.info(
            "Starting MCP TCP server version %s on %s:%s",
            PACKAGE_VERSION,
            host,
            port,
        )

        event = shutdown_event or asyncio.Event()
        self._shutdown_event = event

        async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
            peer = writer.get_extra_info("peername")
            await self._serve_connection(reader, writer, peer)

        server = await asyncio.start_server(handler, host, port)
        sockets = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
        logger.info("MCP server listening on %s", sockets)

        try:
            await event.wait()
        except asyncio.CancelledError:  # pragma: no cover - triggered on cancellation
            pass
        finally:
            server.close()
            await server.wait_closed()
            self._shutdown_event = None
            logger.info("MCP server stopped")

    async def serve_stdio(self, *, shutdown_event: asyncio.Event | None = None) -> None:
        """Serve MCP requests over standard input/output streams."""

        logger.info("Starting MCP stdio server version %s", PACKAGE_VERSION)
        loop = asyncio.get_running_loop()
        event = shutdown_event or asyncio.Event()
        self._shutdown_event = event

        reader = asyncio.StreamReader()
        reader_protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: reader_protocol, sys.stdin)

        writer_transport, writer_protocol = await loop.connect_write_pipe(
            lambda: asyncio.StreamReaderProtocol(asyncio.StreamReader()), sys.stdout
        )
        writer = asyncio.StreamWriter(writer_transport, writer_protocol, reader, loop)

        serve_task = asyncio.create_task(
            self._serve_connection(reader, writer, "stdio")
        )
        wait_task = asyncio.create_task(event.wait())

        try:
            await asyncio.wait(
                [serve_task, wait_task], return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            serve_task.cancel()
            wait_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await serve_task
            with contextlib.suppress(asyncio.CancelledError):
                await wait_task
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            logger.info("MCP stdio server stopped")
            self._shutdown_event = None

    @staticmethod
    def _error_response(message_id: Any, error: MCPError) -> Dict[str, Any]:
        """Return a JSON-RPC error response payload."""

        return {"jsonrpc": "2.0", "id": message_id, "error": error.to_dict()}

    @staticmethod
    async def _read_transport_message(
        reader: asyncio.StreamReader,
    ) -> tuple[Optional[str], str]:
        """Read a JSON message supporting newline and content-length framing."""

        while True:
            line = await reader.readline()
            if not line:
                return None, "newline"
            if line in {b"\r\n", b"\n", b""}:
                continue
            break

        if line.lower().startswith(b"content-length:"):
            try:
                length = int(line.split(b":", 1)[1].strip())
            except ValueError as exc:  # pragma: no cover - malformed client
                raise MCPError(
                    code=-32600,
                    message="Invalid Content-Length header",
                    data={"detail": line.decode("utf-8", errors="replace")},
                ) from exc

            while True:
                separator = await reader.readline()
                if not separator:
                    return None, "content-length"
                if separator in {b"\r\n", b"\n", b""}:
                    break

            body = await reader.readexactly(length)
            return body.decode("utf-8"), "content-length"

        buffer = line
        while not buffer.rstrip().endswith(b"}"):
            more = await reader.readline()
            if not more:
                break
            buffer += more
        return buffer.decode("utf-8").strip(), "newline"

    @staticmethod
    def _encode_message(message: Dict[str, Any], framing: str) -> bytes:
        """Serialize ``message`` using the provided framing mode."""

        payload = json.dumps(message)
        if framing == "content-length":
            header = f"Content-Length: {len(payload.encode('utf-8'))}\r\n\r\n"
            return (header + payload).encode("utf-8")
        return (payload + "\n").encode("utf-8")

    @staticmethod
    def _legacy_capabilities() -> List[Dict[str, str]]:
        """Return legacy capability descriptions for backwards compatibility."""

        return [
            {
                "name": "listRepositories",
                "description": "List all repositories tracked by the CI service.",
            },
            {
                "name": "addRepository",
                "description": "Register a repository for automated CI monitoring.",
            },
            {
                "name": "removeRepository",
                "description": "Remove a repository so it is no longer monitored.",
            },
            {
                "name": "runTests",
                "description": "Execute the CI toolchain synchronously for a commit.",
            },
            {
                "name": "queueTestRun",
                "description": "Queue a test run for a repository/commit pair.",
            },
            {
                "name": "getLatestResults",
                "description": "Fetch recent test runs with tool results for a repository.",
            },
            {
                "name": "shutdown",
                "description": "Request the MCP server to terminate.",
            },
        ]

    @classmethod
    def _server_capabilities(cls) -> Dict[str, Any]:
        """Expose server capabilities in MCP-compliant format."""

        return {
            "resources": {
                "list": False,
                "get": False,
                "subscribe": False,
                "listChanged": False,
            },
            "prompts": {
                "list": True,
                "get": False,
            },
            "tools": {
                "list": True,
                "call": False,
            },
            "logging": {
                "subscribe": False,
                "setLevel": True,
            },
            "experimental": {
                "fullAutoCI": {
                    "methods": [cap["name"] for cap in cls._legacy_capabilities()],
                }
            }
        }

    async def _serve_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        peer: Any,
    ) -> None:
        logger.info("MCP client connected: %s", peer)
        try:
            while True:
                try:
                    raw_message, framing = await self._read_transport_message(reader)
                except MCPError as transport_error:
                    error_response = self._error_response(None, transport_error)
                    logger.debug("Transport error for %s: %s", peer, error_response)
                    encoded = self._encode_message(error_response, "newline")
                    writer.write(encoded)
                    await writer.drain()
                    continue
                if raw_message is None:
                    break
                logger.info("Raw MCP payload (%s): %s", framing, raw_message)
                logger.debug(
                    "Decoding MCP message from %s with framing %s", peer, framing
                )
                try:
                    message = json.loads(raw_message)
                except json.JSONDecodeError as exc:
                    error = MCPError(
                        code=-32700,
                        message="Parse error",
                        data={"detail": str(exc)},
                    )
                    error_response = self._error_response(None, error)
                    logger.debug("Parse error for %s: %s", peer, error_response)
                    encoded = self._encode_message(error_response, framing)
                    writer.write(encoded)
                    await writer.drain()
                    continue

                try:
                    response = await self.handle_message(message)
                except MCPError as mcp_error:
                    response = self._error_response(message.get("id"), mcp_error)
                    logger.debug("Handler returned error for %s: %s", peer, response)

                if response is None:
                    logger.debug(
                        "No response required for message id=%s", message.get("id")
                    )
                    continue

                encoded = self._encode_message(response, framing)
                logger.debug(
                    "Encoded MCP response (%s bytes, framing=%s) for id=%s",
                    len(encoded),
                    framing,
                    message.get("id"),
                )
                writer.write(encoded)
                await writer.drain()
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            logger.info("MCP client disconnected: %s", peer)
