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
            "tools/call": self._handle_tools_call,
            "prompts/list": self._handle_prompts_list,
            "shutdown": self._handle_shutdown,
        }

    async def handle_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single JSON-RPC message and return the response object.

        Returns ``None`` when the message is a JSON-RPC notification that does not
        require a response.
        """

        method, message_id, params, handler = self._validate_request(message)
        logger.info("MCP request %s (id=%s)", method, message_id)
        logger.debug("MCP request payload: %s", message)

        is_notification = message_id is None
        if handler is None:
            logger.debug("Ignoring unknown notification %s", method)
            return None
        if not is_notification:
            self._verify_token(method, params)

        result = await self._invoke_handler(method, handler, params)
        if is_notification:
            logger.debug("Notification %s handled without response", method)
            return None

        response = {"jsonrpc": "2.0", "id": message_id, "result": result}
        logger.debug(
            "MCP response payload for %s (id=%s): %s",
            method,
            message_id,
            response,
        )
        logger.info("MCP response for %s (id=%s)", method, message_id)
        return response

    def _validate_request(self, message: Dict[str, Any]) -> tuple[
        str,
        Any,
        Dict[str, Any],
        Optional[Callable[[Dict[str, Any]], Awaitable[Any]]],
    ]:
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
                return method, message_id, {}, None
            raise MCPError(code=-32601, message=f"Method not found: {method}")

        params = message.get("params") or {}
        if not isinstance(params, dict):
            raise MCPError(code=-32602, message="Params must be an object")

        return method, message_id, params, handler

    async def _invoke_handler(
        self,
        method: str,
        handler: Callable[[Dict[str, Any]], Awaitable[Any]],
        params: Dict[str, Any],
    ) -> Any:
        try:
            return await handler(params)
        except MCPError:
            raise
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Unhandled exception in MCP handler %s", method)
            raise MCPError(code=-32000, message=str(exc)) from exc

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

        protocol_candidates = self._protocol_candidates(params)
        negotiated_version = self._negotiate_protocol_version(protocol_candidates)

        client_info = self._dict_param(params, "clientInfo")
        capabilities = self._dict_param(params, "capabilities")

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

        response = self._initialize_response(negotiated_version)
        logger.info(
            "Completed initialize handshake (client=%s %s, protocol=%s)",
            client_info.get("name"),
            client_info.get("version"),
            response["protocolVersion"],
        )
        return response

    @staticmethod
    def _dict_param(params: Dict[str, Any], key: str) -> Dict[str, Any]:
        value = params.get(key)
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _protocol_candidates(params: Dict[str, Any]) -> List[str]:
        candidates: List[str] = []
        requested_version = params.get("protocolVersion")
        if isinstance(requested_version, str) and requested_version.strip():
            candidates.append(requested_version.strip())

        requested_versions = params.get("protocolVersions")
        if isinstance(requested_versions, list):
            candidates.extend(
                version.strip()
                for version in requested_versions
                if isinstance(version, str) and version.strip()
            )
        return candidates

    def _negotiate_protocol_version(self, candidates: List[str]) -> str:
        for candidate in candidates:
            if candidate in self._SUPPORTED_PROTOCOL_VERSIONS:
                return candidate

        if candidates:
            logger.warning("Unsupported protocol requested: %s", candidates)
            raise MCPError(
                code=-32602,
                message="Unsupported protocol version",
                data={"supportedVersions": sorted(self._SUPPORTED_PROTOCOL_VERSIONS)},
            )

        return self._DEFAULT_PROTOCOL_VERSION

    def _initialize_response(self, negotiated_version: str) -> Dict[str, Any]:
        return {
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

        repo_id, commit_hash, limit, result_limit = self._latest_results_params(params)

        test_runs = self.service.get_test_results(
            repo_id,
            limit=limit,
            commit_hash=commit_hash,
        )

        self._limit_test_run_results(test_runs, result_limit)
        return {"testRuns": test_runs}

    def _latest_results_params(
        self, params: Dict[str, Any]
    ) -> tuple[int, Optional[str], int, int]:
        repo_id = params.get("repositoryId")
        commit_hash = params.get("commit")
        limit = params.get("limit", 1)
        result_limit = params.get("maxResults", 1)

        if not isinstance(repo_id, int):
            raise MCPError(code=-32602, message="repositoryId must be an integer")
        if not isinstance(limit, int) or limit <= 0:
            raise MCPError(code=-32602, message="limit must be a positive integer")
        if not isinstance(result_limit, int) or result_limit <= 0:
            raise MCPError(code=-32602, message="maxResults must be a positive integer")

        normalized_commit: Optional[str] = None
        if commit_hash is not None:
            if not isinstance(commit_hash, str) or not commit_hash.strip():
                raise MCPError(
                    code=-32602,
                    message="commit must be a non-empty string when provided",
                )
            normalized_commit = commit_hash.strip()

        return repo_id, normalized_commit, limit, result_limit

    def _limit_test_run_results(
        self, test_runs: List[Dict[str, Any]], limit: int
    ) -> None:
        for run in test_runs:
            results = run.get("results")
            if not isinstance(results, list) or not results:
                continue
            ordered = sorted(results, key=self._result_sort_key)
            run["results"] = ordered[:limit]

    def _result_sort_key(self, item: Any) -> tuple[int, int]:
        tool = item.get("tool") if isinstance(item, dict) else None
        return (self._status_priority(item), self._tool_weight(tool))

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

        repo_id, commit_hash, max_results = self._run_tests_params(params)
        result = self.service.run_tests(repo_id, commit_hash)
        status = result.get("status")
        if status != "success":
            raise MCPError(code=-32004, message="Test run failed", data=result)

        trimmed = dict(result)
        trimmed["tools"] = self._trim_tools(trimmed.get("tools"), max_results)
        return {"status": status, "results": trimmed}

    def _run_tests_params(self, params: Dict[str, Any]) -> tuple[int, str, int]:
        repo_id = params.get("repositoryId")
        commit_hash = params.get("commit")
        max_results = params.get("maxResults", 1)

        if not isinstance(repo_id, int) or repo_id <= 0:
            raise MCPError(
                code=-32602, message="repositoryId must be a positive integer"
            )
        if not isinstance(commit_hash, str) or not commit_hash.strip():
            raise MCPError(code=-32602, message="commit must be a non-empty string")
        if not isinstance(max_results, int) or max_results <= 0:
            raise MCPError(code=-32602, message="maxResults must be a positive integer")

        return repo_id, commit_hash.strip(), max_results

    def _trim_tools(self, tools: Any, max_results: int) -> Any:
        if not isinstance(tools, dict) or not tools:
            return tools

        ordered_tools = sorted(
            tools.items(),
            key=lambda pair: (
                self._status_priority(pair[1] if isinstance(pair[1], dict) else {}),
                self._tool_weight(pair[0]),
            ),
        )
        return {name: payload for name, payload in ordered_tools[:max_results]}

    @staticmethod
    def _status_priority(result: Any) -> int:
        """Assign a priority rank to a result status (lower is higher priority)."""

        if not isinstance(result, dict):
            return 100

        status = str(result.get("status", "")).strip().lower()
        if status in {"error", "failed", "failure"}:
            return 0
        if status in {"warning", "warn"}:
            return 10
        if status in {"success", "passed", "ok"}:
            return 50
        if status in {"running", "queued", "pending"}:
            return 60
        return 90

    @staticmethod
    def _tool_weight(tool_name: Any) -> int:
        """Prefer smaller/high-signal tools when truncating results."""

        name = str(tool_name or "").strip().lower()
        weights = {
            "pylint": 0,
            "coverage": 1,
            "lizard": 2,
        }
        return weights.get(name, 5)

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

        return {"tools": self._tool_definitions()}

    async def _handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke a named tool and return MCP tool-call content."""

        name = params.get("name")
        arguments = params.get("arguments")

        if not isinstance(name, str) or not name.strip():
            raise MCPError(code=-32602, message="name must be a non-empty string")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise MCPError(code=-32602, message="arguments must be an object")

        tool_name = name.strip()
        handler = self._tool_handlers().get(tool_name)
        if handler is None:
            raise MCPError(code=-32602, message=f"Unknown tool: {tool_name}")

        try:
            result = await handler(arguments)
        except MCPError as exc:
            if -32100 < exc.code <= -32000:
                return {
                    "isError": True,
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps({"error": exc.to_dict()}, indent=2),
                        }
                    ],
                }
            raise

        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(result, indent=2, sort_keys=True),
                }
            ]
        }

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
        provided = self._extract_token(method, params)
        if provided != self.auth_token:
            logger.warning("MCP authentication failed for method %s", method)
            raise MCPError(code=-32604, message="Unauthorized")
        logger.debug("Authentication succeeded for method %s", method)

    def _extract_token(self, method: str, params: Dict[str, Any]) -> Any:
        provided = params.get("token")
        if provided is not None:
            logger.debug("Token received directly in params")
            return provided

        if method != "initialize":
            return None

        token = self._extract_token_from_capabilities(params.get("capabilities"))
        if token is not None:
            logger.debug("Token received via experimental capabilities")
        return token

    @staticmethod
    def _extract_token_from_capabilities(capabilities: Any) -> Any:
        if not isinstance(capabilities, dict):
            return None
        experimental_caps = capabilities.get("experimental")
        if not isinstance(experimental_caps, dict):
            return None
        token_container = experimental_caps.get("fullAutoCI")
        if not isinstance(token_container, dict):
            return None
        return token_container.get("token") or token_container.get("authToken")

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
        first_line = await MCPServer._read_first_content_line(reader)
        if first_line is None:
            return None, "newline"

        if first_line.lower().startswith(b"content-length:"):
            payload = await MCPServer._read_content_length_body(reader, first_line)
            if payload is None:
                return None, "content-length"
            return payload, "content-length"

        payload = await MCPServer._read_newline_body(reader, first_line)
        return payload, "newline"

    @staticmethod
    async def _read_first_content_line(
        reader: asyncio.StreamReader,
    ) -> Optional[bytes]:
        while True:
            line = await reader.readline()
            if not line:
                return None
            if line in {b"\r\n", b"\n", b""}:
                continue
            return line

    @staticmethod
    async def _read_content_length_body(
        reader: asyncio.StreamReader,
        header_line: bytes,
    ) -> Optional[str]:
        try:
            length = int(header_line.split(b":", 1)[1].strip())
        except ValueError as exc:  # pragma: no cover - malformed client
            raise MCPError(
                code=-32600,
                message="Invalid Content-Length header",
                data={"detail": header_line.decode("utf-8", errors="replace")},
            ) from exc

        while True:
            separator = await reader.readline()
            if not separator:
                return None
            if separator in {b"\r\n", b"\n", b""}:
                break

        body = await reader.readexactly(length)
        return body.decode("utf-8")

    @staticmethod
    async def _read_newline_body(
        reader: asyncio.StreamReader,
        first_line: bytes,
    ) -> str:
        buffer = first_line
        while not buffer.rstrip().endswith(b"}"):
            more = await reader.readline()
            if not more:
                break
            buffer += more
        return buffer.decode("utf-8").strip()

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
                "call": True,
            },
            "logging": {
                "subscribe": False,
                "setLevel": True,
            },
            "experimental": {
                "fullAutoCI": {
                    "methods": [cap["name"] for cap in cls._legacy_capabilities()],
                }
            },
        }

    def _tool_handlers(
        self,
    ) -> Dict[str, Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]]:
        """Return tool-name to handler mapping using bound methods."""

        return {
            "listRepositories": self._handle_list_repositories,
            "addRepository": self._handle_add_repository,
            "removeRepository": self._handle_remove_repository,
            "queueTestRun": self._handle_queue_test_run,
            "getLatestResults": self._handle_get_latest_results,
            "runTests": self._handle_run_tests,
            "shutdown": self._handle_shutdown,
        }

    @staticmethod
    def _tool_definitions() -> List[Dict[str, Any]]:
        return [
            {
                "name": "listRepositories",
                "description": "List repositories tracked by Full Auto CI.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
            {
                "name": "addRepository",
                "description": "Register a repository for CI monitoring.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "url": {"type": "string"},
                        "branch": {"type": "string", "default": "main"},
                    },
                    "required": ["name", "url"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "removeRepository",
                "description": "Remove a repository from CI monitoring.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"repositoryId": {"type": "integer"}},
                    "required": ["repositoryId"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "queueTestRun",
                "description": "Queue a CI run for a repository/commit pair.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "repositoryId": {"type": "integer"},
                        "commit": {"type": "string"},
                    },
                    "required": ["repositoryId", "commit"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "getLatestResults",
                "description": "Fetch recent CI runs (default limit=1, maxResults=1).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "repositoryId": {"type": "integer"},
                        "commit": {"type": "string"},
                        "limit": {"type": "integer", "default": 1},
                        "maxResults": {"type": "integer", "default": 1},
                    },
                    "required": ["repositoryId"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "runTests",
                "description": "Run the CI toolchain synchronously for a commit.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "repositoryId": {"type": "integer"},
                        "commit": {"type": "string"},
                        "maxResults": {"type": "integer", "default": 1},
                    },
                    "required": ["repositoryId", "commit"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "shutdown",
                "description": "Request the MCP server to terminate.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        ]

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
