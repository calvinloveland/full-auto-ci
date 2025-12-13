"""Unit tests for the MCP server implementation."""

from __future__ import annotations

import asyncio
import json
import socket
from typing import Any, Dict, List, Optional, Tuple

import pytest

from src import __version__ as PACKAGE_VERSION
from src.mcp.server import MCPError, MCPServer


class DummyData:
    def __init__(self) -> None:
        self._runs = [
            {
                "id": 1,
                "repository_id": 1,
                "commit_hash": "abcdef1",
                "status": "completed",
                "created_at": 1710000000,
                "started_at": 1710000001,
                "completed_at": 1710000005,
                "error": None,
            }
        ]
        self._results = {
            1: [
                {
                    "tool": "pylint",
                    "status": "success",
                    "output": "All good",
                    "duration": 3.5,
                    "created_at": 1710000006,
                }
            ]
        }

    def fetch_recent_test_runs(
        self, repo_id: int, limit: int = 20, commit_hash: str | None = None
    ):
        if repo_id != 1:
            return []
        runs = self._runs
        if commit_hash is not None:
            runs = [run for run in runs if run["commit_hash"] == commit_hash]
        return runs[:limit]

    def fetch_results_for_test_run(self, test_run_id: int):
        return self._results.get(test_run_id, [])

    def fetch_commit_for_test_run(self, test_run_id: int):
        return {
            "id": test_run_id,
            "hash": "abcdef1",
            "message": "Initial commit",
            "author": "Alice",
            "timestamp": 1710000000,
            "repository_id": 1,
        }


class DummyService:
    def __init__(self) -> None:
        self.repositories = [
            {
                "id": 1,
                "name": "Repo One",
                "url": "https://example.com/one.git",
                "branch": "main",
            }
        ]
        self.data = DummyData()
        self.queued = []
        self.add_test_task_should_succeed = True
        self.add_repository_should_fail = False
        self._next_repo_id = 2
        self.run_tests_calls: List[Tuple[int, str]] = []
        self.run_tests_result: Dict[str, Any] = {
            "status": "success",
            "tools": {
                "pylint": {"status": "success", "score": 10},
                "coverage": {"status": "success", "percent": 95},
            },
        }
        self.get_results_calls: List[Tuple[int, Optional[str], int]] = []

    def list_repositories(self):
        return self.repositories

    def add_test_task(self, repo_id: int, commit_hash: str) -> bool:
        self.queued.append((repo_id, commit_hash))
        return self.add_test_task_should_succeed

    def get_test_results(
        self, repo_id: int, *, commit_hash: str | None = None, limit: int = 20
    ):
        self.get_results_calls.append((repo_id, commit_hash, limit))
        runs = self.data.fetch_recent_test_runs(
            repo_id, limit=limit, commit_hash=commit_hash
        )
        enriched = []
        for run in runs:
            enriched.append(
                {**run, "results": self.data.fetch_results_for_test_run(run["id"])}
            )
        return enriched

    def add_repository(self, name: str, url: str, branch: str = "main") -> int:
        if self.add_repository_should_fail:
            return 0
        repo_id = self._next_repo_id
        self._next_repo_id += 1
        record = {"id": repo_id, "name": name, "url": url, "branch": branch}
        self.repositories.append(record)
        return repo_id

    def remove_repository(self, repo_id: int) -> bool:
        if not self.repositories:
            return False
        index = next(
            (i for i, repo in enumerate(self.repositories) if repo["id"] == repo_id),
            None,
        )
        if index is None:
            return False
        self.repositories.pop(index)
        return True

    def run_tests(self, repo_id: int, commit_hash: str) -> Dict[str, Any]:
        self.run_tests_calls.append((repo_id, commit_hash))
        return self.run_tests_result


@pytest.fixture()
def dummy_service():
    return DummyService()


def _run(coro):
    return asyncio.run(coro)


async def _open_connection_with_retry(host: str, port: int, timeout: float = 2.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    last_error: Exception | None = None
    while True:
        try:
            return await asyncio.open_connection(host, port)
        except (ConnectionRefusedError, OSError) as exc:
            last_error = exc
            if loop.time() >= deadline:
                raise AssertionError(
                    f"Timed out waiting for MCP server on {host}:{port}"
                ) from last_error
            await asyncio.sleep(0.05)


def test_handshake_announces_capabilities(dummy_service):
    server = MCPServer(dummy_service)
    response = _run(
        server.handle_message({"jsonrpc": "2.0", "id": 1, "method": "handshake"})
    )
    assert response["result"]["name"] == "full-auto-ci"
    assert response["result"]["version"] == PACKAGE_VERSION
    capability_names = {cap["name"] for cap in response["result"]["capabilities"]}
    assert capability_names == {
        "listRepositories",
        "addRepository",
        "removeRepository",
        "queueTestRun",
        "getLatestResults",
        "runTests",
        "shutdown",
    }


def test_initialize_negotiates_protocol_and_capabilities(dummy_service):
    server = MCPServer(dummy_service)
    response = _run(
        server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "clientInfo": {"name": "client", "version": "1.0"},
                    "capabilities": {},
                },
            }
        )
    )

    result = response["result"]
    assert result["protocolVersion"] == "2025-06-18"
    assert result["serverInfo"]["name"] == "full-auto-ci"
    assert result["serverInfo"]["version"] == PACKAGE_VERSION
    assert "sessionId" in result and isinstance(result["sessionId"], str)
    capabilities = result["capabilities"]
    assert set(capabilities.keys()) >= {
        "resources",
        "prompts",
        "tools",
        "logging",
        "experimental",
    }
    assert "instructions" in result


def test_initialize_handles_protocol_version_list(dummy_service):
    server = MCPServer(dummy_service)
    response = _run(
        server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 10,
                "method": "initialize",
                "params": {
                    "protocolVersions": ["2025-07-01", "2024-12-06"],
                    "clientInfo": {"name": "client", "version": "1.0"},
                },
            }
        )
    )

    result = response["result"]
    assert result["protocolVersion"] == "2024-12-06"


def test_initialize_defaults_missing_client_fields(dummy_service):
    server = MCPServer(dummy_service)
    response = _run(
        server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 11,
                "method": "initialize",
                "params": {},
            }
        )
    )

    result = response["result"]
    assert result["protocolVersion"] == "2025-06-18"
    assert result["serverInfo"]["name"] == "full-auto-ci"


def test_initialize_rejects_unsupported_protocol(dummy_service):
    server = MCPServer(dummy_service)
    with pytest.raises(MCPError) as excinfo:
        _run(
            server.handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 12,
                    "method": "initialize",
                    "params": {"protocolVersion": "1999-01-01"},
                }
            )
        )

    assert excinfo.value.code == -32602
    assert excinfo.value.data is not None
    assert "supportedVersions" in excinfo.value.data


def test_shutdown_handler_signals_event(dummy_service):
    async def scenario():
        server = MCPServer(dummy_service)
        shutdown_event = asyncio.Event()
        server._shutdown_event = shutdown_event

        response = await server.handle_message(
            {"jsonrpc": "2.0", "id": 99, "method": "shutdown", "params": {}}
        )

        assert response["result"]["shuttingDown"] is True
        assert shutdown_event.is_set()

    _run(scenario())


def test_tcp_server_emits_initialize_response(dummy_service):
    async def scenario():
        server = MCPServer(dummy_service)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            host, port = probe.getsockname()

        shutdown_event = asyncio.Event()
        serve_task = asyncio.create_task(
            server.serve_tcp(host=host, port=port, shutdown_event=shutdown_event)
        )

        try:
            reader, writer = await _open_connection_with_retry(host, port)
            try:
                message = {
                    "jsonrpc": "2.0",
                    "id": 100,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-06-18",
                        "clientInfo": {
                            "name": "test-client",
                            "version": "0.0",
                        },
                        "capabilities": {},
                    },
                }
                payload = json.dumps(message) + "\n"
                writer.write(payload.encode("utf-8"))
                await writer.drain()

                raw_response = await reader.readline()
                assert raw_response, "Server did not respond to initialize request"
                response = json.loads(raw_response.decode("utf-8"))
            finally:
                writer.close()
                await writer.wait_closed()
        finally:
            shutdown_event.set()
            await serve_task

        assert response["id"] == 100
        result = response["result"]
        assert result["serverInfo"]["name"] == "full-auto-ci"
        assert result["serverInfo"]["version"] == PACKAGE_VERSION
        assert result["protocolVersion"] in MCPServer._SUPPORTED_PROTOCOL_VERSIONS
        assert "instructions" in result and "Full Auto CI" in result["instructions"]

    _run(scenario())


def test_list_repositories_returns_service_data(dummy_service):
    server = MCPServer(dummy_service)
    response = _run(
        server.handle_message({"jsonrpc": "2.0", "id": 2, "method": "listRepositories"})
    )
    repos = response["result"]["repositories"]
    assert repos[0]["name"] == "Repo One"


def test_queue_test_run_success(dummy_service):
    server = MCPServer(dummy_service)
    response = _run(
        server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "queueTestRun",
                "params": {"repositoryId": 1, "commit": "deadbeef"},
            }
        )
    )
    assert response["result"]["queued"] is True
    assert dummy_service.queued == [(1, "deadbeef")]


def test_queue_test_run_failure(dummy_service):
    dummy_service.add_test_task_should_succeed = False
    server = MCPServer(dummy_service)
    with pytest.raises(MCPError) as excinfo:
        _run(
            server.handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "queueTestRun",
                    "params": {"repositoryId": 1, "commit": "deadbeef"},
                }
            )
        )
    assert excinfo.value.code == -32001


def test_get_latest_results_enriches_runs(dummy_service):
    server = MCPServer(dummy_service)
    response = _run(
        server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "getLatestResults",
                "params": {"repositoryId": 1, "limit": 5},
            }
        )
    )
    runs = response["result"]["testRuns"]
    assert runs[0]["results"][0]["tool"] == "pylint"


def test_get_latest_results_allows_commit_filter(dummy_service):
    server = MCPServer(dummy_service)
    response = _run(
        server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 6,
                "method": "getLatestResults",
                "params": {"repositoryId": 1, "commit": "abcdef1", "limit": 2},
            }
        )
    )

    assert response["result"]["testRuns"]
    assert dummy_service.get_results_calls[-1] == (1, "abcdef1", 2)


def test_requires_token_when_configured(dummy_service):
    server = MCPServer(dummy_service, auth_token="secret")
    with pytest.raises(MCPError) as excinfo:
        _run(
            server.handle_message(
                {"jsonrpc": "2.0", "id": 6, "method": "listRepositories"}
            )
        )
    assert excinfo.value.code == -32604

    response = _run(
        server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "listRepositories",
                "params": {"token": "secret"},
            }
        )
    )
    assert "repositories" in response["result"]


def test_initialize_requires_token_when_configured(dummy_service):
    server = MCPServer(dummy_service, auth_token="secret")
    with pytest.raises(MCPError) as excinfo:
        _run(
            server.handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 8,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-06-18",
                        "clientInfo": {"name": "client", "version": "1.2"},
                        "capabilities": {},
                    },
                }
            )
        )
    assert excinfo.value.code == -32604

    response = _run(
        server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 9,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "clientInfo": {"name": "client", "version": "1.2"},
                    "capabilities": {
                        "experimental": {"fullAutoCI": {"token": "secret"}}
                    },
                },
            }
        )
    )
    assert response["result"]["serverInfo"]["name"] == "full-auto-ci"


def test_add_repository_registers_repo(dummy_service):
    server = MCPServer(dummy_service)
    response = _run(
        server.handle_message(  # type: ignore[arg-type]
            {
                "jsonrpc": "2.0",
                "id": 10,
                "method": "addRepository",
                "params": {
                    "name": "New Repo",
                    "url": "https://example.com/new.git",
                    "branch": "develop",
                },
            }
        )
    )

    result = response["result"]
    assert result["repositoryId"] == 2
    assert any(repo["name"] == "New Repo" for repo in dummy_service.repositories)


def test_add_repository_validates_params(dummy_service):
    server = MCPServer(dummy_service)
    with pytest.raises(MCPError) as excinfo:
        _run(
            server.handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 11,
                    "method": "addRepository",
                    "params": {"name": "", "url": "git://example"},
                }
            )
        )
    assert excinfo.value.code == -32602


def test_add_repository_failure_raises(dummy_service):
    dummy_service.add_repository_should_fail = True
    server = MCPServer(dummy_service)
    with pytest.raises(MCPError) as excinfo:
        _run(
            server.handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 12,
                    "method": "addRepository",
                    "params": {
                        "name": "Bad Repo",
                        "url": "https://example.com/bad.git",
                    },
                }
            )
        )
    assert excinfo.value.code == -32002


def test_remove_repository_success(dummy_service):
    server = MCPServer(dummy_service)
    response = _run(
        server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 13,
                "method": "removeRepository",
                "params": {"repositoryId": 1},
            }
        )
    )

    result = response["result"]
    assert result["removed"] is True
    assert not any(repo["id"] == 1 for repo in dummy_service.repositories)


def test_remove_repository_requires_valid_id(dummy_service):
    server = MCPServer(dummy_service)
    with pytest.raises(MCPError) as excinfo:
        _run(
            server.handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 14,
                    "method": "removeRepository",
                    "params": {"repositoryId": "bad"},
                }
            )
        )
    assert excinfo.value.code == -32602


def test_remove_repository_failure(dummy_service):
    server = MCPServer(dummy_service)
    with pytest.raises(MCPError) as excinfo:
        _run(
            server.handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 15,
                    "method": "removeRepository",
                    "params": {"repositoryId": 999},
                }
            )
        )
    assert excinfo.value.code == -32003


def test_run_tests_returns_results(dummy_service):
    server = MCPServer(dummy_service)
    response = _run(
        server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 16,
                "method": "runTests",
                "params": {"repositoryId": 1, "commit": "abcdef1"},
            }
        )
    )

    assert response["result"]["status"] == "success"
    assert dummy_service.run_tests_calls[-1] == (1, "abcdef1")


def test_run_tests_failure_raises(dummy_service):
    dummy_service.run_tests_result = {"status": "error", "error": "boom"}
    server = MCPServer(dummy_service)
    with pytest.raises(MCPError) as excinfo:
        _run(
            server.handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 17,
                    "method": "runTests",
                    "params": {"repositoryId": 1, "commit": "abcdef1"},
                }
            )
        )
    assert excinfo.value.code == -32004
    assert excinfo.value.data["error"] == "boom"


def test_transport_reader_handles_content_length():
    async def _run():
        reader = asyncio.StreamReader()
        payload = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
        raw = json.dumps(payload).encode("utf-8")
        reader.feed_data(f"Content-Length: {len(raw)}\r\n\r\n".encode("utf-8") + raw)
        reader.feed_eof()

        data, framing = await MCPServer._read_transport_message(reader)
        assert framing == "content-length"
        assert json.loads(data) == payload

    asyncio.run(_run())


def test_transport_reader_handles_newline():
    async def _run():
        reader = asyncio.StreamReader()
        payload = {"jsonrpc": "2.0", "id": 2, "method": "handshake"}
        reader.feed_data((json.dumps(payload) + "\n").encode("utf-8"))
        reader.feed_eof()

        data, framing = await MCPServer._read_transport_message(reader)
        assert framing == "newline"
        assert json.loads(data) == payload

    asyncio.run(_run())


def test_encode_message_content_length_round_trip():
    payload = {"jsonrpc": "2.0", "id": 3, "result": {}}
    encoded = MCPServer._encode_message(payload, "content-length")
    header, body = encoded.split(b"\r\n\r\n", 1)
    assert header.startswith(b"Content-Length: ")
    assert int(header.split(b": ")[1]) == len(body)
    assert json.loads(body.decode("utf-8")) == payload


def test_unknown_method_raises(dummy_service):
    server = MCPServer(dummy_service)
    with pytest.raises(MCPError) as excinfo:
        _run(server.handle_message({"jsonrpc": "2.0", "id": 8, "method": "unknown"}))
    assert excinfo.value.code == -32601


def test_notifications_without_id_are_ignored(dummy_service):
    server = MCPServer(dummy_service)
    response = _run(
        server.handle_message(
            {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {},
            }
        )
    )
    assert response is None
