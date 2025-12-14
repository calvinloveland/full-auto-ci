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
    """In-memory DataAccess stand-in for MCP tests."""

    def __init__(self) -> None:
        """Seed predictable test-run + tool result fixtures."""

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
                },
                {
                    "tool": "coverage",
                    "status": "success",
                    "output": "95%",
                    "duration": 4.2,
                    "created_at": 1710000007,
                },
            ]
        }

    def fetch_recent_test_runs(
        self, repo_id: int, limit: int = 20, commit_hash: str | None = None
    ):
        """Return recent test runs for a repository."""

        if repo_id != 1:
            return []
        runs = self._runs
        if commit_hash is not None:
            runs = [run for run in runs if run["commit_hash"] == commit_hash]
        return runs[:limit]

    def fetch_results_for_test_run(self, test_run_id: int):
        """Return tool results for a given test run id."""

        return self._results.get(test_run_id, [])

    def fetch_commit_for_test_run(self, test_run_id: int):
        """Return commit metadata for a given test run id."""

        return {
            "id": test_run_id,
            "hash": "abcdef1",
            "message": "Initial commit",
            "author": "Alice",
            "timestamp": 1710000000,
            "repository_id": 1,
        }


class DummyService:
    """Minimal CI service stand-in used by MCPServer tests."""

    def __init__(self) -> None:
        """Initialize predictable repository and tool runner fixtures."""

        self.data = DummyData()
        self._state: Dict[str, Any] = {
            "repositories": [
                {
                    "id": 1,
                    "name": "Repo One",
                    "url": "https://example.com/one.git",
                    "branch": "main",
                }
            ],
            "queued": [],
            "add_test_task_should_succeed": True,
            "add_repository_should_fail": False,
            "next_repo_id": 2,
            "run_tests_calls": [],
            "run_tests_result": {
                "status": "success",
                "tools": {
                    "pylint": {"status": "success", "score": 10},
                    "coverage": {"status": "success", "percent": 95},
                },
            },
            "get_results_calls": [],
        }
        self.tool_runner = DummyToolRunner(
            {
                "pylint": {
                    "status": "success",
                    "score": 10.0,
                    "issues": {},
                    "details": [],
                },
                "lizard": {"status": "success", "summary": {"above_threshold": 0}},
            }
        )

    @property
    def repositories(self) -> List[Dict[str, Any]]:
        """Repository records exposed by the service."""

        return self._state["repositories"]

    @property
    def queued(self) -> List[Tuple[int, str]]:
        """Queued test tasks (repo_id, commit_hash)."""

        return self._state["queued"]

    @property
    def add_test_task_should_succeed(self) -> bool:
        """Whether add_test_task should report success."""

        return bool(self._state["add_test_task_should_succeed"])

    @add_test_task_should_succeed.setter
    def add_test_task_should_succeed(self, value: bool) -> None:
        """Control add_test_task success behavior."""

        self._state["add_test_task_should_succeed"] = bool(value)

    @property
    def add_repository_should_fail(self) -> bool:
        """Whether add_repository should fail by returning id 0."""

        return bool(self._state["add_repository_should_fail"])

    @add_repository_should_fail.setter
    def add_repository_should_fail(self, value: bool) -> None:
        """Control add_repository failure behavior."""

        self._state["add_repository_should_fail"] = bool(value)

    @property
    def run_tests_calls(self) -> List[Tuple[int, str, bool]]:
        """Record of run_tests calls."""

        return self._state["run_tests_calls"]

    @property
    def run_tests_result(self) -> Dict[str, Any]:
        """Result payload returned by run_tests."""

        return self._state["run_tests_result"]

    @run_tests_result.setter
    def run_tests_result(self, value: Dict[str, Any]) -> None:
        """Override the run_tests return payload."""

        self._state["run_tests_result"] = value

    @property
    def get_results_calls(self) -> List[Tuple[int, Optional[str], int]]:
        """Record of get_test_results invocations."""

        return self._state["get_results_calls"]

    def list_repositories(self):
        """Return tracked repositories."""

        return self.repositories

    def add_test_task(self, repo_id: int, commit_hash: str) -> bool:
        """Queue a test task for later processing."""

        self.queued.append((repo_id, commit_hash))
        return self.add_test_task_should_succeed

    def get_test_results(
        self, repo_id: int, *, commit_hash: str | None = None, limit: int = 20
    ):
        """Return historical runs enriched with tool results."""

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
        """Register a repository and return its id, or 0 on failure."""

        if self.add_repository_should_fail:
            return 0
        repo_id = int(self._state["next_repo_id"])
        self._state["next_repo_id"] = repo_id + 1
        record = {"id": repo_id, "name": name, "url": url, "branch": branch}
        self.repositories.append(record)
        return repo_id

    def remove_repository(self, repo_id: int) -> bool:
        """Remove a repository by id."""

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

    def run_tests(
        self,
        repo_id: int,
        commit_hash: str,
        *,
        include_working_tree: bool = False,
    ) -> Dict[str, Any]:
        """Record and return the configured run_tests_result payload."""

        self.run_tests_calls.append((repo_id, commit_hash, include_working_tree))
        return self.run_tests_result


class DummyToolRunner:
    """Minimal ToolRunner stand-in returning canned results."""

    def __init__(self, results: Dict[str, Any]):
        """Initialize runner with pre-computed tool results."""

        self._results = results
        self.calls: List[str] = []

    @property
    def results(self) -> Dict[str, Any]:
        """Return the current tool results payload."""

        return self._results

    @results.setter
    def results(self, value: Dict[str, Any]) -> None:
        """Replace the tool results payload."""

        self._results = value

    def run_all(self, repo_path: str) -> Dict[str, Any]:
        """Pretend to run tools by returning the canned results."""

        self.calls.append(repo_path)
        return self._results


@pytest.fixture()
def service_stub():
    """Provide a DummyService instance for tests."""

    return DummyService()


def _run(coro):
    """Run an async coroutine synchronously for tests."""

    return asyncio.run(coro)


async def _open_connection_with_retry(host: str, port: int, timeout: float = 2.0):
    """Open a TCP connection, retrying briefly while a server starts."""

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


def test_handshake_announces_capabilities(service_stub):
    """Handshake should announce server capabilities."""

    server = MCPServer(service_stub)
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
        "getWorkingTreeIssues",
        "shutdown",
    }


def test_initialize_negotiates_protocol_and_capabilities(service_stub):
    """Initialize should negotiate protocol version and return capabilities."""

    server = MCPServer(service_stub)
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


def test_initialize_handles_protocol_version_list(service_stub):
    """Initialize should accept protocolVersions and choose a supported value."""

    server = MCPServer(service_stub)
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


def test_tools_list_exposes_full_auto_ci_tools(service_stub):
    """tools/list should expose Full Auto CI tool definitions."""

    server = MCPServer(service_stub)
    response = _run(
        server.handle_message({"jsonrpc": "2.0", "id": 20, "method": "tools/list"})
    )

    tools = response["result"]["tools"]
    assert isinstance(tools, list)
    assert tools
    names = {tool["name"] for tool in tools}
    assert {
        "listRepositories",
        "addRepository",
        "removeRepository",
        "queueTestRun",
        "getLatestResults",
        "runTests",
        "getWorkingTreeIssues",
        "shutdown",
    } <= names
    assert all("inputSchema" in tool for tool in tools)


def test_get_working_tree_issues_returns_structured_issues(service_stub):
    """getWorkingTreeIssues should return structured issues and summary."""

    service_stub.tool_runner.results = {
        "pylint": {
            "status": "success",
            "details": [
                {
                    "type": "warning",
                    "path": "src/cli.py",
                    "line": 12,
                    "column": 3,
                    "message": "Unused import",
                    "message-id": "W0611",
                    "symbol": "unused-import",
                }
            ],
        }
    }

    server = MCPServer(service_stub)
    response = _run(
        server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 200,
                "method": "getWorkingTreeIssues",
                "params": {"repoPath": ".", "maxIssues": 10},
            }
        )
    )

    result = response["result"]
    assert result["truncated"] is False
    assert isinstance(result["issues"], list)
    assert result["issues"]
    issue = result["issues"][0]
    assert issue["tool"] == "pylint"
    assert issue["severity"] in {"warning", "error", "info"}
    assert issue["path"] == "src/cli.py"
    assert issue["line"] == 12
    assert "summary" in result
    assert result["summary"]["total"] == 1


def test_get_working_tree_issues_is_not_cached(service_stub):
    """getWorkingTreeIssues should not cache tool runner output."""

    server = MCPServer(service_stub)

    service_stub.tool_runner.results = {
        "pylint": {
            "status": "success",
            "details": [
                {
                    "type": "error",
                    "path": "main.py",
                    "line": 1,
                    "column": 1,
                    "message": "Synthetic error",
                    "message-id": "E9999",
                }
            ],
        }
    }
    first = _run(
        server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 201,
                "method": "getWorkingTreeIssues",
                "params": {"repoPath": ".", "maxIssues": 50},
            }
        )
    )["result"]
    assert first["issues"]

    service_stub.tool_runner.results = {"pylint": {"status": "success", "details": []}}
    second = _run(
        server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 202,
                "method": "getWorkingTreeIssues",
                "params": {"repoPath": ".", "maxIssues": 50},
            }
        )
    )["result"]
    assert second["issues"] == []


def test_tools_call_can_invoke_get_working_tree_issues(service_stub):
    """tools/call should invoke getWorkingTreeIssues."""

    service_stub.tool_runner.results = {
        "lizard": {
            "status": "success",
            "top_offenders": [
                {"name": "f", "filename": "src/tools.py", "line": 10, "ccn": 42}
            ],
        }
    }
    server = MCPServer(service_stub)
    response = _run(
        server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 203,
                "method": "tools/call",
                "params": {
                    "name": "getWorkingTreeIssues",
                    "arguments": {"repoPath": ".", "maxIssues": 5},
                },
            }
        )
    )

    result = response["result"]
    payload = json.loads(result["content"][0]["text"])
    assert payload["summary"]["total"] == 1
    assert payload["issues"][0]["tool"] == "lizard"


def test_tools_call_returns_text_content(service_stub):
    """tools/call should return text content payloads."""

    server = MCPServer(service_stub)
    response = _run(
        server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 21,
                "method": "tools/call",
                "params": {"name": "listRepositories", "arguments": {}},
            }
        )
    )

    result = response["result"]
    assert "content" in result
    assert result["content"][0]["type"] == "text"
    payload = json.loads(result["content"][0]["text"])
    assert payload["repositories"] == service_stub.repositories


def test_tools_call_wraps_tool_failures_as_is_error(service_stub):
    """tools/call should wrap tool failures as isError payloads."""

    service_stub.run_tests_result = {"status": "failed", "tools": {}}
    server = MCPServer(service_stub)
    response = _run(
        server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 22,
                "method": "tools/call",
                "params": {
                    "name": "runTests",
                    "arguments": {"repositoryId": 1, "commit": "abcdef1"},
                },
            }
        )
    )

    result = response["result"]
    assert result["isError"] is True
    error_payload = json.loads(result["content"][0]["text"])["error"]
    assert error_payload["code"] == -32004


def test_initialize_defaults_missing_client_fields(service_stub):
    """Initialize should fill defaults when client fields are missing."""

    server = MCPServer(service_stub)
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


def test_initialize_rejects_unsupported_protocol(service_stub):
    """Initialize should reject unsupported protocol versions."""

    server = MCPServer(service_stub)
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


def test_shutdown_handler_signals_event(service_stub):
    """shutdown should signal the server shutdown event."""

    async def scenario():
        """Execute shutdown request flow."""

        server = MCPServer(service_stub)
        shutdown_event = asyncio.Event()
        server._shutdown_event = shutdown_event

        response = await server.handle_message(
            {"jsonrpc": "2.0", "id": 99, "method": "shutdown", "params": {}}
        )

        assert response["result"]["shuttingDown"] is True
        assert shutdown_event.is_set()

    _run(scenario())


def test_tcp_server_emits_initialize_response(service_stub):
    """TCP server should emit a valid initialize response."""

    async def scenario():
        """Start TCP server and perform an initialize exchange."""

        server = MCPServer(service_stub)

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


def test_list_repositories_returns_service_data(service_stub):
    """listRepositories should return repository data from the service."""

    server = MCPServer(service_stub)
    response = _run(
        server.handle_message({"jsonrpc": "2.0", "id": 2, "method": "listRepositories"})
    )
    repos = response["result"]["repositories"]
    assert repos[0]["name"] == "Repo One"


def test_queue_test_run_success(service_stub):
    """queueTestRun should enqueue tasks when the service accepts it."""

    server = MCPServer(service_stub)
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
    assert service_stub.queued == [(1, "deadbeef")]


def test_queue_test_run_failure(service_stub):
    """queueTestRun should raise when the service rejects the task."""

    service_stub.add_test_task_should_succeed = False
    server = MCPServer(service_stub)
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


def test_get_latest_results_enriches_runs(service_stub):
    """getLatestResults should return test runs enriched with tool results."""

    server = MCPServer(service_stub)
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


def test_get_latest_results_defaults_to_single_run_and_single_result(service_stub):
    """getLatestResults should default to a single run and result."""

    server = MCPServer(service_stub)
    response = _run(
        server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 50,
                "method": "getLatestResults",
                "params": {"repositoryId": 1},
            }
        )
    )

    runs = response["result"]["testRuns"]
    assert len(runs) == 1
    assert len(runs[0]["results"]) == 1


def test_get_latest_results_allows_multiple_results(service_stub):
    """getLatestResults should allow returning multiple tool results per run."""

    server = MCPServer(service_stub)
    response = _run(
        server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 51,
                "method": "getLatestResults",
                "params": {"repositoryId": 1, "limit": 1, "maxResults": 2},
            }
        )
    )

    runs = response["result"]["testRuns"]
    assert len(runs) == 1
    assert len(runs[0]["results"]) == 2


def test_get_latest_results_allows_commit_filter(service_stub):
    """getLatestResults should filter by commit hash when provided."""

    server = MCPServer(service_stub)
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
    assert service_stub.get_results_calls[-1] == (1, "abcdef1", 2)


def test_requires_token_when_configured(service_stub):
    """Requests should require a matching token when configured."""

    server = MCPServer(service_stub, auth_token="secret")
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


def test_initialize_requires_token_when_configured(service_stub):
    """Initialize should require a token when server auth is enabled."""

    server = MCPServer(service_stub, auth_token="secret")
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


def test_add_repository_registers_repo(service_stub):
    """addRepository should register a repository via the service."""

    server = MCPServer(service_stub)
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
    assert any(repo["name"] == "New Repo" for repo in service_stub.repositories)


def test_add_repository_validates_params(service_stub):
    """addRepository should validate inputs."""

    server = MCPServer(service_stub)
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


def test_add_repository_failure_raises(service_stub):
    """addRepository should raise when the service fails to add."""

    service_stub.add_repository_should_fail = True
    server = MCPServer(service_stub)
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


def test_remove_repository_success(service_stub):
    """removeRepository should remove a repository by id."""

    server = MCPServer(service_stub)
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
    assert not any(repo["id"] == 1 for repo in service_stub.repositories)


def test_remove_repository_requires_valid_id(service_stub):
    """removeRepository should validate repositoryId type."""

    server = MCPServer(service_stub)
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


def test_remove_repository_failure(service_stub):
    """removeRepository should raise when removing a missing repository."""

    server = MCPServer(service_stub)
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


def test_run_tests_returns_results(service_stub):
    """runTests should execute and return results."""

    server = MCPServer(service_stub)
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
    assert len(response["result"]["results"]["tools"]) == 1
    assert service_stub.run_tests_calls[-1] == (1, "abcdef1", False)


def test_run_tests_allows_multiple_results(service_stub):
    """runTests should allow returning multiple results per tool."""

    server = MCPServer(service_stub)
    response = _run(
        server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 160,
                "method": "runTests",
                "params": {"repositoryId": 1, "commit": "abcdef1", "maxResults": 2},
            }
        )
    )

    assert response["result"]["status"] == "success"
    assert len(response["result"]["results"]["tools"]) == 2


def test_run_tests_allows_working_tree_flag(service_stub):
    """runTests should pass through includeWorkingTree."""

    server = MCPServer(service_stub)
    response = _run(
        server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 161,
                "method": "runTests",
                "params": {
                    "repositoryId": 1,
                    "commit": "abcdef1",
                    "includeWorkingTree": True,
                },
            }
        )
    )

    assert response["result"]["status"] == "success"
    assert service_stub.run_tests_calls[-1] == (1, "abcdef1", True)


def test_run_tests_rejects_non_boolean_working_tree_flag(service_stub):
    """runTests should validate includeWorkingTree is boolean."""

    server = MCPServer(service_stub)
    with pytest.raises(MCPError) as excinfo:
        _run(
            server.handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 162,
                    "method": "runTests",
                    "params": {
                        "repositoryId": 1,
                        "commit": "abcdef1",
                        "includeWorkingTree": "yes",
                    },
                }
            )
        )
    assert excinfo.value.code == -32602


def test_run_tests_failure_raises(service_stub):
    """runTests should raise if the service returns an error status."""

    service_stub.run_tests_result = {"status": "error", "error": "boom"}
    server = MCPServer(service_stub)
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
    """Transport reader should decode Content-Length framing."""

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
    """Transport reader should decode newline-delimited framing."""

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
    """Content-length encoding should round-trip back to the original payload."""

    payload = {"jsonrpc": "2.0", "id": 3, "result": {}}
    encoded = MCPServer._encode_message(payload, "content-length")
    header, body = encoded.split(b"\r\n\r\n", 1)
    assert header.startswith(b"Content-Length: ")
    assert int(header.split(b": ")[1]) == len(body)
    assert json.loads(body.decode("utf-8")) == payload


def test_unknown_method_raises(service_stub):
    """Unknown methods should return a JSON-RPC method-not-found error."""

    server = MCPServer(service_stub)
    with pytest.raises(MCPError) as excinfo:
        _run(server.handle_message({"jsonrpc": "2.0", "id": 8, "method": "unknown"}))
    assert excinfo.value.code == -32601


def test_notifications_without_id_are_ignored(service_stub):
    """Notifications (no id) should not return a response."""

    server = MCPServer(service_stub)
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
