"""Unit tests for the MCP server implementation."""

from __future__ import annotations

import asyncio
import json

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
        return self._runs[:limit]

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

    def list_repositories(self):
        return self.repositories

    def add_test_task(self, repo_id: int, commit_hash: str) -> bool:
        self.queued.append((repo_id, commit_hash))
        return self.add_test_task_should_succeed

    def get_test_results(
        self, repo_id: int, *, commit_hash: str | None = None, limit: int = 20
    ):
        runs = self.data.fetch_recent_test_runs(
            repo_id, limit=limit, commit_hash=commit_hash
        )
        enriched = []
        for run in runs:
            enriched.append(
                {**run, "results": self.data.fetch_results_for_test_run(run["id"])}
            )
        return enriched


@pytest.fixture()
def dummy_service():
    return DummyService()


def _run(coro):
    return asyncio.run(coro)


def test_handshake_announces_capabilities(dummy_service):
    server = MCPServer(dummy_service)
    response = _run(
        server.handle_message({"jsonrpc": "2.0", "id": 1, "method": "handshake"})
    )
    assert response["result"]["name"] == "full-auto-ci"
    assert response["result"]["version"] == PACKAGE_VERSION
    capability_names = {cap["name"] for cap in response["result"]["capabilities"]}
    assert capability_names == {"listRepositories", "queueTestRun", "getLatestResults"}


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
    assert "capabilities" in result
    assert "instructions" in result


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
