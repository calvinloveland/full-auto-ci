"""Integration tests covering end-to-end CIService flows."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import pytest

from src.cli import CLI
from src.service import CIService
from src.tools import ToolRunner


class DummyTool:
    """Minimal tool that always succeeds for test purposes."""

    name = "dummy"

    def run(self, repo_path: str):  # noqa: D401 - inherited documentation
        assert os.path.isdir(
            repo_path
        ), "Tool should receive an existing repository path"
        return {
            "status": "success",
            "duration": 0.05,
            "notes": "integration-pass",
        }


class FakeRepo:
    """Static git repository stub used by the integration harness."""

    def __init__(self, repo_id: int, repo_path: Path):
        self.repo_id = repo_id
        self.repo_path = str(repo_path)
        self.url = "https://example.com/fake.git"
        self.branch = "main"
        self.checked_out: list[str] = []

    def clone(self) -> bool:  # pragma: no cover - unused fallback
        return True

    def pull(self) -> bool:  # pragma: no cover - unused fallback
        return True

    def checkout_commit(self, commit_hash: str) -> bool:
        self.checked_out.append(commit_hash)
        return True


class StubGitTracker:
    """Minimal GitTracker replacement that serves predefined repositories."""

    def __init__(self, repos: Dict[int, FakeRepo]):
        self.repos = repos

    def get_repository(self, repo_id: int):
        return self.repos.get(repo_id)

    # The following hooks keep the stub compatible with CIService management APIs.
    def add_repository(
        self, repo_id: int, name: str, url: str, branch: str = "main"
    ) -> bool:  # noqa: D401 - interface parity
        if repo_id in self.repos:
            return True
        path = (
            Path(self.repos[next(iter(self.repos))].repo_path).parent
            / f"repo_{repo_id}"
        )
        path.mkdir(parents=True, exist_ok=True)
        self.repos[repo_id] = FakeRepo(repo_id, path)
        return True

    def remove_repository(self, repo_id: int) -> bool:  # noqa: D401 - interface parity
        return self.repos.pop(repo_id, None) is not None


@pytest.fixture()
def integration_service(monkeypatch, tmp_path):
    """Provision a CIService wired with stubbed git + tool layers."""

    monkeypatch.setenv("FULL_AUTO_CI_DOGFOOD", "0")

    db_path = tmp_path / "ci.sqlite"
    service = CIService(config_path=None, db_path=str(db_path))

    repo_dir = tmp_path / "workspace"
    repo_dir.mkdir()
    (repo_dir / "README.md").write_text("integration repo", encoding="utf-8")

    repo_id = service.data.create_repository(
        "Fixture Repo", "https://example.com/fake.git", "main"
    )
    fake_repo = FakeRepo(repo_id, repo_dir)

    service.git_tracker = StubGitTracker({repo_id: fake_repo})
    service.tool_runner = ToolRunner([DummyTool()])

    return {
        "service": service,
        "repo_id": repo_id,
        "repo_dir": repo_dir,
        "db_path": db_path,
        "fake_repo": fake_repo,
    }


def test_run_tests_persists_results(integration_service):
    service = integration_service["service"]
    repo_id = integration_service["repo_id"]

    outcome = service.run_tests(repo_id, "deadbeef")

    assert outcome["status"] == "success"
    assert outcome["tools"]["dummy"]["status"] == "success"

    test_run_id = outcome["test_run_id"]
    stored_runs = service.data.fetch_recent_test_runs(repo_id, limit=5)
    assert stored_runs
    assert stored_runs[0]["status"].lower() == "completed"

    stored_results = service.data.fetch_results_for_test_run(test_run_id)
    assert stored_results
    assert stored_results[0]["tool"] == "dummy"
    assert stored_results[0]["status"] == "success"


def test_cli_test_run_end_to_end(capsys, integration_service):
    service = integration_service["service"]
    repo_id = integration_service["repo_id"]
    db_path = integration_service["db_path"]

    cli = CLI(config_path=None, db_path=str(db_path))
    cli.service = service

    exit_code = cli.run(["test", "run", str(repo_id), "abc1234"])

    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "Overall" in captured
    assert "dummy" in captured
