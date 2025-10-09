"""Tests for the Full Auto CI dashboard."""

from __future__ import annotations

import os
import tempfile
import time
from unittest.mock import MagicMock

import pytest

from src.dashboard import create_app
from src.db import DataAccess


@pytest.fixture()
def dashboard_app(monkeypatch):
    monkeypatch.setenv("FULL_AUTO_CI_DOGFOOD", "0")
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test.sqlite")
        data = DataAccess(db_path)
        data.initialize_schema()

        repo_id = data.create_repository("Demo", "https://example.com/demo.git", "main")
        timestamp = int(time.time())
        run_id = data.create_test_run(repo_id, "abc1234", "completed", timestamp)
        data.update_test_run(run_id, status="completed", completed_at=timestamp)

        commit_id = data.create_commit(
            repo_id,
            "abc1234",
            author="Dev Bot",
            message="Initial commit",
            timestamp=timestamp,
        )
        data.insert_result(
            commit_id,
            tool="pylint",
            status="success",
            output="{}",
            duration=1.2,
        )

        app = create_app(db_path=db_path)
        app.config.update(TESTING=True)
        yield app


@pytest.fixture()
def client(dashboard_app):
    return dashboard_app.test_client()


def test_index_lists_repositories(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Demo" in response.data
    assert b"Repositories" in response.data


def test_repository_detail(client):
    # repository id is 1 because DataAccess autoincrements starting at 1
    response = client.get("/repo/1")
    assert response.status_code == 200
    body = response.data.decode()
    assert "Demo" in body
    assert "abc1234" in body
    assert "pylint" in body


def test_repositories_partial(client):
    response = client.get("/partials/repositories")
    assert response.status_code == 200
    body = response.data.decode()
    assert "Demo" in body
    assert "status-card" in body


def test_repository_insights_partial(client):
    response = client.get("/repo/1/insights")
    assert response.status_code == 200
    body = response.data.decode()
    assert "Recent Test Runs" in body
    assert "abc1234" in body


def test_dashboard_main_runs(monkeypatch):
    monkeypatch.setenv("FULL_AUTO_CI_DOGFOOD", "0")
    from src.dashboard import __main__ as dashboard_main

    mock_app = MagicMock()
    mock_service = MagicMock()
    mock_service.config.get.return_value = {
        "host": "0.0.0.0",
        "port": 9100,
        "debug": True,
    }
    mock_app.config = {"CI_SERVICE": mock_service}

    monkeypatch.setattr(dashboard_main, "create_app", lambda: mock_app)

    dashboard_main.main()

    mock_app.run.assert_called_once_with(host="0.0.0.0", port=9100, debug=True)
