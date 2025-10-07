"""Tests for the REST API layer."""

from __future__ import annotations

import hashlib
import logging
import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.api import API


def _init_api_tables(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY,
                key_hash TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS repositories (
                id INTEGER PRIMARY KEY,
                url TEXT NOT NULL,
                name TEXT NOT NULL,
                status TEXT DEFAULT 'active',
                last_check INTEGER
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS test_runs (
                id INTEGER PRIMARY KEY,
                repository_id INTEGER NOT NULL,
                commit_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                completed_at INTEGER,
                FOREIGN KEY (repository_id) REFERENCES repositories (id)
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture(name="api_app")
def _api_app(tmp_path, monkeypatch):
    db_path = tmp_path / "api.sqlite"
    _init_api_tables(db_path)

    service = MagicMock()
    webhook = MagicMock()

    monkeypatch.setattr("src.api.CIService", MagicMock(return_value=service))
    monkeypatch.setattr("src.api.WebhookHandler", MagicMock(return_value=webhook))

    api_instance = API(db_path=str(db_path))
    assert api_instance.app is not None  # Flask should be available in test env
    return api_instance, service, webhook, db_path


def test_health_check(api_app):
    api_instance, _service, _webhook, _db = api_app
    client = api_instance.app.test_client()

    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "healthy"


def test_generate_and_verify_api_key(api_app, monkeypatch):
    api_instance, _service, _webhook, db_path = api_app
    client = api_instance.app.test_client()

    monkeypatch.setattr("src.api.secrets.token_hex", lambda n=32: "deadbeef")

    generate_response = client.post("/api/key/generate")
    assert generate_response.status_code == 200
    api_key = generate_response.get_json()["api_key"]
    assert api_key == "deadbeef"

    verify_response = client.post("/api/key/verify", json={"api_key": api_key})
    assert verify_response.status_code == 200
    assert verify_response.get_json()["valid"] is True

    # Unknown key should be rejected
    invalid_response = client.post("/api/key/verify", json={"api_key": "nope"})
    assert invalid_response.status_code == 401

    # Confirm the stored hash matches our generated key
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT key_hash FROM api_keys")
        stored_hash = cursor.fetchone()[0]
    finally:
        conn.close()
    assert stored_hash == hashlib.sha256(api_key.encode()).hexdigest()


def test_list_and_get_repository(api_app):
    api_instance, _service, _webhook, db_path = api_app
    client = api_instance.app.test_client()

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO repositories (id, url, name, status, last_check) VALUES (?, ?, ?, ?, ?)",
            (1, "https://example.com/repo.git", "Repo", "active", 1234),
        )
        conn.commit()
    finally:
        conn.close()

    list_response = client.get("/api/repositories")
    assert list_response.status_code == 200
    repositories = list_response.get_json()["repositories"]
    assert repositories[0]["name"] == "Repo"

    detail_response = client.get("/api/repository/1")
    assert detail_response.status_code == 200
    assert detail_response.get_json()["id"] == 1

    missing_response = client.get("/api/repository/99")
    assert missing_response.status_code == 404


def test_repository_modification_endpoints(api_app):
    api_instance, service, _webhook, _db = api_app
    client = api_instance.app.test_client()

    service.add_repository.return_value = 7
    add_response = client.post(
        "/api/repository/add",
        json={"url": "https://example.com/new.git", "name": "New"},
    )
    assert add_response.status_code == 200
    service.add_repository.assert_called_once()

    service.remove_repository.return_value = False
    remove_response = client.delete("/api/repository/remove/7")
    assert remove_response.status_code == 404

    service.remove_repository.return_value = True
    success_response = client.delete("/api/repository/remove/7")
    assert success_response.status_code == 200


def test_test_results_endpoints(api_app):
    api_instance, _service, _webhook, db_path = api_app
    client = api_instance.app.test_client()

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO repositories (id, url, name, status, last_check) VALUES (1, 'url', 'Repo', 'active', 0)"
        )
        cursor.execute(
            "INSERT INTO test_runs (id, repository_id, commit_hash, status, created_at, completed_at)"
            " VALUES (1, 1, 'abc', 'completed', 10, 11)"
        )
        cursor.execute(
            "INSERT INTO test_runs (id, repository_id, commit_hash, status, created_at, completed_at)"
            " VALUES (2, 1, 'def', 'queued', 20, NULL)"
        )
        conn.commit()
    finally:
        conn.close()

    results_response = client.get("/api/tests/1")
    assert results_response.status_code == 200
    runs = results_response.get_json()["test_runs"]
    assert len(runs) == 2

    latest_response = client.get("/api/tests/latest")
    assert latest_response.status_code == 200
    latest_runs = latest_response.get_json()["test_runs"]
    assert latest_runs[0]["repository_name"] == "Repo"


def test_webhook_endpoints(api_app):
    api_instance, service, webhook, _db = api_app
    client = api_instance.app.test_client()

    webhook.handle.return_value = {"repository_id": 1, "hash": "abc"}
    response = client.post(
        "/webhook/github",
        data=json.dumps({"repository": {}}),
        content_type="application/json",
    )
    assert response.status_code == 200
    service.add_test_task.assert_called_with(1, "abc")

    webhook.handle.return_value = None
    failure = client.post(
        "/webhook/gitlab",
        data=json.dumps({"repository": {}}),
        content_type="application/json",
    )
    assert failure.status_code == 400

    webhook.handle.return_value = {"repository_id": 2, "hash": "def"}
    bitbucket = client.post(
        "/webhook/bitbucket",
        data=json.dumps({"repository": {}}),
        content_type="application/json",
    )
    assert bitbucket.status_code == 200
    service.add_test_task.assert_called_with(2, "def")


def test_api_run_logs(api_app, caplog):
    api_instance, _service, _webhook, _db = api_app
    caplog.clear()
    caplog.set_level(logging.INFO, logger="src.api")
    api_instance.run(host="0.0.0.0", port=9999, debug=True)
    messages = [record.message for record in caplog.records]
    assert any("API server would run on" in msg for msg in messages)
    assert any("Flask is not installed yet" in msg for msg in messages)
