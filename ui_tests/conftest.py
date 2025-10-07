"""Fixtures and helpers for Playwright UI tests."""

from __future__ import annotations

import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict

import pytest
import yaml
from werkzeug.serving import make_server

from src.dashboard import create_app
from src.db import DataAccess

pytest_plugins = ["pytest_playwright.pytest_playwright"]


class _DashboardServer(threading.Thread):
    """Run the Flask dashboard inside a background thread."""

    def __init__(self, app, host: str, port: int) -> None:
        super().__init__(daemon=True)
        self._app = app
        self._host = host
        self._port = port
        self._server = make_server(host, port, app)
        self._context = app.app_context()

    def run(self) -> None:  # noqa: D401 - inherited docstring
        self._context.push()
        try:
            self._server.serve_forever()
        finally:
            self._context.pop()

    def stop(self) -> None:
        self._server.shutdown()


def _write_dashboard_config(config_path: Path, host: str, port: int) -> None:
    config_data: Dict[str, Dict[str, object]] = {
        "dashboard": {
            "host": host,
            "port": port,
            "debug": False,
            "auto_open": False,
            "auto_start": False,
        },
        "dogfood": {
            "enabled": False,
        },
    }
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config_data, handle, default_flow_style=False)


def _wait_for_server(url: str, timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1):
                return
        except (urllib.error.URLError, ConnectionError):
            time.sleep(0.1)
    raise RuntimeError(f"Dashboard did not become ready at {url}")


@pytest.fixture(scope="session")
def dashboard_server(tmp_path_factory: pytest.TempPathFactory):
    """Start a dashboard instance that Playwright tests can target."""

    work_dir = tmp_path_factory.mktemp("full_auto_ci_ui")
    config_path = work_dir / "config.yml"
    db_path = work_dir / "database.sqlite"

    host = os.getenv("FULL_AUTO_CI_UI_TEST_HOST", "127.0.0.1")
    port = int(os.getenv("FULL_AUTO_CI_UI_TEST_PORT", "8123"))

    _write_dashboard_config(config_path, host, port)

    data_access = DataAccess(str(db_path))
    data_access.initialize_schema()

    app = create_app(config_path=str(config_path), db_path=str(db_path))

    os.environ.setdefault("FULL_AUTO_CI_OPEN_BROWSER", "0")
    os.environ.setdefault("FULL_AUTO_CI_START_DASHBOARD", "1")

    server = _DashboardServer(app, host, port)
    server.start()
    _wait_for_server(f"http://{host}:{port}/")

    try:
        yield {
            "base_url": f"http://{host}:{port}",
            "data_access": data_access,
            "config_path": str(config_path),
            "db_path": str(db_path),
        }
    finally:
        server.stop()
        server.join(timeout=5)


@pytest.fixture()
def dashboard_data_access(dashboard_server):
    """Expose the dashboard's data access helper for test setup."""

    return dashboard_server["data_access"]


@pytest.fixture(scope="session")
def base_url(dashboard_server):
    """Provide the dashboard base URL for pytest-playwright."""

    return dashboard_server["base_url"]
