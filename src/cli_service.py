"""Service command helpers for Full Auto CI CLI."""

from __future__ import annotations

import logging
import multiprocessing
import os
import signal
import sys
import time
import webbrowser
from typing import TYPE_CHECKING, Any, Optional

from .dashboard import create_app
from .service import CIService

if TYPE_CHECKING:  # pragma: no cover - typing only
    from argparse import Namespace

    from .cli import CLI


logger = logging.getLogger(__name__)


def register_service_commands(subparsers) -> None:
    """Register the top-level service subcommands."""

    service_parser = subparsers.add_parser("service", help="Service management")
    service_subparsers = service_parser.add_subparsers(dest="service_command")
    service_subparsers.add_parser("start", help="Start the CI service")
    service_subparsers.add_parser("stop", help="Stop the CI service")
    service_subparsers.add_parser("status", help="Check service status")


def handle_service_command(cli: "CLI", args: "Namespace") -> int:
    """Dispatch a parsed service command."""

    handler_map = {
        "start": _service_start,
        "stop": _service_stop,
        "status": _service_status,
    }

    command = getattr(args, "service_command", None)
    handler = handler_map.get(command)
    if handler is None:
        print(f"Error: Unknown service command {command}")
        return 1
    return handler(cli)


def _service_start(cli: "CLI") -> int:
    existing_pid = _read_pid(cli)
    if existing_pid and _is_pid_running(existing_pid):
        print(f"Service already running (PID {existing_pid})")
        return 0

    dashboard_cfg = _dashboard_config(cli)
    dashboard_url = _dashboard_url(dashboard_cfg)

    process = _launch_service_process(cli)
    if process is None:
        print("Error: Service failed to start. Check logs for details.")
        return 1

    _write_pid_file(cli, process.pid)
    print(f"Service started in background (PID {process.pid}).")
    if dashboard_url:
        print(f"Dashboard available at {dashboard_url}")
    _maybe_start_dashboard(cli, dashboard_cfg)
    if dashboard_url:
        _maybe_open_dashboard(dashboard_url, dashboard_cfg)
    return 0


def _service_stop(cli: "CLI") -> int:
    pid = _read_pid(cli)
    if not pid or not _is_pid_running(pid):
        print("Service is not running")
        _remove_pid_file(cli)
        _maybe_cleanup_dashboard(cli)
        return 0

    print(f"Stopping service (PID {pid})...")
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as error:
        logger.error("Failed to signal service process %s: %s", pid, error)
        return 1

    if not _wait_for_pid_exit(pid, timeout=10.0):
        print(f"Service did not terminate in time (PID {pid})")
        return 1

    _remove_pid_file(cli)
    _stop_dashboard_process(cli)
    print("Service stopped")
    return 0


def _service_status(cli: "CLI") -> int:
    pid = _read_pid(cli)
    if pid and _is_pid_running(pid):
        print(f"Service is running (PID {pid})")
    else:
        print("Service is not running")
        _remove_pid_file(cli)

    dashboard_pid = _read_dashboard_pid(cli)
    if dashboard_pid and _is_pid_running(dashboard_pid):
        print(f"Dashboard is running (PID {dashboard_pid})")
    elif dashboard_pid:
        _remove_dashboard_pid(cli)

    return 0


def _dashboard_config(cli: "CLI") -> dict[str, Any]:
    config = cli.service.config.get("dashboard") or {}
    return dict(config)


def _dashboard_url(config: dict[str, Any]) -> str:
    if not config:
        host = "127.0.0.1"
        port = 8000
    else:
        host = str(config.get("host", "127.0.0.1"))
        raw_port = config.get("port", 8000)
        port = int(raw_port or 8000)

    visible_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    return f"http://{visible_host}:{port}"


def _launch_service_process(cli: "CLI") -> Optional[multiprocessing.Process]:
    process = multiprocessing.Process(
        target=_run_service_process,
        args=(cli.service.config.config_path, cli.service.db_path),
        daemon=False,
    )
    process.start()
    time.sleep(0.5)
    if not process.is_alive():
        return None
    return process


def _run_service_process(config_path: Optional[str], db_path: Optional[str]) -> None:
    service = CIService(config_path=config_path, db_path=db_path)

    def _shutdown_handler(_signum, _frame):
        logging.getLogger(__name__).info("Shutdown signal received; stopping service")
        service.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown_handler)
    signal.signal(signal.SIGINT, _shutdown_handler)

    service.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        service.stop()


def _run_dashboard_process(config_path: Optional[str], db_path: Optional[str]) -> None:
    app = create_app(config_path=config_path, db_path=db_path)
    service = app.config["CI_SERVICE"]
    dashboard_cfg = service.config.get("dashboard") or {}

    host = str(dashboard_cfg.get("host", "127.0.0.1"))
    port = int(dashboard_cfg.get("port", 8000) or 8000)
    debug = bool(dashboard_cfg.get("debug", False))

    logger.info("Starting dashboard on %s:%s", host, port)
    try:
        app.run(host=host, port=port, debug=debug, use_reloader=False)
    finally:
        logger.info("Dashboard process exiting")


def _maybe_start_dashboard(cli: "CLI", dashboard_cfg: dict[str, Any]) -> Optional[int]:
    if not _should_auto_start_dashboard(dashboard_cfg):
        return None

    existing_pid = _read_dashboard_pid(cli)
    if existing_pid and _is_pid_running(existing_pid):
        print(f"Dashboard already running (PID {existing_pid}).")
        return existing_pid

    if existing_pid:
        _remove_dashboard_pid(cli)

    process = multiprocessing.Process(
        target=_run_dashboard_process,
        args=(cli.service.config.config_path, cli.service.db_path),
        daemon=False,
    )
    process.start()
    time.sleep(0.5)

    if not process.is_alive():
        print("Warning: Dashboard failed to start. Check logs for details.")
        return None

    _write_dashboard_pid(cli, process.pid)
    print(f"Dashboard started in background (PID {process.pid}).")
    return process.pid


def _should_auto_start_dashboard(dashboard_cfg: dict[str, Any]) -> bool:
    env_flag = os.getenv("FULL_AUTO_CI_START_DASHBOARD")
    if env_flag is not None:
        return env_flag.strip().lower() not in {"0", "false", "no"}
    return bool(dashboard_cfg.get("auto_start", True))


def _stop_dashboard_process(cli: "CLI") -> None:
    pid = _read_dashboard_pid(cli)
    if not pid:
        _remove_dashboard_pid(cli)
        return

    if not _is_pid_running(pid):
        _remove_dashboard_pid(cli)
        return

    print(f"Stopping dashboard (PID {pid})...")
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as error:
        logger.warning("Failed to signal dashboard process %s: %s", pid, error)
        _remove_dashboard_pid(cli)
        return

    if not _wait_for_pid_exit(pid, timeout=5.0):
        print(f"Dashboard did not terminate in time (PID {pid})")
        return

    print("Dashboard stopped")
    _remove_dashboard_pid(cli)


def _maybe_cleanup_dashboard(cli: "CLI") -> None:
    pid = _read_dashboard_pid(cli)
    if pid and not _is_pid_running(pid):
        _remove_dashboard_pid(cli)


def _maybe_open_dashboard(url: str, dashboard_cfg: dict[str, Any]) -> None:
    if not _should_open_dashboard(dashboard_cfg):
        return

    if not bool(dashboard_cfg.get("auto_start", True)):
        logger.info(
            "dashboard.auto_start disabled; ensure the server is running before using %s",
            url,
        )

    try:
        if webbrowser.open(url, new=2):
            print(f"Opened {url} in your browser.")
        else:
            logger.info("Browser reported failure to open %s", url)
    except Exception as error:  # pylint: disable=broad-except
        logger.warning("Unable to open browser for %s: %s", url, error)


def _should_open_dashboard(dashboard_cfg: dict[str, Any]) -> bool:
    env_flag = os.getenv("FULL_AUTO_CI_OPEN_BROWSER")
    if env_flag is not None:
        return env_flag.strip().lower() not in {"0", "false", "no"}
    return bool(dashboard_cfg.get("auto_open", True))


def _wait_for_pid_exit(pid: int, *, timeout: float) -> bool:
    waited = 0.0
    while _is_pid_running(pid) and waited < timeout:
        time.sleep(0.2)
        waited += 0.2
    return not _is_pid_running(pid)


def _pid_file_path(cli: "CLI") -> str:
    base_dir = cli.service.config.config.get(
        "data_directory", os.path.expanduser("~/.fullautoci")
    )
    base_dir = os.path.expanduser(str(base_dir))
    return os.path.join(base_dir, "service.pid")


def _write_pid_file(cli: "CLI", pid: int) -> None:
    pid_path = _pid_file_path(cli)
    try:
        os.makedirs(os.path.dirname(pid_path), exist_ok=True)
        with open(pid_path, "w", encoding="utf-8") as handle:
            handle.write(str(pid))
    except OSError as error:
        logger.warning("Unable to write PID file %s: %s", pid_path, error)


def _remove_pid_file(cli: "CLI") -> None:
    pid_path = _pid_file_path(cli)
    try:
        os.remove(pid_path)
    except FileNotFoundError:
        pass
    except OSError as error:
        logger.warning("Unable to remove PID file %s: %s", pid_path, error)


def _read_pid(cli: "CLI") -> Optional[int]:
    pid_path = _pid_file_path(cli)
    try:
        with open(pid_path, "r", encoding="utf-8") as handle:
            return int(handle.read().strip())
    except (FileNotFoundError, ValueError):
        return None


def _dashboard_pid_path(cli: "CLI") -> str:
    base_dir = cli.service.config.config.get(
        "data_directory", os.path.expanduser("~/.fullautoci")
    )
    base_dir = os.path.expanduser(str(base_dir))
    return os.path.join(base_dir, "dashboard.pid")


def _write_dashboard_pid(cli: "CLI", pid: int) -> None:
    pid_path = _dashboard_pid_path(cli)
    try:
        os.makedirs(os.path.dirname(pid_path), exist_ok=True)
        with open(pid_path, "w", encoding="utf-8") as handle:
            handle.write(str(pid))
    except OSError as error:
        logger.warning("Unable to write dashboard PID file %s: %s", pid_path, error)


def _remove_dashboard_pid(cli: "CLI") -> None:
    pid_path = _dashboard_pid_path(cli)
    try:
        os.remove(pid_path)
    except FileNotFoundError:
        pass
    except OSError as error:
        logger.warning("Unable to remove dashboard PID file %s: %s", pid_path, error)


def _read_dashboard_pid(cli: "CLI") -> Optional[int]:
    pid_path = _dashboard_pid_path(cli)
    try:
        with open(pid_path, "r", encoding="utf-8") as handle:
            return int(handle.read().strip())
    except (FileNotFoundError, ValueError):
        return None


def _is_pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True
