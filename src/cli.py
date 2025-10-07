"""Command-line interface for Full Auto CI."""

import argparse
import asyncio
import json
import logging
import multiprocessing
import os
import signal
import sys
import time
import webbrowser
from typing import Any, Dict, List, Optional

from .dashboard import create_app
from .mcp import MCPServer
from .service import CIService

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _run_service_process(config_path: Optional[str], db_path: Optional[str]) -> None:
    """Entry point for the background service process."""

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


class CLI:
    """Command-line interface for interacting with the CI service."""

    def __init__(
        self, config_path: Optional[str] = None, db_path: Optional[str] = None
    ):
        """Initialize the CLI.

        Args:
            config_path: Path to the configuration file
            db_path: Path to the SQLite database
        """
        self.config_path = config_path
        self.db_path = db_path
        self.service = CIService(config_path=config_path, db_path=db_path)

    def parse_args(self, args: List[str]) -> argparse.Namespace:
        """Parse command line arguments.

        Args:
            args: Command line arguments

        Returns:
            Parsed arguments
        """
        parser = argparse.ArgumentParser(
            description="Full Auto CI - Automated Continuous Integration"
        )
        subparsers = parser.add_subparsers(dest="command", help="Command to run")

        # Service commands
        service_parser = subparsers.add_parser("service", help="Service management")
        service_subparsers = service_parser.add_subparsers(dest="service_command")
        service_subparsers.add_parser("start", help="Start the CI service")
        service_subparsers.add_parser("stop", help="Stop the CI service")
        service_subparsers.add_parser("status", help="Check service status")

        # Repository commands
        repo_parser = subparsers.add_parser("repo", help="Repository management")
        repo_subparsers = repo_parser.add_subparsers(dest="repo_command")
        repo_subparsers.add_parser("list", help="List repositories")

        add_parser = repo_subparsers.add_parser("add", help="Add a repository")
        add_parser.add_argument("name", help="Repository name")
        add_parser.add_argument("url", help="Repository URL")
        add_parser.add_argument("--branch", default="main", help="Branch to monitor")

        remove_parser = repo_subparsers.add_parser("remove", help="Remove a repository")
        remove_parser.add_argument("repo_id", type=int, help="Repository ID")

        # Test commands
        test_parser = subparsers.add_parser("test", help="Test management")
        test_subparsers = test_parser.add_subparsers(dest="test_command")

        run_parser = test_subparsers.add_parser("run", help="Run tests manually")
        run_parser.add_argument("repo_id", type=int, help="Repository ID")
        run_parser.add_argument("commit", help="Commit hash")

        results_parser = test_subparsers.add_parser("results", help="Get test results")
        results_parser.add_argument("repo_id", type=int, help="Repository ID")
        results_parser.add_argument("--commit", help="Commit hash (optional)")

        # Configuration commands
        config_parser = subparsers.add_parser("config", help="Configuration management")
        config_subparsers = config_parser.add_subparsers(dest="config_command")

        show_parser = config_subparsers.add_parser(
            "show", help="Show configuration values"
        )
        show_parser.add_argument(
            "section", nargs="?", help="Configuration section to display"
        )
        show_parser.add_argument(
            "key", nargs="?", help="Specific key within the section"
        )
        show_parser.add_argument(
            "--json", action="store_true", help="Output configuration in JSON format"
        )

        set_parser = config_subparsers.add_parser(
            "set", help="Update a configuration value"
        )
        set_parser.add_argument("section", help="Configuration section")
        set_parser.add_argument("key", help="Configuration key")
        set_parser.add_argument("value", help="New value (use JSON for complex types)")

        config_subparsers.add_parser("path", help="Show configuration file path")

        # User management commands
        user_parser = subparsers.add_parser("user", help="User management")
        user_subparsers = user_parser.add_subparsers(dest="user_command")

        user_subparsers.add_parser("list", help="List users")

        user_add = user_subparsers.add_parser("add", help="Add a user")
        user_add.add_argument("username", help="Username")
        user_add.add_argument("password", help="Password")
        user_add.add_argument("--role", default="user", help="Role (default: user)")
        user_add.add_argument("--api-key", dest="api_key", help="Optional API key")

        user_remove = user_subparsers.add_parser("remove", help="Remove a user")
        user_remove.add_argument("username", help="Username to remove")

        # MCP commands
        mcp_parser = subparsers.add_parser("mcp", help="Model Context Protocol server")
        mcp_subparsers = mcp_parser.add_subparsers(dest="mcp_command")

        mcp_serve = mcp_subparsers.add_parser("serve", help="Start an MCP server endpoint")
        mcp_serve.add_argument("--host", default="127.0.0.1", help="Bind host")
        mcp_serve.add_argument("--port", type=int, default=8765, help="Bind port")
        mcp_serve.add_argument(
            "--token",
            help="Authentication token (defaults to FULL_AUTO_CI_MCP_TOKEN)",
        )
        mcp_serve.add_argument(
            "--no-token",
            action="store_true",
            help="Disable token requirement even if FULL_AUTO_CI_MCP_TOKEN is set",
        )

        return parser.parse_args(args)

    def run(self, args: Optional[List[str]] = None) -> int:
        """Run the CLI with the given arguments.

        Args:
            args: Command line arguments, defaults to sys.argv[1:]

        Returns:
            Exit code
        """
        if args is None:
            args = sys.argv[1:]

        try:
            parsed_args = self.parse_args(args)
        except SystemExit:
            return 1

        if not parsed_args.command:
            print("Error: No command specified")
            return 1

        if parsed_args.command == "service":
            return self._handle_service_command(parsed_args)
        elif parsed_args.command == "repo":
            return self._handle_repo_command(parsed_args)
        elif parsed_args.command == "test":
            return self._handle_test_command(parsed_args)
        elif parsed_args.command == "config":
            return self._handle_config_command(parsed_args)
        elif parsed_args.command == "user":
            return self._handle_user_command(parsed_args)
        elif parsed_args.command == "mcp":
            return self._handle_mcp_command(parsed_args)
        else:
            print(f"Error: Unknown command {parsed_args.command}")
            return 1

    def _handle_service_command(self, args: argparse.Namespace) -> int:
        """Handle service commands.

        Args:
            args: Parsed arguments

        Returns:
            Exit code
        """
        if args.service_command == "start":
            existing_pid = self._read_pid()
            if existing_pid and self._is_pid_running(existing_pid):
                print(f"Service already running (PID {existing_pid})")
                return 0

            dashboard_cfg = self.service.config.get("dashboard") or {}
            host = str(dashboard_cfg.get("host", "127.0.0.1"))
            port = dashboard_cfg.get("port", 8000)
            visible_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
            dashboard_url = f"http://{visible_host}:{port}"

            process = multiprocessing.Process(
                target=_run_service_process,
                args=(self.service.config.config_path, self.service.db_path),
                daemon=False,
            )
            process.start()
            time.sleep(0.5)

            if not process.is_alive():
                print("Error: Service failed to start. Check logs for details.")
                return 1

            self._write_pid_file(process.pid)
            print(f"Service started in background (PID {process.pid}).")
            print(f"Dashboard available at {dashboard_url}")
            self._maybe_start_dashboard(dashboard_cfg)
            self._maybe_open_dashboard(dashboard_url, dashboard_cfg)
            return 0
        elif args.service_command == "stop":
            pid = self._read_pid()
            if not pid:
                print("Service is not running")
                self._remove_pid_file()
                self._maybe_cleanup_dashboard()
                return 0

            if not self._is_pid_running(pid):
                print("Service is not running")
                self._remove_pid_file()
                self._maybe_cleanup_dashboard()
                return 0

            print(f"Stopping service (PID {pid})...")
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError as error:
                logger.error("Failed to signal service process %s: %s", pid, error)
                return 1

            waited = 0.0
            while self._is_pid_running(pid) and waited < 10.0:
                time.sleep(0.2)
                waited += 0.2

            if self._is_pid_running(pid):
                print(f"Service did not terminate in time (PID {pid})")
                return 1

            self._remove_pid_file()
            self._stop_dashboard_process()
            print("Service stopped")
            return 0
        elif args.service_command == "status":
            pid = self._read_pid()
            if pid and self._is_pid_running(pid):
                print(f"Service is running (PID {pid})")
            else:
                print("Service is not running")
                self._remove_pid_file()
            dashboard_pid = self._read_dashboard_pid()
            if dashboard_pid and self._is_pid_running(dashboard_pid):
                print(f"Dashboard is running (PID {dashboard_pid})")
            elif dashboard_pid:
                self._remove_dashboard_pid()
            return 0
        else:
            print(f"Error: Unknown service command {args.service_command}")
            return 1

    def _handle_mcp_command(self, args: argparse.Namespace) -> int:
        if args.mcp_command == "serve":
            token = None
            if not args.no_token:
                token = args.token or os.getenv("FULL_AUTO_CI_MCP_TOKEN")

            server = MCPServer(self.service, auth_token=token)
            host = args.host
            port = args.port

            print(f"Starting MCP server on {host}:{port} (token={'enabled' if token else 'disabled'})")

            async def runner() -> None:
                try:
                    await server.serve_tcp(host=host, port=port)
                except asyncio.CancelledError:  # pragma: no cover
                    pass

            try:
                asyncio.run(runner())
            except KeyboardInterrupt:
                print("MCP server stopped")
                return 0
            return 0

        print(f"Error: Unknown MCP command {args.mcp_command}")
        return 1

    def _handle_repo_command(self, args: argparse.Namespace) -> int:
        """Handle repository commands.

        Args:
            args: Parsed arguments

        Returns:
            Exit code
        """
        if args.repo_command == "list":
            repos = self.service.list_repositories()
            if not repos:
                print("No repositories configured")
            else:
                print("ID | Name | URL | Branch")
                print("-" * 50)
                for repo in repos:
                    print(
                        f"{repo['id']} | {repo['name']} | {repo['url']} | {repo['branch']}"
                    )
            return 0
        elif args.repo_command == "add":
            repo_id = self.service.add_repository(args.name, args.url, args.branch)
            print(f"Repository added with ID: {repo_id}")
            return 0
        elif args.repo_command == "remove":
            success = self.service.remove_repository(args.repo_id)
            if success:
                print(f"Repository with ID {args.repo_id} removed")
                return 0
            else:
                print(f"Error: Repository with ID {args.repo_id} not found")
                return 1
        else:
            print(f"Error: Unknown repo command {args.repo_command}")
            return 1

    def _handle_test_command(self, args: argparse.Namespace) -> int:
        """Handle test commands.

        Args:
            args: Parsed arguments

        Returns:
            Exit code
        """
        if args.test_command == "run":
            results = self.service.run_tests(args.repo_id, args.commit)
            print("Test results:")
            print(f"Overall status: {results['status']}")
            for warning in results.get("warnings", []):
                print(f"Warning: {warning}")
            for tool, result in results["tools"].items():
                print(f"- {tool}: {result['status']}")
            return 0
        elif args.test_command == "results":
            runs = self.service.get_test_results(
                args.repo_id, commit_hash=getattr(args, "commit", None), limit=10
            )

            if not runs:
                if args.commit:
                    print(
                        f"No test runs found for repository {args.repo_id} and commit {args.commit}."
                    )
                else:
                    print(f"No test runs found for repository {args.repo_id}.")
                return 0

            print(f"Test runs for repository {args.repo_id}")
            if args.commit:
                print(f"Filtered by commit: {args.commit}")

            for run in runs:
                print(
                    f"Run {run['id']} | Commit: {run['commit_hash']} | Status: {run['status']}"
                )
                created = run.get("created_at", "-")
                started = run.get("started_at") or "-"
                completed = run.get("completed_at") or "-"
                print(
                    f"  Created: {created} | Started: {started} | Completed: {completed}"
                )

                commit = run.get("commit") or {}
                message = commit.get("message")
                if message:
                    print(f"  Message: {message}")

                if run.get("error"):
                    print(f"  Error: {run['error']}")

                results = run.get("results") or []
                if not results:
                    print("  (no tool results persisted)")
                    continue

                for result in results:
                    tool = result.get("tool", "unknown")
                    status = result.get("status", "unknown")
                    summary = self._summarize_tool_output(
                        result.get("output"), status
                    )
                    line = f"  - {tool}: {status}"
                    if summary:
                        line += f" ({summary})"
                    print(line)

            return 0
        else:
            print(f"Error: Unknown test command {args.test_command}")
            return 1

    def _handle_config_command(self, args: argparse.Namespace) -> int:
        """Handle configuration commands."""
        command = getattr(args, "config_command", None)
        config = self.service.config

        if command == "show":
            section = getattr(args, "section", None)
            key = getattr(args, "key", None)
            data: Any

            if section is None:
                data = config.config
            else:
                section_data = config.get(section)
                if section_data is None:
                    print(f"Error: Configuration section '{section}' not found")
                    return 1
                if key is None:
                    data = section_data
                else:
                    value = config.get(section, key)
                    if value is None:
                        print(f"Error: Key '{key}' not found in section '{section}'")
                        return 1
                    data = value

            self._print_config_data(data, args.json)
            return 0

        if command == "set":
            section = args.section
            key = args.key
            value = self._parse_config_value(args.value)
            config.set(section, key, value)
            if config.save():
                print(f"Updated {section}.{key} = {value}")
                return 0
            print("Error: Failed to save configuration")
            return 1

        if command == "path":
            print(config.config_path)
            return 0

        print(f"Error: Unknown config command {command}")
        return 1

    def _handle_user_command(self, args: argparse.Namespace) -> int:
        """Handle user management commands."""
        command = getattr(args, "user_command", None)

        if command == "list":
            users = self.service.list_users()
            if not users:
                print("No users found")
                return 0
            print("ID | Username | Role | Created")
            print("-" * 60)
            for user in users:
                created = user.get("created_at") or ""
                print(
                    f"{user['id']} | {user['username']} | {user.get('role', 'user')} | {created}"
                )
            return 0

        if command == "add":
            try:
                user_id = self.service.create_user(
                    args.username,
                    args.password,
                    role=getattr(args, "role", "user"),
                    api_key=getattr(args, "api_key", None),
                )
            except ValueError as exc:
                print(f"Error: {exc}")
                return 1
            print(f"User '{args.username}' created with id {user_id}")
            return 0

        if command == "remove":
            success = self.service.remove_user(args.username)
            if success:
                print(f"User '{args.username}' removed")
                return 0
            print(f"Error: User '{args.username}' not found")
            return 1

        print(f"Error: Unknown user command {command}")
        return 1

    @staticmethod
    def _print_config_data(data: Any, as_json: bool):
        if as_json or isinstance(data, (dict, list)):
            try:
                print(json.dumps(data, indent=2, sort_keys=True))
            except TypeError:
                print(str(data))
        else:
            print(data)

    @staticmethod
    def _parse_config_value(raw: str) -> Any:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            lowered = raw.lower()
            if lowered == "true":
                return True
            if lowered == "false":
                return False
            if lowered == "null":
                return None
            try:
                if "." in raw:
                    return float(raw)
                return int(raw)
            except ValueError:
                return raw

    @staticmethod
    def _summarize_tool_output(
        raw_output: Optional[str], result_status: Optional[str] = None
    ) -> Optional[str]:
        if not raw_output:
            return None

        try:
            payload = json.loads(raw_output)
        except (TypeError, json.JSONDecodeError):
            return None

        if not isinstance(payload, dict):
            return None

        extras: List[str] = []

        status = payload.get("status")
        if status and status != result_status:
            extras.append(str(status))

        score = payload.get("score")
        if isinstance(score, (int, float)):
            extras.append(f"score {score:g}")

        percentage = payload.get("percentage")
        if isinstance(percentage, (int, float)):
            extras.append(f"{percentage:.2f}%")

        # Provide a single detail message if available when no other extras
        if not extras:
            details = payload.get("details")
            if isinstance(details, list) and details:
                first_detail = details[0]
                if isinstance(first_detail, dict) and first_detail.get("message"):
                    extras.append(first_detail["message"])

        if not extras:
            return None

        return ", ".join(str(item) for item in extras)

    def _pid_file_path(self) -> str:
        base_dir = self.service.config.config.get(
            "data_directory", os.path.expanduser("~/.fullautoci")
        )
        base_dir = os.path.expanduser(str(base_dir))
        return os.path.join(base_dir, "service.pid")

    def _write_pid_file(self, pid: int) -> None:
        pid_path = self._pid_file_path()
        try:
            os.makedirs(os.path.dirname(pid_path), exist_ok=True)
            with open(pid_path, "w", encoding="utf-8") as handle:
                handle.write(str(pid))
        except OSError as error:
            logger.warning("Unable to write PID file %s: %s", pid_path, error)

    def _remove_pid_file(self) -> None:
        pid_path = self._pid_file_path()
        try:
            os.remove(pid_path)
        except FileNotFoundError:
            pass
        except OSError as error:
            logger.warning("Unable to remove PID file %s: %s", pid_path, error)

    def _read_pid(self) -> Optional[int]:
        pid_path = self._pid_file_path()
        try:
            with open(pid_path, "r", encoding="utf-8") as handle:
                return int(handle.read().strip())
        except (FileNotFoundError, ValueError):
            return None

    @staticmethod
    def _is_pid_running(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def _dashboard_pid_path(self) -> str:
        base_dir = self.service.config.config.get(
            "data_directory", os.path.expanduser("~/.fullautoci")
        )
        base_dir = os.path.expanduser(str(base_dir))
        return os.path.join(base_dir, "dashboard.pid")

    def _write_dashboard_pid(self, pid: int) -> None:
        pid_path = self._dashboard_pid_path()
        try:
            os.makedirs(os.path.dirname(pid_path), exist_ok=True)
            with open(pid_path, "w", encoding="utf-8") as handle:
                handle.write(str(pid))
        except OSError as error:
            logger.warning("Unable to write dashboard PID file %s: %s", pid_path, error)

    def _remove_dashboard_pid(self) -> None:
        pid_path = self._dashboard_pid_path()
        try:
            os.remove(pid_path)
        except FileNotFoundError:
            pass
        except OSError as error:
            logger.warning(
                "Unable to remove dashboard PID file %s: %s", pid_path, error
            )

    def _read_dashboard_pid(self) -> Optional[int]:
        pid_path = self._dashboard_pid_path()
        try:
            with open(pid_path, "r", encoding="utf-8") as handle:
                return int(handle.read().strip())
        except (FileNotFoundError, ValueError):
            return None

    def _maybe_start_dashboard(self, dashboard_cfg: Dict[str, Any]) -> Optional[int]:
        env_flag = os.getenv("FULL_AUTO_CI_START_DASHBOARD")
        if env_flag is not None:
            auto_start = env_flag.strip().lower() not in {"0", "false", "no"}
        else:
            auto_start = bool(dashboard_cfg.get("auto_start", True))

        if not auto_start:
            return None

        existing_pid = self._read_dashboard_pid()
        if existing_pid and self._is_pid_running(existing_pid):
            print(f"Dashboard already running (PID {existing_pid}).")
            return existing_pid
        elif existing_pid:
            self._remove_dashboard_pid()

        process = multiprocessing.Process(
            target=_run_dashboard_process,
            args=(self.service.config.config_path, self.service.db_path),
            daemon=False,
        )
        process.start()
        time.sleep(0.5)

        if not process.is_alive():
            print("Warning: Dashboard failed to start. Check logs for details.")
            return None

        self._write_dashboard_pid(process.pid)
        print(f"Dashboard started in background (PID {process.pid}).")
        return process.pid

    def _stop_dashboard_process(self) -> None:
        pid = self._read_dashboard_pid()
        if not pid:
            self._remove_dashboard_pid()
            return

        if not self._is_pid_running(pid):
            self._remove_dashboard_pid()
            return

        print(f"Stopping dashboard (PID {pid})...")
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as error:
            logger.warning("Failed to signal dashboard process %s: %s", pid, error)
            self._remove_dashboard_pid()
            return

        waited = 0.0
        while self._is_pid_running(pid) and waited < 5.0:
            time.sleep(0.2)
            waited += 0.2

        if self._is_pid_running(pid):
            print(f"Dashboard did not terminate in time (PID {pid})")
        else:
            print("Dashboard stopped")
            self._remove_dashboard_pid()

    def _maybe_cleanup_dashboard(self) -> None:
        pid = self._read_dashboard_pid()
        if pid and not self._is_pid_running(pid):
            self._remove_dashboard_pid()

    def _maybe_open_dashboard(self, url: str, dashboard_cfg: Dict[str, Any]) -> None:
        env_flag = os.getenv("FULL_AUTO_CI_OPEN_BROWSER")

        if env_flag is not None:
            auto_open = env_flag.strip().lower() not in {"0", "false", "no"}
        else:
            auto_open = bool(dashboard_cfg.get("auto_open", True))

        if not auto_open:
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


def main() -> int:
    """Entry point for the CLI."""
    cli = CLI()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())
