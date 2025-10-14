"""Command-line interface for Full Auto CI."""

import argparse
import asyncio
import errno
import json
import logging
import multiprocessing
import os
import signal
import socket
import sys
import time
import webbrowser
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import __version__ as PACKAGE_VERSION
from .dashboard import create_app
from .mcp import MCPServer
from .providers import ProviderConfigError
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
        parser = self._build_parser()
        return parser.parse_args(args)

    def _build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Full Auto CI - Automated Continuous Integration"
        )
        subparsers = parser.add_subparsers(dest="command", help="Command to run")
        self._register_service_commands(subparsers)
        self._register_repo_commands(subparsers)
        self._register_test_commands(subparsers)
        self._register_config_commands(subparsers)
        self._register_user_commands(subparsers)
        self._register_mcp_commands(subparsers)
        self._register_provider_commands(subparsers)
        return parser

    def _register_service_commands(self, subparsers) -> None:
        service_parser = subparsers.add_parser("service", help="Service management")
        service_subparsers = service_parser.add_subparsers(dest="service_command")
        service_subparsers.add_parser("start", help="Start the CI service")
        service_subparsers.add_parser("stop", help="Stop the CI service")
        service_subparsers.add_parser("status", help="Check service status")

    def _register_repo_commands(self, subparsers) -> None:
        repo_parser = subparsers.add_parser("repo", help="Repository management")
        repo_subparsers = repo_parser.add_subparsers(dest="repo_command")
        repo_subparsers.add_parser("list", help="List repositories")

        add_parser = repo_subparsers.add_parser("add", help="Add a repository")
        add_parser.add_argument("name", help="Repository name")
        add_parser.add_argument("url", help="Repository URL")
        add_parser.add_argument("--branch", default="main", help="Branch to monitor")

        remove_parser = repo_subparsers.add_parser("remove", help="Remove a repository")
        remove_parser.add_argument("repo_id", type=int, help="Repository ID")

    def _register_test_commands(self, subparsers) -> None:
        test_parser = subparsers.add_parser("test", help="Test management")
        test_subparsers = test_parser.add_subparsers(dest="test_command")

        run_parser = test_subparsers.add_parser("run", help="Run tests manually")
        run_parser.add_argument("repo_id", type=int, help="Repository ID")
        run_parser.add_argument("commit", help="Commit hash")

        results_parser = test_subparsers.add_parser("results", help="Get test results")
        results_parser.add_argument("repo_id", type=int, help="Repository ID")
        results_parser.add_argument("--commit", help="Commit hash (optional)")

    def _register_config_commands(self, subparsers) -> None:
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

    def _register_user_commands(self, subparsers) -> None:
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

    def _register_mcp_commands(self, subparsers) -> None:
        mcp_parser = subparsers.add_parser("mcp", help="Model Context Protocol server")
        mcp_subparsers = mcp_parser.add_subparsers(dest="mcp_command")

        mcp_serve = mcp_subparsers.add_parser(
            "serve", help="Start an MCP server endpoint"
        )
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
        mcp_serve.add_argument(
            "--log-level",
            default="INFO",
            help="Log level for the MCP server (e.g., DEBUG, INFO)",
        )
        mcp_serve.add_argument(
            "--stdio",
            action="store_true",
            help="Serve the MCP protocol over standard input/output instead of TCP",
        )

    def _register_provider_commands(self, subparsers) -> None:
        provider_parser = subparsers.add_parser(
            "provider", help="External CI provider management"
        )
        provider_subparsers = provider_parser.add_subparsers(dest="provider_command")

        provider_subparsers.add_parser("list", help="List configured providers")
        provider_subparsers.add_parser("types", help="List available provider types")

        add_parser = provider_subparsers.add_parser(
            "add", help="Register a new external provider"
        )
        add_parser.add_argument("type", help="Provider type identifier (e.g. github)")
        add_parser.add_argument("name", help="Display name for the provider")
        add_parser.add_argument(
            "--config",
            help="Inline JSON configuration for the provider",
        )
        add_parser.add_argument(
            "--config-file",
            dest="config_file",
            help="Path to a JSON file containing provider configuration",
        )

        remove_parser = provider_subparsers.add_parser(
            "remove", help="Remove a configured provider"
        )
        remove_parser.add_argument("provider_id", type=int, help="Provider identifier")

        sync_parser = provider_subparsers.add_parser(
            "sync", help="Run a provider synchronization cycle"
        )
        sync_parser.add_argument("provider_id", type=int, help="Provider identifier")
        sync_parser.add_argument(
            "--limit",
            type=int,
            default=50,
            help="Maximum number of runs to fetch (default: 50)",
        )

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

        handler_map = {
            "service": self._handle_service_command,
            "repo": self._handle_repo_command,
            "test": self._handle_test_command,
            "config": self._handle_config_command,
            "user": self._handle_user_command,
            "mcp": self._handle_mcp_command,
            "provider": self._handle_provider_command,
        }

        handler = handler_map.get(parsed_args.command)
        if handler is None:
            print(f"Error: Unknown command {parsed_args.command}")
            return 1

        return handler(parsed_args)

    def _handle_service_command(self, args: argparse.Namespace) -> int:
        """Handle service commands.

        Args:
            args: Parsed arguments

        Returns:
            Exit code
        """
        handler_map = {
            "start": self._service_start,
            "stop": self._service_stop,
            "status": self._service_status,
        }

        handler = handler_map.get(args.service_command)
        if handler is None:
            print(f"Error: Unknown service command {args.service_command}")
            return 1

        return handler(args)

    def _service_start(self, _args: argparse.Namespace) -> int:
        """Start the CI service in a background process."""

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

    def _service_stop(self, _args: argparse.Namespace) -> int:
        """Stop the CI service if it is running."""

        pid = self._read_pid()
        if not pid or not self._is_pid_running(pid):
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

    def _service_status(self, _args: argparse.Namespace) -> int:
        """Report service and dashboard process status."""

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

    def _handle_mcp_command(self, args: argparse.Namespace) -> int:
        handler_map = {"serve": self._mcp_serve}
        handler = handler_map.get(args.mcp_command)
        if handler is None:
            print(f"Error: Unknown MCP command {args.mcp_command}")
            return 1
        return handler(args)

    def _mcp_serve(self, args: argparse.Namespace) -> int:
        try:
            log_level = self._resolve_log_level(getattr(args, "log_level", "INFO"))
        except ValueError as exc:
            print(f"Error: {exc}")
            return 1

        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        for handler in root_logger.handlers:
            handler.setLevel(log_level)

        logging.getLogger("src").setLevel(log_level)
        logging.getLogger(__name__).setLevel(log_level)
        logging.getLogger("src.mcp.server").setLevel(log_level)

        logger.info(
            "Launching MCP server version %s with log level %s",
            PACKAGE_VERSION,
            logging.getLevelName(log_level),
        )

        token = None
        if not args.no_token:
            token = args.token or os.getenv("FULL_AUTO_CI_MCP_TOKEN")

        server = MCPServer(self.service, auth_token=token)
        host = args.host
        port = args.port
        token_state = "enabled" if token else "disabled"

        if args.stdio:
            print(f"Starting MCP server on stdio (token={token_state})")
        else:
            probe_state = self._probe_mcp_server(host, port, token)
            if probe_state == "available":
                print(
                    f"MCP server already running on {host}:{port} (token={token_state})"
                )
                return 0
            if probe_state == "unauthorized":
                print(
                    "Error: MCP server is already running but rejected the provided token."
                )
                return 1

            print(f"Starting MCP server on {host}:{port} (token={token_state})")

        async def runner() -> None:
            try:
                if args.stdio:
                    await server.serve_stdio()
                else:
                    await server.serve_tcp(host=host, port=port)
            except OSError as exc:  # pragma: no cover - depends on environment
                if not args.stdio and exc.errno == errno.EADDRINUSE:
                    print(f"Error: MCP server port {host}:{port} is already in use.")
                    return
                raise
            except asyncio.CancelledError:  # pragma: no cover
                pass

        try:
            asyncio.run(runner())
        except KeyboardInterrupt:
            print("MCP server stopped")
        return 0

    def _handle_provider_command(self, args: argparse.Namespace) -> int:
        handler_map = {
            "list": self._provider_list,
            "types": self._provider_types,
            "add": self._provider_add,
            "remove": self._provider_remove,
            "sync": self._provider_sync,
        }

        command = getattr(args, "provider_command", None)
        handler = handler_map.get(command)
        if handler is None:
            print(f"Error: Unknown provider command {command}")
            return 1
        return handler(args)

    def _provider_list(self, _args: argparse.Namespace) -> int:
        providers = self.service.list_providers()
        if not providers:
            print("No external providers configured.")
            return 0

        print("Configured providers:")
        for provider in providers:
            descriptor = provider.get("display_name") or provider["type"]
            print(
                f"  [{provider['id']}] {provider['name']} ({provider['type']}) -> {descriptor}"
            )
        return 0

    def _provider_types(self, _args: argparse.Namespace) -> int:
        types = list(self.service.get_provider_types())
        if not types:
            print("No provider types registered.")
            return 0

        print("Available provider types:")
        for entry in types:
            description = entry.get("description") or ""
            suffix = f" - {description}" if description else ""
            print(f"  {entry['type']}: {entry['display_name']}{suffix}")
        return 0

    def _provider_add(self, args: argparse.Namespace) -> int:
        try:
            config = self._load_provider_config(
                inline=getattr(args, "config", None),
                file_path=getattr(args, "config_file", None),
            )
        except ValueError as exc:
            print(f"Error: {exc}")
            return 1

        try:
            provider = self.service.add_provider(args.type, args.name, config=config)
        except (ProviderConfigError, ValueError) as exc:
            print(f"Error: {exc}")
            return 1

        print(
            f"Provider '{provider.get('name', args.name)}' registered with id {provider.get('id')}"
        )
        return 0

    def _provider_remove(self, args: argparse.Namespace) -> int:
        removed = self.service.remove_provider(args.provider_id)
        if removed:
            print(f"Provider {args.provider_id} removed")
            return 0
        print(f"Error: Provider {args.provider_id} not found")
        return 1

    def _provider_sync(self, args: argparse.Namespace) -> int:
        try:
            runs = self.service.sync_provider(args.provider_id, limit=args.limit)
        except KeyError:
            print(f"Error: Provider {args.provider_id} not found")
            return 1
        except RuntimeError as exc:
            print(f"Error: {exc}")
            return 1

        count = len(runs) if isinstance(runs, list) else 0
        print(f"Synced provider {args.provider_id}; fetched {count} run(s)")
        return 0

    def _load_provider_config(
        self,
        *,
        inline: Optional[str],
        file_path: Optional[str],
    ) -> Dict[str, Any]:
        if inline and file_path:
            raise ValueError("Specify either --config or --config-file, not both")

        if file_path:
            try:
                with open(file_path, "r", encoding="utf-8") as handle:
                    return json.load(handle)
            except FileNotFoundError as exc:  # pragma: no cover - pass through
                raise ValueError(f"Configuration file not found: {file_path}") from exc
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in configuration file: {exc}") from exc

        if inline:
            try:
                return json.loads(inline)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON payload: {exc}") from exc

        return {}

    def _probe_mcp_server(
        self, host: str, port: int, token: Optional[str]
    ) -> Optional[str]:
        """Check whether an MCP server is already listening on the target port."""

        try:
            with socket.create_connection((host, port), timeout=1.0) as sock:
                params: Dict[str, Any] = {}
                if token:
                    params["token"] = token
                message = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "handshake",
                    "params": params,
                }
                sock.sendall((json.dumps(message) + "\n").encode("utf-8"))
                sock.settimeout(1.0)
                buffer = b""
                while not buffer.endswith(b"\n"):
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    buffer += chunk
        except (OSError, ValueError):
            return None

        if not buffer:
            return None

        try:
            response = json.loads(buffer.decode("utf-8"))
        except json.JSONDecodeError:
            return None

        if isinstance(response, dict) and "result" in response:
            return "available"

        error = response.get("error") if isinstance(response, dict) else None
        if isinstance(error, dict) and error.get("code") == -32604:
            return "unauthorized"

        return None

    def _handle_repo_command(self, args: argparse.Namespace) -> int:
        """Handle repository commands.

        Args:
            args: Parsed arguments

        Returns:
            Exit code
        """
        handler_map = {
            "list": self._repo_list,
            "add": self._repo_add,
            "remove": self._repo_remove,
        }

        handler = handler_map.get(args.repo_command)
        if handler is None:
            print(f"Error: Unknown repo command {args.repo_command}")
            return 1

        return handler(args)

    def _repo_list(self, _args: argparse.Namespace) -> int:
        """Print the repository catalog."""

        repos = self.service.list_repositories()
        if not repos:
            print("No repositories configured")
            return 0

        headers = ("ID", "Name", "Branch", "URL")
        rows: List[Sequence[str]] = [
            (str(repo["id"]), repo["name"], repo["branch"], repo["url"])
            for repo in repos
        ]
        self._print_table(
            headers,
            rows,
            title="Repositories",
            alignments=("right", "left", "left", "left"),
        )
        return 0

    def _repo_add(self, args: argparse.Namespace) -> int:
        """Create a new repository entry."""

        repo_id = self.service.add_repository(args.name, args.url, args.branch)
        print(f"Repository added with ID: {repo_id}")
        return 0

    def _repo_remove(self, args: argparse.Namespace) -> int:
        """Remove a repository by identifier."""

        success = self.service.remove_repository(args.repo_id)
        if success:
            print(f"Repository with ID {args.repo_id} removed")
            return 0

        print(f"Error: Repository with ID {args.repo_id} not found")
        return 1

    def _handle_test_command(self, args: argparse.Namespace) -> int:
        """Handle test commands.

        Args:
            args: Parsed arguments

        Returns:
            Exit code
        """
        handler_map = {"run": self._test_run, "results": self._test_results}
        handler = handler_map.get(args.test_command)
        if handler is None:
            print(f"Error: Unknown test command {args.test_command}")
            return 1
        return handler(args)

    def _test_run(self, args: argparse.Namespace) -> int:
        results = self.service.run_tests(args.repo_id, args.commit)

        title = f"Test results for repository {args.repo_id}"
        subtitle = f"Commit: {args.commit}"
        self._print_heading(title)
        print(subtitle)

        overall = self._format_status(results.get("status"))
        print(f"Overall: {overall}")

        warnings = results.get("warnings", []) or []
        for warning in warnings:
            print(f"[WARN] {warning}")

        tool_rows: List[Sequence[str]] = []
        for tool, result in results.get("tools", {}).items():
            status = self._format_status(result.get("status"))
            duration = self._format_duration(result.get("duration"))
            summary = self._summarize_tool_output(
                json.dumps(result), result.get("status")
            )
            tool_rows.append(
                (
                    tool,
                    status,
                    duration,
                    summary or "—",
                )
            )

        if tool_rows:
            self._print_table(
                ("Tool", "Status", "Duration", "Details"),
                tool_rows,
                title="Tool results",
                alignments=("left", "left", "right", "left"),
            )
        else:
            print("No tool results produced")
        return 0

    def _test_results(self, args: argparse.Namespace) -> int:
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

        self._print_heading(f"Test runs for repository {args.repo_id}")
        if args.commit:
            print(f"Filtered by commit: {args.commit}")

        for index, run in enumerate(runs, start=1):
            if index > 1:
                print("")

            status_label = self._format_status(run.get("status"))
            heading = f"Run {run['id']} • {status_label}"
            self._print_subheading(heading)

            commit_hash = run.get("commit_hash", "-")
            created = run.get("created_at", "-")
            started = run.get("started_at") or "-"
            completed = run.get("completed_at") or "-"
            message = (run.get("commit") or {}).get("message")

            meta_rows = [
                ("Commit", commit_hash),
                ("Created", created),
                ("Started", started),
                ("Completed", completed),
            ]
            if message:
                meta_rows.append(("Message", message))

            self._print_key_values(meta_rows, indent=2)

            if run.get("error"):
                print(f"  [ERROR] {run['error']}")

            results = run.get("results") or []
            if results:
                tool_rows = []
                for result in results:
                    tool = result.get("tool", "unknown")
                    status = self._format_status(result.get("status"))
                    summary = self._summarize_tool_output(
                        result.get("output"), result.get("status")
                    )
                    tool_rows.append(
                        (
                            tool,
                            status,
                            summary or "—",
                        )
                    )

                self._print_table(
                    ("Tool", "Status", "Details"),
                    tool_rows,
                    title="  Tool results",
                    alignments=("left", "left", "left"),
                )
            else:
                print("  No tool results persisted")

        return 0

    def _handle_config_command(self, args: argparse.Namespace) -> int:
        """Handle configuration commands."""
        command = getattr(args, "config_command", None)
        handler_map = {
            "show": self._config_show,
            "set": self._config_set,
            "path": self._config_path,
        }
        handler = handler_map.get(command)
        if handler is None:
            print(f"Error: Unknown config command {command}")
            return 1
        return handler(args)

    def _config_show(self, args: argparse.Namespace) -> int:
        config = self.service.config
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

    def _config_set(self, args: argparse.Namespace) -> int:
        config = self.service.config
        section = args.section
        key = args.key
        value = self._parse_config_value(args.value)
        config.set(section, key, value)
        if config.save():
            print(f"Updated {section}.{key} = {value}")
            return 0
        print("Error: Failed to save configuration")
        return 1

    def _config_path(self, _args: argparse.Namespace) -> int:
        print(self.service.config.config_path)
        return 0

    def _handle_user_command(self, args: argparse.Namespace) -> int:
        """Handle user management commands."""
        command = getattr(args, "user_command", None)
        handler_map = {
            "list": self._user_list,
            "add": self._user_add,
            "remove": self._user_remove,
        }
        handler = handler_map.get(command)
        if handler is None:
            print(f"Error: Unknown user command {command}")
            return 1
        return handler(args)

    def _user_list(self, _args: argparse.Namespace) -> int:
        users = self.service.list_users()
        if not users:
            print("No users found")
            return 0

        headers = ("ID", "Username", "Role", "Created")
        rows = [
            (
                str(user.get("id", "")),
                user.get("username", ""),
                user.get("role", "user"),
                user.get("created_at", ""),
            )
            for user in users
        ]
        self._print_table(
            headers, rows, title="Users", alignments=("right", "left", "left", "left")
        )
        return 0

    def _user_add(self, args: argparse.Namespace) -> int:
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

    def _user_remove(self, args: argparse.Namespace) -> int:
        success = self.service.remove_user(args.username)
        if success:
            print(f"User '{args.username}' removed")
            return 0
        print(f"Error: User '{args.username}' not found")
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
            pass

        lowered = raw.lower()
        literal_map = {"true": True, "false": False, "null": None}
        if lowered in literal_map:
            return literal_map[lowered]

        numeric_value = CLI._parse_numeric_literal(raw)
        if numeric_value is not None:
            return numeric_value

        return raw

    @staticmethod
    def _parse_numeric_literal(raw: str) -> Optional[float | int]:
        try:
            if "." in raw:
                return float(raw)
            return int(raw)
        except ValueError:
            return None

    @staticmethod
    def _format_duration(value: Optional[Any]) -> str:
        if value in (None, ""):
            return "—"

        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)

        return f"{numeric:.2f}s"

    @staticmethod
    def _format_status(status: Optional[str]) -> str:
        normalized = (status or "").strip().lower()
        mapping = {
            "success": "✔ success",
            "completed": "✔ completed",
            "error": "✖ error",
            "failed": "✖ failed",
            "running": "… running",
            "pending": "… pending",
            "queued": "… queued",
        }
        return mapping.get(normalized, status or "unknown")

    @staticmethod
    def _print_heading(text: str):
        title = text.strip()
        underline = "=" * len(title)
        print(title)
        print(underline)

    @staticmethod
    def _print_subheading(text: str):
        label = text.strip()
        print(label)
        print("-" * len(label))

    @staticmethod
    def _print_key_values(pairs: Sequence[Tuple[str, Any]], *, indent: int = 0) -> None:
        if not pairs:
            return

        width = max(len(label) for label, _ in pairs)
        prefix = " " * max(indent, 0)
        for label, value in pairs:
            value_text = str(value) if value is not None else ""
            print(f"{prefix}{label.ljust(width)} : {value_text}")

    def _print_table(
        self,
        headers: Sequence[str],
        rows: Iterable[Sequence[Any]],
        *,
        title: Optional[str] = None,
        alignments: Optional[Sequence[str]] = None,
    ) -> None:
        rendered_rows = [
            tuple("" if cell is None else str(cell) for cell in row) for row in rows
        ]
        column_count = len(headers)
        if alignments is None:
            alignments = tuple("left" for _ in range(column_count))
        else:
            alignments = tuple((alignment or "left") for alignment in alignments)

        widths = [len(str(header)) for header in headers]
        for row in rendered_rows:
            for index, cell in enumerate(row):
                widths[index] = max(widths[index], len(cell))

        horizontal = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
        header_sep = "+" + "+".join("=" * (width + 2) for width in widths) + "+"

        def format_row(row_values: Sequence[str]) -> str:
            cells: List[str] = []
            for idx, raw_value in enumerate(row_values):
                align = alignments[idx] if idx < len(alignments) else "left"
                if align == "right":
                    cell = raw_value.rjust(widths[idx])
                elif align == "center":
                    cell = raw_value.center(widths[idx])
                else:
                    cell = raw_value.ljust(widths[idx])
                cells.append(f" {cell} ")
            return "|" + "|".join(cells) + "|"

        output_lines: List[str] = []
        if title:
            output_lines.append(title)
        output_lines.append(horizontal)
        output_lines.append(format_row(tuple(str(header) for header in headers)))
        output_lines.append(header_sep)
        for row in rendered_rows:
            output_lines.append(format_row(row))
        output_lines.append(horizontal)

        for line in output_lines:
            print(line)

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

        if existing_pid:
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

    @staticmethod
    def _resolve_log_level(value: str) -> int:
        if not value:
            raise ValueError("Log level cannot be empty")

        normalized = value.upper()
        if normalized == "WARN":
            normalized = "WARNING"

        level = logging.getLevelName(normalized)
        if isinstance(level, str):  # logging returns level name when unknown
            raise ValueError(
                "Invalid log level. Choose from CRITICAL, ERROR, WARNING, INFO, DEBUG, or NOTSET."
            )

        return level


def main() -> int:
    """Entry point for the CLI."""
    cli = CLI()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())
