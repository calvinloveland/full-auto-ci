"""Command-line interface for Full Auto CI."""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import __version__ as PACKAGE_VERSION
from .cli_mcp import serve as run_mcp_serve
from .cli_providers import handle_provider_command as run_provider_command
from .cli_service import handle_service_command as run_service_command
from .cli_service import register_service_commands as add_service_commands
from .service import CIService

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _TableRenderContext:
    headers: Tuple[str, ...]
    rows: Sequence[Sequence[str]]
    alignments: Sequence[str]
    widths: Sequence[int]


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
        add_service_commands(subparsers)

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
        return run_service_command(self, args)

    def _handle_mcp_command(self, args: argparse.Namespace) -> int:
        if args.mcp_command == "serve":
            return run_mcp_serve(
                args,
                service=self.service,
                resolve_log_level=self._resolve_log_level,
            )
        print(f"Error: Unknown MCP command {args.mcp_command}")
        return 1

    def _handle_provider_command(self, args: argparse.Namespace) -> int:
        return run_provider_command(self, args)

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
            self._notify_missing_runs(args.repo_id, getattr(args, "commit", None))
            return 0

        self._print_heading(f"Test runs for repository {args.repo_id}")
        if args.commit:
            print(f"Filtered by commit: {args.commit}")

        for index, run in enumerate(runs, start=1):
            if index > 1:
                print("")
            self._render_test_run(run)

        return 0

    def _notify_missing_runs(self, repo_id: int, commit: Optional[str]) -> None:
        if commit:
            print(f"No test runs found for repository {repo_id} and commit {commit}.")
        else:
            print(f"No test runs found for repository {repo_id}.")

    def _render_test_run(self, run: Dict[str, Any]) -> None:
        status_label = self._format_status(run.get("status"))
        heading = f"Run {run['id']} • {status_label}"
        self._print_subheading(heading)
        self._print_run_metadata(run)

        if run.get("error"):
            print(f"  [ERROR] {run['error']}")

        self._print_run_results(run.get("results") or [])

    def _print_run_metadata(self, run: Dict[str, Any]) -> None:
        commit_hash = run.get("commit_hash", "-")
        created = run.get("created_at", "-")
        started = run.get("started_at") or "-"
        completed = run.get("completed_at") or "-"
        meta_rows = [
            ("Commit", commit_hash),
            ("Created", created),
            ("Started", started),
            ("Completed", completed),
        ]

        message = (run.get("commit") or {}).get("message")
        if message:
            meta_rows.append(("Message", message))

        self._print_key_values(meta_rows, indent=2)

    def _print_run_results(self, results: Sequence[Dict[str, Any]]) -> None:
        if not results:
            print("  No tool results persisted")
            return

        tool_rows = []
        for result in results:
            tool_name = result.get("tool", "unknown")
            status = self._format_status(result.get("status"))
            summary = self._summarize_tool_output(
                result.get("output"), result.get("status")
            )
            tool_rows.append((tool_name, status, summary or "—"))

        self._print_table(
            ("Tool", "Status", "Details"),
            tool_rows,
            title="  Tool results",
            alignments=("left", "left", "left"),
        )

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
        rendered_rows = self._stringify_table_rows(rows)
        normalized_alignments = self._normalize_alignments(alignments, len(headers))
        widths = self._calculate_column_widths(headers, rendered_rows)
        context = _TableRenderContext(
            headers=tuple(str(header) for header in headers),
            rows=rendered_rows,
            alignments=normalized_alignments,
            widths=widths,
        )
        lines = self._generate_table_lines(title, context)

        for line in lines:
            print(line)

    @staticmethod
    def _stringify_table_rows(
        rows: Iterable[Sequence[Any]],
    ) -> List[Tuple[str, ...]]:
        return [
            tuple("" if cell is None else str(cell) for cell in row) for row in rows
        ]

    @staticmethod
    def _normalize_alignments(
        alignments: Optional[Sequence[str]], column_count: int
    ) -> Tuple[str, ...]:
        if alignments is None:
            return tuple("left" for _ in range(column_count))
        return tuple((alignment or "left") for alignment in alignments)

    @staticmethod
    def _calculate_column_widths(
        headers: Sequence[str], rendered_rows: Sequence[Sequence[str]]
    ) -> List[int]:
        widths = [len(str(header)) for header in headers]
        for row in rendered_rows:
            for index, cell in enumerate(row):
                widths[index] = max(widths[index], len(cell))
        return widths

    @staticmethod
    def _build_border(widths: Sequence[int], fill: str) -> str:
        return "+" + "+".join(fill * (width + 2) for width in widths) + "+"

    def _generate_table_lines(
        self,
        title: Optional[str],
        context: _TableRenderContext,
    ) -> List[str]:
        lines: List[str] = []
        if title:
            lines.append(title)
        horizontal = self._build_border(context.widths, "-")
        header_sep = self._build_border(context.widths, "=")
        lines.append(horizontal)
        lines.append(
            self._format_table_row(context.headers, context.alignments, context.widths)
        )
        lines.append(header_sep)
        for row in context.rows:
            lines.append(
                self._format_table_row(row, context.alignments, context.widths)
            )
        lines.append(horizontal)
        return lines

    @staticmethod
    def _format_table_row(
        row_values: Sequence[str],
        alignments: Sequence[str],
        widths: Sequence[int],
    ) -> str:
        cells: List[str] = []
        for idx, raw_value in enumerate(row_values):
            align = alignments[idx] if idx < len(alignments) else "left"
            cell_width = widths[idx] if idx < len(widths) else 0
            if align == "right":
                cell_text = raw_value.rjust(cell_width)
            elif align == "center":
                cell_text = raw_value.center(cell_width)
            else:
                cell_text = raw_value.ljust(cell_width)
            cells.append(f" {cell_text} ")
        return "|" + "|".join(cells) + "|"

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
