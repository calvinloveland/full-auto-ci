"""Command-line interface for Full Auto CI."""

import argparse
import json
import logging
import sys
from typing import Any, List, Optional

from .service import CIService

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
            self.service.start()
            print("Service started")
            return 0
        elif args.service_command == "stop":
            self.service.stop()
            print("Service stopped")
            return 0
        elif args.service_command == "status":
            if hasattr(self.service, "running") and self.service.running:
                print("Service is running")
            else:
                print("Service is not running")
            return 0
        else:
            print(f"Error: Unknown service command {args.service_command}")
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
            for tool, result in results["tools"].items():
                print(f"- {tool}: {result['status']}")
            return 0
        elif args.test_command == "results":
            print(f"Getting results for repository {args.repo_id}")
            if args.commit:
                print(f"Commit: {args.commit}")
            else:
                print("All commits")
            # In the future, implement fetching actual results
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


def main() -> int:
    """Entry point for the CLI."""
    cli = CLI()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())
