"""Tests for the CLI module."""

import os
import tempfile
import unittest
from unittest.mock import patch

from src.cli import CLI


class TestCLI(unittest.TestCase):
    """Test cases for CLI."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary database
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp()
        self.temp_config_fd, self.temp_config_path = tempfile.mkstemp(suffix=".yml")
        os.close(self.temp_config_fd)
        os.unlink(self.temp_config_path)
        self.cli = CLI(config_path=self.temp_config_path, db_path=self.temp_db_path)

    def tearDown(self):
        """Tear down test fixtures."""
        os.close(self.temp_db_fd)
        os.unlink(self.temp_db_path)
        if os.path.exists(self.temp_config_path):
            os.unlink(self.temp_config_path)

    def test_parse_args(self):
        """Test parsing arguments."""
        args = self.cli.parse_args(["service", "status"])
        self.assertEqual(args.command, "service")
        self.assertEqual(args.service_command, "status")

        args = self.cli.parse_args(["repo", "list"])
        self.assertEqual(args.command, "repo")
        self.assertEqual(args.repo_command, "list")

        args = self.cli.parse_args(
            ["repo", "add", "test", "https://github.com/test/test.git"]
        )
        self.assertEqual(args.command, "repo")
        self.assertEqual(args.repo_command, "add")
        self.assertEqual(args.name, "test")
        self.assertEqual(args.url, "https://github.com/test/test.git")
        self.assertEqual(args.branch, "main")
        args = self.cli.parse_args(["config", "show"])
        self.assertEqual(args.command, "config")
        self.assertEqual(args.config_command, "show")

        args = self.cli.parse_args(["user", "add", "alice", "pw123", "--role", "admin"])
        self.assertEqual(args.command, "user")
        self.assertEqual(args.user_command, "add")
        self.assertEqual(args.username, "alice")
        self.assertEqual(args.password, "pw123")
        self.assertEqual(args.role, "admin")

        args = self.cli.parse_args(["user", "list"])
        self.assertEqual(args.command, "user")
        self.assertEqual(args.user_command, "list")

    @patch("src.cli.CLI._handle_service_command")
    def test_run_service_command(self, mock_handle):
        """Test running service commands."""
        mock_handle.return_value = 0

        # Call the run method with service command
        exit_code = self.cli.run(["service", "status"])

        # Verify that _handle_service_command was called
        mock_handle.assert_called_once()
        self.assertEqual(exit_code, 0)

    @patch("src.cli.CLI._handle_repo_command")
    def test_run_repo_command(self, mock_handle):
        """Test running repo commands."""
        mock_handle.return_value = 0

        # Call the run method with repo command
        exit_code = self.cli.run(["repo", "list"])

        # Verify that _handle_repo_command was called
        mock_handle.assert_called_once()
        self.assertEqual(exit_code, 0)

    @patch("src.cli.CLI._handle_test_command")
    def test_run_test_command(self, mock_handle):
        """Test running test commands."""
        mock_handle.return_value = 0

        # Call the run method with test command
        exit_code = self.cli.run(["test", "run", "1", "abcdef"])

        # Verify that _handle_test_command was called
        mock_handle.assert_called_once()
        self.assertEqual(exit_code, 0)

    def test_run_unknown_command(self):
        """Test running an unknown command."""
        # Call the run method with an unknown command
        exit_code = self.cli.run(["unknown"])

        # Verify that the exit code is 1
        self.assertEqual(exit_code, 1)

    def test_run_no_command(self):
        """Test running with no command."""
        # Call the run method with no command
        exit_code = self.cli.run([])

        # Verify that the exit code is 1
        self.assertEqual(exit_code, 1)

    @patch("src.cli.CLI._handle_config_command")
    def test_run_config_command(self, mock_handle):
        """Test routing to config command handler."""
        mock_handle.return_value = 0

        exit_code = self.cli.run(["config", "show"])

        mock_handle.assert_called_once()
        self.assertEqual(exit_code, 0)

    @patch("src.cli.CLI._handle_user_command")
    def test_run_user_command(self, mock_handle):
        """Test routing to user command handler."""
        mock_handle.return_value = 0

        exit_code = self.cli.run(["user", "list"])

        mock_handle.assert_called_once()
        self.assertEqual(exit_code, 0)

    @patch("builtins.print")
    def test_config_set_updates_value(self, mock_print):
        """Setting a config value should persist to Config."""
        exit_code = self.cli.run(["config", "set", "service", "max_workers", "10"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(self.cli.service.config.get("service", "max_workers"), 10)
        mock_print.assert_any_call("Updated service.max_workers = 10")

    @patch("builtins.print")
    def test_config_show_handles_missing_section(self, mock_print):
        """Show should error when section missing."""
        exit_code = self.cli.run(["config", "show", "missing"])

        self.assertEqual(exit_code, 1)
        mock_print.assert_any_call("Error: Configuration section 'missing' not found")

    @patch("builtins.print")
    def test_config_path_outputs_path(self, mock_print):
        """Path command should print config file location."""
        exit_code = self.cli.run(["config", "path"])

        self.assertEqual(exit_code, 0)
        mock_print.assert_any_call(self.temp_config_path)

    @patch("builtins.print")
    def test_user_list_outputs_table(self, mock_print):
        """List command prints users when present."""
        self.cli.service.list_users = lambda: [
            {"id": 1, "username": "alice", "role": "admin", "created_at": "2024-01-01"}
        ]

        exit_code = self.cli.run(["user", "list"])

        self.assertEqual(exit_code, 0)
        mock_print.assert_any_call("ID | Username | Role | Created")
        mock_print.assert_any_call("1 | alice | admin | 2024-01-01")

    @patch("builtins.print")
    def test_user_add_success(self, mock_print):
        """Add command delegates to service and prints id."""
        self.cli.service.create_user = lambda *args, **kwargs: 7

        exit_code = self.cli.run(["user", "add", "alice", "pw"])

        self.assertEqual(exit_code, 0)
        mock_print.assert_any_call("User 'alice' created with id 7")

    @patch("builtins.print")
    def test_user_add_validation_error(self, mock_print):
        """Add command surfaces validation errors."""

        def raise_error(*_args, **_kwargs):
            raise ValueError("Username is required")

        self.cli.service.create_user = raise_error

        exit_code = self.cli.run(["user", "add", "", "pw"])

        self.assertEqual(exit_code, 1)
        mock_print.assert_any_call("Error: Username is required")

    @patch("builtins.print")
    def test_user_remove(self, mock_print):
        """Remove command reports success or failure."""
        self.cli.service.remove_user = lambda username: username == "alice"

        exit_code_success = self.cli.run(["user", "remove", "alice"])
        exit_code_failure = self.cli.run(["user", "remove", "bob"])

        self.assertEqual(exit_code_success, 0)
        self.assertEqual(exit_code_failure, 1)
        mock_print.assert_any_call("User 'alice' removed")
        mock_print.assert_any_call("Error: User 'bob' not found")


if __name__ == "__main__":
    unittest.main()
