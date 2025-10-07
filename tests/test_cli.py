"""Tests for the CLI module."""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

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
        self.cli.service.config.set("dashboard", "auto_open", True)
        self.cli.service.config.set("dashboard", "auto_start", False)

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

    @patch("builtins.print")
    def test_test_run_outputs_warnings(self, mock_print):
        """Warnings from run_tests should be surfaced in CLI output."""

        self.cli.service.run_tests = lambda *_args: {
            "status": "success",
            "tools": {"pylint": {"status": "success"}},
            "warnings": ["dirty repo"],
        }

        exit_code = self.cli.run(["test", "run", "1", "abcdef"])

        self.assertEqual(exit_code, 0)
        mock_print.assert_any_call("Warning: dirty repo")

    @patch("builtins.print")
    def test_test_results_no_runs(self, mock_print):
        """Results command should report when no runs are found."""

        def fake_get_results(repo_id, *, commit_hash=None, limit=10):
            self.assertEqual(repo_id, 1)
            self.assertEqual(limit, 10)
            self.assertIsNone(commit_hash)
            return []

        self.cli.service.get_test_results = fake_get_results

        exit_code = self.cli.run(["test", "results", "1"])

        self.assertEqual(exit_code, 0)
        mock_print.assert_any_call("No test runs found for repository 1.")

    @patch("builtins.print")
    def test_test_results_outputs_runs(self, mock_print):
        """Results command should display run metadata and tool summaries."""

        sample_run = {
            "id": 42,
            "commit_hash": "abc1234",
            "status": "completed",
            "created_at": 1000,
            "started_at": 1001,
            "completed_at": 1002,
            "error": None,
            "commit": {"message": "Update docs"},
            "results": [
                {
                    "tool": "pylint",
                    "status": "success",
                    "output": json.dumps({"status": "success", "score": 9.5}),
                    "duration": 0.5,
                },
                {
                    "tool": "coverage",
                    "status": "success",
                    "output": json.dumps({"status": "success", "percentage": 49.73}),
                    "duration": 1.2,
                },
            ],
        }

        self.cli.service.get_test_results = (
            lambda repo_id, *, commit_hash=None, limit=10: [sample_run]
        )

        exit_code = self.cli.run(["test", "results", "1"])

        self.assertEqual(exit_code, 0)
        mock_print.assert_any_call("Test runs for repository 1")
        mock_print.assert_any_call("Run 42 | Commit: abc1234 | Status: completed")
        mock_print.assert_any_call("  Created: 1000 | Started: 1001 | Completed: 1002")
        mock_print.assert_any_call("  Message: Update docs")
        mock_print.assert_any_call("  - pylint: success (score 9.5)")
        mock_print.assert_any_call("  - coverage: success (49.73%)")

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

    @patch("src.cli.webbrowser.open", return_value=True)
    @patch("builtins.print")
    def test_service_start_background_process(self, mock_print, mock_open):
        """Service start launches background process and records pid."""
        mock_proc = MagicMock()
        mock_proc.pid = 4321
        mock_proc.is_alive.return_value = True

        with patch.object(self.cli, "_read_pid", return_value=None), patch(
            "src.cli.multiprocessing.Process", return_value=mock_proc
        ) as mock_proc_cls, patch.object(self.cli, "_write_pid_file") as mock_write:
            exit_code = self.cli.run(["service", "start"])

        self.assertEqual(exit_code, 0)
        mock_proc_cls.assert_called_once()
        mock_proc.start.assert_called_once()
        mock_write.assert_called_once_with(4321)
        mock_print.assert_any_call("Service started in background (PID 4321).")
        mock_print.assert_any_call("Dashboard available at http://127.0.0.1:8000")
        mock_open.assert_called_once_with("http://127.0.0.1:8000", new=2)
        mock_print.assert_any_call("Opened http://127.0.0.1:8000 in your browser.")

    @patch("src.cli.webbrowser.open")
    @patch("builtins.print")
    def test_service_start_auto_open_disabled(self, mock_print, mock_open):
        self.cli.service.config.set("dashboard", "auto_open", False)
        mock_proc = MagicMock()
        mock_proc.pid = 55
        mock_proc.is_alive.return_value = True

        with patch.object(self.cli, "_read_pid", return_value=None), patch(
            "src.cli.multiprocessing.Process", return_value=mock_proc
        ), patch.object(self.cli, "_write_pid_file"), patch.object(
            self.cli, "_maybe_start_dashboard"
        ):
            exit_code = self.cli.run(["service", "start"])

        self.assertEqual(exit_code, 0)
        mock_open.assert_not_called()
        mock_print.assert_any_call("Dashboard available at http://127.0.0.1:8000")

    @patch("src.cli.webbrowser.open", return_value=False)
    @patch("builtins.print")
    def test_service_start_auto_start_dashboard(self, mock_print, _mock_open):
        self.cli.service.config.set("dashboard", "auto_start", True)
        service_proc = MagicMock()
        service_proc.pid = 600
        service_proc.is_alive.return_value = True
        dash_proc = MagicMock()
        dash_proc.pid = 601
        dash_proc.is_alive.return_value = True

        with patch.object(self.cli, "_read_pid", return_value=None), patch(
            "src.cli.multiprocessing.Process", side_effect=[service_proc, dash_proc]
        ) as mock_process, patch.object(
            self.cli, "_write_pid_file"
        ) as mock_write_pid, patch.object(
            self.cli, "_write_dashboard_pid"
        ) as mock_write_dash:
            exit_code = self.cli.run(["service", "start"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(mock_process.call_count, 2)
        mock_write_pid.assert_called_once_with(600)
        mock_write_dash.assert_called_once_with(601)
        mock_print.assert_any_call("Dashboard started in background (PID 601).")

    @patch("builtins.print")
    def test_service_start_when_already_running(self, mock_print):
        with patch.object(self.cli, "_read_pid", return_value=101), patch.object(
            self.cli, "_is_pid_running", return_value=True
        ), patch("src.cli.multiprocessing.Process") as mock_proc_cls:
            exit_code = self.cli.run(["service", "start"])

        self.assertEqual(exit_code, 0)
        mock_proc_cls.assert_not_called()
        mock_print.assert_any_call("Service already running (PID 101)")

    @patch("builtins.print")
    def test_service_stop_running(self, mock_print):
        with patch.object(self.cli, "_read_pid", return_value=202), patch.object(
            self.cli, "_is_pid_running", side_effect=[True, True, False, False]
        ) as mock_running, patch("src.cli.os.kill") as mock_kill, patch.object(
            self.cli, "_remove_pid_file"
        ) as mock_remove, patch(
            "src.cli.time.sleep"
        ), patch.object(
            self.cli, "_stop_dashboard_process"
        ) as mock_stop_dash:
            exit_code = self.cli.run(["service", "stop"])

        self.assertEqual(exit_code, 0)
        mock_kill.assert_called_once()
        self.assertGreaterEqual(mock_running.call_count, 3)
        mock_remove.assert_called_once()
        mock_stop_dash.assert_called_once()
        mock_print.assert_any_call("Service stopped")

    @patch("builtins.print")
    def test_service_stop_not_running(self, mock_print):
        with patch.object(self.cli, "_read_pid", return_value=None), patch.object(
            self.cli, "_remove_pid_file"
        ) as mock_remove, patch.object(
            self.cli, "_maybe_cleanup_dashboard"
        ) as mock_cleanup:
            exit_code = self.cli.run(["service", "stop"])

        self.assertEqual(exit_code, 0)
        mock_remove.assert_called_once()
        mock_cleanup.assert_called_once()
        mock_print.assert_any_call("Service is not running")

    @patch("builtins.print")
    def test_service_status_running(self, mock_print):
        """Status should report running when pid is alive."""
        with patch.object(self.cli, "_read_pid", return_value=123), patch.object(
            self.cli, "_is_pid_running", return_value=True
        ), patch.object(self.cli, "_read_dashboard_pid", return_value=None):
            exit_code = self.cli.run(["service", "status"])

        self.assertEqual(exit_code, 0)
        mock_print.assert_any_call("Service is running (PID 123)")

    @patch("builtins.print")
    def test_service_status_not_running_cleans_pid(self, mock_print):
        """Status clears stale pid files when process is gone."""
        with patch.object(self.cli, "_read_pid", return_value=999), patch.object(
            self.cli, "_is_pid_running", return_value=False
        ), patch.object(self.cli, "_remove_pid_file") as mock_remove, patch.object(
            self.cli, "_read_dashboard_pid", return_value=None
        ):
            exit_code = self.cli.run(["service", "status"])

        self.assertEqual(exit_code, 0)
        mock_remove.assert_called_once()
        mock_print.assert_any_call("Service is not running")

    @patch("src.cli.asyncio.run")
    @patch("src.cli.MCPServer")
    def test_mcp_serve_command(self, mock_server_cls, mock_async_run):
        mock_server = MagicMock()
        mock_server_cls.return_value = mock_server
        def fake_run(coro):
            coro.close()
            return None

        mock_async_run.side_effect = fake_run

        exit_code = self.cli.run(
            ["mcp", "serve", "--host", "0.0.0.0", "--port", "9001", "--token", "tok"]
        )

        self.assertEqual(exit_code, 0)
        mock_server_cls.assert_called_once_with(self.cli.service, auth_token="tok")
        mock_async_run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
