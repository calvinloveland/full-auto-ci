"""Tests for the CI service."""

import os
import sqlite3
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.service import CIService


class TestCIService(unittest.TestCase):
    """Test cases for CIService."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary database
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp()
        self._dogfood_env = os.environ.get("FULL_AUTO_CI_DOGFOOD")
        os.environ["FULL_AUTO_CI_DOGFOOD"] = "0"
        self.service = CIService(db_path=self.temp_db_path)

    def tearDown(self):
        """Tear down test fixtures."""
        if self._dogfood_env is None:
            os.environ.pop("FULL_AUTO_CI_DOGFOOD", None)
        else:
            os.environ["FULL_AUTO_CI_DOGFOOD"] = self._dogfood_env
        os.close(self.temp_db_fd)
        os.unlink(self.temp_db_path)

    def test_init(self):
        """Test initialization."""
        self.assertEqual(self.service.db_path, self.temp_db_path)
        self.assertFalse(self.service.running)

    def test_add_repository(self):
        """Test adding a repository."""
        repo_id = self.service.add_repository(
            "test", "https://github.com/test/test.git"
        )
        self.assertGreater(repo_id, 0)

        # Verify that the repository was added
        repo = self.service.get_repository(repo_id)
        self.assertIsNotNone(repo)
        self.assertEqual(repo["name"], "test")
        self.assertEqual(repo["url"], "https://github.com/test/test.git")
        self.assertEqual(repo["branch"], "main")

    def test_remove_repository(self):
        """Test removing a repository."""
        repo_id = self.service.add_repository(
            "test", "https://github.com/test/test.git"
        )
        success = self.service.remove_repository(repo_id)
        self.assertTrue(success)

        # Verify that the repository was removed
        repo = self.service.get_repository(repo_id)
        self.assertIsNone(repo)

    def test_list_repositories(self):
        """Test listing repositories."""
        # Add some repositories
        repo1_id = self.service.add_repository(
            "test1", "https://github.com/test/test1.git"
        )
        repo2_id = self.service.add_repository(
            "test2", "https://github.com/test/test2.git"
        )

        # List repositories
        repos = self.service.list_repositories()
        self.assertEqual(len(repos), 2)

        # Verify repository details
        repo1 = next((r for r in repos if r["id"] == repo1_id), None)
        self.assertIsNotNone(repo1)
        self.assertEqual(repo1["name"], "test1")

        repo2 = next((r for r in repos if r["id"] == repo2_id), None)
        self.assertIsNotNone(repo2)
        self.assertEqual(repo2["name"], "test2")

    def test_create_user_and_list(self):
        """Users can be created and enumerated."""
        user_id = self.service.create_user(
            "alice", "s3cret", role="admin", api_key="apikey"
        )
        self.assertGreater(user_id, 0)

        users = self.service.list_users()
        self.assertEqual(len(users), 1)
        self.assertEqual(users[0]["username"], "alice")
        self.assertEqual(users[0]["role"], "admin")

        conn = sqlite3.connect(self.service.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT password_hash, api_key_hash FROM users WHERE username = ?",
                ("alice",),
            )
            row = cursor.fetchone()
        finally:
            conn.close()

        self.assertIsNotNone(row)
        password_hash, api_key_hash = row
        self.assertNotEqual(password_hash, "s3cret")
        self.assertEqual(len(password_hash), 64)
        self.assertNotEqual(api_key_hash, "apikey")
        self.assertEqual(len(api_key_hash), 64)

    def test_remove_user(self):
        """Users can be removed."""
        self.service.create_user("bob", "pw")
        success = self.service.remove_user("bob")
        self.assertTrue(success)

        users = self.service.list_users()
        self.assertEqual(users, [])

        self.assertFalse(self.service.remove_user("bob"))

    @patch("threading.Thread")
    def test_start_stop(self, mock_thread):
        """Test starting and stopping the service."""
        # Mock the thread to avoid actually running it
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance

        # Start the service
        self.service.start()
        self.assertTrue(self.service.running)
        expected_threads = (self.service.config.get("service", "max_workers") or 4) + 1
        self.assertEqual(mock_thread.call_count, expected_threads)
        self.assertEqual(mock_thread_instance.start.call_count, expected_threads)

        # Stop the service
        self.service.stop()
        self.assertFalse(self.service.running)
        self.assertTrue(mock_thread_instance.join.called)

    @patch("src.service.CIService._create_test_run")
    @patch("src.service.CIService._store_results")
    @patch("src.service.CIService._summarize_tool_results")
    @patch("src.service.CIService._update_test_run")
    @patch("src.git.GitTracker.get_repository")
    def test_run_tests(
        self,
        mock_get_repo,
        mock_update_run,
        mock_summarize,
        mock_store,
        mock_create_run,
    ):
        """Test running tests synchronously via run_tests."""

        mock_repo = MagicMock()
        mock_repo.repo_path = "/tmp/repo"
        mock_repo.clone.return_value = True
        mock_repo.checkout_commit.return_value = True
        mock_get_repo.return_value = mock_repo

        mock_create_run.return_value = 42
        mock_summarize.return_value = ("success", None)

        with patch("src.service.os.path.exists", return_value=True):
            with patch.object(self.service, "tool_runner") as mock_tool_runner:
                mock_tool_runner.run_all.return_value = {
                    "pylint": {"status": "success"},
                    "coverage": {"status": "success"},
                }

                result = self.service.run_tests(1, "abcdef")

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["test_run_id"], 42)
        self.assertIn("pylint", result["tools"])
        self.assertIn("coverage", result["tools"])
        mock_update_run.assert_any_call(42, "running")
        mock_update_run.assert_any_call(42, "completed")
        mock_store.assert_called_once()


if __name__ == "__main__":
    unittest.main()
