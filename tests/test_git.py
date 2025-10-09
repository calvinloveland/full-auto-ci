"""Tests for the git module."""

import os
import shutil
import sqlite3
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.git import GitRepo, GitTracker, RepositoryConfig


class TestGitRepo(unittest.TestCase):
    """Test cases for GitRepo."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.repo = GitRepo(
            RepositoryConfig(
                repo_id=1,
                name="test",
                url="https://github.com/test/test.git",
                work_dir=self.temp_dir,
            )
        )

    def tearDown(self):
        """Tear down test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_init(self):
        """Test initialization."""
        self.assertEqual(self.repo.repo_id, 1)
        self.assertEqual(self.repo.name, "test")
        self.assertEqual(self.repo.url, "https://github.com/test/test.git")
        self.assertEqual(self.repo.branch, "main")
        self.assertEqual(self.repo.work_dir, self.temp_dir)
        self.assertEqual(self.repo.repo_path, os.path.join(self.temp_dir, "1_test"))

    @patch("subprocess.run")
    def test_clone_success(self, mock_run):
        """Test cloning a repository successfully."""
        # Mock the subprocess.run result
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_run.return_value = mock_process

        # Clone the repository
        success = self.repo.clone()

        # Verify the result
        self.assertTrue(success)
        mock_run.assert_called_once()
        args, _ = mock_run.call_args
        self.assertEqual(args[0][0], "git")
        self.assertEqual(args[0][1], "clone")

    @patch("subprocess.run")
    def test_clone_failure(self, mock_run):
        """Test cloning a repository with failure."""
        # Mock the subprocess.run result
        mock_run.side_effect = Exception("Error cloning repository")

        # Clone the repository
        success = self.repo.clone()

        # Verify the result
        self.assertFalse(success)

    @patch("subprocess.run")
    @patch("os.path.exists")
    def test_pull_success(self, mock_exists, mock_run):
        """Test pulling changes successfully."""
        # Mock os.path.exists to return True
        mock_exists.return_value = True

        # Mock the subprocess.run result
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_run.return_value = mock_process

        # Pull changes
        success = self.repo.pull()

        # Verify the result
        self.assertTrue(success)
        mock_run.assert_called_once()
        args, _ = mock_run.call_args
        self.assertEqual(args[0][0], "git")
        self.assertEqual(args[0][1], "pull")

    @patch("subprocess.run")
    @patch("os.path.exists")
    def test_get_latest_commit(self, mock_exists, mock_run):
        """Test getting the latest commit."""
        # Mock os.path.exists to return True
        mock_exists.return_value = True

        # Mock the subprocess.run result
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "abcdef1234567890\nTest User\ntest@example.com\n1626912000\nTest commit message"
        mock_run.return_value = mock_process

        # Get the latest commit
        commit = self.repo.get_latest_commit()

        # Verify the result
        self.assertIsNotNone(commit)
        self.assertEqual(commit["hash"], "abcdef1234567890")
        self.assertEqual(commit["author"], "Test User")
        self.assertEqual(commit["author_email"], "test@example.com")
        self.assertEqual(commit["timestamp"], 1626912000)
        self.assertEqual(commit["message"], "Test commit message")
        self.assertEqual(commit["repository_id"], 1)

    @patch("subprocess.run")
    @patch("os.path.exists")
    def test_get_commits_since(self, mock_exists, mock_run):
        """Test getting commits since a specific commit."""
        # Mock os.path.exists to return True
        mock_exists.return_value = True

        # Mock the subprocess.run result
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = (
            "abcdef1\nTest User 1\ntest1@example.com\n1626912000\nTest commit 1\n\n"
            "abcdef2\nTest User 2\ntest2@example.com\n1626998400\nTest commit 2"
        )
        mock_run.return_value = mock_process

        # Get commits since a specific commit
        commits = self.repo.get_commits_since("0123456789abcdef")

        # Verify the result
        self.assertEqual(len(commits), 2)
        self.assertEqual(commits[0]["hash"], "abcdef1")
        self.assertEqual(commits[0]["author"], "Test User 1")
        self.assertEqual(commits[0]["message"], "Test commit 1")
        self.assertEqual(commits[1]["hash"], "abcdef2")
        self.assertEqual(commits[1]["author"], "Test User 2")
        self.assertEqual(commits[1]["message"], "Test commit 2")


class TestGitTracker(unittest.TestCase):
    """Test cases for GitTracker."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary database
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp()
        self.tracker = GitTracker(db_path=self.temp_db_path)

        # Setup basic database structure
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()

        # Create repositories table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS repositories (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            url TEXT NOT NULL,
            branch TEXT DEFAULT 'main',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        )

        # Create commits table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS commits (
            id INTEGER PRIMARY KEY,
            repository_id INTEGER,
            commit_hash TEXT NOT NULL,
            author TEXT,
            message TEXT,
            timestamp TIMESTAMP,
            FOREIGN KEY (repository_id) REFERENCES repositories (id)
        )
        """
        )

        conn.commit()
        conn.close()

    def tearDown(self):
        """Tear down test fixtures."""
        os.close(self.temp_db_fd)
        os.unlink(self.temp_db_path)

    @patch("src.git.GitRepo")
    def test_add_repository(self, mock_git_repo):
        """Test adding a repository."""
        # Mock the GitRepo instance
        mock_repo = MagicMock()
        mock_repo.clone.return_value = True
        mock_repo.get_latest_commit.return_value = {
            "hash": "abcdef1234567890",
            "author": "Test User",
            "author_email": "test@example.com",
            "timestamp": 1626912000,
            "message": "Test commit message",
            "repository_id": 1,
        }
        mock_git_repo.return_value = mock_repo

        # Add the repository
        success = self.tracker.add_repository(
            1, "test", "https://github.com/test/test.git"
        )

        # Verify the result
        self.assertTrue(success)
        self.assertIn(1, self.tracker.repos)

    @patch("src.git.GitRepo")
    @patch("os.path.exists")
    @patch("shutil.rmtree")
    def test_remove_repository(self, mock_rmtree, mock_exists, _mock_git_repo):
        """Test removing a repository."""
        # Setup
        mock_exists.return_value = True
        self.tracker.repos[1] = MagicMock()
        self.tracker.repos[1].repo_path = "/path/to/repo"

        # Remove the repository
        success = self.tracker.remove_repository(1)

        # Verify the result
        self.assertTrue(success)
        self.assertNotIn(1, self.tracker.repos)
        mock_rmtree.assert_called_once_with("/path/to/repo")

    @patch("src.git.GitRepo")
    def test_get_repository(self, mock_git_repo):
        """Test getting a repository."""
        # Setup
        mock_repo = MagicMock()
        mock_git_repo.return_value = mock_repo
        self.tracker.repos[1] = mock_repo

        # Get the repository
        repo = self.tracker.get_repository(1)

        # Verify the result
        self.assertIs(repo, mock_repo)


if __name__ == "__main__":
    unittest.main()
