"""Git integration for Full Auto CI."""

import logging
import os
import shutil
import sqlite3
import subprocess
import tempfile
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GitRepo:
    """Git repository handler."""

    def __init__(
        self,
        repo_id: int,
        name: str,
        url: str,
        branch: str = "main",
        work_dir: Optional[str] = None,
    ):
        """Initialize a Git repository handler.

        Args:
            repo_id: Repository ID
            name: Repository name
            url: Repository URL
            branch: Branch to monitor
            work_dir: Working directory for clones (defaults to temporary directory)
        """
        self.repo_id = repo_id
        self.name = name
        self.url = url
        self.branch = branch
        self.work_dir = work_dir or os.path.expanduser("~/.fullautoci/repos")
        self.repo_path = os.path.join(
            self.work_dir, f"{self.repo_id}_{self.name.replace(' ', '_')}"
        )

        # Make sure work_dir exists
        os.makedirs(self.work_dir, exist_ok=True)

    def clone(self) -> bool:
        """Clone the repository.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Remove existing clone if it exists
            if os.path.exists(self.repo_path):
                logger.info(f"Removing existing repository at {self.repo_path}")
                shutil.rmtree(self.repo_path)

            # Clone the repository
            logger.info(f"Cloning repository {self.name} from {self.url}")
            result = subprocess.run(
                ["git", "clone", "-b", self.branch, self.url, self.repo_path],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info(f"Cloned repository {self.name} to {self.repo_path}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone repository {self.name}: {e}")
            logger.error(f"Stdout: {e.stdout}")
            logger.error(f"Stderr: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Error cloning repository {self.name}: {e}")
            return False

    def pull(self) -> bool:
        """Pull the latest changes.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if repository exists
            if not os.path.exists(self.repo_path):
                logger.warning(
                    f"Repository {self.name} not found at {self.repo_path}, cloning instead"
                )
                return self.clone()

            # Pull the latest changes
            logger.info(f"Pulling latest changes for repository {self.name}")
            result = subprocess.run(
                ["git", "pull", "origin", self.branch],
                cwd=self.repo_path,
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info(f"Pulled latest changes for repository {self.name}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to pull repository {self.name}: {e}")
            logger.error(f"Stdout: {e.stdout}")
            logger.error(f"Stderr: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Error pulling repository {self.name}: {e}")
            return False

    def get_latest_commit(self) -> Optional[Dict[str, Any]]:
        """Get the latest commit.

        Returns:
            Dictionary with commit information or None if error
        """
        try:
            # Check if repository exists
            if not os.path.exists(self.repo_path):
                logger.warning(f"Repository {self.name} not found at {self.repo_path}")
                return None

            # Get the latest commit
            logger.info(f"Getting latest commit for repository {self.name}")
            result = subprocess.run(
                ["git", "log", "-1", "--format=%H%n%an%n%ae%n%at%n%s"],
                cwd=self.repo_path,
                check=True,
                capture_output=True,
                text=True,
            )
            lines = result.stdout.strip().split("\n")

            if len(lines) < 5:
                logger.error(f"Unexpected git log output format: {result.stdout}")
                return None

            commit_hash = lines[0]
            author_name = lines[1]
            author_email = lines[2]
            timestamp = int(lines[3])
            message = lines[4]

            commit = {
                "hash": commit_hash,
                "author": author_name,
                "author_email": author_email,
                "timestamp": timestamp,
                "datetime": datetime.fromtimestamp(timestamp).isoformat(),
                "message": message,
                "repository_id": self.repo_id,
            }

            return commit
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get latest commit for repository {self.name}: {e}")
            logger.error(f"Stdout: {e.stdout}")
            logger.error(f"Stderr: {e.stderr}")
            return None
        except Exception as e:
            logger.error(f"Error getting latest commit for repository {self.name}: {e}")
            return None

    def get_commits_since(self, since_commit: str) -> List[Dict[str, Any]]:
        """Get all commits since a specific commit.

        Args:
            since_commit: Commit hash to start from (exclusive)

        Returns:
            List of commit dictionaries, from oldest to newest
        """
        try:
            # Check if repository exists
            if not os.path.exists(self.repo_path):
                logger.warning(f"Repository {self.name} not found at {self.repo_path}")
                return []

            # Get all commits since the given commit
            logger.info(
                f"Getting commits since {since_commit} for repository {self.name}"
            )
            result = subprocess.run(
                [
                    "git",
                    "log",
                    f"{since_commit}..HEAD",
                    "--format=%H%n%an%n%ae%n%at%n%s",
                    "--reverse",
                ],
                cwd=self.repo_path,
                check=True,
                capture_output=True,
                text=True,
            )
            output = result.stdout.strip()

            if not output:
                logger.info(f"No new commits found since {since_commit}")
                return []

            # Parse the output
            commits = []
            commit_blocks = output.split("\n\n")

            for block in commit_blocks:
                lines = block.strip().split("\n")
                if len(lines) < 5:
                    logger.warning(
                        f"Unexpected git log output format in block: {block}"
                    )
                    continue

                commit_hash = lines[0]
                author_name = lines[1]
                author_email = lines[2]
                timestamp = int(lines[3])
                message = lines[4]

                commits.append(
                    {
                        "hash": commit_hash,
                        "author": author_name,
                        "author_email": author_email,
                        "timestamp": timestamp,
                        "datetime": datetime.fromtimestamp(timestamp).isoformat(),
                        "message": message,
                        "repository_id": self.repo_id,
                    }
                )

            return commits
        except subprocess.CalledProcessError as e:
            logger.error(
                f"Failed to get commits since {since_commit} for repository {self.name}: {e}"
            )
            logger.error(f"Stdout: {e.stdout}")
            logger.error(f"Stderr: {e.stderr}")
            return []
        except Exception as e:
            logger.error(
                f"Error getting commits since {since_commit} for repository {self.name}: {e}"
            )
            return []

    def checkout_commit(self, commit_hash: str) -> bool:
        """Checkout a specific commit.

        Args:
            commit_hash: Commit hash to checkout

        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if repository exists
            if not os.path.exists(self.repo_path):
                logger.warning(f"Repository {self.name} not found at {self.repo_path}")
                return False

            # Checkout the commit
            logger.info(f"Checking out commit {commit_hash} for repository {self.name}")
            result = subprocess.run(
                ["git", "checkout", commit_hash],
                cwd=self.repo_path,
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info(f"Checked out commit {commit_hash} for repository {self.name}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(
                f"Failed to checkout commit {commit_hash} for repository {self.name}: {e}"
            )
            logger.error(f"Stdout: {e.stdout}")
            logger.error(f"Stderr: {e.stderr}")
            return False
        except Exception as e:
            logger.error(
                f"Error checking out commit {commit_hash} for repository {self.name}: {e}"
            )
            return False


class GitTracker:
    """Git repository tracker."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize the git tracker.

        Args:
            db_path: Path to the SQLite database
        """
        self.db_path = db_path or os.path.expanduser("~/.fullautoci/database.sqlite")
        self.repos = {}  # type: Dict[int, GitRepo]

    def add_repository(
        self, repo_id: int, name: str, url: str, branch: str = "main"
    ) -> bool:
        """Add a repository to track.

        Args:
            repo_id: Repository ID
            name: Repository name
            url: Repository URL
            branch: Branch to monitor

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create GitRepo instance
            repo = GitRepo(repo_id, name, url, branch)

            # Try to clone the repository
            if not repo.clone():
                return False

            # Add to repos dictionary
            self.repos[repo_id] = repo

            # Get the latest commit and store it in the database
            commit = repo.get_latest_commit()
            if commit:
                self._store_commit(commit)

            return True
        except Exception as e:
            logger.error(f"Error adding repository {name}: {e}")
            return False

    def remove_repository(self, repo_id: int) -> bool:
        """Remove a repository from tracking.

        Args:
            repo_id: Repository ID

        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if repository exists in tracker
            if repo_id not in self.repos:
                logger.warning(f"Repository {repo_id} not found in tracker")
                return False

            # Get the repository
            repo = self.repos[repo_id]

            # Remove the local clone
            if os.path.exists(repo.repo_path):
                shutil.rmtree(repo.repo_path)

            # Remove from repos dictionary
            del self.repos[repo_id]

            return True
        except Exception as e:
            logger.error(f"Error removing repository {repo_id}: {e}")
            return False

    def check_for_updates(self) -> Dict[int, List[Dict[str, Any]]]:
        """Check all repositories for updates.

        Returns:
            Dictionary mapping repository IDs to lists of new commits
        """
        new_commits = {}

        # Load repositories from database
        self._load_repositories()

        # Check each repository for updates
        for repo_id, repo in self.repos.items():
            try:
                # Pull the latest changes
                if not repo.pull():
                    logger.error(f"Failed to pull repository {repo.name}")
                    continue

                # Get the latest commit we have stored
                latest_commit_hash = self._get_latest_commit_hash(repo_id)

                if not latest_commit_hash:
                    # No commits stored yet, get the latest and store it
                    commit = repo.get_latest_commit()
                    if commit:
                        self._store_commit(commit)
                        new_commits[repo_id] = [commit]
                else:
                    # Get all commits since the latest one
                    commits = repo.get_commits_since(latest_commit_hash)
                    if commits:
                        for commit in commits:
                            self._store_commit(commit)
                        new_commits[repo_id] = commits
            except Exception as e:
                logger.error(f"Error checking for updates in repository {repo_id}: {e}")

        return new_commits

    def get_repository(self, repo_id: int) -> Optional[GitRepo]:
        """Get a repository.

        Args:
            repo_id: Repository ID

        Returns:
            GitRepo instance or None if not found
        """
        # Load repositories if not already loaded
        if not self.repos:
            self._load_repositories()

        return self.repos.get(repo_id)

    def _load_repositories(self):
        """Load repositories from database."""
        try:
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get all repositories
            cursor.execute("SELECT id, name, url, branch FROM repositories")
            repositories = cursor.fetchall()

            # Close the connection
            conn.close()

            # Create GitRepo instances
            for repo_id, name, url, branch in repositories:
                # Skip if already loaded
                if repo_id in self.repos:
                    continue

                # Create GitRepo instance
                self.repos[repo_id] = GitRepo(repo_id, name, url, branch)

            logger.info(f"Loaded {len(self.repos)} repositories from database")
        except Exception as e:
            logger.error(f"Error loading repositories from database: {e}")

    def _get_latest_commit_hash(self, repo_id: int) -> Optional[str]:
        """Get the latest commit hash for a repository from the database.

        Args:
            repo_id: Repository ID

        Returns:
            Commit hash or None if no commits found
        """
        try:
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get the latest commit
            cursor.execute(
                "SELECT commit_hash FROM commits WHERE repository_id = ? ORDER BY timestamp DESC LIMIT 1",
                (repo_id,),
            )
            result = cursor.fetchone()

            # Close the connection
            conn.close()

            return result[0] if result else None
        except Exception as e:
            logger.error(
                f"Error getting latest commit hash for repository {repo_id}: {e}"
            )
            return None

    def _store_commit(self, commit: Dict[str, Any]) -> bool:
        """Store a commit in the database.

        Args:
            commit: Commit information

        Returns:
            True if successful, False otherwise
        """
        try:
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Insert the commit
            cursor.execute(
                """
                INSERT INTO commits (repository_id, commit_hash, author, message, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    commit["repository_id"],
                    commit["hash"],
                    commit["author"],
                    commit["message"],
                    commit["timestamp"],
                ),
            )

            # Commit the transaction
            conn.commit()

            # Close the connection
            conn.close()

            return True
        except Exception as e:
            logger.error(f"Error storing commit {commit['hash']}: {e}")
            return False
