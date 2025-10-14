"""Core service module for Full Auto CI."""

import hashlib
import json
import logging
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .config import Config
from .db import DataAccess
from .git import GitTracker
from .tools import Coverage, Pylint, ToolRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ServiceRuntime:
    """Mutable runtime state for :class:`CIService`."""

    running: bool = False
    task_queue: "queue.Queue[Dict[str, Any]]" = field(default_factory=queue.Queue)
    workers: List[threading.Thread] = field(default_factory=list)
    monitor_thread: Optional[threading.Thread] = None


@dataclass
class ServiceComponents:
    """Container for service dependencies."""

    git_tracker: GitTracker
    tool_runner: ToolRunner
    data: DataAccess


class CIService:
    """Main service class that runs the continuous integration process."""

    def __init__(
        self, config_path: Optional[str] = None, db_path: Optional[str] = None
    ):
        """Initialize the CI service.

        Args:
            config_path: Path to the configuration file
            db_path: Path to the SQLite database
        """
        self.config = Config(config_path)
        self.db_path = (
            db_path
            or self.config.get("database", "path")
            or os.path.expanduser("~/.fullautoci/database.sqlite")
        )
        self._runtime = ServiceRuntime()
        self._component_overrides: Dict[str, Any] = {}
        self._components = ServiceComponents(
            git_tracker=GitTracker(db_path=self.db_path),
            tool_runner=ToolRunner([Pylint(), Coverage()]),
            data=DataAccess(self.db_path),
        )
        self.data.initialize_schema()
        self._bootstrap_dogfood_repository()
        logger.info("CI Service initialized")

    @property
    def running(self) -> bool:
        """Return whether the service has been started."""

        return self._runtime.running

    @running.setter
    def running(self, value: bool) -> None:
        """Update the running flag for the service."""

        self._runtime.running = value

    @property
    def task_queue(self) -> "queue.Queue[Dict[str, Any]]":
        """Expose the background task queue."""

        return self._runtime.task_queue

    @property
    def workers(self) -> List[threading.Thread]:
        """Return worker threads spawned by the service."""

        return self._runtime.workers

    @property
    def monitor_thread(self) -> Optional[threading.Thread]:
        """Return the monitor thread if it has been started."""

        return self._runtime.monitor_thread

    @monitor_thread.setter
    def monitor_thread(self, value: Optional[threading.Thread]) -> None:
        """Update the active monitor thread reference."""

        self._runtime.monitor_thread = value

    def _set_component(self, name: str, value: Any) -> None:
        """Assign a component while keeping track of the original value."""

        if name not in self._component_overrides:
            self._component_overrides[name] = getattr(self._components, name)
        setattr(self._components, name, value)

    def _reset_component(self, name: str) -> None:
        """Restore a previously overridden component."""

        original = self._component_overrides.pop(name, None)
        if original is not None:
            setattr(self._components, name, original)

    @property
    def tool_runner(self) -> ToolRunner:
        """Access the tool runner component."""

        return self._components.tool_runner

    @tool_runner.setter
    def tool_runner(self, value: ToolRunner) -> None:
        """Override the tool runner component."""

        self._set_component("tool_runner", value)

    @tool_runner.deleter
    def tool_runner(self) -> None:
        """Reset the tool runner override."""

        self._reset_component("tool_runner")

    @property
    def data(self) -> DataAccess:
        """Access the database layer component."""

        return self._components.data

    @data.setter
    def data(self, value: DataAccess) -> None:
        """Override the data access component."""

        self._set_component("data", value)

    @data.deleter
    def data(self) -> None:
        """Reset the data component override."""

        self._reset_component("data")

    @property
    def git_tracker(self) -> GitTracker:
        """Access the git tracker component."""

        return self._components.git_tracker

    @git_tracker.setter
    def git_tracker(self, value: GitTracker) -> None:
        """Override the git tracker component."""

        self._set_component("git_tracker", value)

    @git_tracker.deleter
    def git_tracker(self) -> None:
        """Reset the git tracker component override."""

        self._reset_component("git_tracker")

    def _create_test_run(
        self, repo_id: int, commit_hash: str, status: str = "pending"
    ) -> int:
        """Create a test run record.

        Args:
            repo_id: Repository ID
            commit_hash: Commit hash under test
            status: Initial status value

        Returns:
            The ID of the created test run
        """
        now = int(time.time())
        return self.data.create_test_run(repo_id, commit_hash, status, now)

    def _update_test_run(
        self, test_run_id: Optional[int], status: str, error: Optional[str] = None
    ):
        """Update the status metadata for a test run."""
        if not test_run_id:
            return

        now = int(time.time())
        started_at = now if status == "running" else None
        completed_at = now if status in ("completed", "error") else None

        self.data.update_test_run(
            test_run_id,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            error=error,
        )

    @staticmethod
    def _hash_secret(secret: str) -> str:
        return hashlib.sha256(secret.encode("utf-8")).hexdigest()

    @staticmethod
    def _coerce_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            normalized = value.strip().lower()
            if not normalized:
                return default
            return normalized not in {"0", "false", "no", "off"}
        return default

    @staticmethod
    def _has_local_changes(repo_url: str) -> bool:
        """Return True when ``repo_url`` points to a working tree with changes."""
        if not repo_url or not os.path.isdir(repo_url):
            return False

        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_url,
                check=False,
                capture_output=True,
                text=True,
            )
        except (OSError, ValueError) as exc:
            logger.warning(
                "Unable to inspect repository at %s for local changes: %s",
                repo_url,
                exc,
            )
            return False

        return bool(result.stdout.strip())

    def _bootstrap_dogfood_repository(self) -> None:
        dogfood_config = self.config.get("dogfood") or {}

        env_flag = os.getenv("FULL_AUTO_CI_DOGFOOD")
        enabled = (
            self._coerce_bool(env_flag)
            if env_flag is not None
            else self._coerce_bool(dogfood_config.get("enabled"))
        )

        if not enabled:
            logger.debug("Dogfooding disabled via configuration")
            return

        repo_url = (
            os.getenv("FULL_AUTO_CI_REPO_URL")
            or dogfood_config.get("url")
            or "https://github.com/calvinloveland/full-auto-ci.git"
        )
        repo_name = (
            os.getenv("FULL_AUTO_CI_REPO_NAME")
            or dogfood_config.get("name")
            or "Full Auto CI"
        )
        repo_branch = (
            os.getenv("FULL_AUTO_CI_REPO_BRANCH")
            or dogfood_config.get("branch")
            or "main"
        )

        repositories = self.list_repositories()
        existing = next(
            (repo for repo in repositories if repo["url"] == repo_url), None
        )

        if existing:
            repo_id = existing["id"]
            logger.info("Dogfooding repository already registered (ID %s)", repo_id)
        else:
            logger.info(
                "Registering dogfooding repository %s (%s)", repo_name, repo_url
            )
            repo_id = self.add_repository(repo_name, repo_url, repo_branch)
            if not repo_id:
                logger.error("Failed to register dogfooding repository %s", repo_url)
                return

        queue_flag = os.getenv("FULL_AUTO_CI_DOGFOOD_QUEUE")
        queue_on_start = (
            self._coerce_bool(queue_flag, True)
            if queue_flag is not None
            else self._coerce_bool(dogfood_config.get("queue_on_start"), True)
        )

        if not queue_on_start:
            logger.debug("Skipping automatic dogfood run queueing")
            return

        repo = self.git_tracker.get_repository(repo_id)
        if not repo:
            logger.debug(
                "Dogfooding repository %s not yet available in git tracker", repo_id
            )
            return

        latest_commit = repo.get_latest_commit()
        if not latest_commit:
            if not repo.pull():
                logger.debug(
                    "Unable to sync dogfooding repository %s for initial run", repo_id
                )
                return
            latest_commit = repo.get_latest_commit()

        if not latest_commit:
            logger.debug("No commits available to queue for dogfooding repository")
            return

        if self._enqueue_commit(repo_id, latest_commit["hash"], latest_commit):
            logger.info(
                "Queued latest dogfood commit %s for repository ID %s",
                latest_commit["hash"][0:7],
                repo_id,
            )

    def _create_or_get_pending_test_run(
        self, repo_id: int, commit_hash: str
    ) -> Tuple[Optional[int], bool]:
        """Create a new test run if no active run exists.

        Returns a tuple `(test_run_id, skipped)` where `skipped` indicates that a run
        is already pending/queued/running and a new one should not be enqueued.
        """
        latest = self.data.get_latest_test_run(repo_id, commit_hash)
        if latest:
            existing_id, existing_status = latest
            if existing_status in {"pending", "queued", "running"}:
                return existing_id, True

        return self._create_test_run(repo_id, commit_hash), False

    def _get_commit_record(self, repo_id: int, commit_hash: str) -> Dict[str, Any]:
        """Fetch commit metadata for queueing purposes."""
        record = self.data.fetch_commit(repo_id, commit_hash)
        if not record:
            return {"repository_id": repo_id, "hash": commit_hash}

        commit: Dict[str, Any] = {
            "repository_id": repo_id,
            "hash": commit_hash,
            "author": record.get("author"),
            "message": record.get("message"),
        }
        timestamp = record.get("timestamp")
        if timestamp is not None:
            commit["timestamp"] = timestamp
            commit["datetime"] = datetime.fromtimestamp(timestamp).isoformat()
        return commit

    def _enqueue_commit(
        self,
        repo_id: int,
        commit_hash: str,
        commit: Optional[Dict[str, Any]] = None,
        repo_info: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Enqueue a commit for testing, creating tracking metadata as needed."""
        repo_info = repo_info or self.get_repository(repo_id)
        if not repo_info:
            logger.error(
                "Repository %s not found when enqueueing commit %s",
                repo_id,
                commit_hash,
            )
            return False

        commit_record = commit or self._get_commit_record(repo_id, commit_hash)
        commit_record.setdefault("repository_id", repo_id)

        test_run_id, skipped = self._create_or_get_pending_test_run(
            repo_id, commit_hash
        )
        if skipped:
            logger.info(
                "Test run already pending for repo %s commit %s (run id %s)",
                repo_id,
                commit_hash,
                test_run_id,
            )
            return True

        self._update_test_run(test_run_id, "queued")
        task = {
            "type": "test",
            "repo_id": repo_id,
            "commit": commit_record,
            "test_run_id": test_run_id,
        }
        self.task_queue.put(task)
        return True

    def _update_repository_last_check(self, repo_id: int):
        """Record the last time a repository was polled."""
        self.data.update_repository_last_check(repo_id, int(time.time()))

    def _summarize_tool_results(
        self, results: Dict[str, Any]
    ) -> Tuple[str, Optional[str]]:
        """Determine aggregate status across tool executions."""
        overall = "success"
        messages: List[str] = []

        for tool_name, tool_result in results.items():
            if tool_result.get("status") != "success":
                overall = "error"
                detail = (
                    tool_result.get("error")
                    or tool_result.get("stderr")
                    or tool_result.get("status")
                )
                messages.append(f"{tool_name}: {detail}")

        message = "\n".join(messages) if messages else None
        return overall, message

    def start(self):
        """Start the CI service."""
        if self.running:
            logger.warning("Service is already running")
            return

        self.running = True

        # Start worker threads
        max_workers = self.config.get("service", "max_workers") or 4
        logger.info("Starting %s worker threads", max_workers)
        for _ in range(max_workers):
            worker = threading.Thread(target=self._worker_loop)
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

        # Start monitor thread
        logger.info("Starting repository monitor thread")
        self.monitor_thread = threading.Thread(target=self._monitor_repositories)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

        logger.info("CI Service started")

    def stop(self):
        """Stop the CI service."""
        if not self.running:
            logger.warning("Service is not running")
            return

        logger.info("Stopping CI Service")
        self.running = False

        # Join monitor thread
        monitor = self.monitor_thread
        if monitor:
            monitor.join(timeout=5.0)
            self.monitor_thread = None

        # Join worker threads
        for worker in self.workers:
            worker.join(timeout=1.0)

        self.workers.clear()

        logger.info("CI Service stopped")

    def _worker_loop(self):
        """Worker loop for processing tasks."""
        while self.running:
            try:
                # Try to get a task from the queue with timeout
                try:
                    task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # Process the task
                if task["type"] == "test":
                    self._process_test_task(task)
                else:
                    logger.warning("Unknown task type: %s", task["type"])

                # Mark the task as done
                self.task_queue.task_done()
            except Exception:  # pylint: disable=broad-except
                logger.exception("Error in worker thread")

    def _process_test_task(self, task):
        """Process a test task.

        Args:
            task: Task dictionary
        """
        repo_id = task["repo_id"]
        commit = task["commit"]
        commit_hash = commit["hash"]
        test_run_id = task.get("test_run_id")

        logger.info(
            "Processing test task for repository %s, commit %s", repo_id, commit_hash
        )

        repo = self.git_tracker.get_repository(repo_id)
        if not repo:
            error_msg = f"Repository {repo_id} not found"
            logger.error(error_msg)
            self._update_test_run(test_run_id, "error", error_msg)
            return

        self._update_test_run(test_run_id, "running")

        if not repo.checkout_commit(commit_hash):
            error_msg = f"Failed to checkout commit {commit_hash}"
            logger.error(error_msg)
            self._update_test_run(test_run_id, "error", error_msg)
            return

        try:
            logger.info(
                "Running tests for repository %s, commit %s", repo_id, commit_hash
            )
            results = self.tool_runner.run_all(repo.repo_path)
            self._store_results(repo_id, commit_hash, results)

            overall_status, message = self._summarize_tool_results(results)
            if overall_status == "success":
                logger.info(
                    "Tests completed for repository %s, commit %s", repo_id, commit_hash
                )
                self._update_test_run(test_run_id, "completed")
            else:
                logger.error(
                    "Tool failures detected for repository %s, commit %s: %s",
                    repo_id,
                    commit_hash,
                    message,
                )
                self._update_test_run(test_run_id, "error", message)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(
                "Error running tests for repository %s, commit %s: %s",
                repo_id,
                commit_hash,
                exc,
            )
            self._update_test_run(test_run_id, "error", str(exc))

    def _store_results(self, repo_id: int, commit_hash: str, results: Dict[str, Any]):
        """Store test results in the database.

        Args:
            repo_id: Repository ID
            commit_hash: Commit hash
            results: Test results
        """
        try:
            commit_id = self.data.get_commit_id(repo_id, commit_hash)
            if commit_id is None:
                logger.warning(
                    "Commit %s not found in database, creating placeholder entry",
                    commit_hash,
                )
                commit_id = self.data.create_commit(
                    repo_id,
                    commit_hash,
                    timestamp=int(time.time()),
                )

            for tool_name, tool_result in results.items():
                if tool_result is None:
                    continue

                payload = dict(tool_result)
                embedded_results = payload.pop("embedded_results", None)
                status = payload.get("status", "unknown")
                duration = float(payload.get("duration", 0.0) or 0.0)

                output = json.dumps(payload)
                self.data.insert_result(
                    commit_id,
                    tool=tool_name,
                    status=status,
                    output=output,
                    duration=duration,
                )

                if embedded_results:
                    for embedded in embedded_results:
                        if not isinstance(embedded, dict):
                            continue

                        embedded_tool = embedded.get("tool")
                        if not embedded_tool:
                            continue

                        embedded_status = embedded.get("status", "unknown")
                        embedded_duration = float(embedded.get("duration", 0.0) or 0.0)
                        embedded_output = embedded.get("output")

                        if isinstance(embedded_output, str):
                            output_text = embedded_output
                        else:
                            output_text = json.dumps(embedded_output or {})

                        self.data.insert_result(
                            commit_id,
                            tool=str(embedded_tool),
                            status=str(embedded_status),
                            output=output_text,
                            duration=embedded_duration,
                        )

            logger.info("Stored test results for commit %s", commit_hash)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(
                "Error storing test results for commit %s: %s", commit_hash, exc
            )

    def _monitor_repositories(self):
        """Monitor repositories for new commits."""
        while self.running:
            try:
                # Check for updates in all repositories
                logger.info("Checking repositories for updates")
                new_commits = self.git_tracker.check_for_updates()

                # Process new commits
                for repo_id, commits in new_commits.items():
                    logger.info(
                        "Found %s new commits for repository %s", len(commits), repo_id
                    )

                    for commit in commits:
                        logger.info("Queuing commit %s for testing", commit["hash"])
                        self._enqueue_commit(repo_id, commit["hash"], commit)

                # Update last check timestamps for all tracked repositories
                for repo_id in list(self.git_tracker.repos.keys()):
                    self._update_repository_last_check(repo_id)

                # Sleep before next check
                poll_interval = self.config.get("service", "poll_interval") or 60
                time.sleep(poll_interval)
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("Error monitoring repositories: %s", exc)
                time.sleep(60)  # Retry after a minute

    def run_tests(self, repo_id: int, commit_hash: str) -> Dict[str, Any]:
        """Run tests for a specific commit.

        Args:
            repo_id: Repository ID
            commit_hash: Git commit hash

        Returns:
            Dictionary with test results
        """
        logger.info("Running tests for repo %s, commit %s", repo_id, commit_hash)

        repo = self.git_tracker.get_repository(repo_id)
        if not repo:
            logger.error("Repository %s not found", repo_id)
            return {"status": "error", "error": f"Repository {repo_id} not found"}

        warnings: List[str] = []

        if self._has_local_changes(getattr(repo, "url", "")):
            warning = (
                "Uncommitted changes detected in the source repository; "
                "only committed files are included in this run."
            )
            logger.warning(
                "%s (repo %s: %s)", warning, repo_id, getattr(repo, "url", "")
            )
            warnings.append(warning)

        test_run_id = self._create_test_run(repo_id, commit_hash)
        self._update_test_run(test_run_id, "running")

        if not os.path.exists(repo.repo_path):
            if not repo.clone():
                error_msg = f"Failed to clone repository {repo_id}"
                logger.error(error_msg)
                self._update_test_run(test_run_id, "error", error_msg)
                return {"status": "error", "error": error_msg}

        if not repo.checkout_commit(commit_hash):
            error_msg = f"Failed to checkout commit {commit_hash}"
            logger.error(error_msg)
            self._update_test_run(test_run_id, "error", error_msg)
            return {"status": "error", "error": error_msg}

        try:
            logger.info(
                "Running tools for repository %s, commit %s", repo_id, commit_hash
            )
            results = self.tool_runner.run_all(repo.repo_path)
            self._store_results(repo_id, commit_hash, results)

            overall_status, message = self._summarize_tool_results(results)
            if overall_status == "success":
                logger.info(
                    "Tests completed for repository %s, commit %s", repo_id, commit_hash
                )
                self._update_test_run(test_run_id, "completed")
            else:
                logger.error(
                    "Tool failures detected for repository %s, commit %s: %s",
                    repo_id,
                    commit_hash,
                    message,
                )
                self._update_test_run(test_run_id, "error", message)

            formatted_results = {
                "status": "success" if overall_status == "success" else "error",
                "tools": results,
                "test_run_id": test_run_id,
            }
            if message:
                formatted_results["error"] = message
            if warnings:
                formatted_results["warnings"] = warnings
            return formatted_results
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(
                "Error running tests for repository %s, commit %s: %s",
                repo_id,
                commit_hash,
                exc,
            )
            self._update_test_run(test_run_id, "error", str(exc))
            return {"status": "error", "error": str(exc)}

    def add_repository(self, name: str, url: str, branch: str = "main") -> int:
        """Add a repository to monitor.

        Args:
            name: Repository name
            url: Repository URL
            branch: Branch to monitor

        Returns:
            Repository ID
        """
        repo_id = self.data.create_repository(name, url, branch)

        # Add to git tracker
        if repo_id is not None and repo_id > 0:
            success = self.git_tracker.add_repository(repo_id, name, url, branch)
            if not success:
                logger.error(
                    "Failed to add repository to git tracker: %s (%s)", name, url
                )
                # Note: We keep the database entry so the tracker can retry later.

        logger.info("Added repository: %s (%s)", name, url)
        return repo_id if repo_id is not None else 0

    def remove_repository(self, repo_id: int) -> bool:
        """Remove a repository from monitoring.

        Args:
            repo_id: Repository ID

        Returns:
            True if successful, False otherwise
        """
        # First remove from git tracker
        git_success = self.git_tracker.remove_repository(repo_id)
        if not git_success:
            logger.warning("Failed to remove repository from git tracker: %s", repo_id)

        # Then remove from database
        db_success = self.data.delete_repository(repo_id)

        if db_success:
            logger.info("Removed repository with ID: %s", repo_id)

        return db_success

    def get_repository(self, repo_id: int) -> Optional[Dict[str, Any]]:
        """Get repository information.

        Args:
            repo_id: Repository ID

        Returns:
            Repository information or None if not found
        """
        repo = self.data.fetch_repository(repo_id)
        if not repo:
            return None
        return {
            "id": repo["id"],
            "name": repo["name"],
            "url": repo["url"],
            "branch": repo["branch"],
        }

    def list_repositories(self) -> List[Dict[str, Any]]:
        """List all monitored repositories.

        Returns:
            List of repository information
        """
        repos = self.data.list_repositories()
        return [
            {
                "id": repo["id"],
                "name": repo["name"],
                "url": repo["url"],
                "branch": repo["branch"],
            }
            for repo in repos
        ]

    def get_test_results(
        self,
        repo_id: int,
        *,
        commit_hash: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Retrieve recent test runs with associated tool outputs."""
        runs = self.data.fetch_recent_test_runs(
            repo_id, limit=limit, commit_hash=commit_hash
        )

        hydrated: List[Dict[str, Any]] = []
        for run in runs:
            commit = self.data.fetch_commit_for_test_run(run["id"])
            results = self.data.fetch_results_for_test_run(run["id"])
            hydrated.append({**run, "commit": commit, "results": results})

        return hydrated

    # User management -----------------------------------------------------

    def create_user(
        self,
        username: str,
        password: str,
        role: str = "user",
        api_key: Optional[str] = None,
    ) -> int:
        """Create a user account with hashed credentials."""
        if not username:
            raise ValueError("Username is required")
        if not password:
            raise ValueError("Password is required")

        password_hash = self._hash_secret(password)
        api_key_hash = self._hash_secret(api_key) if api_key else None

        user_id = self.data.create_user(username, password_hash, role, api_key_hash)
        logger.info("Created user %s with role %s", username, role)
        return user_id

    def list_users(self) -> List[Dict[str, Any]]:
        """Return all known user records."""
        return self.data.list_users()

    def remove_user(self, username: str) -> bool:
        """Delete a user by username and report whether removal succeeded."""
        success = self.data.delete_user(username)
        if success:
            logger.info("Removed user %s", username)
        else:
            logger.warning("Attempted to remove non-existent user %s", username)
        return success

    def add_test_task(self, repo_id: int, commit_hash: str) -> bool:
        """Add a test task to the queue.

        Args:
            repo_id: Repository ID
            commit_hash: Commit hash to test

        Returns:
            True if the task was added, False otherwise
        """
        logger.info(
            "Adding test task for repository %s, commit %s", repo_id, commit_hash
        )

        repo_info = self.get_repository(repo_id)
        if not repo_info:
            logger.error("Repository not found: %s", repo_id)
            return False

        return self._enqueue_commit(repo_id, commit_hash, repo_info=repo_info)
