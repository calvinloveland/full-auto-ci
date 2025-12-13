"""Core service module for Full Auto CI."""

import hashlib
import json
import logging
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse

from .config import Config
from .db import DataAccess
from .git import GitTracker
from .providers import BaseProvider, ProviderConfigError
from .providers import registry as provider_registry
from .ratchet import RatchetManager
from .tools import Coverage, Lizard, Pylint, Tool, ToolRunner

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
    ratchet: RatchetManager


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
        data_access = DataAccess(self.db_path)
        self._components = ServiceComponents(
            git_tracker=GitTracker(db_path=self.db_path),
            tool_runner=self._build_tool_runner(),
            data=data_access,
            ratchet=RatchetManager(data_access, self.config),
        )
        self.data.initialize_schema()
        self._bootstrap_dogfood_repository()
        self.provider_registry = provider_registry
        self._providers: Dict[int, BaseProvider] = {}
        self._load_providers()
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

    def _build_tool_runner(self) -> ToolRunner:
        """Instantiate the ToolRunner using configuration flags."""

        tools_config = self.config.get("tools") or {}
        tools: List[Tool] = []

        pylint_config = tools_config.get("pylint", {})
        if self._tool_enabled(pylint_config):
            tools.append(Pylint())

        coverage_config = tools_config.get("coverage", {})
        if self._tool_enabled(coverage_config):
            run_cmd = self._normalize_run_tests_cmd(coverage_config)
            timeout = self._coerce_positive_float(
                coverage_config.get("timeout_seconds")
            )
            xml_timeout = self._coerce_positive_float(
                coverage_config.get("xml_timeout_seconds")
            )
            tools.append(
                Coverage(
                    run_tests_cmd=run_cmd,
                    timeout=timeout,
                    xml_timeout=xml_timeout,
                )
            )

        lizard_config = tools_config.get("lizard", {})
        if self._tool_enabled(lizard_config):
            max_ccn = lizard_config.get("max_ccn")
            if isinstance(max_ccn, (int, float)):
                tools.append(Lizard(max_ccn=int(max_ccn)))
            else:
                tools.append(Lizard())

        return ToolRunner(tools)

    @staticmethod
    def _tool_enabled(config: Any, default: bool = True) -> bool:
        if not isinstance(config, dict):
            return default
        return bool(config.get("enabled", default))

    @staticmethod
    def _normalize_run_tests_cmd(config: Any) -> Optional[List[str]]:
        if not isinstance(config, dict):
            return None

        value = config.get("run_tests_cmd")
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return value
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return None

    @staticmethod
    def _coerce_positive_float(value: Any) -> Optional[float]:
        if isinstance(value, (int, float)):
            candidate = float(value)
        elif isinstance(value, str):
            try:
                candidate = float(value.strip())
            except ValueError:
                return None
        else:
            return None

        if candidate <= 0:
            return None

        return candidate

    def _instantiate_provider(self, record: Dict[str, Any]) -> Optional[BaseProvider]:
        provider_type = record.get("type") or ""
        try:
            provider = self.provider_registry.create(provider_type, self, record)
        except KeyError:
            logger.error(
                "Unable to instantiate provider %s: type '%s' is not registered",
                record.get("id"),
                provider_type,
            )
            return None
        except ProviderConfigError as exc:
            logger.error(
                "Provider %s configuration invalid: %s",
                record.get("id"),
                exc,
            )
            return None
        return provider

    def _load_providers(self) -> None:
        self._providers.clear()
        for record in self.data.list_external_providers():
            provider = self._instantiate_provider(record)
            if provider is None:
                continue
            self._providers[provider.provider_id] = provider

    # Provider management -----------------------------------------------------
    def list_providers(self) -> List[Dict[str, Any]]:
        """Return enriched provider descriptors for dashboard/CLI consumption."""
        providers: List[Dict[str, Any]] = []
        for record in self.data.list_external_providers():
            descriptor: Dict[str, Any] = {
                "id": record["id"],
                "name": record["name"],
                "type": record["type"],
                "created_at": record.get("created_at"),
                "config": record.get("config") or {},
            }

            instance = self._providers.get(record["id"])
            if instance is not None:
                descriptor.update(
                    {
                        "display_name": instance.display_name,
                        "description": instance.description,
                    }
                )
            else:
                try:
                    provider_cls = self.provider_registry.get(record["type"])
                except KeyError:
                    provider_cls = None
                if provider_cls is not None:
                    descriptor.setdefault(
                        "display_name",
                        getattr(provider_cls, "display_name", record["type"]),
                    )
                    descriptor.setdefault(
                        "description",
                        getattr(provider_cls, "description", ""),
                    )
            providers.append(descriptor)
        return providers

    def get_provider_types(self) -> List[Dict[str, Any]]:
        """Expose metadata describing the registered provider types."""
        return list(self.provider_registry.available_types())

    def add_provider(
        self,
        provider_type: str,
        name: str,
        *,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Register a new external provider and return its descriptor."""
        provider_cls = self.provider_registry.get(provider_type)
        config_dict = dict(config or {})
        errors = list(provider_cls.validate_static_config(config_dict))
        if errors:
            raise ProviderConfigError("; ".join(errors))

        provider_id = self.data.create_external_provider(
            name, provider_type, config_dict
        )
        record = self.data.fetch_external_provider(provider_id)
        if not record:
            raise RuntimeError("Failed to load provider after creation")

        provider = self._instantiate_provider(record)
        if provider is not None:
            self._providers[provider_id] = provider
            return provider.to_dict()

        return {
            "id": provider_id,
            "name": name,
            "type": provider_type,
            "config": config_dict,
        }

    def remove_provider(self, provider_id: int) -> bool:
        """Delete a provider definition and drop any cached instance."""
        removed = self.data.delete_external_provider(provider_id)
        if removed:
            self._providers.pop(provider_id, None)
        return removed

    def sync_provider(
        self, provider_id: int, *, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Invoke ``sync_runs`` on the provider, returning normalized job data."""
        provider = self._providers.get(provider_id)
        if provider is None:
            record = self.data.fetch_external_provider(provider_id)
            if record is None:
                raise KeyError(f"Provider {provider_id} not found")
            provider = self._instantiate_provider(record)
            if provider is None:
                raise RuntimeError(f"Provider {provider_id} could not be loaded")
            self._providers[provider_id] = provider
        return provider.sync_runs(limit=limit)

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

    @property
    def ratchet_manager(self) -> RatchetManager:
        """Access the ratchet manager component."""

        return self._components.ratchet

    @ratchet_manager.setter
    def ratchet_manager(self, value: RatchetManager) -> None:
        """Override the ratchet manager component."""

        self._set_component("ratchet", value)

    @ratchet_manager.deleter
    def ratchet_manager(self) -> None:
        """Reset the ratchet manager component override."""

        self._reset_component("ratchet")

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

        if not self._dogfood_enabled(dogfood_config):
            logger.debug("Dogfooding disabled via configuration")
            return

        repo_info = self._resolve_dogfood_repo_info(dogfood_config)
        repo_id = self._ensure_dogfood_repository(repo_info)
        if not repo_id:
            return

        if not self._should_queue_dogfood_run(dogfood_config):
            logger.debug("Skipping automatic dogfood run queueing")
            return

        latest_commit = self._resolve_latest_dogfood_commit(repo_id)
        if not latest_commit:
            return

        if self._enqueue_commit(repo_id, latest_commit["hash"], latest_commit):
            logger.info(
                "Queued latest dogfood commit %s for repository ID %s",
                latest_commit["hash"][0:7],
                repo_id,
            )

    def _dogfood_enabled(self, dogfood_config: Dict[str, Any]) -> bool:
        env_flag = os.getenv("FULL_AUTO_CI_DOGFOOD")
        if env_flag is not None:
            return self._coerce_bool(env_flag)
        return self._coerce_bool(dogfood_config.get("enabled"))

    def _resolve_dogfood_repo_info(
        self, dogfood_config: Dict[str, Any]
    ) -> Dict[str, str]:
        return {
            "url": (
                os.getenv("FULL_AUTO_CI_REPO_URL")
                or dogfood_config.get("url")
                or "https://github.com/calvinloveland/full-auto-ci.git"
            ),
            "name": (
                os.getenv("FULL_AUTO_CI_REPO_NAME")
                or dogfood_config.get("name")
                or "Full Auto CI"
            ),
            "branch": (
                os.getenv("FULL_AUTO_CI_REPO_BRANCH")
                or dogfood_config.get("branch")
                or "main"
            ),
        }

    def _ensure_dogfood_repository(self, repo_info: Dict[str, str]) -> Optional[int]:
        repositories = self.list_repositories()
        existing = next(
            (repo for repo in repositories if repo["url"] == repo_info["url"]),
            None,
        )

        if existing:
            repo_id = existing["id"]
            logger.info("Dogfooding repository already registered (ID %s)", repo_id)
            return repo_id

        logger.info(
            "Registering dogfooding repository %s (%s)",
            repo_info["name"],
            repo_info["url"],
        )
        repo_id = self.add_repository(
            repo_info["name"],
            repo_info["url"],
            repo_info["branch"],
        )
        if not repo_id:
            logger.error(
                "Failed to register dogfooding repository %s", repo_info["url"]
            )
            return None
        return repo_id

    def _should_queue_dogfood_run(self, dogfood_config: Dict[str, Any]) -> bool:
        queue_flag = os.getenv("FULL_AUTO_CI_DOGFOOD_QUEUE")
        if queue_flag is not None:
            return self._coerce_bool(queue_flag, True)
        return self._coerce_bool(dogfood_config.get("queue_on_start"), True)

    def _resolve_latest_dogfood_commit(self, repo_id: int) -> Optional[Dict[str, Any]]:
        repo = self.git_tracker.get_repository(repo_id)
        if not repo:
            logger.debug(
                "Dogfooding repository %s not yet available in git tracker", repo_id
            )
            return None

        latest_commit = repo.get_latest_commit()
        if latest_commit:
            return latest_commit

        if not repo.pull():
            logger.debug(
                "Unable to sync dogfooding repository %s for initial run", repo_id
            )
            return None

        latest_commit = repo.get_latest_commit()
        if latest_commit:
            return latest_commit

        logger.debug("No commits available to queue for dogfooding repository")
        return None

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
            try:
                self.ratchet_manager.apply(repo_id, results)
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("Ratchet evaluation failed: %s", exc)
                for tool_result in results.values():
                    if (
                        isinstance(tool_result, dict)
                        and tool_result.get("status") == "success"
                    ):
                        tool_result["status"] = "error"
                        tool_result["error"] = f"Ratchet evaluation failed: {exc}"
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
            commit_id = self._ensure_commit_id(repo_id, commit_hash)
            if commit_id is None:
                return

            for tool_name, tool_result in results.items():
                self._store_tool_result(commit_id, tool_name, tool_result)

            logger.info("Stored test results for commit %s", commit_hash)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(
                "Error storing test results for commit %s: %s", commit_hash, exc
            )

    def _ensure_commit_id(self, repo_id: int, commit_hash: str) -> Optional[int]:
        commit_id = self.data.get_commit_id(repo_id, commit_hash)
        if commit_id is not None:
            return commit_id

        logger.warning(
            "Commit %s not found in database, creating placeholder entry",
            commit_hash,
        )
        return self.data.create_commit(
            repo_id,
            commit_hash,
            timestamp=int(time.time()),
        )

    def _store_tool_result(
        self,
        commit_id: int,
        tool_name: str,
        tool_result: Any,
    ) -> None:
        if tool_result is None:
            return

        payload = dict(tool_result)
        embedded_results = payload.pop("embedded_results", None)
        status = payload.get("status", "unknown")
        duration = float(payload.get("duration", 0.0) or 0.0)

        self.data.insert_result(
            commit_id,
            tool=tool_name,
            status=status,
            output=json.dumps(payload),
            duration=duration,
        )

        self._store_embedded_results(commit_id, embedded_results)

    def _store_embedded_results(self, commit_id: int, embedded_results: Any) -> None:
        if not embedded_results:
            return
        if not isinstance(embedded_results, list):
            return

        for embedded in embedded_results:
            stored = self._coerce_embedded_result(embedded)
            if stored is None:
                continue
            self.data.insert_result(commit_id, **stored)

    @staticmethod
    def _coerce_embedded_result(embedded: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(embedded, dict):
            return None

        embedded_tool = embedded.get("tool")
        if not embedded_tool:
            return None

        embedded_status = embedded.get("status", "unknown")
        embedded_duration = float(embedded.get("duration", 0.0) or 0.0)
        embedded_output = embedded.get("output")

        if isinstance(embedded_output, str):
            output_text = embedded_output
        else:
            output_text = json.dumps(embedded_output or {})

        return {
            "tool": str(embedded_tool),
            "status": str(embedded_status),
            "output": output_text,
            "duration": embedded_duration,
        }

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

    def run_tests(
        self,
        repo_id: int,
        commit_hash: str,
        *,
        include_working_tree: bool = False,
    ) -> Dict[str, Any]:
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

        repo_url = self._coerce_repo_url(repo)
        include_working_tree, local_repo_path = self._resolve_working_tree_inclusion(
            repo_url,
            include_working_tree=include_working_tree,
        )

        warnings = self._collect_local_change_warnings(
            repo,
            repo_id,
            include_working_tree=include_working_tree,
        )

        working_tree_setup = self._prepare_working_tree_run(
            repo_id,
            commit_hash,
            repo_url,
            local_repo_path,
            include_working_tree=include_working_tree,
        )
        if isinstance(working_tree_setup, dict):
            return working_tree_setup
        resolved_commit_hash, working_snapshot, run_root = working_tree_setup

        test_run_id = self._create_test_run(repo_id, resolved_commit_hash)
        self._update_test_run(test_run_id, "running")

        try:
            early_error = self._maybe_prepare_repo_for_run(
                repo,
                repo_id,
                commit_hash,
                test_run_id,
                include_working_tree=include_working_tree,
            )
            if early_error is not None:
                return early_error

            return self._execute_tool_run(
                repo_id,
                resolved_commit_hash,
                repo,
                test_run_id,
                warnings,
                run_root=run_root,
            )
        finally:
            self._cleanup_working_snapshot(working_snapshot)

    @staticmethod
    def _coerce_repo_url(repo: Any) -> str:
        repo_url = getattr(repo, "url", "")
        return repo_url if isinstance(repo_url, str) else ""

    def _resolve_working_tree_inclusion(
        self,
        repo_url: str,
        *,
        include_working_tree: bool,
    ) -> tuple[bool, str | None]:
        local_repo_path = self._resolve_local_repo_path(repo_url)
        if include_working_tree:
            return True, local_repo_path
        if not local_repo_path:
            return False, None
        if not os.path.isdir(local_repo_path):
            return False, local_repo_path
        if self._has_local_changes(local_repo_path):
            return True, local_repo_path
        return False, local_repo_path

    def _maybe_prepare_repo_for_run(
        self,
        repo: Any,
        repo_id: int,
        commit_hash: str,
        test_run_id: int,
        *,
        include_working_tree: bool,
    ) -> Optional[Dict[str, Any]]:
        if include_working_tree:
            return None
        return self._prepare_repo_for_run(repo, repo_id, commit_hash, test_run_id)

    @staticmethod
    def _cleanup_working_snapshot(working_snapshot: str | None) -> None:
        if not working_snapshot:
            return
        if os.path.isdir(working_snapshot):
            shutil.rmtree(working_snapshot, ignore_errors=True)

    def _prepare_working_tree_run(
        self,
        repo_id: int,
        commit_hash: str,
        repo_url: str,
        local_repo_path: str | None,
        *,
        include_working_tree: bool,
    ) -> tuple[str, str | None, str | None] | Dict[str, Any]:
        if not include_working_tree:
            return commit_hash, None, None

        if not local_repo_path:
            error_msg = "includeWorkingTree is only supported for local repository URLs"
            logger.error("%s (repo %s: %s)", error_msg, repo_id, repo_url)
            return {"status": "error", "error": error_msg}

        if not os.path.isdir(local_repo_path):
            error_msg = "Local repository path not found"
            logger.error(
                "%s (repo %s: %s -> %s)",
                error_msg,
                repo_id,
                repo_url,
                local_repo_path,
            )
            return {"status": "error", "error": error_msg}

        head = self._resolve_head_commit(local_repo_path)
        suffix = f"working-tree-{int(time.time())}"
        resolved_commit_hash = f"{head}+{suffix}" if head else f"{commit_hash}+{suffix}"

        working_snapshot = self._snapshot_working_tree(local_repo_path, repo_id)
        return resolved_commit_hash, working_snapshot, working_snapshot

    def _collect_local_change_warnings(
        self,
        repo: Any,
        repo_id: int,
        *,
        include_working_tree: bool = False,
    ) -> List[str]:
        warnings: List[str] = []

        repo_url = getattr(repo, "url", "")
        if not isinstance(repo_url, str):
            repo_url = ""
        local_path = self._resolve_local_repo_path(repo_url)
        if not local_path:
            return warnings

        if not self._has_local_changes(local_path):
            return warnings

        if include_working_tree:
            warning = (
                "Uncommitted changes detected in the source repository; "
                "working tree files are included in this run."
            )
            logger.info(
                "%s (repo %s: %s)",
                warning,
                repo_id,
                repo_url,
            )
            warnings.append(warning)
        else:
            warning = (
                "Uncommitted changes detected in the source repository; "
                "only committed files are included in this run."
            )
            logger.warning("%s (repo %s: %s)", warning, repo_id, repo_url)
            warnings.append(warning)
        return warnings

    @staticmethod
    def _resolve_local_repo_path(repo_url: str) -> Optional[str]:
        if not isinstance(repo_url, str) or not repo_url.strip():
            return None

        parsed = urlparse(repo_url)
        if parsed.scheme and parsed.scheme != "file":
            return None
        if parsed.scheme == "file":
            path = unquote(parsed.path or "")
            return path or None
        return repo_url

    @staticmethod
    def _resolve_head_commit(repo_path: str) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path,
                check=False,
                capture_output=True,
                text=True,
            )
        except (OSError, ValueError):
            return None
        value = (result.stdout or "").strip()
        return value if result.returncode == 0 and value else None

    def _snapshot_working_tree(self, repo_path: str, repo_id: int) -> str:
        base_dir = os.path.expanduser("~/.fullautoci/working_tree_snapshots")
        os.makedirs(base_dir, exist_ok=True)

        snapshot_dir = tempfile.mkdtemp(prefix=f"repo_{repo_id}_", dir=base_dir)

        ignore = shutil.ignore_patterns(
            ".git",
            ".hg",
            ".svn",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            ".tox",
            ".venv",
            "venv",
            "dist",
            "build",
            "*.egg-info",
            ".coverage",
            "coverage.xml",
        )

        shutil.rmtree(snapshot_dir, ignore_errors=True)
        shutil.copytree(
            repo_path,
            snapshot_dir,
            symlinks=True,
            ignore=ignore,
            dirs_exist_ok=False,
        )
        return snapshot_dir

    def _prepare_repo_for_run(
        self,
        repo: Any,
        repo_id: int,
        commit_hash: str,
        test_run_id: int,
    ) -> Optional[Dict[str, Any]]:
        if not os.path.exists(repo.repo_path) and not repo.clone():
            error_msg = f"Failed to clone repository {repo_id}"
            logger.error(error_msg)
            self._update_test_run(test_run_id, "error", error_msg)
            return {"status": "error", "error": error_msg}

        if not repo.checkout_commit(commit_hash):
            error_msg = f"Failed to checkout commit {commit_hash}"
            logger.error(error_msg)
            self._update_test_run(test_run_id, "error", error_msg)
            return {"status": "error", "error": error_msg}

        return None

    def _execute_tool_run(
        self,
        repo_id: int,
        commit_hash: str,
        repo: Any,
        test_run_id: int,
        warnings: List[str],
        *,
        run_root: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            logger.info(
                "Running tools for repository %s, commit %s", repo_id, commit_hash
            )
            root = run_root or repo.repo_path
            results = self.tool_runner.run_all(root)
            self._store_results(repo_id, commit_hash, results)

            overall_status, message = self._summarize_tool_results(results)
            self._finalize_test_run(
                test_run_id, repo_id, commit_hash, overall_status, message
            )
            return self._format_run_results(
                overall_status, message, results, test_run_id, warnings
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(
                "Error running tests for repository %s, commit %s: %s",
                repo_id,
                commit_hash,
                exc,
            )
            self._update_test_run(test_run_id, "error", str(exc))
            return {"status": "error", "error": str(exc)}

    def _finalize_test_run(
        self,
        test_run_id: int,
        repo_id: int,
        commit_hash: str,
        overall_status: str,
        message: Optional[str],
    ) -> None:
        if overall_status == "success":
            logger.info(
                "Tests completed for repository %s, commit %s", repo_id, commit_hash
            )
            self._update_test_run(test_run_id, "completed")
            return

        logger.error(
            "Tool failures detected for repository %s, commit %s: %s",
            repo_id,
            commit_hash,
            message,
        )
        self._update_test_run(test_run_id, "error", message)

    @staticmethod
    def _format_run_results(
        overall_status: str,
        message: Optional[str],
        results: Dict[str, Any],
        test_run_id: int,
        warnings: List[str],
    ) -> Dict[str, Any]:
        formatted_results: Dict[str, Any] = {
            "status": "success" if overall_status == "success" else "error",
            "tools": results,
            "test_run_id": test_run_id,
        }
        if message:
            formatted_results["error"] = message
        if warnings:
            formatted_results["warnings"] = warnings
        return formatted_results

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
