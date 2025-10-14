"""Database access layer for Full Auto CI."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple


class DataAccess:
    """Lightweight data-access helper around the SQLite database."""

    # pylint: disable=too-many-public-methods

    def __init__(self, db_path: str):
        self.db_path = db_path

    @contextmanager
    def _transaction(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def initialize_schema(self):
        """Ensure all required tables, columns, and indexes exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with self._transaction() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS repositories (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    url TEXT NOT NULL,
                    branch TEXT DEFAULT 'main',
                    status TEXT DEFAULT 'active',
                    last_check INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

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

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY,
                    commit_id INTEGER,
                    tool TEXT NOT NULL,
                    status TEXT NOT NULL,
                    output TEXT,
                    duration REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (commit_id) REFERENCES commits (id)
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    api_key_hash TEXT,
                    role TEXT DEFAULT 'user',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS api_keys (
                    id INTEGER PRIMARY KEY,
                    key_hash TEXT UNIQUE NOT NULL,
                    created_at INTEGER NOT NULL
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS test_runs (
                    id INTEGER PRIMARY KEY,
                    repository_id INTEGER NOT NULL,
                    commit_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    started_at INTEGER,
                    completed_at INTEGER,
                    error TEXT,
                    FOREIGN KEY (repository_id) REFERENCES repositories (id)
                )
                """
            )

            # Backfill columns that might be missing in older installations
            added_status = self._ensure_column(
                cursor, "repositories", "status", "TEXT DEFAULT 'active'"
            )
            if added_status:
                cursor.execute(
                    "UPDATE repositories SET status = 'active' WHERE status IS NULL"
                )
            self._ensure_column(cursor, "repositories", "last_check", "INTEGER")

            # Helpful indexes for frequent lookups
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_repositories_status
                ON repositories(status)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_commits_repo_hash
                ON commits(repository_id, commit_hash)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_results_commit
                ON results(commit_id)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_test_runs_repo_status
                ON test_runs(repository_id, status)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_test_runs_repo_commit
                ON test_runs(repository_id, commit_hash)
                """
            )

    @staticmethod
    def _ensure_column(
        cursor: sqlite3.Cursor, table: str, column: str, definition_suffix: str
    ) -> bool:
        """Add a column to ``table`` when it's missing.

        Returns ``True`` when the column was added.
        """
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [row[1] for row in cursor.fetchall()]
        if column in columns:
            return False
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition_suffix}")
        return True

    def create_repository(self, name: str, url: str, branch: str) -> int:
        """Insert a repository record and return its identifier."""
        with self._transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO repositories (name, url, branch) VALUES (?, ?, ?)",
                (name, url, branch),
            )
            return int(cursor.lastrowid or 0)

    def delete_repository(self, repo_id: int) -> bool:
        """Delete a repository by ``repo_id`` and report success."""
        with self._transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM repositories WHERE id = ?", (repo_id,))
            return cursor.rowcount > 0

    def fetch_repository(self, repo_id: int) -> Optional[Dict[str, Any]]:
        """Return a repository row as a dictionary or ``None`` when missing."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, name, url, branch, status, last_check
                FROM repositories
                WHERE id = ?
                """,
                (repo_id,),
            )
            row = cursor.fetchone()

        if not row:
            return None

        return {
            "id": row[0],
            "name": row[1],
            "url": row[2],
            "branch": row[3],
            "status": row[4],
            "last_check": row[5],
        }

    def list_repositories(self) -> List[Dict[str, Any]]:
        """Return all repositories as a list of dictionaries."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, name, url, branch, status, last_check
                FROM repositories
                """
            )
            rows = cursor.fetchall()

        return [
            {
                "id": row[0],
                "name": row[1],
                "url": row[2],
                "branch": row[3],
                "status": row[4],
                "last_check": row[5],
            }
            for row in rows
        ]

    def update_repository_last_check(self, repo_id: int, timestamp: int):
        """Persist the ``timestamp`` of the last repository polling cycle."""
        with self._transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE repositories SET last_check = ? WHERE id = ?",
                (timestamp, repo_id),
            )

    def get_commit_id(self, repo_id: int, commit_hash: str) -> Optional[int]:
        """Return the primary key for ``commit_hash`` within ``repo_id``."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM commits WHERE repository_id = ? AND commit_hash = ?",
                (repo_id, commit_hash),
            )
            result = cursor.fetchone()
        return int(result[0]) if result else None

    def create_commit(self, repo_id: int, commit_hash: str, **metadata: Any) -> int:
        """Insert a commit row and return its identifier.

        Accepted keyword arguments: ``author``, ``message``, and ``timestamp``.
        """
        allowed_keys = {"author", "message", "timestamp"}
        unexpected_keys = set(metadata) - allowed_keys
        if unexpected_keys:
            invalid = ", ".join(sorted(unexpected_keys))
            raise ValueError(f"Unsupported commit fields: {invalid}")

        author = metadata.get("author")
        message = metadata.get("message")
        timestamp = metadata.get("timestamp")

        with self._transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO commits (repository_id, commit_hash, author, message, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (repo_id, commit_hash, author, message, timestamp),
            )
            return int(cursor.lastrowid or 0)

    def fetch_commit(self, repo_id: int, commit_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve commit metadata for ``commit_hash`` within ``repo_id``."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT commit_hash, author, message, timestamp
                FROM commits
                WHERE repository_id = ? AND commit_hash = ?
                """,
                (repo_id, commit_hash),
            )
            row = cursor.fetchone()
        if not row:
            return None
        return {
            "hash": row[0],
            "author": row[1],
            "message": row[2],
            "timestamp": row[3],
        }

    def insert_result(self, commit_id: int, **result: Any):
        """Persist a tool execution result for ``commit_id``.

        Expects keyword arguments ``tool``, ``status``, ``output``, and
        ``duration``.
        """
        required_fields = ("tool", "status", "output", "duration")
        missing_fields = [field for field in required_fields if field not in result]
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"Missing result fields: {missing}")

        tool = str(result["tool"])
        status = str(result["status"])
        output = str(result["output"])
        duration = float(result["duration"])

        with self._transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO results (commit_id, tool, status, output, duration)
                VALUES (?, ?, ?, ?, ?)
                """,
                (commit_id, tool, status, output, duration),
            )

    def create_test_run(
        self, repo_id: int, commit_hash: str, status: str, created_at: int
    ) -> int:
        """Create a test run record and return its ID."""
        with self._transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO test_runs (repository_id, commit_hash, status, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (repo_id, commit_hash, status, created_at),
            )
            return int(cursor.lastrowid or 0)

    def get_latest_test_run(
        self, repo_id: int, commit_hash: str
    ) -> Optional[Tuple[int, str]]:
        """Return the most recent test run ID and status for a commit."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, status
                FROM test_runs
                WHERE repository_id = ? AND commit_hash = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (repo_id, commit_hash),
            )
            row = cursor.fetchone()
        if not row:
            return None
        return int(row[0]), str(row[1])

    def update_test_run(self, test_run_id: int, **updates: Optional[Any]):
        """Apply partial updates to a test run row.

        Supported keyword arguments are ``status``, ``started_at``,
        ``completed_at``, and ``error``.
        """
        allowed_keys = {"status", "started_at", "completed_at", "error"}
        unexpected_keys = set(updates) - allowed_keys
        if unexpected_keys:
            invalid = ", ".join(sorted(unexpected_keys))
            raise ValueError(f"Unsupported test run fields: {invalid}")

        status = updates.get("status")
        started_at = updates.get("started_at")
        completed_at = updates.get("completed_at")
        error = updates.get("error")

        assignments: List[str] = []
        params: List[Any] = []

        if status is not None:
            assignments.append("status = ?")
            params.append(status)
        if started_at is not None:
            assignments.append("started_at = ?")
            params.append(started_at)
        if completed_at is not None:
            assignments.append("completed_at = ?")
            params.append(completed_at)
        if "error" in updates or (status in {"completed", "error"} and error is None):
            assignments.append("error = ?")
            params.append(error)

        if not assignments:
            return

        params.append(test_run_id)
        with self._transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE test_runs SET {', '.join(assignments)} WHERE id = ?",
                params,
            )

    def summarize_test_runs(self, repo_id: int) -> Dict[str, int]:
        """Return counts of test runs by status for a repository."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT status, COUNT(*)
                FROM test_runs
                WHERE repository_id = ?
                GROUP BY status
                """,
                (repo_id,),
            )
            rows = cursor.fetchall()

        return {row[0]: int(row[1]) for row in rows}

    def fetch_recent_test_runs(
        self, repo_id: int, limit: int = 20, commit_hash: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch recent test runs for a repository."""
        query = [
            "SELECT id, commit_hash, status, created_at, started_at, completed_at, error",
            "FROM test_runs",
            "WHERE repository_id = ?",
        ]
        params: List[Any] = [repo_id]

        if commit_hash:
            query.append("AND commit_hash = ?")
            params.append(commit_hash)

        query.append("ORDER BY created_at DESC, id DESC")

        if limit:
            query.append("LIMIT ?")
            params.append(limit)

        sql = " ".join(query)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            rows = cursor.fetchall()

        return [
            {
                "id": int(row[0]),
                "commit_hash": row[1],
                "status": row[2],
                "created_at": row[3],
                "started_at": row[4],
                "completed_at": row[5],
                "error": row[6],
            }
            for row in rows
        ]

    def fetch_commit_for_test_run(self, test_run_id: int) -> Optional[Dict[str, Any]]:
        """Fetch commit metadata associated with a test run."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT c.id, c.commit_hash, c.author, c.message, c.timestamp, c.repository_id
                FROM commits c
                JOIN test_runs tr
                  ON tr.repository_id = c.repository_id
                 AND tr.commit_hash = c.commit_hash
                WHERE tr.id = ?
                LIMIT 1
                """,
                (test_run_id,),
            )
            row = cursor.fetchone()

        if not row:
            return None

        return {
            "id": int(row[0]),
            "hash": row[1],
            "author": row[2],
            "message": row[3],
            "timestamp": row[4],
            "repository_id": row[5],
        }

    def fetch_results_for_test_run(self, test_run_id: int) -> List[Dict[str, Any]]:
        """Fetch tool results for a specific test run."""
        commit = self.fetch_commit_for_test_run(test_run_id)
        if not commit or commit.get("id") is None:
            return []

        commit_id = commit["id"]

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT tool, status, output, duration, created_at
                FROM results
                WHERE commit_id = ?
                ORDER BY created_at DESC, id DESC
                """,
                (commit_id,),
            )
            rows = cursor.fetchall()

        return [self._hydrate_result_row(row) for row in rows]

    @staticmethod
    def _hydrate_result_row(row: Tuple[Any, ...]) -> Dict[str, Any]:
        tool, status, output_text, duration, created_at = row
        parsed_output: Optional[Any] = None
        if output_text:
            try:
                parsed_output = json.loads(output_text)
            except (TypeError, json.JSONDecodeError):
                parsed_output = None

        return {
            "tool": tool,
            "status": status,
            "output": output_text,
            "duration": duration,
            "created_at": created_at,
            "data": parsed_output,
        }

    # User management helpers

    def create_user(
        self,
        username: str,
        password_hash: str,
        role: str = "user",
        api_key_hash: Optional[str] = None,
    ) -> int:
        """Create a user account and return its identifier."""
        with self._transaction() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """
                    INSERT INTO users (username, password_hash, api_key_hash, role)
                    VALUES (?, ?, ?, ?)
                    """,
                    (username, password_hash, api_key_hash, role),
                )
            except sqlite3.IntegrityError as exc:  # username uniqueness
                raise ValueError(f"User '{username}' already exists") from exc
            return int(cursor.lastrowid or 0)

    def list_users(self) -> List[Dict[str, Any]]:
        """Return all users sorted by username."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, username, role, created_at
                FROM users
                ORDER BY username COLLATE NOCASE
                """
            )
            rows = cursor.fetchall()
        return [
            {
                "id": row[0],
                "username": row[1],
                "role": row[2],
                "created_at": row[3],
            }
            for row in rows
        ]

    def delete_user(self, username: str) -> bool:
        """Remove a user by username and report whether a row was deleted."""
        with self._transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM users WHERE username = ?", (username,))
            return cursor.rowcount > 0

    def get_user_credentials(self, username: str) -> Optional[Dict[str, Any]]:
        """Return credential data for ``username`` or ``None`` when missing."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, username, password_hash, role
                FROM users
                WHERE username = ?
                """,
                (username,),
            )
            row = cursor.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "username": row[1],
            "password_hash": row[2],
            "role": row[3],
        }
