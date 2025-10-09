"""Webhook handlers for Full Auto CI."""

import hashlib
import hmac
import json
import logging
import os
import sqlite3
import time
from datetime import datetime
from typing import Any, Callable, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class _WebhookAbort(Exception):
    """Internal exception used to abort webhook processing."""


HandlerFunc = Callable[[Dict[str, str], Dict[str, Any]], Optional[Dict[str, Any]]]


class WebhookHandler:
    """Webhook handler for Git providers."""

    def __init__(self, db_path: Optional[str] = None, secret: Optional[str] = None):
        """Initialize the webhook handler.

        Args:
            db_path: Path to the SQLite database
            secret: Secret for webhook signature verification
        """
        self.db_path = db_path or os.path.expanduser("~/.fullautoci/database.sqlite")
        self.secret = secret
        self.handlers: Dict[str, HandlerFunc] = {
            "github": self._handle_github,
            "gitlab": self._handle_gitlab,
            "bitbucket": self._handle_bitbucket,
        }

    def register_handler(self, provider: str, handler: HandlerFunc) -> None:
        """Register or override the handler for ``provider``."""

        self.handlers[provider.lower()] = handler

    def handle(
        self, provider: str, headers: Dict[str, str], payload: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Handle a webhook from a Git provider.

        Args:
            provider: Git provider name (github, gitlab, bitbucket)
            headers: HTTP headers
            payload: Webhook payload

        Returns:
            Dictionary with commit information or None if not a push event
        """
        logger.info("Received webhook from %s", provider)

        # Verify signature if secret is set
        result: Optional[Dict[str, Any]] = None
        try:
            if self.secret and not self._verify_signature(provider, headers, payload):
                self._abort(
                    logging.WARNING, "Invalid signature for %s webhook", provider
                )

            handler = self.handlers.get(provider.lower())
            if not handler:
                self._abort(logging.WARNING, "No handler for provider %s", provider)

            result = handler(headers, payload)
        except _WebhookAbort:
            result = None
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("Error handling %s webhook", provider)
            result = None

        return result

    def _verify_signature(
        self, provider: str, headers: Dict[str, str], payload: Dict[str, Any]
    ) -> bool:
        """Verify the webhook signature.

        Args:
            provider: Git provider name
            headers: HTTP headers
            payload: Webhook payload

        Returns:
            True if signature is valid, False otherwise
        """
        # If no secret is set, skip verification
        if not self.secret:
            logger.warning("Webhook secret not set, skipping signature verification")
            return True

        normalized_provider = provider.lower()
        is_valid = False

        if normalized_provider == "github":
            signature_header = headers.get("X-Hub-Signature-256")
            if signature_header:
                payload_bytes = json.dumps(payload).encode("utf-8")
                hmac_obj = hmac.new(
                    self.secret.encode("utf-8"), payload_bytes, hashlib.sha256
                )
                expected_signature = f"sha256={hmac_obj.hexdigest()}"
                is_valid = hmac.compare_digest(signature_header, expected_signature)
        elif normalized_provider == "gitlab":
            token_header = headers.get("X-Gitlab-Token")
            if token_header:
                is_valid = hmac.compare_digest(token_header, self.secret)
        elif normalized_provider == "bitbucket":
            # Bitbucket doesn't have a standard signature header
            is_valid = True

        return is_valid

    def _handle_github(
        self, headers: Dict[str, str], payload: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Handle a GitHub webhook.

        Args:
            headers: HTTP headers
            payload: Webhook payload

        Returns:
            Dictionary with commit information or None if not a push event
        """
        try:
            event_type = headers.get("X-GitHub-Event")
            if event_type != "push":
                self._abort(logging.INFO, "Ignoring GitHub event: %s", event_type)

            repository = payload.get("repository", {})
            repo_url = repository.get("clone_url")
            if not repository.get("full_name") or not repo_url:
                self._abort(
                    logging.WARNING, "Missing repository information in GitHub webhook"
                )

            repo_id = self._get_repo_id_by_url(repo_url)
            if not repo_id:
                self._abort(
                    logging.WARNING,
                    "Repository not found in database: %s",
                    repo_url,
                )

            commits = payload.get("commits") or []
            if not commits:
                self._abort(logging.INFO, "No commits in GitHub push event")

            commit = commits[-1]
            timestamp_raw = commit.get("timestamp")
            if not timestamp_raw:
                self._abort(logging.WARNING, "Commit missing timestamp in GitHub push")

            timestamp = self._parse_iso_timestamp(timestamp_raw)

            return {
                "provider": "github",
                "repository_id": repo_id,
                "hash": commit.get("id"),
                "author": commit.get("author", {}).get("name"),
                "author_email": commit.get("author", {}).get("email"),
                "timestamp": timestamp,
                "message": commit.get("message"),
                "datetime": datetime.fromtimestamp(timestamp).isoformat(),
            }
        except _WebhookAbort:
            return None

    def _handle_gitlab(
        self, headers: Dict[str, str], payload: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Handle a GitLab webhook.

        Args:
            headers: HTTP headers
            payload: Webhook payload

        Returns:
            Dictionary with commit information or None if not a push event
        """
        try:
            event_type = headers.get("X-Gitlab-Event")
            if event_type != "Push Hook":
                self._abort(logging.INFO, "Ignoring GitLab event: %s", event_type)

            project = payload.get("project", {})
            repo_url = project.get("git_http_url")
            if not project.get("path_with_namespace") or not repo_url:
                self._abort(
                    logging.WARNING, "Missing repository information in GitLab webhook"
                )

            repo_id = self._get_repo_id_by_url(repo_url)
            if not repo_id:
                self._abort(
                    logging.WARNING,
                    "Repository not found in database: %s",
                    repo_url,
                )

            commits = payload.get("commits") or []
            if not commits:
                self._abort(logging.INFO, "No commits in GitLab push event")

            commit = commits[-1]
            timestamp_raw = commit.get("timestamp")
            if not timestamp_raw:
                self._abort(logging.WARNING, "Commit missing timestamp in GitLab push")

            timestamp = self._parse_iso_timestamp(timestamp_raw)

            return {
                "provider": "gitlab",
                "repository_id": repo_id,
                "hash": commit.get("id"),
                "author": commit.get("author", {}).get("name"),
                "author_email": commit.get("author", {}).get("email"),
                "timestamp": timestamp,
                "message": commit.get("message"),
                "datetime": datetime.fromtimestamp(timestamp).isoformat(),
            }
        except _WebhookAbort:
            return None

    def _handle_bitbucket(
        self, headers: Dict[str, str], payload: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Handle a Bitbucket webhook.

        Args:
            headers: HTTP headers
            payload: Webhook payload

        Returns:
            Dictionary with commit information or None if not a push event
        """
        try:
            event_type = headers.get("X-Event-Key")
            if event_type != "repo:push":
                self._abort(logging.INFO, "Ignoring Bitbucket event: %s", event_type)

            repository = payload.get("repository", {})
            repo_url = repository.get("links", {}).get("html", {}).get("href")
            if not repository.get("full_name") or not repo_url:
                self._abort(
                    logging.WARNING,
                    "Missing repository information in Bitbucket webhook",
                )

            if repo_url.endswith("/"):
                repo_url = repo_url[:-1]
            repo_url = f"{repo_url}.git"

            repo_id = self._get_repo_id_by_url(repo_url)
            if not repo_id:
                self._abort(
                    logging.WARNING,
                    "Repository not found in database: %s",
                    repo_url,
                )

            changes = payload.get("push", {}).get("changes") or []
            if not changes:
                self._abort(logging.INFO, "No changes in Bitbucket push event")

            commits = changes[0].get("commits") or []
            if not commits:
                self._abort(logging.INFO, "No commits in Bitbucket push event")

            commit = commits[-1]
            timestamp_raw = commit.get("date")
            if not timestamp_raw:
                self._abort(
                    logging.WARNING, "Commit missing timestamp in Bitbucket push"
                )

            timestamp = self._parse_iso_timestamp(timestamp_raw)
            author_email = commit.get("author", {}).get("raw")
            if author_email:
                author_email = author_email.split("<")[-1].split(">")[0]

            return {
                "provider": "bitbucket",
                "repository_id": repo_id,
                "hash": commit.get("hash"),
                "author": commit.get("author", {}).get("user", {}).get("display_name"),
                "author_email": author_email,
                "timestamp": timestamp,
                "message": commit.get("message"),
                "datetime": datetime.fromtimestamp(timestamp).isoformat(),
            }
        except _WebhookAbort:
            return None

    def _get_repo_id_by_url(self, url: str) -> Optional[int]:
        """Get repository ID by URL.

        Args:
            url: Repository URL

        Returns:
            Repository ID or None if not found
        """
        try:
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get the repository ID
            cursor.execute("SELECT id FROM repositories WHERE url = ?", (url,))
            result = cursor.fetchone()

            # Close the connection
            conn.close()

            return result[0] if result else None
        except (sqlite3.Error, OSError):
            logger.exception("Error getting repository ID by URL %s", url)
            return None

    @staticmethod
    def _parse_iso_timestamp(timestamp: str) -> int:
        """Convert an ISO-8601 formatted timestamp into epoch seconds."""

        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return int(time.mktime(parsed.timetuple()))

    @staticmethod
    def _abort(level: int, message: str, *args: Any) -> None:
        """Log at ``level`` and raise an internal abort sentinel."""

        logger.log(level, message, *args)
        raise _WebhookAbort()
