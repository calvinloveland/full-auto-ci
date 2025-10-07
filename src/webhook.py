"""Webhook handlers for Full Auto CI."""
import os
import logging
import json
import hmac
import hashlib
from typing import Dict, Any, Optional, Callable
import sqlite3

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
        self.handlers = {
            "github": self._handle_github,
            "gitlab": self._handle_gitlab,
            "bitbucket": self._handle_bitbucket,
        }

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
        logger.info(f"Received webhook from {provider}")

        # Verify signature if secret is set
        if self.secret and not self._verify_signature(provider, headers, payload):
            logger.warning(f"Invalid signature for {provider} webhook")
            return None

        # Get the handler for the provider
        handler = self.handlers.get(provider.lower())
        if not handler:
            logger.warning(f"No handler for provider {provider}")
            return None

        # Handle the webhook
        try:
            return handler(headers, payload)
        except Exception as e:
            logger.error(f"Error handling {provider} webhook: {e}")
            return None

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

        if provider.lower() == "github":
            # GitHub uses X-Hub-Signature-256 header
            signature_header = headers.get("X-Hub-Signature-256")
            if not signature_header:
                return False

            # Calculate the signature
            payload_bytes = json.dumps(payload).encode("utf-8")
            hmac_obj = hmac.new(
                self.secret.encode("utf-8"), payload_bytes, hashlib.sha256
            )
            expected_signature = f"sha256={hmac_obj.hexdigest()}"

            return hmac.compare_digest(signature_header, expected_signature)
        elif provider.lower() == "gitlab":
            # GitLab uses X-Gitlab-Token header
            token_header = headers.get("X-Gitlab-Token")
            if not token_header:
                return False

            return hmac.compare_digest(token_header, self.secret)
        elif provider.lower() == "bitbucket":
            # Bitbucket doesn't have a standard signature header
            # Would need to implement custom verification
            return True
        else:
            return False

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
        # Check if it's a push event
        event_type = headers.get("X-GitHub-Event")
        if event_type != "push":
            logger.info(f"Ignoring GitHub event: {event_type}")
            return None

        # Get repository information
        repo_name = payload.get("repository", {}).get("full_name")
        repo_url = payload.get("repository", {}).get("clone_url")
        if not repo_name or not repo_url:
            logger.warning("Missing repository information in GitHub webhook")
            return None

        # Find the repository in the database
        repo_id = self._get_repo_id_by_url(repo_url)
        if not repo_id:
            logger.warning(f"Repository not found in database: {repo_url}")
            return None

        # Get commit information
        commits = payload.get("commits", [])
        if not commits:
            logger.info("No commits in GitHub push event")
            return None

        # Get the latest commit
        commit = commits[-1]
        commit_hash = commit.get("id")
        author = commit.get("author", {}).get("name")
        author_email = commit.get("author", {}).get("email")
        message = commit.get("message")
        timestamp = commit.get("timestamp")  # ISO 8601 format

        # Convert timestamp to Unix timestamp
        from datetime import datetime
        import time

        timestamp = int(
            time.mktime(
                datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timetuple()
            )
        )

        return {
            "provider": "github",
            "repository_id": repo_id,
            "hash": commit_hash,
            "author": author,
            "author_email": author_email,
            "timestamp": timestamp,
            "message": message,
            "datetime": datetime.fromtimestamp(timestamp).isoformat(),
        }

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
        # Check if it's a push event
        event_type = headers.get("X-Gitlab-Event")
        if event_type != "Push Hook":
            logger.info(f"Ignoring GitLab event: {event_type}")
            return None

        # Get repository information
        repo_name = payload.get("project", {}).get("path_with_namespace")
        repo_url = payload.get("project", {}).get("git_http_url")
        if not repo_name or not repo_url:
            logger.warning("Missing repository information in GitLab webhook")
            return None

        # Find the repository in the database
        repo_id = self._get_repo_id_by_url(repo_url)
        if not repo_id:
            logger.warning(f"Repository not found in database: {repo_url}")
            return None

        # Get commit information
        commits = payload.get("commits", [])
        if not commits:
            logger.info("No commits in GitLab push event")
            return None

        # Get the latest commit
        commit = commits[-1]
        commit_hash = commit.get("id")
        author = commit.get("author", {}).get("name")
        author_email = commit.get("author", {}).get("email")
        message = commit.get("message")
        timestamp = commit.get("timestamp")  # ISO 8601 format

        # Convert timestamp to Unix timestamp
        from datetime import datetime
        import time

        timestamp = int(
            time.mktime(
                datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timetuple()
            )
        )

        return {
            "provider": "gitlab",
            "repository_id": repo_id,
            "hash": commit_hash,
            "author": author,
            "author_email": author_email,
            "timestamp": timestamp,
            "message": message,
            "datetime": datetime.fromtimestamp(timestamp).isoformat(),
        }

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
        # Check if it's a push event
        event_type = headers.get("X-Event-Key")
        if event_type != "repo:push":
            logger.info(f"Ignoring Bitbucket event: {event_type}")
            return None

        # Get repository information
        repo_name = payload.get("repository", {}).get("full_name")
        repo_url = (
            payload.get("repository", {}).get("links", {}).get("html", {}).get("href")
        )
        if not repo_name or not repo_url:
            logger.warning("Missing repository information in Bitbucket webhook")
            return None

        # Convert HTTP URL to clone URL
        if repo_url.endswith("/"):
            repo_url = repo_url[:-1]
        repo_url = f"{repo_url}.git"

        # Find the repository in the database
        repo_id = self._get_repo_id_by_url(repo_url)
        if not repo_id:
            logger.warning(f"Repository not found in database: {repo_url}")
            return None

        # Get commit information
        changes = payload.get("push", {}).get("changes", [])
        if not changes:
            logger.info("No changes in Bitbucket push event")
            return None

        # Get the latest commit from the first change
        commits = changes[0].get("commits", [])
        if not commits:
            logger.info("No commits in Bitbucket push event")
            return None

        commit = commits[-1]
        commit_hash = commit.get("hash")
        author = commit.get("author", {}).get("user", {}).get("display_name")
        author_email = commit.get("author", {}).get("raw")  # Contains "Name <email>"
        if author_email:
            author_email = author_email.split("<")[-1].split(">")[0]
        message = commit.get("message")
        timestamp = commit.get("date")  # ISO 8601 format

        # Convert timestamp to Unix timestamp
        from datetime import datetime
        import time

        timestamp = int(
            time.mktime(
                datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timetuple()
            )
        )

        return {
            "provider": "bitbucket",
            "repository_id": repo_id,
            "hash": commit_hash,
            "author": author,
            "author_email": author_email,
            "timestamp": timestamp,
            "message": message,
            "datetime": datetime.fromtimestamp(timestamp).isoformat(),
        }

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
        except Exception as e:
            logger.error(f"Error getting repository ID by URL: {e}")
            return None
