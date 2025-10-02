"""REST API for Full Auto CI."""
import os
import logging
import secrets
from typing import Dict, Any, Optional, Tuple, Union
import hashlib
import time
import sqlite3
import json

# Try to import Flask
try:
    from flask import Flask, request, jsonify, Response
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    
from .service import CIService
from .webhook import WebhookHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class API:
    """REST API for Full Auto CI.
    
    This class provides a REST API for the Full Auto CI service.
    """
    
    def __init__(self, config_path: Optional[str] = None, db_path: Optional[str] = None, webhook_secret: Optional[str] = None):
        """Initialize the API.
        
        Args:
            config_path: Path to the configuration file
            db_path: Path to the SQLite database
            webhook_secret: Secret for webhook signature verification
        """
        self.config_path = config_path or os.path.expanduser("~/.fullautoci/config.yml")
        self.db_path = db_path or os.path.expanduser("~/.fullautoci/database.sqlite")
        self.webhook_secret = webhook_secret
        self.service = CIService(config_path=self.config_path, db_path=self.db_path)
        self.webhook_handler = WebhookHandler(db_path=self.db_path, secret=webhook_secret)
        
        if FLASK_AVAILABLE:
            self.app = Flask(__name__)
            self._setup_routes()
        else:
            logger.warning("Flask is not installed. API server will not be available.")
            self.app = None
    
    def _setup_routes(self):
        """Set up API routes."""
        if not FLASK_AVAILABLE or self.app is None:
            return
            
        # API Key endpoints
        @self.app.route('/api/key/generate', methods=['POST'])
        def generate_api_key_route():
            return self.generate_api_key_endpoint()
            
        @self.app.route('/api/key/verify', methods=['POST'])
        def verify_api_key_route():
            return self.verify_api_key_endpoint()
        
        # Repository endpoints
        @self.app.route('/api/repositories', methods=['GET'])
        def list_repositories_route():
            return self.list_repositories()
            
        @self.app.route('/api/repository/<int:repo_id>', methods=['GET'])
        def get_repository_route(repo_id):
            return self.get_repository(repo_id)
            
        @self.app.route('/api/repository/add', methods=['POST'])
        def add_repository_route():
            return self.add_repository()
            
        @self.app.route('/api/repository/remove/<int:repo_id>', methods=['DELETE'])
        def remove_repository_route(repo_id):
            return self.remove_repository(repo_id)
        
        # Test results endpoints
        @self.app.route('/api/tests/<int:repo_id>', methods=['GET'])
        def get_test_results_route(repo_id):
            return self.get_test_results(repo_id)
            
        @self.app.route('/api/tests/latest', methods=['GET'])
        def get_latest_test_results_route():
            return self.get_latest_test_results()
        
        # Webhook endpoints
        @self.app.route('/webhook/github', methods=['POST'])
        def github_webhook_route():
            return self.github_webhook()
            
        @self.app.route('/webhook/gitlab', methods=['POST'])
        def gitlab_webhook_route():
            return self.gitlab_webhook()
            
        @self.app.route('/webhook/bitbucket', methods=['POST'])
        def bitbucket_webhook_route():
            return self.bitbucket_webhook()
        
        # Health check endpoint
        @self.app.route('/health', methods=['GET'])
        def health_check_route():
            return self.health_check()
    
    def health_check(self):
        """Health check endpoint."""
        if not FLASK_AVAILABLE:
            return {"status": "API server available but Flask not installed"}, 200
            
        return jsonify({"status": "healthy", "version": "0.1.0"})
    
    def github_webhook(self):
        """Handle GitHub webhook."""
        if not FLASK_AVAILABLE:
            return {"error": "Flask not installed"}, 501
            
        # Get the headers and payload
        headers = dict(request.headers)
        payload = request.json
        
        # Handle the webhook
        result = self.webhook_handler.handle("github", headers, payload)
        
        if result:
            # Trigger a test run for the repository
            self.service.add_test_task(result["repository_id"], result["hash"])
            return jsonify({"status": "success", "message": "Webhook processed"})
        else:
            return jsonify({"status": "error", "message": "Invalid webhook payload"}), 400
    
    def gitlab_webhook(self):
        """Handle GitLab webhook."""
        if not FLASK_AVAILABLE:
            return {"error": "Flask not installed"}, 501
            
        # Get the headers and payload
        headers = dict(request.headers)
        payload = request.json
        
        # Handle the webhook
        result = self.webhook_handler.handle("gitlab", headers, payload)
        
        if result:
            # Trigger a test run for the repository
            self.service.add_test_task(result["repository_id"], result["hash"])
            return jsonify({"status": "success", "message": "Webhook processed"})
        else:
            return jsonify({"status": "error", "message": "Invalid webhook payload"}), 400
    
    def bitbucket_webhook(self):
        """Handle Bitbucket webhook."""
        if not FLASK_AVAILABLE:
            return {"error": "Flask not installed"}, 501
            
        # Get the headers and payload
        headers = dict(request.headers)
        payload = request.json
        
        # Handle the webhook
        result = self.webhook_handler.handle("bitbucket", headers, payload)
        
        if result:
            # Trigger a test run for the repository
            self.service.add_test_task(result["repository_id"], result["hash"])
            return jsonify({"status": "success", "message": "Webhook processed"})
        else:
            return jsonify({"status": "error", "message": "Invalid webhook payload"}), 400
            
    def generate_api_key_endpoint(self):
        """Generate API key endpoint."""
        if not FLASK_AVAILABLE:
            return {"error": "Flask not installed"}, 501
            
        api_key = self.generate_api_key()
        
        # Store API key in database
        hashed_key = hashlib.sha256(api_key.encode()).hexdigest()
        timestamp = int(time.time())
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO api_keys (key_hash, created_at) VALUES (?, ?)",
                (hashed_key, timestamp)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error storing API key: {e}")
            return jsonify({"error": "Error storing API key"}), 500
            
        return jsonify({"api_key": api_key})
    
    def verify_api_key_endpoint(self):
        """Verify API key endpoint."""
        if not FLASK_AVAILABLE:
            return {"error": "Flask not installed"}, 501
            
        data = request.json
        if not data or "api_key" not in data:
            return jsonify({"error": "Missing API key"}), 400
            
        api_key = data["api_key"]
        hashed_key = hashlib.sha256(api_key.encode()).hexdigest()
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM api_keys WHERE key_hash = ?", (hashed_key,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return jsonify({"valid": True})
            else:
                return jsonify({"valid": False}), 401
        except Exception as e:
            logger.error(f"Error verifying API key: {e}")
            return jsonify({"error": "Error verifying API key"}), 500
    
    def list_repositories(self):
        """List repositories endpoint."""
        if not FLASK_AVAILABLE:
            return {"error": "Flask not installed"}, 501
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT id, url, name, status, last_check FROM repositories")
            repositories = cursor.fetchall()
            conn.close()
            
            result = [
                {
                    "id": repo[0],
                    "url": repo[1],
                    "name": repo[2],
                    "status": repo[3],
                    "last_check": repo[4]
                }
                for repo in repositories
            ]
            
            return jsonify({"repositories": result})
        except Exception as e:
            logger.error(f"Error listing repositories: {e}")
            return jsonify({"error": "Error listing repositories"}), 500
    
    def get_repository(self, repo_id):
        """Get repository endpoint."""
        if not FLASK_AVAILABLE:
            return {"error": "Flask not installed"}, 501
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT id, url, name, status, last_check FROM repositories WHERE id = ?", (repo_id,))
            repository = cursor.fetchone()
            conn.close()
            
            if repository:
                result = {
                    "id": repository[0],
                    "url": repository[1],
                    "name": repository[2],
                    "status": repository[3],
                    "last_check": repository[4]
                }
                return jsonify(result)
            else:
                return jsonify({"error": "Repository not found"}), 404
        except Exception as e:
            logger.error(f"Error getting repository: {e}")
            return jsonify({"error": "Error getting repository"}), 500
    
    def add_repository(self):
        """Add repository endpoint."""
        if not FLASK_AVAILABLE:
            return {"error": "Flask not installed"}, 501
            
        data = request.json
        if not data or "url" not in data:
            return jsonify({"error": "Missing URL"}), 400
            
        url = data["url"]
        name = data.get("name", url.split("/")[-1].split(".")[0])
        
        try:
            result = self.service.add_repository(url, name)
            if result:
                return jsonify({"status": "success", "repository_id": result})
            else:
                return jsonify({"error": "Error adding repository"}), 500
        except Exception as e:
            logger.error(f"Error adding repository: {e}")
            return jsonify({"error": str(e)}), 500
    
    def remove_repository(self, repo_id):
        """Remove repository endpoint."""
        if not FLASK_AVAILABLE:
            return {"error": "Flask not installed"}, 501
            
        try:
            result = self.service.remove_repository(repo_id)
            if result:
                return jsonify({"status": "success"})
            else:
                return jsonify({"error": "Repository not found"}), 404
        except Exception as e:
            logger.error(f"Error removing repository: {e}")
            return jsonify({"error": str(e)}), 500
    
    def get_test_results(self, repo_id):
        """Get test results endpoint."""
        if not FLASK_AVAILABLE:
            return {"error": "Flask not installed"}, 501
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, repository_id, commit_hash, status, created_at, completed_at FROM test_runs "
                "WHERE repository_id = ? ORDER BY created_at DESC",
                (repo_id,)
            )
            test_runs = cursor.fetchall()
            
            result = [
                {
                    "id": run[0],
                    "repository_id": run[1],
                    "commit_hash": run[2],
                    "status": run[3],
                    "created_at": run[4],
                    "completed_at": run[5]
                }
                for run in test_runs
            ]
            
            conn.close()
            return jsonify({"test_runs": result})
        except Exception as e:
            logger.error(f"Error getting test results: {e}")
            return jsonify({"error": "Error getting test results"}), 500
    
    def get_latest_test_results(self):
        """Get latest test results endpoint."""
        if not FLASK_AVAILABLE:
            return {"error": "Flask not installed"}, 501
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT tr.id, tr.repository_id, r.name, tr.commit_hash, tr.status, tr.created_at, tr.completed_at "
                "FROM test_runs tr JOIN repositories r ON tr.repository_id = r.id "
                "WHERE tr.id IN (SELECT MAX(id) FROM test_runs GROUP BY repository_id)"
            )
            test_runs = cursor.fetchall()
            
            result = [
                {
                    "id": run[0],
                    "repository_id": run[1],
                    "repository_name": run[2],
                    "commit_hash": run[3],
                    "status": run[4],
                    "created_at": run[5],
                    "completed_at": run[6]
                }
                for run in test_runs
            ]
            
            conn.close()
            return jsonify({"test_runs": result})
        except Exception as e:
            logger.error(f"Error getting latest test results: {e}")
            return jsonify({"error": "Error getting latest test results"}), 500
    
    def generate_api_key(self) -> str:
        """Generate a new API key.
        
        Returns:
            New API key
        """
        return secrets.token_hex(32)
    
    def run(self, host: str = "127.0.0.1", port: int = 5000, debug: bool = False):
        """Run the API server.
        
        Args:
            host: Host to bind to
            port: Port to bind to
            debug: Whether to run in debug mode
        """
        # self.app.run(host=host, port=port, debug=debug)  # Uncomment when Flask is installed
        logger.info(f"API server would run on {host}:{port} (debug={debug})")
        logger.info("Flask is not installed yet, so this is just a placeholder")


def create_app(config_path: Optional[str] = None, db_path: Optional[str] = None, webhook_secret: Optional[str] = None):
    """Create a Flask app for the API.
    
    Args:
        config_path: Path to the configuration file
        db_path: Path to the SQLite database
        webhook_secret: Secret for webhook signature verification
        
    Returns:
        Flask app or None if Flask is not installed
    """
    api = API(config_path=config_path, db_path=db_path, webhook_secret=webhook_secret)
    
    if FLASK_AVAILABLE and api.app is not None:
        return api.app
    else:
        logger.warning("Flask is not installed, API server will not be available")
        return None
