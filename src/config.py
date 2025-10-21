"""Configuration handling for Full Auto CI."""

import copy
import logging
import os
from typing import Any, Dict, Optional

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Config:
    """Configuration handler for Full Auto CI."""

    DEFAULT_CONFIG = {
        "service": {
            "poll_interval": 60,  # seconds
            "log_level": "INFO",
            "max_workers": 4,
        },
        "database": {
            "path": "~/.fullautoci/database.sqlite",
        },
        "api": {
            "host": "127.0.0.1",
            "port": 5000,
            "debug": False,
        },
        "dashboard": {
            "host": "127.0.0.1",
            "port": 8000,
            "debug": False,
            "secret_key": None,
            "auto_open": True,
            "auto_start": True,
        },
        "tools": {
            "pylint": {
                "enabled": True,
                "config_file": None,  # Use default pylintrc
            },
            "coverage": {
                "enabled": True,
                "run_tests_cmd": ["pytest"],
                "timeout_seconds": 300,
                "xml_timeout_seconds": 120,
            },
            "lizard": {
                "enabled": True,
                "max_ccn": 10,
            },
        },
        "dogfood": {
            "enabled": False,
            "name": "Full Auto CI",
            "url": "https://github.com/calvinloveland/full-auto-ci.git",
            "branch": "main",
            "queue_on_start": True,
        },
    }

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.

        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path or os.path.expanduser("~/.fullautoci/config.yml")
        self.config = copy.deepcopy(self.DEFAULT_CONFIG)
        self._load_config()

    def _load_config(self):
        """Load configuration from file."""
        if not os.path.exists(self.config_path):
            logger.warning("Configuration file not found at %s", self.config_path)
            logger.info("Using default configuration")
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as config_file:
                user_config = yaml.safe_load(config_file)
                if user_config:
                    self._merge_config(user_config)
            logger.info("Loaded configuration from %s", self.config_path)
        except (OSError, yaml.YAMLError) as error:
            logger.error(
                "Error loading configuration from %s: %s",
                self.config_path,
                error,
            )
            logger.info("Using default configuration")

    def _merge_config(self, user_config: Dict[str, Any]):
        """Merge user configuration with default configuration.

        Args:
            user_config: User configuration
        """
        # Simple recursive merge
        for section, values in user_config.items():
            if section in self.config and isinstance(values, dict):
                self.config[section].update(values)
            else:
                self.config[section] = values

    def get(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        """Get configuration value.

        Args:
            section: Configuration section
            key: Configuration key (optional, if None returns the entire section)
            default: Default value if the key is not found

        Returns:
            Configuration value or default
        """
        if section not in self.config:
            return default

        if key is None:
            return self.config[section]

        return self.config[section].get(key, default)

    def set(self, section: str, key: str, value: Any):
        """Set configuration value.

        Args:
            section: Configuration section
            key: Configuration key
            value: Configuration value
        """
        if section not in self.config:
            self.config[section] = {}

        self.config[section][key] = value

    def save(self):
        """Save configuration to file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

            with open(self.config_path, "w", encoding="utf-8") as config_file:
                yaml.dump(self.config, config_file, default_flow_style=False)
            logger.info("Saved configuration to %s", self.config_path)
            return True
        except OSError as error:
            logger.error(
                "Error saving configuration to %s: %s", self.config_path, error
            )
            return False
