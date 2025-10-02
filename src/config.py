"""Configuration handling for Full Auto CI."""
import os
import logging
from typing import Dict, Any, Optional
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
        "tools": {
            "pylint": {
                "enabled": True,
                "config_file": None,  # Use default pylintrc
            },
            "coverage": {
                "enabled": True,
                "run_tests_cmd": ["pytest"],
            },
        },
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path or os.path.expanduser("~/.fullautoci/config.yml")
        self.config = self.DEFAULT_CONFIG.copy()
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file."""
        if not os.path.exists(self.config_path):
            logger.warning(f"Configuration file not found at {self.config_path}")
            logger.info("Using default configuration")
            return
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
                if user_config:
                    self._merge_config(user_config)
            logger.info(f"Loaded configuration from {self.config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
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
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            logger.info(f"Saved configuration to {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
