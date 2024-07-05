# textpredict/config_management.py

import json
import logging
import os

logger = logging.getLogger(__name__)


class ConfigManager:
    def __init__(self, config_file: str):
        """
        Initialize the ConfigManager with the specified configuration file.

        Args:
            config_file (str): Path to the configuration file.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
        """
        self.config_file = config_file
        if not os.path.exists(config_file):
            logger.error(f"Configuration file {config_file} not found")
            raise FileNotFoundError(f"Configuration file {config_file} not found")
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """
        Load the configuration from the file.

        Returns:
            dict: The configuration dictionary.
        """
        try:
            with open(self.config_file, "r") as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {self.config_file}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    def get(self, key: str, default=None):
        """
        Get a configuration value by key.

        Args:
            key (str): The configuration key.
            default: The default value to return if the key is not found.

        Returns:
            The configuration value.
        """
        return self.config.get(key, default)

    def set(self, key: str, value):
        """
        Set a configuration value by key.

        Args:
            key (str): The configuration key.
            value: The value to set.
        """
        self.config[key] = value
        self._save_config()

    def _save_config(self):
        """
        Save the current configuration to the file.
        """
        try:
            with open(self.config_file, "w") as f:
                json.dump(self.config, f, indent=4)
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise
