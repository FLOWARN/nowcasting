import yaml
from typing import Dict, Any
import logging
import os

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {str(e)}")
        raise

def save_config(config: Dict[str, Any], save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f)
    logger.info(f"Saved config to {save_path}")