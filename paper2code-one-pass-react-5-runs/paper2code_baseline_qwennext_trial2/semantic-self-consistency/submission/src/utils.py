import os
import json
from typing import Any

def load_json(filepath: str) -> Any:
    """Load JSON file"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data: Any, filepath: str) -> None:
    """Save JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def ensure_dir(filepath: str) -> None:
    """Ensure directory exists"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

def get_env(key: str, default: str) -> str:
    """Get environment variable"""
    return os.getenv(key, default)