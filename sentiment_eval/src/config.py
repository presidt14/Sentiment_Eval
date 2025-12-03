import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]

load_dotenv(BASE_DIR / ".env", override=False)


def load_settings(path: Path | str | None = None) -> Dict[str, Any]:
    """Load settings from YAML, defaulting to config/settings.example.yaml."""

    if path is None:
        resolved_path = BASE_DIR / "config" / "settings.example.yaml"
    else:
        resolved_path = Path(path)
        if not resolved_path.is_absolute():
            resolved_path = BASE_DIR / resolved_path

    with open(resolved_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

def get_env(key: str, default: str | None = None) -> str | None:
    return os.getenv(key, default)
