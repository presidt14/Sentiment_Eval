import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]

load_dotenv(BASE_DIR / ".env", override=False)

# Cache for prompts configuration
_prompts_cache: Optional[Dict[str, Any]] = None


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


def load_prompts(
    path: Path | str | None = None, force_reload: bool = False
) -> Dict[str, Any]:
    """
    Load prompt strategies from YAML, defaulting to config/prompts.yaml.

    Args:
        path: Optional path to prompts YAML file.
        force_reload: If True, bypass cache and reload from disk.

    Returns:
        Dictionary of prompt strategies.
    """
    global _prompts_cache

    if _prompts_cache is not None and not force_reload:
        return _prompts_cache

    if path is None:
        resolved_path = BASE_DIR / "config" / "prompts.yaml"
    else:
        resolved_path = Path(path)
        if not resolved_path.is_absolute():
            resolved_path = BASE_DIR / resolved_path

    if not resolved_path.exists():
        # Return default prompt if file doesn't exist
        _prompts_cache = {
            "default_sentiment": {
                "name": "Default Sentiment Classifier",
                "description": "Standard sentiment analysis",
                "system": "You are a sentiment classifier. Classify as positive, neutral, or negative. Return JSON with sentiment, confidence, and reason.",
                "user_template": "{text}",
            }
        }
        return _prompts_cache

    with open(resolved_path, "r", encoding="utf-8") as f:
        _prompts_cache = yaml.safe_load(f)

    return _prompts_cache


def get_prompt_strategy(strategy_name: str = "default_sentiment") -> Dict[str, Any]:
    """
    Get a specific prompt strategy by name.

    Args:
        strategy_name: Name of the strategy (key in prompts.yaml).

    Returns:
        Dictionary with 'system' and 'user_template' keys.

    Raises:
        KeyError: If strategy_name is not found.
    """
    prompts = load_prompts()
    if strategy_name not in prompts:
        available = list(prompts.keys())
        raise KeyError(
            f"Prompt strategy '{strategy_name}' not found. Available: {available}"
        )
    return prompts[strategy_name]


def list_prompt_strategies() -> Dict[str, str]:
    """
    List all available prompt strategies with their descriptions.

    Returns:
        Dictionary mapping strategy names to their descriptions.
    """
    prompts = load_prompts()
    return {key: cfg.get("name", key) for key, cfg in prompts.items()}


def get_env(key: str, default: str | None = None) -> str | None:
    return os.getenv(key, default)
