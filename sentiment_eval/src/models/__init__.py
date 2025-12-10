from typing import Any, Dict, List, Optional

from .base import SentimentModel
from .claude import ClaudeSentimentModel
from .deepseek import DeepseekSentimentModel
from .gemini import GeminiSentimentModel
from .gemma import GemmaSentimentModel
from .mock import MockSentimentModel
from .openai_model import OpenAISentimentModel


def get_active_models(
    cfg: Dict[str, Any] | None = None,
    prompt_config: Optional[Dict[str, Any]] = None,
) -> List[SentimentModel]:
    """
    Get list of active sentiment models based on configuration.

    Args:
        cfg: Settings configuration. If None, loads from default settings.
        prompt_config: Optional prompt configuration to apply to all models.
                      Should contain 'system' and 'user_template' keys.

    Returns:
        List of instantiated SentimentModel instances.
    """
    from ..config import load_settings

    if cfg is None:
        cfg = load_settings()

    active = cfg.get("active_models", [])
    instances: List[SentimentModel] = []

    if "gemma" in active:
        instances.append(GemmaSentimentModel(prompt_config=prompt_config))
    if "claude" in active:
        instances.append(ClaudeSentimentModel(prompt_config=prompt_config))
    if "gemini" in active:
        instances.append(GeminiSentimentModel(prompt_config=prompt_config))
    if "deepseek" in active:
        instances.append(DeepseekSentimentModel(prompt_config=prompt_config))
    if "mock" in active:
        mock_cfg = cfg.get("mock", {})
        instances.append(
            MockSentimentModel(seed=mock_cfg.get("seed"), prompt_config=prompt_config)
        )

    return instances


def get_active_models_with_strategy(
    strategy_name: str = "default_sentiment",
    cfg: Dict[str, Any] | None = None,
) -> List[SentimentModel]:
    """
    Get list of active sentiment models with a specific prompt strategy.

    Args:
        strategy_name: Name of the prompt strategy from prompts.yaml.
        cfg: Settings configuration. If None, loads from default settings.

    Returns:
        List of instantiated SentimentModel instances with the strategy applied.
    """
    from ..config import get_prompt_strategy

    prompt_config = get_prompt_strategy(strategy_name)
    return get_active_models(cfg=cfg, prompt_config=prompt_config)
