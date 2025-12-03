from typing import Any, Dict, List

from .gemma import GemmaSentimentModel
from .claude import ClaudeSentimentModel
from .gemini import GeminiSentimentModel
from .deepseek import DeepseekSentimentModel
from .mock import MockSentimentModel
from .base import SentimentModel


def get_active_models(cfg: Dict[str, Any] | None = None) -> List[SentimentModel]:
    from ..config import load_settings

    if cfg is None:
        cfg = load_settings()

    active = cfg.get("active_models", [])
    instances: List[SentimentModel] = []

    if "gemma" in active:
        instances.append(GemmaSentimentModel())
    if "claude" in active:
        instances.append(ClaudeSentimentModel())
    if "gemini" in active:
        instances.append(GeminiSentimentModel())
    if "deepseek" in active:
        instances.append(DeepseekSentimentModel())
    if "mock" in active:
        mock_cfg = cfg.get("mock", {})
        instances.append(MockSentimentModel(seed=mock_cfg.get("seed")))

    return instances
