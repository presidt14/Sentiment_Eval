import random
from typing import Any, Dict, Optional

from .base import SentimentModel

_POSITIVE_KEYWORDS = ["love", "best", "great", "good", "amazing", "excellent", "fantastic"]
_NEGATIVE_KEYWORDS = ["hate", "awful", "bad", "worst", "terrible", "slow", "down", "poor"]


class MockSentimentModel(SentimentModel):
    def __init__(self, seed: Optional[int] = None) -> None:
        self.name = "mock"
        self._random = random.Random()
        if seed is not None:
            random.seed(seed)
            self._random.seed(seed)

    def classify(self, text: str) -> Dict[str, Any]:
        text_lower = text.lower()

        # Keyword heuristic matching
        for word in _POSITIVE_KEYWORDS:
            if word in text_lower:
                return {
                    "sentiment": "positive",
                    "confidence": round(self._random.uniform(0.7, 0.99), 2),
                    "reason": f"Heuristic match: found keyword '{word}'",
                }

        for word in _NEGATIVE_KEYWORDS:
            if word in text_lower:
                return {
                    "sentiment": "negative",
                    "confidence": round(self._random.uniform(0.7, 0.99), 2),
                    "reason": f"Heuristic match: found keyword '{word}'",
                }

        # Default to neutral
        return {
            "sentiment": "neutral",
            "confidence": round(self._random.uniform(0.5, 0.7), 2),
            "reason": "Heuristic: no strong sentiment keywords detected",
        }
