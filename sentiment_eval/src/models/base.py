from abc import ABC, abstractmethod
from typing import Dict, Any

class SentimentModel(ABC):
    name: str

    @abstractmethod
    def classify(self, text: str) -> Dict[str, Any]:
        """
        Return dict with keys: sentiment, confidence, reason
        """
        raise NotImplementedError
