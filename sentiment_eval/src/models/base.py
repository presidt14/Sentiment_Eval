from abc import ABC, abstractmethod
from typing import Dict, Any
import asyncio


class SentimentModel(ABC):
    name: str

    @abstractmethod
    def classify(self, text: str) -> Dict[str, Any]:
        """
        Synchronous classification.
        Return dict with keys: sentiment, confidence, reason
        """
        raise NotImplementedError

    @abstractmethod
    async def aclassify(self, text: str) -> Dict[str, Any]:
        """
        Asynchronous classification.
        Return dict with keys: sentiment, confidence, reason
        """
        raise NotImplementedError

    async def _run_sync_in_executor(self, text: str) -> Dict[str, Any]:
        """
        Helper to run the synchronous classify() in a thread pool executor.
        Useful for models that don't have native async support.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.classify, text)
