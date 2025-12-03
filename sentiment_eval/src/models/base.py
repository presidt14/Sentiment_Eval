from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import asyncio

from ..config import get_prompt_strategy


class SentimentModel(ABC):
    """
    Base class for sentiment analysis models.
    
    Supports externalized prompts via prompt_config parameter.
    """
    name: str
    _prompt_config: Optional[Dict[str, Any]] = None
    
    def set_prompt_config(self, prompt_config: Dict[str, Any]) -> None:
        """
        Set the prompt configuration for this model.
        
        Args:
            prompt_config: Dictionary with 'system' and 'user_template' keys.
        """
        self._prompt_config = prompt_config
    
    def set_prompt_strategy(self, strategy_name: str) -> None:
        """
        Set the prompt strategy by name (loads from prompts.yaml).
        
        Args:
            strategy_name: Name of the strategy in prompts.yaml.
        """
        self._prompt_config = get_prompt_strategy(strategy_name)
    
    def get_system_prompt(self) -> str:
        """Get the current system prompt."""
        if self._prompt_config and "system" in self._prompt_config:
            return self._prompt_config["system"]
        return self._get_default_system_prompt()
    
    def get_user_message(self, text: str) -> str:
        """
        Get the formatted user message for the given text.
        
        Args:
            text: The text to analyze.
            
        Returns:
            Formatted user message with text substituted.
        """
        if self._prompt_config and "user_template" in self._prompt_config:
            return self._prompt_config["user_template"].format(text=text)
        return text
    
    def _get_default_system_prompt(self) -> str:
        """
        Return the default system prompt for this model.
        Subclasses can override this to provide model-specific defaults.
        """
        return """You are a sentiment classifier for social-media posts.

Classify sentiment as:
- positive
- neutral
- negative

Return ONLY strict JSON with keys:
{
  "sentiment": "positive|neutral|negative",
  "confidence": 0-1 float,
  "reason": "Short explanation"
}"""

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
