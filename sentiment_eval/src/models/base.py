import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ..config import get_prompt_strategy


class SentimentModel(ABC):
    """
    Base class for sentiment analysis models.

    Supports externalized prompts via prompt_config parameter.
    
    CRITICAL: All models inherit the Zone of Control guardrail via
    _apply_zone_of_control_guardrail() which enforces:
    - If brand_relevance=False → sentiment must be "neutral"
    
    This ensures consistent safety behavior across ALL providers.
    """

    name: str
    _prompt_config: Optional[Dict[str, Any]] = None

    def _apply_zone_of_control_guardrail(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enforce Zone of Control Contract.
        
        If brand_relevance is False, the post is not about the brand's controlled
        aspects (e.g., sport outcomes, user's own gambling choices). Therefore,
        it cannot be a negative sentiment ABOUT the brand.
        
        This guardrail ensures consistent behavior across all consumers of this
        model (API, CLI, notebooks, etc.) and ALL model providers.
        
        Args:
            result: Raw model output dict with sentiment, brand_relevance, etc.
            
        Returns:
            Result dict with guardrail applied.
        """
        brand_relevance = result.get("brand_relevance")
        
        # Parse brand_relevance if it's a string
        if isinstance(brand_relevance, str):
            brand_relevance = brand_relevance.lower() in ("true", "1", "yes")
        
        if brand_relevance is False:
            original_sentiment = result.get("sentiment")
            if original_sentiment != "neutral":
                result["sentiment"] = "neutral"
                result["negative_type"] = None
                # Append guardrail note to reason
                original_reason = result.get("reason", "")
                result["reason"] = f"{original_reason} [Guardrail: not brand-relevant]"
        
        return result

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

    def get_user_message(self, text: str, **kwargs) -> str:
        """
        Get the formatted user message for the given text.

        Args:
            text: The text to analyze.
            **kwargs: Additional template variables (e.g., brand_name).

        Returns:
            Formatted user message with text and other variables substituted.
        """
        if self._prompt_config and "user_template" in self._prompt_config:
            template = self._prompt_config["user_template"]
            # Build format dict with text and any additional kwargs
            format_dict = {"text": text, **kwargs}
            try:
                return template.format(**format_dict)
            except KeyError as e:
                # If a template variable is missing, fall back to just text
                return template.format(text=text, brand_name=kwargs.get("brand_name", "Unknown"))
        return text
    
    def get_system_prompt_formatted(self, **kwargs) -> str:
        """
        Get the system prompt with optional variable substitution.
        
        Args:
            **kwargs: Variables to substitute in the system prompt (e.g., brand_name).
            
        Returns:
            Formatted system prompt.
        """
        system = self.get_system_prompt()
        if kwargs:
            try:
                return system.format(**kwargs)
            except KeyError:
                return system
        return system

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
