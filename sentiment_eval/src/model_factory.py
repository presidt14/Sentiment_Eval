"""
Model Factory

Factory pattern for instantiating sentiment models by provider name.
Ensures consistent initialization and guardrail application across all providers.
"""

from typing import Any, Dict, Optional

from .config import get_env, get_prompt_strategy
from .models.base import SentimentModel


def get_model(
    provider_name: str,
    prompt_strategy: str = "compliance_risk_assessor",
    prompt_config: Optional[Dict[str, Any]] = None,
) -> SentimentModel:
    """
    Factory function to get a sentiment model by provider name.
    
    All models returned inherit the Zone of Control guardrail from base class.
    
    Args:
        provider_name: One of "gemma", "claude", "openai", "deepseek", "gemini", "mock"
        prompt_strategy: Name of prompt strategy from prompts.yaml (default: compliance_risk_assessor)
        prompt_config: Optional direct prompt config dict (overrides prompt_strategy)
        
    Returns:
        Instantiated SentimentModel subclass
        
    Raises:
        ValueError: If provider_name is not recognized
    """
    # Get prompt config
    if prompt_config is None:
        prompt_config = get_prompt_strategy(prompt_strategy)
    
    provider = provider_name.lower().strip()
    
    if provider == "gemma":
        from .models.gemma import GemmaSentimentModel
        return GemmaSentimentModel(prompt_config=prompt_config)
    
    elif provider == "claude":
        from .models.claude import ClaudeSentimentModel
        return ClaudeSentimentModel(prompt_config=prompt_config)
    
    elif provider == "openai" or provider == "gpt4" or provider == "gpt-4":
        from .models.openai_model import OpenAISentimentModel
        return OpenAISentimentModel(prompt_config=prompt_config)
    
    elif provider == "deepseek":
        from .models.deepseek import DeepseekSentimentModel
        return DeepseekSentimentModel(prompt_config=prompt_config)
    
    elif provider == "gemini":
        from .models.gemini import GeminiSentimentModel
        return GeminiSentimentModel(prompt_config=prompt_config)
    
    elif provider == "mock":
        from .models.mock import MockSentimentModel
        return MockSentimentModel(seed=42, prompt_config=prompt_config)
    
    else:
        available = ["gemma", "claude", "openai", "deepseek", "gemini", "mock"]
        raise ValueError(
            f"Unknown provider: '{provider_name}'. "
            f"Available providers: {', '.join(available)}"
        )


def list_providers() -> list:
    """Return list of available provider names."""
    return ["gemma", "claude", "openai", "deepseek", "gemini", "mock"]


def get_provider_info() -> Dict[str, Dict[str, str]]:
    """Return information about each provider."""
    return {
        "gemma": {
            "name": "Gemma 3 27B",
            "api": "Nebius TokenFactory",
            "env_var": "NEBIUS_API_KEY",
        },
        "claude": {
            "name": "Claude 3.5 Sonnet",
            "api": "Anthropic",
            "env_var": "ANTHROPIC_API_KEY",
        },
        "openai": {
            "name": "GPT-4o",
            "api": "OpenAI",
            "env_var": "OPENAI_API_KEY",
        },
        "deepseek": {
            "name": "DeepSeek Chat",
            "api": "DeepSeek",
            "env_var": "DEEPSEEK_API_KEY",
        },
        "gemini": {
            "name": "Gemini Pro",
            "api": "Google AI",
            "env_var": "GOOGLE_API_KEY",
        },
        "mock": {
            "name": "Mock Model",
            "api": "None (deterministic)",
            "env_var": "None",
        },
    }
