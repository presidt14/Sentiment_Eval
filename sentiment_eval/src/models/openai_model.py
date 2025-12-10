"""
OpenAI GPT-4 Sentiment Model

Uses the OpenAI API for sentiment classification with Zone of Control guardrail.
"""

import json
from typing import Any, Dict, Optional

from openai import OpenAI

from ..config import get_env, load_settings
from .base import SentimentModel

cfg = load_settings()


class OpenAISentimentModel(SentimentModel):
    """
    OpenAI GPT-4 sentiment model.
    
    Uses OpenAI chat completions API.
    Requires OPENAI_API_KEY environment variable.
    """
    
    def __init__(self, prompt_config: Optional[Dict[str, Any]] = None):
        self.name = "openai"
        
        # Get config with fallbacks
        openai_cfg = cfg.get("openai", {})
        self.model = openai_cfg.get("model", "gpt-4o")
        self.timeout = openai_cfg.get("timeout_seconds", 30)
        self.temperature = openai_cfg.get("temperature", 0.1)
        self.max_tokens = openai_cfg.get("max_tokens", 512)
        
        # API key from env
        self.api_key = get_env("OPENAI_API_KEY", "")
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            timeout=self.timeout,
        )
        
        if prompt_config:
            self.set_prompt_config(prompt_config)

    def classify(self, text: str) -> Dict[str, Any]:
        """
        Classify text sentiment using OpenAI API.
        
        Returns:
            Dict with sentiment, confidence, reason, brand_relevance, negative_type
        """
        if not self.api_key:
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "reason": "Error: OPENAI_API_KEY not set. Add to .env file.",
                "brand_relevance": None,
                "negative_type": None,
            }
        
        system_prompt = self.get_system_prompt()
        user_message = self.get_user_message(text)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
            
            content = response.choices[0].message.content
            result = json.loads(content)
            
            # Apply Zone of Control guardrail (inherited from base)
            return self._apply_zone_of_control_guardrail(result)
            
        except Exception as e:
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "reason": f"API Error (OpenAI): {e.__class__.__name__}: {str(e)[:100]}",
                "brand_relevance": None,
                "negative_type": None,
            }

    async def aclassify(self, text: str) -> Dict[str, Any]:
        """
        Async classification using thread pool executor for sync HTTP call.
        """
        return await self._run_sync_in_executor(text)
