import json
import re
from typing import Any, Dict, Optional

from openai import OpenAI

from ..config import get_env, load_settings
from .base import SentimentModel

cfg = load_settings()

# Nebius TokenFactory API configuration
NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
NEBIUS_MODEL = "google/gemma-3-27b-it"


class GemmaSentimentModel(SentimentModel):
    """
    Gemma sentiment model using Nebius TokenFactory API.
    
    Uses OpenAI-compatible chat completions endpoint.
    Requires NEBIUS_API_KEY environment variable.
    """
    
    def __init__(self, prompt_config: Optional[Dict[str, Any]] = None):
        self.name = "gemma"
        
        # Get config with fallbacks
        gemma_cfg = cfg.get("gemma", {})
        self.model = gemma_cfg.get("model", NEBIUS_MODEL)
        self.base_url = gemma_cfg.get("base_url", NEBIUS_BASE_URL)
        self.timeout = gemma_cfg.get("timeout_seconds", 30)
        self.temperature = gemma_cfg.get("temperature", 0.1)
        self.max_tokens = gemma_cfg.get("max_tokens", 512)
        
        # API key from env
        self.api_key = get_env("NEBIUS_API_KEY", "")
        
        # Initialize OpenAI client with Nebius endpoint
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )
        
        if prompt_config:
            self.set_prompt_config(prompt_config)

    def _apply_zone_of_control_guardrail(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enforce Zone of Control Contract.
        
        If brand_relevance is False, the post is not about the brand's controlled
        aspects (e.g., sport outcomes, user's own gambling choices). Therefore,
        it cannot be a negative sentiment ABOUT the brand.
        
        This guardrail ensures consistent behavior across all consumers of this
        model (API, CLI, notebooks, etc.).
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

    def _parse_response(self, content: str) -> Dict[str, Any]:
        """Parse LLM response to extract structured sentiment data."""
        # Try to extract JSON from response
        try:
            # Look for JSON block in response
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                # Apply guardrail to JSON-parsed results
                return self._apply_zone_of_control_guardrail(parsed)
        except json.JSONDecodeError:
            pass
        
        # Fallback: parse text response
        content_lower = content.lower()
        
        # Detect sentiment from text
        if "negative" in content_lower:
            sentiment = "negative"
        elif "positive" in content_lower:
            sentiment = "positive"
        else:
            sentiment = "neutral"
        
        # Detect brand relevance
        brand_relevance = True
        if "not relevant" in content_lower or "brand_relevance\": false" in content_lower:
            brand_relevance = False
        elif "irrelevant" in content_lower or "not about the brand" in content_lower:
            brand_relevance = False
        
        # Detect negative type
        negative_type = None
        if sentiment == "negative":
            if "scam" in content_lower or "fraud" in content_lower:
                negative_type = "scam_accusation"
            elif "regulatory" in content_lower or "underage" in content_lower:
                negative_type = "regulatory_criticism"
            elif "dissatisf" in content_lower or "complaint" in content_lower:
                negative_type = "customer_dissatisfaction"
            else:
                negative_type = "general_negativity"
        
        result = {
            "sentiment": sentiment,
            "confidence": 0.7,  # Default confidence for text parsing
            "reason": content[:200],
            "brand_relevance": brand_relevance,
            "negative_type": negative_type,
        }
        
        # Apply guardrail to text-parsed results
        return self._apply_zone_of_control_guardrail(result)

    def classify(self, text: str) -> Dict[str, Any]:
        """
        Classify text sentiment using Nebius TokenFactory Gemma API.
        
        Returns:
            Dict with sentiment, confidence, reason, brand_relevance, negative_type
        """
        if not self.api_key:
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "reason": "Error: NEBIUS_API_KEY not set. Add to .env file.",
                "brand_relevance": None,
                "negative_type": None,
            }
        
        system_prompt = self.get_system_prompt()
        user_message = self.get_user_message(text)
        
        messages = [
            {"role": "user", "content": f"{system_prompt}\n\n{user_message}"}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            content = response.choices[0].message.content
            result = self._parse_response(content)
            
            # Add raw response for debugging
            result["_raw_response"] = content[:500]
            
            return result
            
        except Exception as e:
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "reason": f"API Error (Gemma/Nebius): {e.__class__.__name__}: {str(e)[:100]}",
                "brand_relevance": None,
                "negative_type": None,
            }

    async def aclassify(self, text: str) -> Dict[str, Any]:
        """
        Async classification using thread pool executor for sync HTTP call.
        """
        return await self._run_sync_in_executor(text)
