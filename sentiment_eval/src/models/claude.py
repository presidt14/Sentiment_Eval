from typing import Dict, Any
import requests
import json

from ..config import get_env, load_settings
from .base import SentimentModel

cfg = load_settings()

SYSTEM_PROMPT = """
You are a sentiment classifier for social-media posts in the i-gaming and regulatory compliance domain.

Classify sentiment as:
- positive
- neutral
- negative

Return ONLY strict JSON with keys:
{
  "sentiment": "positive|neutral|negative",
  "confidence": 0-1 float,
  "reason": "Short explanation"
}
"""

class ClaudeSentimentModel(SentimentModel):
    def __init__(self):
        self.name = "claude"
        self.model = cfg["claude"]["model"]
        self.timeout = cfg["claude"].get("timeout_seconds", 10)
        self.api_key = get_env("ANTHROPIC_API_KEY", "")

    def classify(self, text: str) -> Dict[str, Any]:
        url = "https://api.anthropic.com/v1/messages"
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        
        payload = {
            "model": self.model,
            "max_tokens": 256,
            "system": SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": text}
            ],
        }
        
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            # Expect content block to contain JSON text
            content = data["content"][0]["text"]
            return json.loads(content)
        except requests.exceptions.RequestException as e:
            # Mock failure for dry run
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "reason": f"API Error (Claude): {e.__class__.__name__}. Check ANTHROPIC_API_KEY.",
            }
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "reason": f"API Error (Claude): Invalid response format. {e.__class__.__name__}.",
            }
