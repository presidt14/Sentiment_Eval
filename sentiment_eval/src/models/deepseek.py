from typing import Dict, Any
import requests
import json

from ..config import get_env, load_settings
from .base import SentimentModel

cfg = load_settings()

DEEPSEEK_PROMPT = """
You are a sentiment classifier for social-media posts in the i-gaming and regulatory compliance domain.

Classify sentiment as:
- positive
- neutral
- negative

Return JSON only, with:
{
  "sentiment": "...",
  "confidence": 0-1,
  "reason": "..."
}
"""

class DeepseekSentimentModel(SentimentModel):
    def __init__(self):
        self.name = "deepseek"
        self.model = cfg["deepseek"]["model"]
        self.timeout = cfg["deepseek"].get("timeout_seconds", 10)
        self.api_key = get_env("DEEPSEEK_API_KEY", "")

    def classify(self, text: str) -> Dict[str, Any]:
        url = "https://api.deepseek.com/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": DEEPSEEK_PROMPT},
                {"role": "user", "content": text}
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": 256,
        }
        
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        except requests.exceptions.RequestException as e:
            # Mock failure for dry run
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "reason": f"API Error (Deepseek): {e.__class__.__name__}. Check DEEPSEEK_API_KEY.",
            }
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "reason": f"API Error (Deepseek): Invalid response format. {e.__class__.__name__}.",
            }
