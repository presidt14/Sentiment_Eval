from typing import Dict, Any
import requests
import json

from ..config import get_env, load_settings
from .base import SentimentModel

cfg = load_settings()

GEMINI_PROMPT = """
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

class GeminiSentimentModel(SentimentModel):
    def __init__(self):
        self.name = "gemini"
        self.model = cfg["gemini"]["model"]
        self.timeout = cfg["gemini"].get("timeout_seconds", 10)
        self.api_key = get_env("GEMINI_API_KEY", "")

    def classify(self, text: str) -> Dict[str, Any]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": GEMINI_PROMPT},
                        {"text": text}
                    ]
                }
            ]
        }
        
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            raw_text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(raw_text)
        except requests.exceptions.RequestException as e:
            # Mock failure for dry run
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "reason": f"API Error (Gemini): {e.__class__.__name__}. Check GEMINI_API_KEY.",
            }
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "reason": f"API Error (Gemini): Invalid response format. {e.__class__.__name__}.",
            }
