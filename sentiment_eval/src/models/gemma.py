from typing import Dict, Any
import requests
import json

from ..config import get_env, load_settings
from .base import SentimentModel

cfg = load_settings()

class GemmaSentimentModel(SentimentModel):
    def __init__(self):
        self.name = "gemma"
        self.url = cfg["gemma"]["gemma url"]
        self.timeout = cfg["gemma"].get("timeout_seconds", 10)
        self.token = get_env("GEMMA_API_TOKEN", "")

    def classify(self, text: str) -> Dict[str, Any]:
        # Expect internal Gemma endpoint to return:
        # {
        #   "sentiment": "positive|neutral|negative",
        #   "confidence": 0.93,
        #   "reason": "short explanation"
        # }
        
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "text": text,
            "task": "sentiment",
        }
        
        try:
            resp = requests.post(self.url, json=payload, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            # Mock failure for dry run
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "reason": f"API Error (Gemma): {e.__class__.__name__}. Check GEMMA_API_TOKEN and URL.",
            }
        except json.JSONDecodeError:
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "reason": "API Error (Gemma): Invalid JSON response.",
            }
