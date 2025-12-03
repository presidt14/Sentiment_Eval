from typing import Dict, Any, Optional
import requests
import json

from ..config import get_env, load_settings
from .base import SentimentModel

cfg = load_settings()


class GeminiSentimentModel(SentimentModel):
    def __init__(self, prompt_config: Optional[Dict[str, Any]] = None):
        self.name = "gemini"
        self.model = cfg["gemini"]["model"]
        self.timeout = cfg["gemini"].get("timeout_seconds", 10)
        self.api_key = get_env("GEMINI_API_KEY", "")
        if prompt_config:
            self.set_prompt_config(prompt_config)

    def classify(self, text: str) -> Dict[str, Any]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        
        # Use externalized prompts
        system_prompt = self.get_system_prompt()
        user_message = self.get_user_message(text)
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": system_prompt},
                        {"text": user_message}
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

    async def aclassify(self, text: str) -> Dict[str, Any]:
        """
        Async classification using thread pool executor for sync HTTP call.
        """
        return await self._run_sync_in_executor(text)
