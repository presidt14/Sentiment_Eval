import json
from typing import Any, Dict, Optional

import requests

from ..config import get_env, load_settings
from .base import SentimentModel

cfg = load_settings()


class GemmaSentimentModel(SentimentModel):
    def __init__(self, prompt_config: Optional[Dict[str, Any]] = None):
        self.name = "gemma"
        self.url = cfg["gemma"]["gemma url"]
        self.timeout = cfg["gemma"].get("timeout_seconds", 10)
        self.token = get_env("GEMMA_API_TOKEN", "")
        if prompt_config:
            self.set_prompt_config(prompt_config)

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

        # Use externalized prompts - pass to internal endpoint
        system_prompt = self.get_system_prompt()
        user_message = self.get_user_message(text)

        payload = {
            "text": user_message,
            "task": "sentiment",
            "system_prompt": system_prompt,  # Internal endpoint can use this
        }

        try:
            resp = requests.post(
                self.url, json=payload, headers=headers, timeout=self.timeout
            )
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

    async def aclassify(self, text: str) -> Dict[str, Any]:
        """
        Async classification using thread pool executor for sync HTTP call.
        """
        return await self._run_sync_in_executor(text)
