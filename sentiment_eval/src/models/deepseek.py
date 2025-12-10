import json
from typing import Any, Dict, Optional

import requests

from ..config import get_env, load_settings
from .base import SentimentModel

cfg = load_settings()


class DeepseekSentimentModel(SentimentModel):
    def __init__(self, prompt_config: Optional[Dict[str, Any]] = None):
        self.name = "deepseek"
        self.model = cfg["deepseek"]["model"]
        self.timeout = cfg["deepseek"].get("timeout_seconds", 10)
        self.api_key = get_env("DEEPSEEK_API_KEY", "")
        if prompt_config:
            self.set_prompt_config(prompt_config)

    def classify(self, text: str) -> Dict[str, Any]:
        url = "https://api.deepseek.com/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Use externalized prompts
        system_prompt = self.get_system_prompt()
        user_message = self.get_user_message(text)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": 256,
        }

        try:
            resp = requests.post(
                url, json=payload, headers=headers, timeout=self.timeout
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            result = json.loads(content)
            # Apply Zone of Control guardrail (inherited from base)
            return self._apply_zone_of_control_guardrail(result)
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

    async def aclassify(self, text: str) -> Dict[str, Any]:
        """
        Async classification using thread pool executor for sync HTTP call.
        """
        return await self._run_sync_in_executor(text)
