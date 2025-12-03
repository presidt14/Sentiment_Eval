from typing import Dict, Any, Optional
import requests
import json

from ..config import get_env, load_settings
from .base import SentimentModel

cfg = load_settings()


class ClaudeSentimentModel(SentimentModel):
    def __init__(self, prompt_config: Optional[Dict[str, Any]] = None):
        self.name = "claude"
        self.model = cfg["claude"]["model"]
        self.timeout = cfg["claude"].get("timeout_seconds", 10)
        self.api_key = get_env("ANTHROPIC_API_KEY", "")
        if prompt_config:
            self.set_prompt_config(prompt_config)

    def classify(self, text: str) -> Dict[str, Any]:
        url = "https://api.anthropic.com/v1/messages"
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        
        # Use externalized prompts
        system_prompt = self.get_system_prompt()
        user_message = self.get_user_message(text)
        
        payload = {
            "model": self.model,
            "max_tokens": 256,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_message}
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

    async def aclassify(self, text: str) -> Dict[str, Any]:
        """
        Async classification using thread pool executor for sync HTTP call.
        """
        return await self._run_sync_in_executor(text)
