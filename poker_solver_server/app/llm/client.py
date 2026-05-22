from __future__ import annotations

import os
import time
import logging
from typing import Any, Dict, List, Optional
import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-flash-3.5")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


class OpenRouterClient:
    def __init__(
        self,
        api_key: str = OPENROUTER_API_KEY,
        model: str = OPENROUTER_MODEL,
        base_url: str = OPENROUTER_BASE_URL,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Call OpenRouter chat completions API with one retry on 429/5xx/parse error."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
        }
        if response_format:
            payload["response_format"] = response_format

        def attempt_request() -> Dict[str, Any]:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, headers=headers, json=payload)
                if response.status_code == 429:
                    raise httpx.HTTPStatusError("429 Too Many Requests", request=response.request, response=response)
                elif response.status_code >= 500:
                    raise httpx.HTTPStatusError(f"{response.status_code} Server Error", request=response.request, response=response)
                
                response.raise_for_status()
                data = response.json()
                if "choices" not in data or not data["choices"]:
                    raise ValueError("Invalid response shape: missing choices")
                return data

        try:
            return attempt_request()
        except (httpx.HTTPError, ValueError) as e:
            logger.warning(f"First attempt failed with error: {e}. Retrying...")
            # Wait briefly before retrying (exponential backoff or just a simple sleep)
            time.sleep(1.0)
            return attempt_request()
