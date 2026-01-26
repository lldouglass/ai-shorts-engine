"""OpenAI LLM provider implementation."""

import json
from typing import Any

import httpx

from shorts_engine.adapters.llm.base import LLMMessage, LLMProvider, LLMResponse
from shorts_engine.config import settings
from shorts_engine.logging import get_logger

logger = get_logger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI API provider for GPT models."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        self.api_key = api_key or settings.openai_api_key
        self.model = model
        self.base_url = base_url

        if not self.api_key:
            logger.warning("OpenAI API key not configured")

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    async def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Generate completion using OpenAI API."""
        if not self.api_key:
            raise ValueError("OpenAI API key not configured")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        logger.debug(
            "openai_request",
            model=self.model,
            message_count=len(messages),
            json_mode=json_mode,
        )

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        choice = data["choices"][0]
        usage = data.get("usage", {})

        logger.info(
            "openai_response",
            model=self.model,
            tokens_used=usage.get("total_tokens", 0),
            finish_reason=choice.get("finish_reason"),
        )

        return LLMResponse(
            content=choice["message"]["content"],
            model=data["model"],
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            raw_response=data,
            finish_reason=choice.get("finish_reason"),
        )

    async def health_check(self) -> bool:
        """Check if OpenAI API is accessible."""
        if not self.api_key:
            return False

        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers=headers,
                )
                return response.status_code == 200
        except Exception as e:
            logger.error("openai_health_check_failed", error=str(e))
            return False
