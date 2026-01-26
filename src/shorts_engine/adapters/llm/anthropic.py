"""Anthropic LLM provider implementation."""

from typing import Any

import httpx

from shorts_engine.adapters.llm.base import LLMMessage, LLMProvider, LLMResponse
from shorts_engine.config import settings
from shorts_engine.logging import get_logger

logger = get_logger(__name__)


class AnthropicProvider(LLMProvider):
    """Anthropic API provider for Claude models."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        base_url: str = "https://api.anthropic.com/v1",
    ) -> None:
        self.api_key = api_key or getattr(settings, "anthropic_api_key", None)
        self.model = model
        self.base_url = base_url

        if not self.api_key:
            logger.warning("Anthropic API key not configured")

    @property
    def name(self) -> str:
        return f"anthropic:{self.model}"

    async def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Generate completion using Anthropic API."""
        if not self.api_key:
            raise ValueError("Anthropic API key not configured")

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        # Separate system message from conversation
        system_message = ""
        conversation_messages = []

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                conversation_messages.append({"role": msg.role, "content": msg.content})

        # Add JSON instruction to system prompt if json_mode
        if json_mode:
            json_instruction = "\n\nIMPORTANT: You must respond with valid JSON only. No other text."
            system_message = system_message + json_instruction if system_message else json_instruction

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": conversation_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if system_message:
            payload["system"] = system_message

        logger.debug(
            "anthropic_request",
            model=self.model,
            message_count=len(conversation_messages),
            json_mode=json_mode,
        )

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/messages",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        content = ""
        if data.get("content"):
            for block in data["content"]:
                if block.get("type") == "text":
                    content += block.get("text", "")

        usage = data.get("usage", {})

        logger.info(
            "anthropic_response",
            model=self.model,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            stop_reason=data.get("stop_reason"),
        )

        return LLMResponse(
            content=content,
            model=data.get("model", self.model),
            usage={
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            },
            raw_response=data,
            finish_reason=data.get("stop_reason"),
        )

    async def health_check(self) -> bool:
        """Check if Anthropic API is accessible."""
        if not self.api_key:
            return False

        # Anthropic doesn't have a simple health endpoint, so we just check auth
        try:
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            }
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Make a minimal request to check auth
                response = await client.post(
                    f"{self.base_url}/messages",
                    headers=headers,
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 1,
                    },
                )
                # 200 = success, 401 = bad auth, anything else = service issue
                return response.status_code == 200
        except Exception as e:
            logger.error("anthropic_health_check_failed", error=str(e))
            return False
