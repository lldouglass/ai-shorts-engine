"""Google Gemini LLM provider with native video support."""

import asyncio
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from shorts_engine.adapters.llm.base import LLMMessage, LLMProvider, LLMResponse, VisionMessage
from shorts_engine.config import settings
from shorts_engine.logging import get_logger

logger = get_logger(__name__)


class GeminiProvider(LLMProvider):
    """Google Gemini API provider with native video understanding.

    Key features:
    - Native video input via File API (no frame extraction needed)
    - Audio understanding for voiceover sync analysis
    - Cost-effective: ~258 tokens/second of video

    Uses the google-genai SDK (Client-based API).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize Gemini provider.

        Args:
            api_key: Google API key (uses GOOGLE_API_KEY from settings if not provided)
            model: Model name (defaults to gemini-3.0-pro)
        """
        self.api_key = api_key or settings.google_api_key
        _model = model or getattr(settings, "gemini_critique_model", "gemini-3.0-pro")
        self.model: str = _model if isinstance(_model, str) else "gemini-3.0-pro"
        self._client: genai.Client | None = None

        if not self.api_key:
            logger.warning("Google API key not configured for Gemini")

    @property
    def client(self) -> genai.Client:
        """Get or create the Gemini client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError("Google API key not configured")
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    @property
    def name(self) -> str:
        return f"gemini:{self.model}"

    @property
    def supports_vision(self) -> bool:
        """Gemini models support vision."""
        return True

    @property
    def supports_video(self) -> bool:
        """Gemini models support native video input."""
        return True

    def _get_generation_config(
        self,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> types.GenerateContentConfig:
        """Build generation config for Gemini API."""
        config_dict: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if json_mode:
            config_dict["response_mime_type"] = "application/json"
        return types.GenerateContentConfig(**config_dict)

    async def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Generate completion using Gemini API."""
        if not self.api_key:
            raise ValueError("Google API key not configured")

        # Convert messages to Gemini format
        # Gemini uses a simpler format - combine system prompt with first user message
        contents: list[types.Content] = []
        system_prompt = ""

        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            elif msg.role == "user":
                content = msg.content
                if system_prompt and not contents:
                    content = f"{system_prompt}\n\n{content}"
                contents.append(types.Content(role="user", parts=[types.Part(text=content)]))
            elif msg.role == "assistant":
                contents.append(types.Content(role="model", parts=[types.Part(text=msg.content)]))

        logger.debug(
            "gemini_request",
            model=self.model,
            message_count=len(contents),
            json_mode=json_mode,
        )

        # Run sync API in executor
        loop = asyncio.get_event_loop()
        config = self._get_generation_config(temperature, max_tokens, json_mode)

        response = await loop.run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=self.model,
                contents=contents,  # type: ignore[arg-type]
                config=config,
            ),
        )

        # Extract usage info
        usage = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = {
                "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0) or 0,
                "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0)
                or 0,
                "total_tokens": getattr(response.usage_metadata, "total_token_count", 0) or 0,
            }

        logger.info(
            "gemini_response",
            model=self.model,
            tokens_used=usage.get("total_tokens", 0),
        )

        return LLMResponse(
            content=response.text or "",
            model=self.model,
            usage=usage,
            raw_response={"candidates": [str(c) for c in (response.candidates or [])]},
            finish_reason=(
                str(response.candidates[0].finish_reason) if response.candidates else None
            ),
        )

    async def complete_with_vision(
        self,
        messages: list[VisionMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Generate completion with vision support using Gemini API."""
        if not self.api_key:
            raise ValueError("Google API key not configured")

        # Build content parts from messages
        parts: list[types.Part] = []
        system_prompt = ""

        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.text
            elif msg.role == "user":
                # Add text
                text = msg.text
                if system_prompt and not parts:
                    text = f"{system_prompt}\n\n{text}"
                parts.append(types.Part(text=text))

                # Add images
                for image_url in msg.image_urls:
                    if image_url.startswith("data:"):
                        # Base64 data URI
                        import base64

                        # Extract mime type and data
                        header, data = image_url.split(",", 1)
                        mime_type = header.split(":")[1].split(";")[0]
                        image_bytes = base64.b64decode(data)
                        parts.append(
                            types.Part(
                                inline_data=types.Blob(
                                    mime_type=mime_type,
                                    data=image_bytes,
                                )
                            )
                        )
                    else:
                        # URL - Gemini doesn't support URLs directly, would need to download
                        logger.warning(
                            "gemini_vision_url_not_supported",
                            url=image_url[:50],
                        )

        image_count = sum(len(m.image_urls) for m in messages)
        logger.debug(
            "gemini_vision_request",
            model=self.model,
            message_count=len(messages),
            image_count=image_count,
            json_mode=json_mode,
        )

        # Run sync API in executor
        loop = asyncio.get_event_loop()
        config = self._get_generation_config(temperature, max_tokens, json_mode)

        response = await loop.run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=self.model,
                contents=[types.Content(role="user", parts=parts)],  # type: ignore[arg-type]
                config=config,
            ),
        )

        # Extract usage info
        usage = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = {
                "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0) or 0,
                "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0)
                or 0,
                "total_tokens": getattr(response.usage_metadata, "total_token_count", 0) or 0,
            }

        logger.info(
            "gemini_vision_response",
            model=self.model,
            tokens_used=usage.get("total_tokens", 0),
        )

        return LLMResponse(
            content=response.text or "",
            model=self.model,
            usage=usage,
            raw_response={"candidates": [str(c) for c in (response.candidates or [])]},
            finish_reason=(
                str(response.candidates[0].finish_reason) if response.candidates else None
            ),
        )

    async def complete_with_video(
        self,
        video_path: Path,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Analyze a video file using Gemini's native video understanding.

        This uploads the video to Google's File API and then analyzes it,
        which is much more efficient than extracting and analyzing frames.

        Args:
            video_path: Path to the video file
            prompt: The analysis prompt
            temperature: Sampling temperature (lower = more consistent)
            max_tokens: Maximum tokens in response
            json_mode: If True, request JSON output format

        Returns:
            LLMResponse with video analysis
        """
        if not self.api_key:
            raise ValueError("Google API key not configured")

        if not video_path.exists():
            raise ValueError(f"Video file not found: {video_path}")

        logger.info(
            "gemini_video_upload_started",
            video_path=str(video_path),
            file_size_mb=video_path.stat().st_size / (1024 * 1024),
        )

        # Upload video to File API (runs in executor since it's sync)
        loop = asyncio.get_event_loop()
        start_time = time.time()

        # Upload file
        video_file = await loop.run_in_executor(
            None,
            lambda: self.client.files.upload(file=video_path),
        )

        # Wait for file to be processed
        while video_file.state == "PROCESSING":
            await asyncio.sleep(2)
            file_name = video_file.name or ""  # Capture for closure
            video_file = await loop.run_in_executor(
                None,
                lambda name=file_name: self.client.files.get(name=name),  # type: ignore[misc]
            )

        if video_file.state == "FAILED":
            raise RuntimeError(f"Video processing failed: {video_file.state}")

        upload_time = time.time() - start_time
        logger.info(
            "gemini_video_upload_completed",
            video_path=str(video_path),
            upload_time_seconds=upload_time,
            file_name=video_file.name,
        )

        # Analyze the video
        logger.debug(
            "gemini_video_request",
            model=self.model,
            video_file=video_file.name,
            json_mode=json_mode,
        )

        config = self._get_generation_config(temperature, max_tokens, json_mode)

        # Create content with video file reference and prompt
        file_part = types.Part(
            file_data=types.FileData(
                file_uri=video_file.uri,
                mime_type=video_file.mime_type,
            )
        )
        prompt_part = types.Part(text=prompt)

        response = await loop.run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=self.model,
                contents=[types.Content(role="user", parts=[file_part, prompt_part])],  # type: ignore[arg-type]
                config=config,
            ),
        )

        # Clean up uploaded file
        try:
            delete_name = video_file.name or ""
            await loop.run_in_executor(
                None,
                lambda: self.client.files.delete(name=delete_name),
            )
            logger.debug("gemini_video_file_deleted", file_name=video_file.name)
        except Exception as e:
            logger.warning("gemini_video_file_delete_failed", error=str(e))

        # Extract usage info
        usage = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = {
                "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0) or 0,
                "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0)
                or 0,
                "total_tokens": getattr(response.usage_metadata, "total_token_count", 0) or 0,
            }

        logger.info(
            "gemini_video_response",
            model=self.model,
            tokens_used=usage.get("total_tokens", 0),
        )

        return LLMResponse(
            content=response.text or "",
            model=self.model,
            usage=usage,
            raw_response={"candidates": [str(c) for c in (response.candidates or [])]},
            finish_reason=(
                str(response.candidates[0].finish_reason) if response.candidates else None
            ),
        )

    async def health_check(self) -> bool:
        """Check if Gemini API is accessible."""
        if not self.api_key:
            return False

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model=self.model,
                    contents="Say 'ok' if you can hear me.",
                ),
            )
            return bool(response.text)
        except Exception as e:
            logger.error("gemini_health_check_failed", error=str(e))
            return False
