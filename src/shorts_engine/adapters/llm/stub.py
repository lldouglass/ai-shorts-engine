"""Stub LLM provider for testing."""

import json

from shorts_engine.adapters.llm.base import LLMMessage, LLMProvider, LLMResponse, VisionMessage
from shorts_engine.logging import get_logger

logger = get_logger(__name__)


class StubLLMProvider(LLMProvider):
    """Stub provider that returns mock LLM responses for testing."""

    @property
    def name(self) -> str:
        return "stub"

    async def complete(
        self,
        messages: list[LLMMessage],
        _temperature: float = 0.7,
        _max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Return a mock completion response."""
        logger.info(
            "stub_llm_complete",
            message_count=len(messages),
            json_mode=json_mode,
        )

        # Extract the user's last message to generate contextual response
        user_message = ""
        for msg in reversed(messages):
            if msg.role == "user":
                user_message = msg.content
                break

        if json_mode:
            # Return a mock video plan JSON
            content = json.dumps(
                {
                    "title": f"Epic Short: {user_message[:30]}...",
                    "description": f"A captivating short video about {user_message[:50]}. "
                    "This video takes viewers on an unforgettable visual journey.",
                    "scenes": [
                        {
                            "scene_number": i + 1,
                            "visual_prompt": f"Scene {i + 1}: {['Opening shot establishing mood', 'Character introduction with dramatic lighting', 'Action sequence with dynamic camera movement', 'Emotional close-up moment', 'Plot twist reveal', 'Climactic confrontation', 'Resolution and reflection', 'Final powerful image'][i % 8]}",
                            "continuity_notes": "Maintain consistent character design, lighting direction from left, color palette with deep shadows",
                            "caption_beat": [
                                "The beginning",
                                "Enter the hero",
                                "Chaos unfolds",
                                "A moment of truth",
                                "Everything changes",
                                "The final stand",
                                "Peace returns",
                                "Remember this",
                            ][i % 8],
                            "duration_seconds": 5.0 + (i % 3),
                        }
                        for i in range(7)
                    ],
                },
                indent=2,
            )
        else:
            content = f"This is a stub response for: {user_message[:100]}"

        return LLMResponse(
            content=content,
            model="stub-model",
            usage={
                "prompt_tokens": len(user_message.split()),
                "completion_tokens": len(content.split()),
                "total_tokens": len(user_message.split()) + len(content.split()),
            },
            finish_reason="stop",
        )

    @property
    def supports_vision(self) -> bool:
        """Stub provider supports vision for testing."""
        return True

    async def complete_with_vision(
        self,
        messages: list[VisionMessage],
        temperature: float = 0.7,  # noqa: ARG002
        max_tokens: int = 4096,  # noqa: ARG002
        json_mode: bool = False,
    ) -> LLMResponse:
        """Return a mock vision completion response."""
        image_count = sum(len(m.image_urls) for m in messages)

        logger.info(
            "stub_llm_vision_complete",
            message_count=len(messages),
            image_count=image_count,
            json_mode=json_mode,
        )

        # Extract the user's last message text
        user_message = ""
        for msg in reversed(messages):
            if msg.role == "user":
                user_message = msg.text
                break

        if json_mode:
            # Return a mock image critique JSON
            content = json.dumps(
                {
                    "visual_coherence_score": 0.85,
                    "style_consistency_score": 0.82,
                    "overall_passed": True,
                    "per_scene_feedback": [
                        {
                            "scene_number": i + 1,
                            "score": 0.80 + (i % 3) * 0.05,
                            "feedback": f"Scene {i + 1} maintains good visual consistency",
                            "issues": [],
                        }
                        for i in range(min(image_count, 7))
                    ],
                    "improvement_suggestions": [
                        "Consider adding more contrast between scenes",
                        "The color palette could be more unified",
                    ],
                    "summary": "Overall the images show good visual coherence and style consistency.",
                },
                indent=2,
            )
        else:
            content = (
                f"Vision analysis of {image_count} images: "
                f"The images show consistent visual style and coherent narrative flow. "
                f"Original prompt: {user_message[:50]}..."
            )

        return LLMResponse(
            content=content,
            model="stub-vision-model",
            usage={
                "prompt_tokens": len(user_message.split()) + image_count * 1000,
                "completion_tokens": len(content.split()),
                "total_tokens": len(user_message.split())
                + image_count * 1000
                + len(content.split()),
            },
            finish_reason="stop",
        )

    async def health_check(self) -> bool:
        """Stub provider is always healthy."""
        return True
