"""LLM-based topic provider for intelligent topic generation."""

import json

from shorts_engine.adapters.llm.base import LLMMessage, LLMProvider
from shorts_engine.adapters.topics.base import GeneratedTopic, TopicContext, TopicProvider
from shorts_engine.logging import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a viral content strategist specializing in short-form video content.
Your job is to generate compelling, high-performing video topic ideas.

You understand:
- What makes content go viral (emotional hooks, curiosity gaps, controversy, relatability)
- Current social media trends and attention patterns
- How to balance creativity with proven formats
- The importance of strong hooks that stop the scroll

Generate topics that:
1. Have strong emotional hooks (curiosity, surprise, fear, joy, anger, awe)
2. Can be explained visually in under 60 seconds
3. Appeal to broad audiences while still being specific enough to be interesting
4. Haven't been overdone but build on proven viral formats
5. Create a curiosity gap that makes viewers need to watch

IMPORTANT: Avoid topics too similar to the recent topics provided. Diversify!"""

USER_PROMPT_TEMPLATE = """Generate {n} unique, viral-worthy video topic ideas for this project:

PROJECT: {project_name}
DESCRIPTION: {project_description}
NICHE: {niche}

CONTEXT FOR OPTIMIZATION:
- Top performing topics (learn from these): {top_performing}
- Recent topics (AVOID similar ones): {recent_topics}
- Best hook types for this audience: {top_hooks}
- Trending topics to consider: {trending}

Generate exactly {n} topics. For each topic, provide:
1. The topic idea (compelling, specific, creates curiosity)
2. Suggested hook type (question, statement, visual, story, contrast, or mystery)
3. Virality score estimate (0.0 to 1.0)
4. Brief reasoning for why this will perform well

Return as JSON array:
[
  {{
    "topic": "Topic text here",
    "hook_suggestion": "question",
    "estimated_virality_score": 0.85,
    "reasoning": "Why this topic will work"
  }}
]

Return ONLY the JSON array, no other text."""


class LLMTopicProvider(TopicProvider):
    """LLM-based topic provider using existing LLM adapters."""

    def __init__(self, llm_provider: LLMProvider):
        """Initialize with an LLM provider.

        Args:
            llm_provider: The LLM provider to use for generation
        """
        self._llm = llm_provider

    @property
    def name(self) -> str:
        return f"llm-{self._llm.name}"

    async def generate_topics(
        self,
        context: TopicContext,
        n: int = 5,
        temperature: float = 0.8,
    ) -> list[GeneratedTopic]:
        """Generate topics using LLM."""
        logger.info(
            "llm_topic_generate_started",
            project_id=str(context.project_id),
            project_name=context.project_name,
            n=n,
            llm_provider=self._llm.name,
        )

        # Build the user prompt with context
        user_prompt = USER_PROMPT_TEMPLATE.format(
            n=n,
            project_name=context.project_name,
            project_description=context.project_description or "General content",
            niche=context.niche or "General interest",
            top_performing=", ".join(context.top_performing_topics[:5]) or "None yet",
            recent_topics=", ".join(context.recent_topics[:10]) or "None",
            top_hooks=", ".join(context.top_hook_types[:3]) or "All types work",
            trending=", ".join(context.trending_topics[:5]) or "None provided",
        )

        messages = [
            LLMMessage(role="system", content=SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_prompt),
        ]

        try:
            response = await self._llm.complete(
                messages=messages,
                temperature=temperature,
                max_tokens=2048,
                json_mode=True,
            )

            # Parse the JSON response
            topics = self._parse_response(response.content, n)

            logger.info(
                "llm_topic_generate_completed",
                project_id=str(context.project_id),
                topics_generated=len(topics),
                llm_model=response.model,
                tokens_used=response.usage.get("total_tokens", 0),
            )

            return topics

        except Exception as e:
            logger.error(
                "llm_topic_generate_failed",
                project_id=str(context.project_id),
                error=str(e),
            )
            raise

    def _parse_response(self, content: str, expected_n: int) -> list[GeneratedTopic]:
        """Parse LLM response into GeneratedTopic objects.

        Args:
            content: Raw LLM response content
            expected_n: Expected number of topics

        Returns:
            List of GeneratedTopic objects
        """
        try:
            # Try to parse as JSON
            data = json.loads(content)

            if not isinstance(data, list):
                logger.warning("llm_response_not_list", content_type=type(data).__name__)
                data = [data]

            topics = []
            for item in data[:expected_n]:
                if isinstance(item, dict):
                    topics.append(
                        GeneratedTopic(
                            topic=item.get("topic", ""),
                            hook_suggestion=item.get("hook_suggestion"),
                            estimated_virality_score=item.get("estimated_virality_score"),
                            reasoning=item.get("reasoning"),
                        )
                    )
                elif isinstance(item, str):
                    topics.append(GeneratedTopic(topic=item))

            return topics

        except json.JSONDecodeError as e:
            logger.warning(
                "llm_response_json_parse_failed",
                error=str(e),
                content_preview=content[:200],
            )
            # Fallback: treat the whole content as a single topic
            return [GeneratedTopic(topic=content.strip())]

    async def health_check(self) -> bool:
        """Check if the underlying LLM provider is available."""
        return await self._llm.health_check()
