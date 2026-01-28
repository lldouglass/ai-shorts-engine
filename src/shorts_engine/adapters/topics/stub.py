"""Stub topic provider for testing."""

import random

from shorts_engine.adapters.topics.base import GeneratedTopic, TopicContext, TopicProvider
from shorts_engine.logging import get_logger

logger = get_logger(__name__)

# Sample topics for testing - diverse viral video ideas
STUB_TOPICS = [
    "What if you could breathe underwater for 24 hours",
    "The hidden psychology behind why we procrastinate",
    "Ancient Roman concrete was better than ours - here's why",
    "This AI can predict your next thought",
    "The one habit that made me a millionaire at 25",
    "Scientists just discovered a new emotion",
    "Why do we dream about falling",
    "The truth about social media algorithms nobody talks about",
    "This optical illusion will break your brain",
    "What happens to your body when you stop eating sugar",
    "The dark side of being too productive",
    "Why billionaires wake up at 4 AM (it's not what you think)",
    "The science of why some songs get stuck in your head",
    "This simple trick doubles your reading speed",
    "What your handwriting reveals about your personality",
    "The real reason you can't remember your dreams",
    "How to become invisible in plain sight",
    "The psychology behind viral content",
    "Why time feels faster as you age",
    "The hidden cost of free apps",
]

HOOK_SUGGESTIONS = [
    "question",
    "statement",
    "visual",
    "story",
    "contrast",
    "mystery",
]


class StubTopicProvider(TopicProvider):
    """Stub provider that returns mock topics for testing."""

    @property
    def name(self) -> str:
        return "stub"

    async def generate_topics(
        self,
        context: TopicContext,
        n: int = 5,
        temperature: float = 0.8,
    ) -> list[GeneratedTopic]:
        """Return mock topics for testing."""
        logger.info(
            "stub_topic_generate",
            project_id=str(context.project_id),
            project_name=context.project_name,
            n=n,
            temperature=temperature,
        )

        # Filter out recent topics to avoid duplicates
        available = [t for t in STUB_TOPICS if t not in context.recent_topics]

        # If not enough available, use all topics
        if len(available) < n:
            available = STUB_TOPICS.copy()

        # Shuffle based on temperature (higher = more random)
        if temperature > 0.5:
            random.shuffle(available)

        selected = available[:n]

        topics = []
        for topic in selected:
            topics.append(
                GeneratedTopic(
                    topic=topic,
                    hook_suggestion=random.choice(HOOK_SUGGESTIONS),
                    estimated_virality_score=round(random.uniform(0.6, 0.95), 2),
                    reasoning=f"Stub topic for testing: {context.project_name}",
                )
            )

        logger.info(
            "stub_topics_generated",
            count=len(topics),
            project_id=str(context.project_id),
        )

        return topics

    async def health_check(self) -> bool:
        """Stub provider is always healthy."""
        return True
