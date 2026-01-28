"""Topic generation provider adapters."""

from shorts_engine.adapters.topics.base import GeneratedTopic, TopicContext, TopicProvider
from shorts_engine.adapters.topics.llm import LLMTopicProvider
from shorts_engine.adapters.topics.stub import StubTopicProvider

__all__ = [
    "GeneratedTopic",
    "TopicContext",
    "TopicProvider",
    "LLMTopicProvider",
    "StubTopicProvider",
]
