"""Content strategies module - centralized engagement and CTR strategies.

All video generation prompts pull from these strategies so there's ONE place
to update when we learn what works.
"""

from shorts_engine.strategies.engagement import ENGAGEMENT_STRATEGIES, get_strategies_for_niche
from shorts_engine.strategies.hooks import HOOK_FORMULAS

__all__ = [
    "ENGAGEMENT_STRATEGIES",
    "HOOK_FORMULAS",
    "get_strategies_for_niche",
]
