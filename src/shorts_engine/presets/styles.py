"""Style preset definitions for video generation.

Each preset defines visual style tokens, aspect ratio, and generation parameters
that are injected into prompts for consistent video generation.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class AspectRatio(StrEnum):
    """Supported aspect ratios for video generation."""

    VERTICAL_9_16 = "9:16"  # Shorts, TikTok, Reels
    HORIZONTAL_16_9 = "16:9"  # YouTube standard
    SQUARE_1_1 = "1:1"  # Instagram feed


@dataclass(frozen=True)
class StylePreset:
    """A visual style preset for video generation.

    Attributes:
        name: Unique identifier for the preset
        display_name: Human-readable name
        description: Brief description of the style
        style_tokens: Core style descriptors injected into every prompt
        negative_tokens: Things to avoid in generation
        continuity_tokens: Tokens for maintaining character/scene consistency
        aspect_ratio: Video aspect ratio
        default_duration_per_scene: Default scene duration in seconds
        color_palette: Primary colors for the style
        camera_style: Default camera movement/framing
        generation_params: Provider-specific generation parameters
    """

    name: str
    display_name: str
    description: str
    style_tokens: list[str]
    negative_tokens: list[str] = field(default_factory=list)
    continuity_tokens: list[str] = field(default_factory=list)
    aspect_ratio: AspectRatio = AspectRatio.VERTICAL_9_16
    default_duration_per_scene: float = 5.0
    color_palette: list[str] = field(default_factory=list)
    camera_style: str = "dynamic"
    generation_params: dict[str, Any] = field(default_factory=dict)

    def format_style_prompt(self) -> str:
        """Format style tokens into a prompt suffix."""
        return ", ".join(self.style_tokens)

    def format_negative_prompt(self) -> str:
        """Format negative tokens into a negative prompt."""
        return ", ".join(self.negative_tokens)

    def format_continuity_prompt(self) -> str:
        """Format continuity tokens for scene-to-scene consistency."""
        return ", ".join(self.continuity_tokens)


# =============================================================================
# PRESET DEFINITIONS
# =============================================================================

DARK_DYSTOPIAN_ANIME = StylePreset(
    name="DARK_DYSTOPIAN_ANIME",
    display_name="Dark Dystopian Anime",
    description="Gritty, cinematic anime style with fog, debris, and ink linework",
    style_tokens=[
        "dark dystopian anime",
        "gritty cinematic",
        "heavy fog",
        "floating debris",
        "ink linework",
        "dramatic lighting",
        "neon accents",
        "urban decay",
        "post-apocalyptic",
        "manga shading",
        "high contrast",
        "noir atmosphere",
    ],
    negative_tokens=[
        "bright colors",
        "cheerful",
        "cartoon",
        "pixar style",
        "watercolor",
        "pastel",
        "low quality",
        "blurry",
        "deformed",
    ],
    continuity_tokens=[
        "consistent character design",
        "same art style",
        "matching color grading",
        "unified atmosphere",
    ],
    aspect_ratio=AspectRatio.VERTICAL_9_16,
    default_duration_per_scene=5.0,
    color_palette=["#1a1a2e", "#16213e", "#0f3460", "#e94560", "#00fff5"],
    camera_style="slow dramatic pans, close-ups with depth of field",
    generation_params={
        "guidance_scale": 7.5,
        "num_inference_steps": 50,
    },
)

VIBRANT_MOTION_GRAPHICS = StylePreset(
    name="VIBRANT_MOTION_GRAPHICS",
    display_name="Vibrant Motion Graphics",
    description="Bold, colorful motion graphics with geometric shapes and smooth transitions",
    style_tokens=[
        "motion graphics",
        "bold geometric shapes",
        "vibrant gradients",
        "smooth transitions",
        "clean lines",
        "modern design",
        "flat illustration",
        "kinetic typography ready",
        "professional broadcast quality",
        "seamless loops",
    ],
    negative_tokens=[
        "photorealistic",
        "3D render",
        "organic textures",
        "noise",
        "grain",
        "low quality",
        "amateur",
    ],
    continuity_tokens=[
        "consistent color scheme",
        "matching motion style",
        "unified visual language",
    ],
    aspect_ratio=AspectRatio.VERTICAL_9_16,
    default_duration_per_scene=4.0,
    color_palette=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"],
    camera_style="smooth zooms, seamless shape morphing",
    generation_params={
        "guidance_scale": 8.0,
        "num_inference_steps": 40,
    },
)

CINEMATIC_REALISM = StylePreset(
    name="CINEMATIC_REALISM",
    display_name="Cinematic Realism",
    description="Photorealistic cinematic footage with film grain and dramatic lighting",
    style_tokens=[
        "cinematic",
        "photorealistic",
        "35mm film",
        "shallow depth of field",
        "dramatic lighting",
        "golden hour",
        "anamorphic lens flare",
        "film grain",
        "color graded",
        "high production value",
        "blockbuster movie quality",
    ],
    negative_tokens=[
        "cartoon",
        "anime",
        "illustration",
        "low quality",
        "amateur",
        "stock footage",
        "oversaturated",
    ],
    continuity_tokens=[
        "consistent lighting direction",
        "matching color temperature",
        "same camera movement style",
        "unified film look",
    ],
    aspect_ratio=AspectRatio.VERTICAL_9_16,
    default_duration_per_scene=6.0,
    color_palette=["#2C3E50", "#E74C3C", "#F39C12", "#ECF0F1", "#1ABC9C"],
    camera_style="cinematic tracking shots, smooth dollys, epic reveals",
    generation_params={
        "guidance_scale": 7.0,
        "num_inference_steps": 50,
    },
)

SURREAL_DREAMSCAPE = StylePreset(
    name="SURREAL_DREAMSCAPE",
    display_name="Surreal Dreamscape",
    description="Ethereal, dream-like visuals with impossible geometry and soft glow",
    style_tokens=[
        "surrealist",
        "dreamlike",
        "ethereal glow",
        "impossible geometry",
        "floating objects",
        "soft focus",
        "iridescent",
        "otherworldly",
        "Salvador Dali inspired",
        "melting reality",
        "cosmic",
        "bioluminescent",
    ],
    negative_tokens=[
        "realistic",
        "mundane",
        "harsh lighting",
        "sharp edges",
        "low quality",
        "boring",
        "ordinary",
    ],
    continuity_tokens=[
        "consistent dream logic",
        "matching ethereal quality",
        "unified color harmony",
        "same level of surrealism",
    ],
    aspect_ratio=AspectRatio.VERTICAL_9_16,
    default_duration_per_scene=5.5,
    color_palette=["#9B59B6", "#3498DB", "#1ABC9C", "#F1C40F", "#E91E63"],
    camera_style="floating camera, impossible angles, smooth morphing transitions",
    generation_params={
        "guidance_scale": 9.0,
        "num_inference_steps": 60,
    },
)


# =============================================================================
# PRESET REGISTRY
# =============================================================================

PRESETS: dict[str, StylePreset] = {
    "DARK_DYSTOPIAN_ANIME": DARK_DYSTOPIAN_ANIME,
    "VIBRANT_MOTION_GRAPHICS": VIBRANT_MOTION_GRAPHICS,
    "CINEMATIC_REALISM": CINEMATIC_REALISM,
    "SURREAL_DREAMSCAPE": SURREAL_DREAMSCAPE,
}


def get_preset(name: str) -> StylePreset | None:
    """Get a preset by name (case-insensitive).

    Args:
        name: Preset name

    Returns:
        StylePreset if found, None otherwise
    """
    return PRESETS.get(name.upper())


def get_preset_names() -> list[str]:
    """Get list of available preset names."""
    return list(PRESETS.keys())
