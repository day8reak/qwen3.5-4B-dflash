"""Accuracy-first Qwen3.5-4B MTP reference and backend contracts."""

from .config import Qwen35MTPConfig
from .generation import GenerationResult, ordinary_generate, speculative_generate
from .mtp import Qwen35MTPDrafter

__all__ = [
    "GenerationResult",
    "Qwen35MTPConfig",
    "Qwen35MTPDrafter",
    "ordinary_generate",
    "speculative_generate",
]
