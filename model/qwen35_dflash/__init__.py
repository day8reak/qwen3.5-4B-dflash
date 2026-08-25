"""Portable golden reference for the Z-Lab Qwen3.5-4B DFlash drafter."""

from .config import (
    DFLASH_CONFIG,
    DFLASH_TARGET_FEATURE_SIZE,
    DFLASH_TARGET_HIDDEN_SIZE,
    DFLASH_TARGET_LAYER_IDS,
    DFLASH_TARGET_NUM_HIDDEN_LAYERS,
    Qwen35DFlashConfig,
    audit_official_4b_dflash_config,
)
from .model import DFlashDraftModel, extract_context_feature
from .ops import DFlashOps, ModuleDFlashOps, TorchDFlashOps
from .target_features import (
    DFlashBaseModelOutputWithPast,
    DFlashCausalLMOutputWithPast,
    DFlashFeatureCollector,
    DFlashTargetFeatureSpec,
    QWEN35_4B_DFLASH_TARGET_FEATURES,
)
from .weights import audit_dflash_checkpoint

__all__ = [
    "DFlashDraftModel",
    "DFlashOps",
    "DFlashBaseModelOutputWithPast",
    "DFlashCausalLMOutputWithPast",
    "DFlashFeatureCollector",
    "DFlashTargetFeatureSpec",
    "DFLASH_CONFIG",
    "DFLASH_TARGET_FEATURE_SIZE",
    "DFLASH_TARGET_HIDDEN_SIZE",
    "DFLASH_TARGET_LAYER_IDS",
    "DFLASH_TARGET_NUM_HIDDEN_LAYERS",
    "ModuleDFlashOps",
    "QWEN35_4B_DFLASH_TARGET_FEATURES",
    "Qwen35DFlashConfig",
    "TorchDFlashOps",
    "audit_dflash_checkpoint",
    "audit_official_4b_dflash_config",
    "extract_context_feature",
]
