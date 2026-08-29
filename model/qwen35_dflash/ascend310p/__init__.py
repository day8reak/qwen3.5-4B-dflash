"""Ascend 310P AIR/OM deployment framework for Qwen3.5 DFlash."""

from .acl_runtime import AclOmRuntime
from .compiler import compile_air_bundle, resolve_atc_executable, validate_soc_version
from .contracts import AirGraphSpec, GenerationStep
from .cpp_runtime import build_cpp_runner, run_cpp_pair
from .exporter import export_air_bundle
from .generation import benchmark_prompt, generate_prompt
from .integrated import IntegratedDFlashRecomputeGraph, integrated_recompute_graph_spec
from .recompute_backend import RecomputeDFlashOmBackend
from .resources import LockedDataResource, resolve_locked_data
from .target_adapter import TransformersDFlashTargetAdapter
from .workflow import run_cpp_target_pipeline, run_om_inference, run_target_pipeline

__all__ = [
    "AclOmRuntime",
    "AirGraphSpec",
    "GenerationStep",
    "IntegratedDFlashRecomputeGraph",
    "LockedDataResource",
    "RecomputeDFlashOmBackend",
    "TransformersDFlashTargetAdapter",
    "benchmark_prompt",
    "build_cpp_runner",
    "compile_air_bundle",
    "export_air_bundle",
    "generate_prompt",
    "integrated_recompute_graph_spec",
    "resolve_locked_data",
    "resolve_atc_executable",
    "run_cpp_pair",
    "run_cpp_target_pipeline",
    "run_om_inference",
    "run_target_pipeline",
    "validate_soc_version",
]
