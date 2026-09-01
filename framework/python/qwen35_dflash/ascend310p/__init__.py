"""Ascend 310P AIR/OM deployment framework for Qwen3.5 DFlash."""

from .acl_runtime import AclOmRuntime
from .compiler import compile_air_bundle, resolve_atc_executable, validate_soc_version
from .contracts import AirGraphSpec, CustomOpExportSpec, GenerationStep
from .cpp_runtime import build_cpp_runner, run_cpp_pair
from .exporter import export_air_bundle
from .generation import benchmark_prompt, generate_prompt
from .integrated import IntegratedDFlashRecomputeGraph, integrated_recompute_graph_spec
from .input_manifest import build_quant_input_manifest, verify_quant_input_manifest
from .incremental import ExactAcceptCommitStateGraph
from .incremental_graphs import (
    DraftProposeStateGraph,
    TargetDecodeOneStateGraph,
    TargetPrefillHeadGraph,
    TargetPrefillStateGraph,
    TargetVerifyCommitStateGraph,
)
from .quant_factory import (
    create_quant_incremental_state_graphs,
    create_quant_recompute_graph,
)
from .recompute_backend import RecomputeDFlashOmBackend
from .workflow import run_cpp_target_pipeline, run_om_inference, run_target_pipeline

__all__ = [
    "AclOmRuntime",
    "AirGraphSpec",
    "CustomOpExportSpec",
    "GenerationStep",
    "ExactAcceptCommitStateGraph",
    "DraftProposeStateGraph",
    "IntegratedDFlashRecomputeGraph",
    "RecomputeDFlashOmBackend",
    "TargetDecodeOneStateGraph",
    "TargetPrefillHeadGraph",
    "TargetPrefillStateGraph",
    "TargetVerifyCommitStateGraph",
    "benchmark_prompt",
    "build_cpp_runner",
    "build_quant_input_manifest",
    "compile_air_bundle",
    "create_quant_recompute_graph",
    "create_quant_incremental_state_graphs",
    "export_air_bundle",
    "generate_prompt",
    "integrated_recompute_graph_spec",
    "resolve_atc_executable",
    "run_cpp_pair",
    "run_cpp_target_pipeline",
    "run_om_inference",
    "run_target_pipeline",
    "validate_soc_version",
    "verify_quant_input_manifest",
]
