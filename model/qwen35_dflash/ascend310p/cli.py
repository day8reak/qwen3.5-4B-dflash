"""Command line entry point for DFlash AIR, OM, and prompt inference."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

import torch

from .cpp_runtime import build_cpp_runner, run_cpp_pair, write_cpp_prompt_report
from .compiler import (
    compile_air_bundle,
    resolve_atc_executable,
    validate_soc_version,
)
from .exporter import export_air_bundle
from .contracts import AirGraphSpec
from .utils import (
    atomic_write_json,
    file_record,
    load_json_object,
    require_run_output,
    resolve_callable,
)
from .workflow import (
    DEFAULT_BACKEND_FACTORY,
    DEFAULT_GRAPH_FACTORY,
    load_tokenizer,
    run_cpp_target_pipeline,
    run_declared_target_preflight,
    run_om_inference,
    run_target_pipeline,
)
from .generation import tokenize_prompt


def _config(path: Path | None) -> dict[str, Any]:
    return {} if path is None else load_json_object(path)


def _print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def command_export(args: argparse.Namespace) -> int:
    payload = export_air_bundle(
        args.factory,
        _config(args.factory_config),
        args.bundle_dir,
    )
    _print(payload)
    return 0


def command_compile(args: argparse.Namespace) -> int:
    payload = compile_air_bundle(
        args.air_manifest,
        soc_version=args.soc_version,
        atc_bin=args.atc,
        extra_args=args.atc_arg,
    )
    _print(payload)
    return 0


def command_build(args: argparse.Namespace) -> int:
    # Resolve every cheap target prerequisite before the factory loads the two
    # 4B checkpoints.
    atc_path = resolve_atc_executable(args.atc)
    exact_soc_version = validate_soc_version(args.soc_version)
    exported = export_air_bundle(
        args.factory,
        _config(args.factory_config),
        args.bundle_dir,
    )
    payload = compile_air_bundle(
        Path(exported["manifest_path"]),
        soc_version=exact_soc_version,
        atc_bin=atc_path,
        extra_args=args.atc_arg,
    )
    _print(payload)
    return 0


def command_build_cpp(args: argparse.Namespace) -> int:
    payload = build_cpp_runner(
        build_dir=args.build_dir,
        output=args.output,
        cmake=args.cmake,
        ascendcl_root=args.ascendcl_root,
    )
    _print(payload)
    return 0


def _synchronize_tensor_device(value: torch.Tensor) -> None:
    if value.device.type == "npu":
        torch.npu.synchronize(value.device)  # type: ignore[attr-defined]
    elif value.device.type == "cuda":
        torch.cuda.synchronize(value.device)


def command_probe(args: argparse.Namespace) -> int:
    """Run one real-weight PyTorch graph call without making a target claim."""

    output = require_run_output(args.output)
    tokens = [int(item.strip()) for item in args.input_token_ids.split(",")]
    if not tokens or any(token < 0 for token in tokens):
        raise ValueError("input-token-ids must contain non-negative integers")
    torch.set_num_threads(max(1, int(args.threads)))
    factory = resolve_callable(args.factory)
    load_start = time.perf_counter_ns()
    value = factory(_config(args.factory_config))
    specs = (value,) if isinstance(value, AirGraphSpec) else tuple(value)
    if len(specs) != 1 or specs[0].role != "generation-recompute":
        raise ValueError("PyTorch probe requires one generation-recompute graph")
    spec = specs[0]
    if spec.input_names != ("input_ids", "attention_mask"):
        raise ValueError("PyTorch probe graph has an incompatible input ABI")
    example_ids, example_mask = spec.example_args
    if len(tokens) > example_ids.shape[1]:
        raise ValueError("probe tokens exceed the graph's fixed sequence gear")
    input_ids = example_ids.clone()
    attention_mask = torch.zeros_like(example_mask)
    input_ids[:, : len(tokens)] = torch.tensor(
        tokens, dtype=input_ids.dtype, device=input_ids.device
    )
    attention_mask[:, : len(tokens)] = 1
    load_end = time.perf_counter_ns()
    _synchronize_tensor_device(input_ids)
    forward_start = time.perf_counter_ns()
    with torch.inference_mode():
        target_top1, draft_top1 = spec.model(input_ids, attention_mask)
    _synchronize_tensor_device(input_ids)
    forward_end = time.perf_counter_ns()
    if target_top1.dtype != torch.long or draft_top1.dtype != torch.long:
        raise RuntimeError("integrated graph Top1 outputs must use int64")
    if (target_top1 < 0).any() or (draft_top1 < 0).any():
        raise RuntimeError("integrated graph returned a negative token ID")
    report = {
        "schema_version": 1,
        "status": "PASS",
        "scope": "real-weight PyTorch integrated-graph probe",
        "cpu_fallback": input_ids.device.type == "cpu",
        "device": str(input_ids.device),
        "input_token_ids": tokens,
        "fixed_sequence_length": int(input_ids.shape[1]),
        "ordinary_next_token_id": int(target_top1[0, len(tokens) - 1].item()),
        "draft_token_ids": [int(item) for item in draft_top1[0].tolist()],
        "output_shapes": {
            "target_top1": list(target_top1.shape),
            "draft_top1": list(draft_top1.shape),
        },
        "graph_metadata": dict(spec.metadata),
        "timing_ms": {
            "load": (load_end - load_start) / 1_000_000.0,
            "forward": (forward_end - forward_start) / 1_000_000.0,
        },
        "claim_boundary": (
            "CPU is simulation evidence only; this report is not AIR, OM, "
            "Ascend 310P accuracy, or target latency evidence."
        ),
    }
    atomic_write_json(output, report)
    _print(report)
    return 0


def command_infer(args: argparse.Namespace) -> int:
    output = require_run_output(args.output)
    payload = run_om_inference(
        deployment_manifest=args.deployment_manifest,
        backend_factory=args.backend,
        backend_options=_config(args.backend_config),
        prompt=args.prompt,
        chat=args.chat,
        device_id=args.device_id,
        max_new_tokens=args.max_new_tokens,
        max_draft_tokens=args.max_draft_tokens,
        warmup=args.warmup,
        repetitions=args.repetitions,
        model_dir=args.model_dir,
        model_asset_id=args.model_asset_id,
        ordinary_reference=args.ordinary_reference,
        allow_simulation=args.allow_simulation,
    )
    atomic_write_json(output, payload)
    _print(payload)
    return 0


def _eos_token_ids(tokenizer: Any) -> tuple[int, ...]:
    value = getattr(tokenizer, "eos_token_id", None)
    if value is None:
        return ()
    if isinstance(value, int):
        return (int(value),)
    return tuple(int(item) for item in value)


def command_infer_cpp(args: argparse.Namespace) -> int:
    """Tokenize once, then run paired 3+10 generation in one C++ process."""

    output = require_run_output(args.output)
    if output.exists():
        raise FileExistsError(f"C++ prompt report already exists: {output}")
    preflight_log = run_declared_target_preflight()
    tokenizer, tokenizer_source = load_tokenizer(
        model_dir=args.model_dir,
        model_asset_id=args.model_asset_id,
    )
    tokenize_start = time.perf_counter_ns()
    prompt_ids = tokenize_prompt(tokenizer, args.prompt, chat=args.chat)
    tokenize_end = time.perf_counter_ns()
    raw_output = output.with_name(f"{output.stem}-runner-raw.json")
    run_root = Path(os.environ["AI_RUN_DIR"]).expanduser().resolve()
    log_output = run_root / "log" / f"{output.stem}-cpp-runner.log"
    payload = run_cpp_pair(
        deployment_manifest=args.deployment_manifest,
        runner=args.runner,
        runner_options=_config(args.runner_config),
        prompt_token_ids=prompt_ids,
        eos_token_ids=_eos_token_ids(tokenizer),
        device_id=args.device_id,
        max_new_tokens=args.max_new_tokens,
        max_draft_tokens=args.max_draft_tokens,
        raw_output=raw_output,
        log_output=log_output,
    )
    payload["control_plane"]["target_preflight"] = file_record(
        preflight_log, relative_to=run_root
    )
    generated = [int(item) for item in payload["dflash"]["stable_generated_token_ids"]]
    detokenize_start = time.perf_counter_ns()
    text_output = tokenizer.decode(generated, skip_special_tokens=True)
    detokenize_end = time.perf_counter_ns()
    report = write_cpp_prompt_report(
        payload=payload,
        output=output,
        prompt=args.prompt,
        chat=args.chat,
        tokenizer_source=tokenizer_source,
        tokenize_ms=(tokenize_end - tokenize_start) / 1_000_000.0,
        detokenize_ms=(detokenize_end - detokenize_start) / 1_000_000.0,
        generated_text=text_output,
    )
    _print(report)
    return 0


def command_run_e2e(args: argparse.Namespace) -> int:
    payload = run_target_pipeline(
        factory=args.factory,
        factory_config=_config(args.factory_config),
        bundle_dir=args.bundle_dir,
        soc_version=args.soc_version,
        atc_bin=args.atc,
        atc_args=args.atc_arg,
        backend_factory=args.backend,
        ordinary_backend_options=_config(args.ordinary_backend_config),
        dflash_backend_options=_config(args.dflash_backend_config),
        report_dir=args.report_dir,
        prompt=args.prompt,
        chat=args.chat,
        device_id=args.device_id,
        max_new_tokens=args.max_new_tokens,
        max_draft_tokens=args.max_draft_tokens,
        model_dir=args.model_dir,
        model_asset_id=args.model_asset_id,
    )
    _print(payload)
    return 0


def command_run_e2e_cpp(args: argparse.Namespace) -> int:
    payload = run_cpp_target_pipeline(
        factory=args.factory,
        factory_config=_config(args.factory_config),
        bundle_dir=args.bundle_dir,
        soc_version=args.soc_version,
        atc_bin=args.atc,
        atc_args=args.atc_arg,
        runner=args.runner,
        runner_options=_config(args.runner_config),
        report_dir=args.report_dir,
        prompt=args.prompt,
        chat=args.chat,
        device_id=args.device_id,
        max_new_tokens=args.max_new_tokens,
        max_draft_tokens=args.max_draft_tokens,
        model_dir=args.model_dir,
        model_asset_id=args.model_asset_id,
    )
    _print(payload)
    return 0


def _add_atc_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--atc",
        type=Path,
        default=os.environ.get("ASCEND310P_ATC_BIN"),
        help="ATC executable from the declared target profile",
    )
    parser.add_argument(
        "--soc-version",
        default=os.environ.get("SOC_VERSION"),
        required=os.environ.get("SOC_VERSION") is None,
        help="exact ATC SoC identity, for example Ascend310P3",
    )
    parser.add_argument(
        "--atc-arg",
        action="append",
        default=[],
        help="additional non-core ATC option, repeatable",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen3.5 DFlash TorchAir -> AIR -> OM -> Ascend 310P framework"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export-air", help="export factory graphs to AIR")
    export.add_argument("--factory", required=True, help="module:function graph factory")
    export.add_argument("--factory-config", type=Path)
    export.add_argument("--bundle-dir", type=Path, required=True)
    export.set_defaults(handler=command_export)

    compile_parser = subparsers.add_parser(
        "compile-om", help="compile a hash-locked AIR bundle with ATC"
    )
    compile_parser.add_argument("--air-manifest", type=Path, required=True)
    _add_atc_arguments(compile_parser)
    compile_parser.set_defaults(handler=command_compile)

    build = subparsers.add_parser("build-om", help="export AIR and compile every graph")
    build.add_argument("--factory", required=True, help="module:function graph factory")
    build.add_argument("--factory-config", type=Path)
    build.add_argument("--bundle-dir", type=Path, required=True)
    _add_atc_arguments(build)
    build.set_defaults(handler=command_build)

    build_cpp = subparsers.add_parser(
        "build-cpp", help="build and host-test the low-overhead AscendCL C++ runner"
    )
    build_cpp.add_argument("--build-dir", type=Path, required=True)
    build_cpp.add_argument("--output", type=Path, required=True)
    build_cpp.add_argument("--cmake", default="cmake")
    build_cpp.add_argument(
        "--ascendcl-root",
        type=Path,
        help="active CANN toolkit root; otherwise use declared environment variables",
    )
    build_cpp.set_defaults(handler=command_build_cpp)

    probe = subparsers.add_parser(
        "probe-pytorch", help="run one integrated real-weight PyTorch graph probe"
    )
    probe.add_argument(
        "--factory",
        default=(
            "qwen35_dflash.ascend310p.factories:"
            "create_integrated_recompute_graph"
        ),
    )
    probe.add_argument("--factory-config", type=Path, required=True)
    probe.add_argument("--input-token-ids", required=True)
    probe.add_argument("--threads", type=int, default=16)
    probe.add_argument("--output", type=Path, required=True)
    probe.set_defaults(handler=command_probe)

    infer = subparsers.add_parser(
        "infer-om", help="run prompt-to-text generation and stage timing"
    )
    infer.add_argument("--deployment-manifest", type=Path, required=True)
    infer.add_argument("--backend", required=True, help="module:create_backend")
    infer.add_argument("--backend-config", type=Path)
    tokenizer_source = infer.add_mutually_exclusive_group()
    tokenizer_source.add_argument("--model-dir", type=Path)
    tokenizer_source.add_argument(
        "--model-asset-id",
        help="project-declared locked tokenizer asset (default: qwen3.5-4b)",
    )
    infer.add_argument("--prompt", required=True)
    infer.add_argument("--chat", action="store_true")
    infer.add_argument("--device-id", type=int, default=0)
    infer.add_argument("--max-new-tokens", type=int, default=32)
    infer.add_argument("--max-draft-tokens", type=int, default=16)
    infer.add_argument("--warmup", type=int, default=3)
    infer.add_argument("--repetitions", type=int, default=10)
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument(
        "--ordinary-reference",
        type=Path,
        help="ordinary-greedy infer-om report from the same OM bundle",
    )
    infer.add_argument(
        "--allow-simulation",
        action="store_true",
        help="allow non-310P metadata for control-flow tests; CPU fallback is still forbidden",
    )
    infer.set_defaults(handler=command_infer)

    infer_cpp = subparsers.add_parser(
        "infer-cpp",
        help="run paired ordinary/DFlash prompt generation in the C++ ACL hot path",
    )
    infer_cpp.add_argument("--deployment-manifest", type=Path, required=True)
    infer_cpp.add_argument("--runner", type=Path, required=True)
    infer_cpp.add_argument("--runner-config", type=Path, required=True)
    infer_cpp_tokenizer = infer_cpp.add_mutually_exclusive_group()
    infer_cpp_tokenizer.add_argument("--model-dir", type=Path)
    infer_cpp_tokenizer.add_argument(
        "--model-asset-id",
        help="project-declared locked tokenizer asset (default: qwen3.5-4b)",
    )
    infer_cpp.add_argument("--prompt", required=True)
    infer_cpp.add_argument("--chat", action="store_true")
    infer_cpp.add_argument("--device-id", type=int, default=0)
    infer_cpp.add_argument("--max-new-tokens", type=int, default=32)
    infer_cpp.add_argument("--max-draft-tokens", type=int, default=15)
    infer_cpp.add_argument("--output", type=Path, required=True)
    infer_cpp.set_defaults(handler=command_infer_cpp)

    run_e2e = subparsers.add_parser(
        "run-e2e",
        help="export AIR, compile OM, and run gated ordinary/DFlash target timing",
    )
    run_e2e.add_argument(
        "--factory",
        default=DEFAULT_GRAPH_FACTORY,
        help="module:function graph factory",
    )
    run_e2e.add_argument("--factory-config", type=Path, required=True)
    run_e2e.add_argument("--bundle-dir", type=Path, required=True)
    _add_atc_arguments(run_e2e)
    run_e2e.add_argument(
        "--backend",
        default=DEFAULT_BACKEND_FACTORY,
        help="module:create_backend",
    )
    run_e2e.add_argument(
        "--ordinary-backend-config", type=Path, required=True
    )
    run_e2e.add_argument("--dflash-backend-config", type=Path, required=True)
    run_e2e_tokenizer = run_e2e.add_mutually_exclusive_group()
    run_e2e_tokenizer.add_argument("--model-dir", type=Path)
    run_e2e_tokenizer.add_argument(
        "--model-asset-id",
        help="project-declared locked tokenizer asset (default: qwen3.5-4b)",
    )
    run_e2e.add_argument("--prompt", required=True)
    run_e2e.add_argument("--chat", action="store_true")
    run_e2e.add_argument("--device-id", type=int, default=0)
    run_e2e.add_argument("--max-new-tokens", type=int, default=32)
    run_e2e.add_argument("--max-draft-tokens", type=int, default=15)
    run_e2e.add_argument("--report-dir", type=Path, required=True)
    run_e2e.set_defaults(handler=command_run_e2e)

    run_e2e_cpp = subparsers.add_parser(
        "run-e2e-cpp",
        help="export/compile and run paired 3+10 in the C++ AscendCL hot path",
    )
    run_e2e_cpp.add_argument(
        "--factory", default=DEFAULT_GRAPH_FACTORY, help="module:function graph factory"
    )
    run_e2e_cpp.add_argument("--factory-config", type=Path, required=True)
    run_e2e_cpp.add_argument("--bundle-dir", type=Path, required=True)
    _add_atc_arguments(run_e2e_cpp)
    run_e2e_cpp.add_argument("--runner", type=Path, required=True)
    run_e2e_cpp.add_argument("--runner-config", type=Path, required=True)
    run_e2e_cpp_tokenizer = run_e2e_cpp.add_mutually_exclusive_group()
    run_e2e_cpp_tokenizer.add_argument("--model-dir", type=Path)
    run_e2e_cpp_tokenizer.add_argument(
        "--model-asset-id",
        help="project-declared locked tokenizer asset (default: qwen3.5-4b)",
    )
    run_e2e_cpp.add_argument("--prompt", required=True)
    run_e2e_cpp.add_argument("--chat", action="store_true")
    run_e2e_cpp.add_argument("--device-id", type=int, default=0)
    run_e2e_cpp.add_argument("--max-new-tokens", type=int, default=32)
    run_e2e_cpp.add_argument("--max-draft-tokens", type=int, default=15)
    run_e2e_cpp.add_argument("--report-dir", type=Path, required=True)
    run_e2e_cpp.set_defaults(handler=command_run_e2e_cpp)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))
