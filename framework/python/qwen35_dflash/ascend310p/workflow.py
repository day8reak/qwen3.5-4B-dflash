"""One-command AIR -> OM -> prompt inference workflow for Ascend 310P."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Mapping, Sequence

from transformers import AutoTokenizer

from .cpp_runtime import (
    preflight_cpp_runner,
    run_cpp_pair,
    validate_cpp_runner_options,
    write_cpp_prompt_report,
)
from .compiler import (
    compile_air_bundle,
    resolve_atc_executable,
    validate_soc_version,
)
from .exporter import export_air_bundle
from .generation import (
    benchmark_prompt,
    load_backend,
    tokenize_prompt,
    verify_ordinary_reference,
)
from .utils import (
    atomic_write_json,
    file_record,
    load_json_object,
    require_run_output,
)


DEFAULT_GRAPH_FACTORY = (
    "qwen35_dflash.ascend310p.quant_factory:create_quant_recompute_graph"
)
DEFAULT_BACKEND_FACTORY = (
    "qwen35_dflash.ascend310p.recompute_backend:create_backend"
)
_TARGET_WARMUP = 3
_TARGET_REPETITIONS = 10
_SUMMARY_LATENCIES = ("prefill", "decode", "model_total", "end_to_end")


def _require_importable(module_name: str, purpose: str) -> None:
    try:
        importlib.import_module(module_name)
    except Exception as error:
        raise RuntimeError(
            f"{module_name} is required for {purpose}; activate the declared "
            "CANN/Ascend 310P target environment"
        ) from error


def _run_declared_target_preflight() -> Path:
    preflight_value = os.environ.get("AI_TARGET_PREFLIGHT")
    model_root_value = os.environ.get("AI_MODEL_ROOT")
    if bool(preflight_value) != bool(model_root_value):
        raise RuntimeError(
            "AI_TARGET_PREFLIGHT and AI_MODEL_ROOT must be configured together"
        )
    if preflight_value and model_root_value:
        preflight = Path(preflight_value).expanduser().resolve()
        model_root = Path(model_root_value).expanduser().resolve()
        if not preflight.is_file() or not os.access(preflight, os.X_OK):
            raise RuntimeError(
                f"declared target preflight is not executable: {preflight}"
            )
        command = [
            str(preflight),
            str(model_root),
            "--require-model-python",
            "--require-atc",
            "--require-device",
        ]
    else:
        npu_smi = shutil.which("npu-smi")
        if npu_smi is None:
            raise RuntimeError(
                "standalone target preflight requires npu-smi in PATH; "
                "CPU/simulation fallback is not accepted"
            )
        device_nodes = sorted(Path("/dev").glob("davinci[0-9]*"))
        if not device_nodes:
            raise RuntimeError(
                "standalone target preflight found no /dev/davinciN device"
            )
        command = [npu_smi, "info"]
    result = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    run_root = Path(os.environ["AI_RUN_DIR"]).expanduser().resolve()
    log_path = require_run_output(run_root / "log" / "dflash-run-e2e-preflight.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout or "", encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            "Ascend 310P preflight failed before checkpoint load: "
            f"exit={result.returncode}, log={log_path}"
        )
    return log_path


def run_declared_target_preflight() -> Path:
    """Run the manifest-declared strict target preflight and retain its log."""

    return _run_declared_target_preflight()


def preflight_target_pipeline(
    *,
    factory: str,
    factory_config: Mapping[str, Any],
    backend: str,
    atc_bin: str | Path | None,
    soc_version: str,
) -> tuple[Path, str, Path]:
    """Fail before loading checkpoints when the target toolchain is incomplete."""

    exact_soc_version = validate_soc_version(soc_version)
    atc_path = resolve_atc_executable(atc_bin)
    preflight_log = _run_declared_target_preflight()
    _require_importable("torchair", "AIR export")
    if factory == DEFAULT_GRAPH_FACTORY:
        device = str(factory_config.get("device", "")).strip().lower()
        if device != "npu" and not device.startswith("npu:"):
            raise ValueError(
                "the built-in target pipeline factory requires an explicit NPU device"
            )
        _require_importable("torch_npu", "the built-in target graph")
    if backend == DEFAULT_BACKEND_FACTORY:
        _require_importable("acl", "the built-in pyACL OM backend")
    return atc_path, exact_soc_version, preflight_log


def validate_backend_pair(
    ordinary_options: Mapping[str, Any],
    dflash_options: Mapping[str, Any],
) -> None:
    """Require one runtime identity whose only mode difference is ordinary_only."""

    ordinary = dict(ordinary_options)
    dflash = dict(dflash_options)
    if ordinary.get("ordinary_only") is not True:
        raise ValueError("ordinary backend config must set ordinary_only=true")
    if dflash.get("ordinary_only") is not False:
        raise ValueError("DFlash backend config must set ordinary_only=false")
    ordinary.pop("ordinary_only")
    dflash.pop("ordinary_only")
    if ordinary != dflash:
        differing = sorted(
            key
            for key in set(ordinary) | set(dflash)
            if ordinary.get(key) != dflash.get(key)
        )
        raise ValueError(
            "ordinary and DFlash backend configs may differ only in ordinary_only; "
            f"different fields={differing}"
        )


def load_tokenizer(
    *,
    model_dir: str | Path | None = None,
    model_asset_id: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load a local Target tokenizer and return its auditable source identity."""

    if model_dir is not None and model_asset_id is not None:
        raise ValueError("model_dir and model_asset_id are mutually exclusive")
    if model_asset_id is not None:
        raise ValueError(
            "the standalone quant branch does not resolve workspace asset IDs; "
            "pass --model-dir from the locked input manifest"
        )
    if model_dir is None:
        raise ValueError("the quant AIR/OM runtime requires --model-dir")
    source_dir = Path(model_dir).expanduser().resolve()
    if source_dir.is_symlink() or not source_dir.is_dir():
        raise FileNotFoundError(f"model_dir is not a regular directory: {source_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        source_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    source = {
        "path": str(source_dir),
        "asset_id": None,
        "manifest_sha256": None,
    }
    return tokenizer, source


def _run_backend_report(
    *,
    deployment_manifest: str | Path,
    backend_factory: str,
    backend_options: Mapping[str, Any],
    tokenizer: Any,
    tokenizer_source: Mapping[str, Any],
    prompt: str,
    chat: bool,
    device_id: int,
    max_new_tokens: int,
    max_draft_tokens: int,
    warmup: int,
    repetitions: int,
    ordinary_reference: str | Path | None,
    allow_simulation: bool,
) -> dict[str, Any]:
    backend = load_backend(
        backend_factory,
        deployment_manifest,
        device_id=device_id,
        options=backend_options,
    )
    try:
        generation_mode = dict(backend.metadata()).get("generation_mode")
        if (
            generation_mode != "ordinary-greedy"
            and ordinary_reference is None
            and not allow_simulation
        ):
            raise ValueError(
                "target DFlash inference requires an ordinary reference from the same OM bundle"
            )
        payload = benchmark_prompt(
            backend,
            tokenizer,
            prompt,
            chat=chat,
            max_new_tokens=max_new_tokens,
            max_draft_tokens=max_draft_tokens,
            warmup=warmup,
            repetitions=repetitions,
            require_target=not allow_simulation,
        )
    finally:
        backend.close()
    payload["tokenizer_source"] = dict(tokenizer_source)
    generation_mode = payload.get("backend_metadata", {}).get("generation_mode")
    if generation_mode == "ordinary-greedy":
        payload["report_kind"] = "ordinary-greedy-reference"
    elif ordinary_reference is not None:
        reference_path = Path(ordinary_reference).expanduser().resolve()
        payload["ordinary_parity"] = verify_ordinary_reference(
            payload,
            load_json_object(reference_path),
            reference_path=reference_path,
        )
        payload["report_kind"] = "dflash-strict-greedy-target"
    else:
        payload["report_kind"] = "dflash-control-flow-simulation"
    return payload


def run_om_inference(
    *,
    deployment_manifest: str | Path,
    backend_factory: str,
    backend_options: Mapping[str, Any],
    prompt: str,
    chat: bool = False,
    device_id: int = 0,
    max_new_tokens: int = 32,
    max_draft_tokens: int = 15,
    warmup: int = _TARGET_WARMUP,
    repetitions: int = _TARGET_REPETITIONS,
    model_dir: str | Path | None = None,
    model_asset_id: str | None = None,
    ordinary_reference: str | Path | None = None,
    allow_simulation: bool = False,
) -> dict[str, Any]:
    tokenizer, source = load_tokenizer(
        model_dir=model_dir,
        model_asset_id=model_asset_id,
    )
    return _run_backend_report(
        deployment_manifest=deployment_manifest,
        backend_factory=backend_factory,
        backend_options=backend_options,
        tokenizer=tokenizer,
        tokenizer_source=source,
        prompt=prompt,
        chat=chat,
        device_id=device_id,
        max_new_tokens=max_new_tokens,
        max_draft_tokens=max_draft_tokens,
        warmup=warmup,
        repetitions=repetitions,
        ordinary_reference=ordinary_reference,
        allow_simulation=allow_simulation,
    )


def _latency_view(report: Mapping[str, Any]) -> dict[str, Any]:
    latency = report.get("latency_ms")
    if not isinstance(latency, Mapping):
        raise ValueError("target report has no latency summary")
    result = {}
    for name in _SUMMARY_LATENCIES:
        value = latency.get(name)
        if not isinstance(value, Mapping):
            raise ValueError(f"target report has no {name} latency summary")
        result[name] = dict(value)
    return result


def _speedup_view(
    ordinary: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, float | None]:
    ordinary_latency = _latency_view(ordinary)
    candidate_latency = _latency_view(candidate)
    result: dict[str, float | None] = {}
    for name in _SUMMARY_LATENCIES:
        baseline = float(ordinary_latency[name]["median"])
        value = float(candidate_latency[name]["median"])
        result[name] = None if value <= 0.0 else baseline / value
    return result


def run_target_pipeline(
    *,
    factory: str,
    factory_config: Mapping[str, Any],
    bundle_dir: str | Path,
    soc_version: str,
    atc_bin: str | Path | None,
    atc_args: Sequence[str],
    backend_factory: str,
    ordinary_backend_options: Mapping[str, Any],
    dflash_backend_options: Mapping[str, Any],
    report_dir: str | Path,
    prompt: str,
    chat: bool = False,
    device_id: int = 0,
    max_new_tokens: int = 32,
    max_draft_tokens: int = 15,
    model_dir: str | Path | None = None,
    model_asset_id: str | None = None,
) -> dict[str, Any]:
    """Build one OM and produce gated ordinary and DFlash 3+10 reports."""

    validate_backend_pair(ordinary_backend_options, dflash_backend_options)
    bundle_root = require_run_output(bundle_dir)
    report_root = require_run_output(report_dir)
    if (
        bundle_root == report_root
        or bundle_root in report_root.parents
        or report_root in bundle_root.parents
    ):
        raise ValueError("bundle_dir and report_dir must not overlap")
    if report_root.exists() and any(report_root.iterdir()):
        raise FileExistsError(f"target report directory is not empty: {report_root}")
    atc_path, exact_soc_version, preflight_log = preflight_target_pipeline(
        factory=factory,
        factory_config=factory_config,
        backend=backend_factory,
        atc_bin=atc_bin,
        soc_version=soc_version,
    )

    exported = export_air_bundle(factory, factory_config, bundle_root)
    deployment = compile_air_bundle(
        Path(exported["manifest_path"]),
        soc_version=exact_soc_version,
        atc_bin=atc_path,
        extra_args=atc_args,
    )
    deployment_manifest = Path(deployment["manifest_path"])
    tokenizer, tokenizer_source = load_tokenizer(
        model_dir=model_dir,
        model_asset_id=model_asset_id,
    )
    report_root.mkdir(parents=True, exist_ok=True)
    ordinary_path = report_root / "ordinary.json"
    ordinary = _run_backend_report(
        deployment_manifest=deployment_manifest,
        backend_factory=backend_factory,
        backend_options=ordinary_backend_options,
        tokenizer=tokenizer,
        tokenizer_source=tokenizer_source,
        prompt=prompt,
        chat=chat,
        device_id=device_id,
        max_new_tokens=max_new_tokens,
        max_draft_tokens=max_draft_tokens,
        warmup=_TARGET_WARMUP,
        repetitions=_TARGET_REPETITIONS,
        ordinary_reference=None,
        allow_simulation=False,
    )
    if ordinary.get("report_kind") != "ordinary-greedy-reference":
        raise RuntimeError("ordinary backend did not produce an ordinary-greedy report")
    atomic_write_json(ordinary_path, ordinary)

    dflash_path = report_root / "dflash.json"
    dflash = _run_backend_report(
        deployment_manifest=deployment_manifest,
        backend_factory=backend_factory,
        backend_options=dflash_backend_options,
        tokenizer=tokenizer,
        tokenizer_source=tokenizer_source,
        prompt=prompt,
        chat=chat,
        device_id=device_id,
        max_new_tokens=max_new_tokens,
        max_draft_tokens=max_draft_tokens,
        warmup=_TARGET_WARMUP,
        repetitions=_TARGET_REPETITIONS,
        ordinary_reference=ordinary_path,
        allow_simulation=False,
    )
    if dflash.get("report_kind") != "dflash-strict-greedy-target":
        raise RuntimeError("DFlash backend did not produce a strict-greedy target report")
    atomic_write_json(dflash_path, dflash)

    run_root = Path(os.environ["AI_RUN_DIR"]).expanduser().resolve()
    graph_artifacts = {}
    for graph in deployment.get("graphs", []):
        graph_name = str(graph["name"])
        graph_artifacts[graph_name] = file_record(
            bundle_root / str(graph["om"]["path"]), relative_to=run_root
        )
    summary = {
        "schema_version": 1,
        "status": "PASS",
        "scope": "AIR -> OM -> Ascend 310P prompt inference",
        "target": dict(deployment["target"]),
        "device": dict(dflash["backend_metadata"]["device"]),
        "runtime_identity": {
            name: dflash["backend_metadata"][name]
            for name in ("cann", "driver", "firmware", "runtime")
        },
        "compiler": dict(deployment["compiler"]),
        "prompt": prompt,
        "chat": bool(chat),
        "limits": {
            "max_new_tokens": max_new_tokens,
            "max_draft_tokens": max_draft_tokens,
        },
        "measurement_protocol": {
            "warmup": _TARGET_WARMUP,
            "repetitions": _TARGET_REPETITIONS,
            "device_synchronization": "before and after every prefill/decode call",
        },
        "output": {
            "token_ids": list(dflash["stable_generated_token_ids"]),
            "text": str(dflash["stable_generated_text"]),
            "stop_reason": str(dflash["stable_stop_reason"]),
        },
        "ordinary_parity": dict(dflash["ordinary_parity"]),
        "latency_ms": {
            "ordinary": _latency_view(ordinary),
            "dflash": _latency_view(dflash),
        },
        "dflash_speedup_over_ordinary_median": _speedup_view(ordinary, dflash),
        "artifacts": {
            "target_preflight": file_record(preflight_log, relative_to=run_root),
            "air_manifest": file_record(
                Path(exported["manifest_path"]), relative_to=run_root
            ),
            "deployment_manifest": file_record(
                deployment_manifest, relative_to=run_root
            ),
            "om": graph_artifacts,
            "ordinary_report": file_record(ordinary_path, relative_to=run_root),
            "dflash_report": file_record(dflash_path, relative_to=run_root),
        },
    }
    summary_path = atomic_write_json(report_root / "summary.json", summary)
    summary["summary_path"] = str(summary_path)
    return summary


def run_cpp_target_pipeline(
    *,
    factory: str,
    factory_config: Mapping[str, Any],
    bundle_dir: str | Path,
    soc_version: str,
    atc_bin: str | Path | None,
    atc_args: Sequence[str],
    runner: str | Path,
    runner_options: Mapping[str, Any],
    report_dir: str | Path,
    prompt: str,
    chat: bool = False,
    device_id: int = 0,
    max_new_tokens: int = 32,
    max_draft_tokens: int = 15,
    model_dir: str | Path | None = None,
    model_asset_id: str | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Build the selected OM topology and run paired 3+10 in the C++ ACL path."""

    validate_cpp_runner_options(runner_options, device_id)
    runner_path = preflight_cpp_runner(runner)
    bundle_root = require_run_output(bundle_dir)
    report_root = require_run_output(report_dir)
    if (
        bundle_root == report_root
        or bundle_root in report_root.parents
        or report_root in bundle_root.parents
    ):
        raise ValueError("bundle_dir and report_dir must not overlap")
    if report_root.exists() and any(report_root.iterdir()):
        raise FileExistsError(f"C++ target report directory is not empty: {report_root}")
    atc_path, exact_soc_version, preflight_log = preflight_target_pipeline(
        factory=factory,
        factory_config=factory_config,
        backend="qwen35-dflash-ascendcl-cpp",
        atc_bin=atc_bin,
        soc_version=soc_version,
    )
    exported = export_air_bundle(factory, factory_config, bundle_root)
    deployment = compile_air_bundle(
        Path(exported["manifest_path"]),
        soc_version=exact_soc_version,
        atc_bin=atc_path,
        extra_args=atc_args,
    )
    tokenizer, tokenizer_source = load_tokenizer(
        model_dir=model_dir,
        model_asset_id=model_asset_id,
    )
    tokenize_start = time.perf_counter_ns()
    prompt_ids = tokenize_prompt(tokenizer, prompt, chat=chat)
    tokenize_end = time.perf_counter_ns()
    eos_value = getattr(tokenizer, "eos_token_id", None)
    if eos_value is None:
        eos_ids: tuple[int, ...] = ()
    elif isinstance(eos_value, int):
        eos_ids = (int(eos_value),)
    else:
        eos_ids = tuple(int(item) for item in eos_value)
    report_root.mkdir(parents=True, exist_ok=True)
    run_root = Path(os.environ["AI_RUN_DIR"]).expanduser().resolve()
    payload = run_cpp_pair(
        deployment_manifest=Path(deployment["manifest_path"]),
        runner=runner_path,
        runner_options=runner_options,
        prompt_token_ids=prompt_ids,
        eos_token_ids=eos_ids,
        device_id=device_id,
        max_new_tokens=max_new_tokens,
        max_draft_tokens=max_draft_tokens,
        raw_output=report_root / "runner-raw.json",
        log_output=run_root / "log" / "dflash-cpp-runner.log",
        progress=progress,
    )
    payload["control_plane"]["target_preflight"] = file_record(
        preflight_log, relative_to=run_root
    )
    payload["control_plane"]["air_manifest"] = file_record(
        Path(exported["manifest_path"]), relative_to=run_root
    )
    generated = [int(item) for item in payload["dflash"]["stable_generated_token_ids"]]
    detokenize_start = time.perf_counter_ns()
    generated_text = tokenizer.decode(generated, skip_special_tokens=True)
    detokenize_end = time.perf_counter_ns()
    summary_path = report_root / "summary.json"
    summary = write_cpp_prompt_report(
        payload=payload,
        output=summary_path,
        prompt=prompt,
        chat=chat,
        tokenizer_source=tokenizer_source,
        tokenize_ms=(tokenize_end - tokenize_start) / 1_000_000.0,
        detokenize_ms=(detokenize_end - detokenize_start) / 1_000_000.0,
        generated_text=generated_text,
    )
    summary["summary_path"] = str(summary_path)
    return summary
