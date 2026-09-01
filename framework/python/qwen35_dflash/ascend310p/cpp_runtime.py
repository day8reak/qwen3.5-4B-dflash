"""Control plane for the low-overhead AscendCL C++ paired OM runner."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

from .utils import (
    atomic_write_json,
    contained_path,
    file_record,
    load_json_object,
    require_run_output,
    sha256_file,
)


CPP_RUNNER_ID = "qwen35-dflash-ascendcl-cpp-v1"
_GENERIC_DEVICE_NAMES = {"310p", "ascend310p", "atlas310p"}


def _progress(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[infer-cpp] {message}", file=sys.stderr, flush=True)


def _execute_streaming(
    command: Sequence[str],
    *,
    log_path: Path,
    echo: bool,
) -> subprocess.CompletedProcess[str]:
    """Tee combined child output to its durable log while it is running."""

    chunks: list[str] = []
    with log_path.open("x", encoding="utf-8") as log_stream:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            process.kill()
            process.wait()
            raise RuntimeError("C++ ACL runner stdout pipe was not created")
        for line in process.stdout:
            chunks.append(line)
            log_stream.write(line)
            log_stream.flush()
            if echo:
                sys.stderr.write(line)
                sys.stderr.flush()
        return_code = process.wait()
    return subprocess.CompletedProcess(
        list(command), return_code, stdout="".join(chunks)
    )


def resolve_cpp_runner(path: str | Path) -> Path:
    executable = Path(path).expanduser().resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise RuntimeError(f"C++ ACL runner is not executable: {executable}")
    return executable


def preflight_cpp_runner(path: str | Path) -> Path:
    """Prove that the target binary and its dynamic AscendCL deps can start."""

    executable = resolve_cpp_runner(path)
    result = subprocess.run(
        [str(executable), "--help"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if (
        result.returncode != 0
        or "qwen35_dflash_acl_runner" not in result.stdout
        or "--progress true|false" not in result.stdout
    ):
        raise RuntimeError(
            "C++ ACL runner cannot start or does not support live progress; "
            "rebuild it from the current framework source: "
            f"exit={result.returncode}, output={result.stdout!r}"
        )
    return executable


def _runtime_identity(options: Mapping[str, Any], device_id: int) -> dict[str, Any]:
    required = ("device_model", "cann", "driver", "firmware", "runtime")
    missing = [name for name in required if not str(options.get(name, "")).strip()]
    if missing:
        raise ValueError(f"C++ runner config is missing identities: {missing}")
    model = str(options["device_model"]).strip()
    normalized = re.sub(r"[^a-z0-9]", "", model.lower())
    if normalized in _GENERIC_DEVICE_NAMES:
        raise ValueError("C++ runner config must name the concrete 310P product")
    graph_name = str(options.get("graph_name", "quant_dflash_recompute"))
    pad_token_id = int(options.get("pad_token_id", 0))
    if pad_token_id < 0:
        raise ValueError("C++ runner pad_token_id must be non-negative")
    return {
        "cpu_fallback": False,
        "device": {
            "target_id": "ascend310p",
            "model": model,
            "device_id": int(device_id),
        },
        "cann": str(options["cann"]),
        "driver": str(options["driver"]),
        "firmware": str(options["firmware"]),
        "runtime": str(options["runtime"]),
        "graph_name": graph_name,
        "pad_token_id": pad_token_id,
    }


def validate_cpp_runner_options(
    options: Mapping[str, Any], device_id: int
) -> dict[str, Any]:
    return _runtime_identity(options, device_id)


def build_cpp_runner(
    *,
    build_dir: str | Path,
    output: str | Path,
    cmake: str | Path = "cmake",
    ascendcl_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build the production ACL binary without writing into the model repo."""

    explicit_source = os.environ.get("QWEN35_DFLASH_CPP_SOURCE")
    candidates: list[Path] = []
    if explicit_source:
        candidates.append(Path(explicit_source).expanduser().resolve())
    candidates.append(Path(__file__).resolve().parents[3] / "runtime" / "cpp")
    model_root_value = os.environ.get("AI_MODEL_ROOT")
    if model_root_value:
        candidates.append(
            Path(model_root_value).expanduser().resolve()
            / "targets"
            / "ascend310p"
            / "runtime"
            / "cpp"
        )
    source = next(
        (item for item in candidates if (item / "CMakeLists.txt").is_file()),
        candidates[0],
    )
    if not (source / "CMakeLists.txt").is_file():
        raise FileNotFoundError(
            "C++ runner source is missing; searched: "
            + ", ".join(str(item) for item in candidates)
        )
    build = require_run_output(build_dir)
    if build.exists() and any(build.iterdir()):
        raise FileExistsError(f"C++ runner build directory is not empty: {build}")
    report_path = require_run_output(output)
    if report_path.exists():
        raise FileExistsError(f"C++ runner build report already exists: {report_path}")
    configured = str(cmake)
    cmake_path = Path(configured).expanduser()
    if cmake_path.parent == Path("."):
        resolved = shutil.which(configured)
        if resolved is None:
            raise RuntimeError(f"CMake executable is unavailable: {configured}")
        cmake_path = Path(resolved)
    cmake_path = cmake_path.resolve()
    if not cmake_path.is_file() or not os.access(cmake_path, os.X_OK):
        raise RuntimeError(f"CMake executable is invalid: {cmake_path}")
    build.mkdir(parents=True, exist_ok=True)
    run_root = Path(os.environ["AI_RUN_DIR"]).expanduser().resolve()
    log_root = run_root / "log" / "dflash-cpp-build"
    log_root.mkdir(parents=True, exist_ok=True)
    configure_command = [
        str(cmake_path),
        "-S",
        str(source),
        "-B",
        str(build),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DQWEN35_DFLASH_BUILD_ACL_RUNNER=ON",
        "-DQWEN35_DFLASH_BUILD_TESTS=ON",
    ]
    if ascendcl_root is not None:
        configure_command.append(
            f"-DASCENDCL_ROOT={Path(ascendcl_root).expanduser().resolve()}"
        )
    build_command = [
        str(cmake_path),
        "--build",
        str(build),
        "--config",
        "Release",
        "--parallel",
    ]
    test_command = [
        "ctest",
        "--test-dir",
        str(build),
        "--build-config",
        "Release",
        "--output-on-failure",
    ]
    commands = (
        ("configure", configure_command),
        ("build", build_command),
        ("host-tests", test_command),
    )
    logs: dict[str, dict[str, Any]] = {}
    for name, command in commands:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log_path = log_root / f"{name}.log"
        log_path.write_text(result.stdout or "", encoding="utf-8")
        logs[name] = file_record(log_path, relative_to=run_root)
        if result.returncode != 0:
            raise RuntimeError(
                f"C++ runner {name} failed with exit {result.returncode}; log={log_path}"
            )
    candidates = (
        build / "qwen35_dflash_acl_runner",
        build / "Release" / "qwen35_dflash_acl_runner",
    )
    runner = next((item for item in candidates if item.is_file()), None)
    if runner is None or not os.access(runner, os.X_OK):
        raise RuntimeError("C++ build succeeded but produced no executable ACL runner")
    preflight_cpp_runner(runner)
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "artifact_kind": "qwen35-dflash-ascendcl-cpp-runner",
        "source": str(source),
        "cmake": str(cmake_path),
        "ascendcl_root": (
            None
            if ascendcl_root is None
            else str(Path(ascendcl_root).expanduser().resolve())
        ),
        "runner": file_record(runner, relative_to=run_root),
        "logs": logs,
        "claim_boundary": (
            "Host scheduler and fake-ACL integration tests passed; physical-device "
            "latency is established only by infer-cpp/run-e2e-cpp target reports."
        ),
    }
    atomic_write_json(report_path, payload)
    payload["report_path"] = str(report_path)
    payload["runner_path"] = str(runner)
    return payload


def _resolve_integrated_om(
    deployment_manifest: str | Path,
    *,
    graph_name: str,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    manifest_path = Path(deployment_manifest).expanduser().resolve()
    manifest = load_json_object(manifest_path)
    if manifest.get("artifact_kind") != "qwen35-dflash-ascend310p-om-bundle":
        raise ValueError("C++ runner requires a DFlash Ascend 310P OM bundle")
    if manifest.get("status") != "PASS":
        raise ValueError("deployment manifest is not passing")
    matches = [
        graph
        for graph in manifest.get("graphs", [])
        if isinstance(graph, Mapping) and str(graph.get("name")) == graph_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"deployment manifest needs one {graph_name!r} graph, got {len(matches)}"
        )
    graph = dict(matches[0])
    if graph.get("role") != "generation-recompute":
        raise ValueError("C++ runner graph role must be generation-recompute")
    if list(graph.get("input_names", [])) != ["input_ids", "attention_mask"]:
        raise ValueError("C++ runner OM input order differs from the locked ABI")
    if list(graph.get("output_names", [])) != ["target_top1", "draft_top1"]:
        raise ValueError("C++ runner OM output order differs from the locked ABI")
    record = graph.get("om")
    if not isinstance(record, Mapping):
        raise ValueError("deployment graph has no OM record")
    om_path = contained_path(manifest_path.parent, str(record["path"]))
    if not om_path.is_file() or sha256_file(om_path) != str(record["sha256"]):
        raise ValueError("OM artifact integrity check failed before C++ runner launch")
    return om_path, manifest, graph


def _token_csv(values: Sequence[int]) -> str:
    result = [int(item) for item in values]
    if any(item < 0 for item in result):
        raise ValueError("token IDs must be non-negative")
    return ",".join(str(item) for item in result)


def _validate_mode_report(
    name: str,
    report: Mapping[str, Any],
    *,
    generation_mode: str,
) -> None:
    if report.get("status") != "PASS":
        raise RuntimeError(f"C++ {name} report is not passing")
    if report.get("generation_mode") != generation_mode:
        raise RuntimeError(f"C++ {name} generation mode differs")
    if report.get("warmup") != 3 or report.get("repetitions") != 10:
        raise RuntimeError(f"C++ {name} report is not a strict 3+10 measurement")
    measurements = report.get("measurements")
    if not isinstance(measurements, list) or len(measurements) != 10:
        raise RuntimeError(f"C++ {name} report must retain ten raw measurements")
    stable_tokens = [int(item) for item in report.get("stable_generated_token_ids", [])]
    stable_stop = report.get("stable_stop_reason")
    if not stable_tokens:
        raise RuntimeError(f"C++ {name} report generated no token")
    for measurement in measurements:
        if measurement.get("generated_token_ids") != stable_tokens:
            raise RuntimeError(f"C++ {name} repetitions are not token-stable")
        if measurement.get("stop_reason") != stable_stop:
            raise RuntimeError(f"C++ {name} repetitions have different stop reasons")


def validate_cpp_runner_report(
    report: Mapping[str, Any],
    *,
    prompt_token_ids: Sequence[int],
    om_sha256: str,
    device_id: int,
    max_new_tokens: int,
    max_draft_tokens: int,
) -> None:
    if report.get("status") != "PASS" or report.get("runner_id") != CPP_RUNNER_ID:
        raise RuntimeError("C++ ACL runner did not produce a passing known report")
    if report.get("cpu_fallback") is not False:
        raise RuntimeError("C++ target report indicates CPU fallback")
    if int(report.get("device_id", -1)) != int(device_id):
        raise RuntimeError("C++ runner used a different device ID")
    model = report.get("model")
    if not isinstance(model, Mapping) or model.get("sha256") != om_sha256:
        raise RuntimeError("C++ runner OM hash differs from the deployment manifest")
    if list(report.get("prompt_token_ids", [])) != [int(item) for item in prompt_token_ids]:
        raise RuntimeError("C++ runner prompt token IDs differ")
    limits = report.get("limits", {})
    if int(limits.get("max_new_tokens", -1)) != int(max_new_tokens):
        raise RuntimeError("C++ runner max_new_tokens differs")
    if int(limits.get("max_draft_tokens", -1)) != int(max_draft_tokens):
        raise RuntimeError("C++ runner max_draft_tokens differs")
    protocol = report.get("protocol", {})
    if protocol.get("warmup") != 3 or protocol.get("repetitions") != 10:
        raise RuntimeError("C++ runner protocol is not the locked 3+10")
    abi = report.get("abi", {})
    if abi.get("input_names") != ["input_ids", "attention_mask"]:
        raise RuntimeError("C++ runner input ABI differs")
    if abi.get("output_names") != ["target_top1", "draft_top1"]:
        raise RuntimeError("C++ runner output ABI differs")
    if str(abi.get("dtype", "")).lower() != "int64":
        raise RuntimeError("C++ runner ABI dtype differs")
    memory_query = report.get("model_memory_query", {})
    if memory_query.get("source") != "aclmdlQuerySize":
        raise RuntimeError("C++ runner omitted the locked OM memory query")
    work_bytes = memory_query.get("work_bytes")
    weight_bytes = memory_query.get("weight_bytes")
    if (
        isinstance(work_bytes, bool)
        or not isinstance(work_bytes, int)
        or work_bytes < 0
        or isinstance(weight_bytes, bool)
        or not isinstance(weight_bytes, int)
        or weight_bytes <= 0
    ):
        raise RuntimeError("C++ runner returned invalid OM work/weight bytes")
    ordinary = report.get("ordinary")
    dflash = report.get("dflash")
    if not isinstance(ordinary, Mapping) or not isinstance(dflash, Mapping):
        raise RuntimeError("C++ runner omitted paired mode reports")
    _validate_mode_report(
        "ordinary", ordinary, generation_mode="ordinary-greedy"
    )
    _validate_mode_report(
        "DFlash", dflash, generation_mode="dflash-strict-greedy"
    )
    if ordinary.get("stable_generated_token_ids") != dflash.get(
        "stable_generated_token_ids"
    ):
        raise RuntimeError("C++ DFlash tokens differ from ordinary authority")
    if ordinary.get("stable_stop_reason") != dflash.get("stable_stop_reason"):
        raise RuntimeError("C++ DFlash EOS/stop reason differs from ordinary authority")
    parity = report.get("ordinary_parity", {})
    if (
        parity.get("status") != "PASS"
        or parity.get("token_id_mismatches") != 0
        or parity.get("eos_mismatches") != 0
    ):
        raise RuntimeError("C++ runner ordinary parity gate failed")


def run_cpp_pair(
    *,
    deployment_manifest: str | Path,
    runner: str | Path,
    runner_options: Mapping[str, Any],
    prompt_token_ids: Sequence[int],
    eos_token_ids: Sequence[int],
    device_id: int,
    max_new_tokens: int,
    max_draft_tokens: int,
    raw_output: str | Path,
    log_output: str | Path,
    progress: bool = True,
    execute: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, Any]:
    """Run paired ordinary/DFlash generation entirely inside one C++ process."""

    if max_new_tokens <= 0 or max_draft_tokens <= 0:
        raise ValueError("C++ generation limits must be positive")
    tokens = [int(item) for item in prompt_token_ids]
    if not tokens:
        raise ValueError("C++ runner prompt tokens must not be empty")
    _progress(progress, "stage=validate-config-start")
    identity = _runtime_identity(runner_options, device_id)
    _progress(progress, "stage=validate-manifest-and-om-start")
    om_path, deployment, graph = _resolve_integrated_om(
        deployment_manifest,
        graph_name=identity["graph_name"],
    )
    _progress(progress, "stage=validate-manifest-and-om-done")
    om_record = dict(graph["om"])
    executable = preflight_cpp_runner(runner)
    raw_path = require_run_output(raw_output)
    log_path = require_run_output(log_output)
    if raw_path.exists() or log_path.exists():
        raise FileExistsError("C++ runner output/log already exists in this run")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(executable),
        "--model",
        str(om_path),
        "--model-sha256",
        str(om_record["sha256"]),
        "--output",
        str(raw_path),
        "--prompt-token-ids",
        _token_csv(tokens),
        "--eos-token-ids",
        _token_csv(eos_token_ids),
        "--pad-token-id",
        str(identity["pad_token_id"]),
        "--max-new-tokens",
        str(int(max_new_tokens)),
        "--max-draft-tokens",
        str(int(max_draft_tokens)),
        "--warmup",
        "3",
        "--repetitions",
        "10",
        "--device-id",
        str(int(device_id)),
        "--progress",
        "true" if progress else "false",
    ]
    _progress(
        progress,
        "stage=runner-start live child output follows; "
        f"durable_log={log_path}",
    )
    start_ns = time.perf_counter_ns()
    if execute is None:
        result = _execute_streaming(command, log_path=log_path, echo=progress)
    else:
        result = execute(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log_path.write_text(result.stdout or "", encoding="utf-8")
        if progress and result.stdout:
            sys.stderr.write(result.stdout)
            sys.stderr.flush()
    end_ns = time.perf_counter_ns()
    _progress(
        progress,
        f"stage=runner-exit code={result.returncode} "
        f"wall_ms={(end_ns - start_ns) / 1_000_000.0:.3f}",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"C++ ACL runner failed with exit {result.returncode}; log={log_path}"
        )
    if not raw_path.is_file():
        raise RuntimeError("C++ ACL runner returned success without a JSON report")
    _progress(progress, "stage=validate-runner-report-start")
    report = load_json_object(raw_path)
    validate_cpp_runner_report(
        report,
        prompt_token_ids=tokens,
        om_sha256=str(om_record["sha256"]),
        device_id=device_id,
        max_new_tokens=max_new_tokens,
        max_draft_tokens=max_draft_tokens,
    )
    _progress(progress, "stage=validate-runner-report-done")
    run_root = Path(os.environ["AI_RUN_DIR"]).expanduser().resolve()
    air_record = deployment.get("air_manifest")
    if not isinstance(air_record, Mapping):
        raise ValueError("deployment manifest has no AIR manifest record")
    air_manifest_path = contained_path(
        Path(deployment_manifest).expanduser().resolve().parent,
        str(air_record.get("path", "")),
    )
    if (
        not air_manifest_path.is_file()
        or sha256_file(air_manifest_path) != str(air_record.get("sha256", ""))
    ):
        raise ValueError("AIR manifest integrity check failed after C++ execution")
    report["backend_metadata"] = {
        **identity,
        "artifacts": {str(graph["name"]): str(om_record["sha256"])},
        "state_policy": "recompute committed prefixes",
        "host_hot_path": "AscendCL C++",
    }
    report["control_plane"] = {
        "process_wall_ms": (end_ns - start_ns) / 1_000_000.0,
        "runner": {
            "path": str(executable),
            "bytes": executable.stat().st_size,
            "sha256": sha256_file(executable),
        },
        "deployment_manifest": file_record(
            Path(deployment_manifest).expanduser().resolve(), relative_to=run_root
        ),
        "air_manifest": file_record(air_manifest_path, relative_to=run_root),
        "runner_raw_report": file_record(raw_path, relative_to=run_root),
        "runner_log": file_record(log_path, relative_to=run_root),
        "compiler": dict(deployment.get("compiler", {})),
        "target": dict(deployment.get("target", {})),
    }
    _progress(progress, "stage=cpp-pair-done status=PASS")
    return report


def write_cpp_prompt_report(
    *,
    payload: Mapping[str, Any],
    output: str | Path,
    prompt: str,
    chat: bool,
    tokenizer_source: Mapping[str, Any],
    tokenize_ms: float,
    detokenize_ms: float,
    generated_text: str,
) -> dict[str, Any]:
    result = dict(payload)
    result["report_kind"] = "cpp-ascendcl-paired-target"
    result["prompt"] = prompt
    result["chat"] = bool(chat)
    result["tokenizer_source"] = dict(tokenizer_source)
    result["output"] = {
        "token_ids": list(result["dflash"]["stable_generated_token_ids"]),
        "text": generated_text,
        "stop_reason": str(result["dflash"]["stable_stop_reason"]),
    }
    result["host_text_stage_ms"] = {
        "tokenize": float(tokenize_ms),
        "detokenize": float(detokenize_ms),
        "note": (
            "single host measurements outside the 3+10 C++ OM model-loop "
            "distribution; do not combine them into a claimed service latency"
        ),
    }
    result["claim_boundary"] = (
        "C++ removes Python from the OM generation hot path and reports paired "
        "synchronized model-loop latency. Comparable closed-runtime latency still "
        "requires same-device A/B evidence and may require incremental target/draft state."
    )
    target = require_run_output(output)
    atomic_write_json(target, result)
    return result
