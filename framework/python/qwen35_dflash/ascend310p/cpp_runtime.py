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
INCREMENTAL_CPP_RUNNER_ID = "qwen35-dflash-ascendcl-cpp-incremental-v2"
RECOMPUTE_STATE_POLICY = "recompute-committed-prefixes"
INCREMENTAL_STATE_POLICY = "incremental-explicit-state-v2"
ASYNC_MEMSET_STATE_RESET_POLICY = "async-memset"
IMMUTABLE_ZERO_STATE_RESET_POLICY = "immutable-zero"
_INCREMENTAL_STATE_RESET_POLICIES = {
    ASYNC_MEMSET_STATE_RESET_POLICY,
    IMMUTABLE_ZERO_STATE_RESET_POLICY,
}
_INCREMENTAL_ABI_ID = (
    "qwen35-4b-dflash-ascend310p-incremental-performance-v2"
)
_INCREMENTAL_GRAPH_ABI: dict[str, tuple[list[str], list[str]]] = {
    "target-prefill": (
        [
            "input_ids", "effective_length", "eos_token_ids",
            "eos_token_count", "target_conv_state",
            "target_recurrent_state", "target_key_cache",
            "target_value_cache", "logical_target_cursor",
        ],
        [
            "committed_token_ids", "commit_count", "finished",
            "target_feature_tail", "committed_input_count",
            "target_conv_state", "target_recurrent_state",
            "target_key_cache", "target_value_cache",
            "logical_target_cursor",
        ],
    ),
    "target-decode1": (
        [
            "input_ids", "eos_token_ids", "eos_token_count",
            "target_conv_state", "target_recurrent_state",
            "target_key_cache", "target_value_cache",
            "logical_target_cursor",
        ],
        [
            "committed_token_ids", "commit_count", "finished",
            "target_conv_state", "target_recurrent_state",
            "target_key_cache", "target_value_cache",
            "logical_target_cursor",
        ],
    ),
    "draft-propose": (
        [
            "target_feature_tail", "committed_input_count",
            "previous_committed_token_ids", "previous_commit_count",
            "logical_proposal_count", "draft_key_cache",
            "draft_value_cache", "logical_draft_cursor",
        ],
        [
            "verify_input_ids", "draft_key_cache", "draft_value_cache",
            "logical_draft_cursor",
        ],
    ),
    "target-verify-commit": (
        [
            "verify_input_ids", "logical_proposal_count", "eos_token_ids",
            "eos_token_count", "target_conv_state",
            "target_recurrent_state", "target_key_cache",
            "target_value_cache", "logical_target_cursor",
        ],
        [
            "committed_token_ids", "commit_count", "drafted_count",
            "accepted_count", "rejected_count", "target_conv_state",
            "target_recurrent_state", "target_feature_tail",
            "committed_input_count", "logical_target_cursor", "finished",
            "target_key_cache", "target_value_cache",
        ],
    ),
}
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
        or not any(
            marker in result.stdout
            for marker in (
                "qwen35_dflash_acl_runner",
                "qwen35_dflash_incremental_acl_runner",
            )
        )
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
    state_policy = str(
        options.get("state_policy", RECOMPUTE_STATE_POLICY)
    ).strip()
    if state_policy not in {RECOMPUTE_STATE_POLICY, INCREMENTAL_STATE_POLICY}:
        raise ValueError(
            "C++ runner state_policy must be "
            f"{RECOMPUTE_STATE_POLICY!r} or {INCREMENTAL_STATE_POLICY!r}"
        )
    graph_name = str(options.get("graph_name", "quant_dflash_recompute"))
    pad_token_id = int(options.get("pad_token_id", 0))
    if pad_token_id < 0:
        raise ValueError("C++ runner pad_token_id must be non-negative")
    state_reset_policy = str(
        options.get(
            "state_reset_policy", ASYNC_MEMSET_STATE_RESET_POLICY
        )
    ).strip()
    if state_reset_policy not in _INCREMENTAL_STATE_RESET_POLICIES:
        raise ValueError(
            "C++ runner state_reset_policy must be "
            f"{ASYNC_MEMSET_STATE_RESET_POLICY!r} or "
            f"{IMMUTABLE_ZERO_STATE_RESET_POLICY!r}"
        )
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
        "state_policy": state_policy,
        "state_reset_policy": state_reset_policy,
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
    incremental_candidates = (
        build / "qwen35_dflash_incremental_acl_runner",
        build / "Release" / "qwen35_dflash_incremental_acl_runner",
    )
    incremental_runner = next(
        (item for item in incremental_candidates if item.is_file()), None
    )
    if incremental_runner is None or not os.access(incremental_runner, os.X_OK):
        raise RuntimeError(
            "C++ build succeeded but produced no incremental ACL runner"
        )
    preflight_cpp_runner(runner)
    preflight_cpp_runner(incremental_runner)
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
        "incremental_runner": file_record(
            incremental_runner, relative_to=run_root
        ),
        "logs": logs,
        "claim_boundary": (
            "Host scheduler and fake-ACL integration tests passed; physical-device "
            "latency is established only by infer-cpp/run-e2e-cpp target reports."
        ),
    }
    atomic_write_json(report_path, payload)
    payload["report_path"] = str(report_path)
    payload["runner_path"] = str(runner)
    payload["incremental_runner_path"] = str(incremental_runner)
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


def _resolve_incremental_oms(
    deployment_manifest: str | Path,
) -> tuple[
    dict[str, tuple[Path, dict[str, Any], dict[str, Any]]],
    dict[str, Any],
]:
    """Resolve and hash-check the exact four-role v2 deployment ABI."""

    manifest_path = Path(deployment_manifest).expanduser().resolve()
    manifest = load_json_object(manifest_path)
    if manifest.get("artifact_kind") != "qwen35-dflash-ascend310p-om-bundle":
        raise ValueError("incremental C++ runner requires an Ascend 310P OM bundle")
    if manifest.get("status") != "PASS":
        raise ValueError("deployment manifest is not passing")
    graphs = manifest.get("graphs", [])
    resolved: dict[str, tuple[Path, dict[str, Any], dict[str, Any]]] = {}
    for role, (input_names, output_names) in _INCREMENTAL_GRAPH_ABI.items():
        matches = [
            graph
            for graph in graphs
            if isinstance(graph, Mapping)
            and str(graph.get("name")) == role
            and str(graph.get("role")) == role
        ]
        if len(matches) != 1:
            raise ValueError(
                f"deployment manifest needs one {role!r} role, got {len(matches)}"
            )
        graph = dict(matches[0])
        if list(graph.get("input_names", [])) != input_names:
            raise ValueError(f"{role} OM input order differs from the locked v2 ABI")
        if list(graph.get("output_names", [])) != output_names:
            raise ValueError(f"{role} OM output order differs from the locked v2 ABI")
        record_value = graph.get("om")
        if not isinstance(record_value, Mapping):
            raise ValueError(f"{role} deployment graph has no OM record")
        record = dict(record_value)
        om_path = contained_path(manifest_path.parent, str(record["path"]))
        expected_hash = str(record.get("sha256", ""))
        if not om_path.is_file() or sha256_file(om_path) != expected_hash:
            raise ValueError(f"{role} OM artifact integrity check failed")
        resolved[role] = (om_path, graph, record)
    return resolved, manifest


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
    execution_io = report.get("execution_io_counters", {})
    if execution_io.get("input_policy") != (
        "persistent device mirror plus changed contiguous ranges"
    ):
        raise RuntimeError("C++ runner did not use the locked incremental input policy")
    if execution_io.get("target_output_policy") != (
        "download only the last draft_width_plus_one rows needed by proposal or verify"
    ):
        raise RuntimeError("C++ runner did not use the locked target output slice")
    executions = execution_io.get("model_executions")
    synchronizations = execution_io.get("stream_synchronizations")
    if (
        isinstance(executions, bool)
        or not isinstance(executions, int)
        or executions <= 0
        or synchronizations != executions
    ):
        raise RuntimeError("C++ runner execution/synchronization counters differ")
    for actual_name, full_name, avoided_name in (
        (
            "host_to_device_bytes",
            "full_host_to_device_bytes",
            "host_to_device_bytes_avoided",
        ),
        (
            "device_to_host_bytes",
            "full_device_to_host_bytes",
            "device_to_host_bytes_avoided",
        ),
    ):
        actual = execution_io.get(actual_name)
        full = execution_io.get(full_name)
        avoided = execution_io.get(avoided_name)
        if (
            isinstance(actual, bool)
            or not isinstance(actual, int)
            or actual < 0
            or isinstance(full, bool)
            or not isinstance(full, int)
            or full <= 0
            or actual > full
            or avoided != full - actual
        ):
            raise RuntimeError(f"C++ runner returned invalid {actual_name}")
    maximum_target = execution_io.get("maximum_target_elements_per_call")
    if (
        isinstance(maximum_target, bool)
        or not isinstance(maximum_target, int)
        or maximum_target <= 0
        or maximum_target > int(abi.get("draft_width", -1)) + 1
    ):
        raise RuntimeError("C++ runner target output slice exceeds K+1")
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


def validate_incremental_cpp_runner_report(
    report: Mapping[str, Any],
    *,
    prompt_token_ids: Sequence[int],
    om_sha256_by_role: Mapping[str, str],
    device_id: int,
    max_new_tokens: int,
    max_draft_tokens: int,
    state_reset_policy: str = ASYNC_MEMSET_STATE_RESET_POLICY,
) -> None:
    """Validate four resident models, device state routing and paired parity."""

    if (
        report.get("status") != "PASS"
        or report.get("runner_id") != INCREMENTAL_CPP_RUNNER_ID
    ):
        raise RuntimeError("incremental C++ runner returned an unknown report")
    if report.get("candidate_status") != "APPROVED_IN_IMPLEMENTATION_NOT_ACTIVE":
        raise RuntimeError("incremental runner candidate status differs")
    if report.get("cpu_fallback") is not False:
        raise RuntimeError("incremental target report indicates CPU fallback")
    if int(report.get("device_id", -1)) != int(device_id):
        raise RuntimeError("incremental C++ runner used a different device ID")
    expected_roles = list(_INCREMENTAL_GRAPH_ABI)
    models = report.get("models")
    if (
        not isinstance(models, list)
        or not all(isinstance(item, Mapping) for item in models)
        or [item.get("role") for item in models] != expected_roles
    ):
        raise RuntimeError("incremental C++ report model role order differs")
    if set(om_sha256_by_role) != set(expected_roles):
        raise RuntimeError("incremental expected OM hash set differs")
    for item in models:
        role = str(item["role"])
        if item.get("sha256") != om_sha256_by_role[role]:
            raise RuntimeError(f"incremental {role} OM hash differs")
        for field in ("work_bytes", "weight_bytes"):
            value = item.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or (value < 0 if field == "work_bytes" else value <= 0)
            ):
                raise RuntimeError(f"incremental {role} {field} is invalid")
    if list(report.get("prompt_token_ids", [])) != [int(item) for item in prompt_token_ids]:
        raise RuntimeError("incremental C++ prompt token IDs differ")
    limits = report.get("limits", {})
    if int(limits.get("max_new_tokens", -1)) != int(max_new_tokens):
        raise RuntimeError("incremental max_new_tokens differs")
    if int(limits.get("max_draft_tokens", -1)) != int(max_draft_tokens):
        raise RuntimeError("incremental max_draft_tokens differs")
    protocol = report.get("protocol", {})
    if protocol.get("warmup") != 3 or protocol.get("repetitions") != 10:
        raise RuntimeError("incremental runner protocol is not locked 3+10")
    if (
        protocol.get("kind") != "evidence"
        or protocol.get("formal_latency_evidence") is not True
    ):
        raise RuntimeError("incremental runner returned diagnostic-only timing")
    if state_reset_policy not in _INCREMENTAL_STATE_RESET_POLICIES:
        raise ValueError("expected incremental state reset policy is invalid")
    if protocol.get("state_reset_policy") != state_reset_policy:
        raise RuntimeError("incremental runner state reset policy differs")
    if protocol.get("state_reset_only_barriers") != 0:
        raise RuntimeError("incremental runner added a reset-only barrier")
    immutable_zero = state_reset_policy == IMMUTABLE_ZERO_STATE_RESET_POLICY
    if protocol.get(
        "state_reset_device_work_included_by_prefill_barrier"
    ) is not (not immutable_zero):
        raise RuntimeError("incremental reset timing scope differs")
    if protocol.get(
        "state_zero_initialization_included_in_startup"
    ) is not immutable_zero:
        raise RuntimeError("incremental zero initialization timing scope differs")
    if protocol.get("prefill_completion_policy") != (
        "intermediate prompt chunks stay queued; final chunk performs the "
        "only compact D2H and stream synchronization"
    ):
        raise RuntimeError("incremental prefill completion policy differs")
    if protocol.get("prefill_control_policy") != (
        "IDs, effective length, proposal count and EOS table share one H2D "
        "carrier with 64-byte-aligned device subsegments per prompt chunk"
    ):
        raise RuntimeError("incremental prefill control policy differs")
    if protocol.get("device_suballocation_policy") != (
        "64-byte segment starts; ALIGN_UP(payload,32)+32 reserved span"
    ):
        raise RuntimeError("incremental device suballocation policy differs")
    abi = report.get("abi", {})
    if abi.get("id") != _INCREMENTAL_ABI_ID:
        raise RuntimeError("incremental runner ABI identity differs")
    if abi.get("state_policy") != "explicit device-resident ping-pong":
        raise RuntimeError("incremental state is not device resident")
    proposal_width = abi.get("proposal_width")
    verify_width = abi.get("verify_width")
    sequence_capacity = abi.get("sequence_capacity")
    prefill_width = abi.get("prefill_width")
    eos_table_width = abi.get("eos_table_width")
    if (
        isinstance(proposal_width, bool)
        or not isinstance(proposal_width, int)
        or proposal_width <= 0
        or verify_width != proposal_width + 1
    ):
        raise RuntimeError("incremental proposal/verify width differs")
    if (
        isinstance(sequence_capacity, bool)
        or not isinstance(sequence_capacity, int)
        or sequence_capacity <= 0
        or isinstance(prefill_width, bool)
        or not isinstance(prefill_width, int)
        or prefill_width != 64
        or isinstance(eos_table_width, bool)
        or not isinstance(eos_table_width, int)
        or eos_table_width <= 0
        or sequence_capacity % prefill_width
        or len(prompt_token_ids) > sequence_capacity
    ):
        raise RuntimeError("incremental sequence/prefill capacity differs")
    memory = report.get("model_memory_query", {})
    if memory.get("source") != "aclmdlQuerySize":
        raise RuntimeError("incremental runner omitted model memory queries")
    if memory.get("load_policy") != (
        "four aclmdlLoadFromFileWithMem sessions; one max-sized serial "
        "workspace; separate per-artifact weights; no cross-OM weight sharing "
        "assumed"
    ):
        raise RuntimeError("incremental runner did not share the serial workspace")
    expected_sum_work = sum(int(item["work_bytes"]) for item in models)
    expected_max_work = max(int(item["work_bytes"]) for item in models)
    expected_sum_weight = sum(int(item["weight_bytes"]) for item in models)
    if (
        memory.get("sum_work_bytes") != expected_sum_work
        or memory.get("max_work_bytes") != expected_max_work
        or memory.get("sum_weight_bytes") != expected_sum_weight
    ):
        raise RuntimeError("incremental model memory totals do not close")
    for field in (
        "state_device_bytes",
        "working_state_device_bytes",
        "state_reset_bytes_per_request",
        "carrier_device_bytes",
        "prefill_control_bytes_per_slot",
        "prefill_staging_pinned_host_bytes",
    ):
        value = memory.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeError(f"incremental {field} is invalid")
    zero_state_bytes = memory.get("immutable_zero_state_device_bytes")
    if (
        isinstance(zero_state_bytes, bool)
        or not isinstance(zero_state_bytes, int)
        or zero_state_bytes < 0
    ):
        raise RuntimeError("incremental immutable zero state bytes are invalid")
    expected_zero_state_bytes = (
        int(memory["state_reset_bytes_per_request"])
        if immutable_zero
        else 0
    )
    if zero_state_bytes != expected_zero_state_bytes:
        raise RuntimeError("incremental immutable zero state size differs")
    if memory["state_device_bytes"] != (
        memory["working_state_device_bytes"] + zero_state_bytes
    ):
        raise RuntimeError("incremental state allocation does not close")
    expected_staging_slots = sequence_capacity // prefill_width
    control_cursor = 0
    for tensor_bytes in (
        prefill_width * 8,
        eos_table_width * 8,
        4,
        4,
        2,
    ):
        control_cursor = (control_cursor + 63) // 64 * 64
        control_cursor += (tensor_bytes + 31) // 32 * 32 + 32
    expected_control_bytes = (control_cursor + 63) // 64 * 64
    expected_staging_host_bytes = (
        expected_staging_slots * expected_control_bytes
    )
    if (
        memory["prefill_control_bytes_per_slot"]
        != expected_control_bytes
        or memory["prefill_staging_pinned_host_bytes"]
        != expected_staging_host_bytes
    ):
        raise RuntimeError("incremental prefill pinned-host staging differs")
    expected_allocated = (
        expected_max_work
        + expected_sum_weight
        + int(memory["state_device_bytes"])
        + int(memory["carrier_device_bytes"])
    )
    if (
        memory.get("explicit_allocated_device_bytes_excluding_runtime")
        != expected_allocated
    ):
        raise RuntimeError("incremental explicit device allocation does not close")
    execution = report.get("execution_io_counters", {})
    role_counts = [
        execution.get("target_prefill_executions"),
        execution.get("target_decode1_executions"),
        execution.get("draft_propose_executions"),
        execution.get("target_verify_commit_executions"),
    ]
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in role_counts
    ) or int(role_counts[0]) <= 0:
        raise RuntimeError("incremental runner returned invalid OM role counters")
    prefill, decode, draft, verify = (int(value) for value in role_counts)
    if execution.get("model_executions") != sum(role_counts):
        raise RuntimeError("incremental model execution counters do not close")
    request_count = 2 * (3 + 10)
    prompt_chunks = (len(prompt_token_ids) + prefill_width - 1) // prefill_width
    expected_prefill = request_count * prompt_chunks
    expected_deferred = request_count * (prompt_chunks - 1)
    prefill_completions = execution.get("prefill_completion_synchronizations")
    deferred_prefill = execution.get("deferred_prefill_chunks")
    if (
        prefill != expected_prefill
        or prefill_completions != request_count
        or deferred_prefill != expected_deferred
        or execution.get("prefill_synchronizations_elided")
        != expected_deferred
        or execution.get("prefill_compact_downloads_elided")
        != expected_deferred
    ):
        raise RuntimeError("incremental prefill chunk counters differ")
    transactions = int(prefill_completions) + decode + verify
    if (
        execution.get("stream_synchronizations") != transactions
        or execution.get("device_to_host_operations") != transactions
    ):
        raise RuntimeError("incremental transaction synchronization policy differs")
    if execution.get("state_resets") != request_count:
        raise RuntimeError("incremental paired run reset count differs")
    if immutable_zero:
        if (
            execution.get("state_memset_operations") != 0
            or execution.get("state_memset_bytes") != 0
            or execution.get("state_initialization_memset_operations") != 2
            or execution.get("state_initialization_memset_bytes")
            != expected_zero_state_bytes
            or execution.get("state_initialization_stream_synchronizations")
            != 1
        ):
            raise RuntimeError("incremental immutable zero counters differ")
    elif (
        execution.get("state_memset_operations")
        != 2 * execution["state_resets"]
        or execution.get("state_memset_bytes")
        != memory["state_reset_bytes_per_request"] * execution["state_resets"]
        or execution.get("state_initialization_memset_operations") != 0
        or execution.get("state_initialization_memset_bytes") != 0
        or execution.get("state_initialization_stream_synchronizations") != 0
    ):
        raise RuntimeError("incremental async state clear counters differ")
    for field in (
        "state_device_bytes",
        "working_state_device_bytes",
        "immutable_zero_state_device_bytes",
        "state_reset_bytes_per_request",
    ):
        if execution.get(field) != memory.get(field):
            raise RuntimeError(f"incremental {field} reports differ")
    if execution.get("carrier_device_bytes") != memory.get("carrier_device_bytes"):
        raise RuntimeError("incremental carrier byte reports differ")
    if (
        execution.get("prefill_staging_slots") != expected_staging_slots
        or execution.get("prefill_control_bytes_per_slot")
        != expected_control_bytes
        or execution.get("prefill_staging_pinned_host_bytes")
        != expected_staging_host_bytes
    ):
        raise RuntimeError("incremental prefill staging reports differ")
    prefill_upload_operations = execution.get(
        "prefill_control_upload_operations"
    )
    decode_upload_operations = execution.get("decode_id_upload_operations")
    proposal_upload_operations = execution.get(
        "proposal_count_upload_operations"
    )
    if (
        prefill_upload_operations != prefill
        or execution.get("prefill_control_upload_bytes")
        != prefill * expected_control_bytes
        or execution.get("prefill_h2d_operations_elided") != prefill
        or decode_upload_operations != decode
        or execution.get("decode_id_upload_bytes") != decode * 8
        or isinstance(proposal_upload_operations, bool)
        or not isinstance(proposal_upload_operations, int)
        or proposal_upload_operations < 0
        or execution.get("proposal_count_upload_bytes")
        != proposal_upload_operations * 4
        or execution.get("host_to_device_operations")
        != (
            prefill_upload_operations
            + decode_upload_operations
            + proposal_upload_operations
        )
        or execution.get("host_to_device_bytes")
        != (
            execution["prefill_control_upload_bytes"]
            + execution["decode_id_upload_bytes"]
            + execution["proposal_count_upload_bytes"]
        )
    ):
        raise RuntimeError("incremental packed H2D counters differ")
    for field in (
        "host_to_device_operations", "host_to_device_bytes",
        "device_to_host_bytes",
    ):
        value = execution.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeError(f"incremental {field} is invalid")
    ordinary = report.get("ordinary")
    dflash = report.get("dflash")
    if not isinstance(ordinary, Mapping) or not isinstance(dflash, Mapping):
        raise RuntimeError("incremental runner omitted paired mode reports")
    _validate_mode_report("ordinary", ordinary, generation_mode="ordinary-greedy")
    _validate_mode_report("DFlash", dflash, generation_mode="dflash-strict-greedy")
    if ordinary.get("stable_generated_token_ids") != dflash.get(
        "stable_generated_token_ids"
    ):
        raise RuntimeError("incremental DFlash tokens differ from ordinary")
    if ordinary.get("stable_stop_reason") != dflash.get("stable_stop_reason"):
        raise RuntimeError("incremental DFlash stop reason differs from ordinary")
    parity = report.get("ordinary_parity", {})
    if (
        parity.get("status") != "PASS"
        or parity.get("token_id_mismatches") != 0
        or parity.get("eos_mismatches") != 0
    ):
        raise RuntimeError("incremental runner ordinary parity gate failed")


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
    incremental = identity["state_policy"] == INCREMENTAL_STATE_POLICY
    resolved_incremental: dict[
        str, tuple[Path, dict[str, Any], dict[str, Any]]
    ] = {}
    graph: dict[str, Any] | None = None
    om_record: dict[str, Any] | None = None
    om_path: Path | None = None
    if incremental:
        resolved_incremental, deployment = _resolve_incremental_oms(
            deployment_manifest
        )
        artifacts = {
            role: str(record["sha256"])
            for role, (_, _, record) in resolved_incremental.items()
        }
    else:
        om_path, deployment, graph = _resolve_integrated_om(
            deployment_manifest,
            graph_name=identity["graph_name"],
        )
        om_record = dict(graph["om"])
        artifacts = {str(graph["name"]): str(om_record["sha256"])}
    _progress(progress, "stage=validate-manifest-and-om-done")
    executable = preflight_cpp_runner(runner)
    raw_path = require_run_output(raw_output)
    log_path = require_run_output(log_output)
    if raw_path.exists() or log_path.exists():
        raise FileExistsError("C++ runner output/log already exists in this run")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [str(executable)]
    if incremental:
        for role, (role_path, _, record) in resolved_incremental.items():
            command.extend(
                [
                    f"--{role}",
                    str(role_path),
                    f"--{role}-sha256",
                    str(record["sha256"]),
                ]
            )
    else:
        assert om_path is not None and om_record is not None
        command.extend(
            [
                "--model",
                str(om_path),
                "--model-sha256",
                str(om_record["sha256"]),
            ]
        )
    command.extend([
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
        "--measurement-protocol",
        "evidence",
        "--device-id",
        str(int(device_id)),
    ])
    if incremental:
        command.extend(
            ["--state-reset-policy", identity["state_reset_policy"]]
        )
    command.extend(["--progress", "true" if progress else "false"])
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
    if incremental:
        validate_incremental_cpp_runner_report(
            report,
            prompt_token_ids=tokens,
            om_sha256_by_role=artifacts,
            device_id=device_id,
            max_new_tokens=max_new_tokens,
            max_draft_tokens=max_draft_tokens,
            state_reset_policy=identity["state_reset_policy"],
        )
    else:
        assert om_record is not None
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
        "artifacts": artifacts,
        "state_implementation": (
            "explicit Target/Draft device state"
            if incremental
            else "recompute committed prefixes"
        ),
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
    if result.get("runner_id") == INCREMENTAL_CPP_RUNNER_ID:
        result["claim_boundary"] = (
            "The four-OM C++ candidate keeps Target/Draft state and proposal carriers "
            "on device. Promotion still requires real Ascend310P zero-mismatch, "
            "complete-set memory and unprofiled same-device latency evidence."
        )
    else:
        result["claim_boundary"] = (
            "C++ removes Python from the OM generation hot path and reports paired "
            "synchronized model-loop latency. Comparable closed-runtime latency still "
            "requires same-device A/B evidence and may require incremental target/draft state."
        )
    target = require_run_output(output)
    atomic_write_json(target, result)
    return result
