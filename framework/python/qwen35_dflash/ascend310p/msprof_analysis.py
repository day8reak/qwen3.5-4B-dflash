"""Strict role/gear analysis for an incremental C++ runner msprof capture."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
from typing import Any, Iterable, Mapping, Sequence


INCREMENTAL_RUNNER_ID = "qwen35-dflash-ascendcl-cpp-incremental-v3"
BASELINE_ROLES = (
    "target-prefill",
    "target-prefill-head",
    "target-decode1",
    "draft-propose",
    "target-verify-commit",
)
UNIFIED_ROLES = (
    "target-prefill",
    "target-prefill-head",
    "draft-propose",
    "target-verify-commit",
)


class MsprofAnalysisError(RuntimeError):
    """Raised when a profile cannot support exact role/gear attribution."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized(value: str) -> str:
    value = value.lstrip("\ufeff").replace("µ", "u").replace("μ", "u")
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _normalized_row(row: Mapping[str, str | None]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in row.items():
        if key is None or value is None:
            continue
        normalized = _normalized(key)
        if normalized:
            result[normalized] = value.strip()
    return result


def _value(
    row: Mapping[str, str],
    names: Sequence[str],
    *,
    required: bool = True,
) -> str | None:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    if required:
        raise MsprofAnalysisError(
            f"msprof CSV omitted required column variants: {', '.join(names)}"
        )
    return None


def _integer(value: str, *, field: str, source: Path) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as error:
        raise MsprofAnalysisError(
            f"{source}: invalid {field} integer {value!r}"
        ) from error
    if parsed < 0:
        raise MsprofAnalysisError(
            f"{source}: {field} must be non-negative"
        )
    return parsed


def _number(value: str, *, field: str, source: Path) -> float:
    cleaned = value.replace(",", "")
    if cleaned.lower() in {"n/a", "na", "none", "null", "-"}:
        raise MsprofAnalysisError(
            f"{source}: {field} is unavailable; collect with task timing on"
        )
    try:
        parsed = float(cleaned)
    except ValueError as error:
        raise MsprofAnalysisError(
            f"{source}: invalid {field} number {value!r}"
        ) from error
    if not math.isfinite(parsed) or parsed < 0:
        raise MsprofAnalysisError(
            f"{source}: {field} must be a finite non-negative number"
        )
    return parsed


def _filename_model_infer(path: Path) -> tuple[int, int] | None:
    match = re.fullmatch(r"op_summary_(\d+)_(\d+)_(\d+)", path.stem)
    if match is None:
        return None
    return int(match.group(2)), int(match.group(3))


def _record(path: Path, *, relative_to: Path, rows: int) -> dict[str, Any]:
    try:
        label = str(path.relative_to(relative_to))
    except ValueError:
        label = str(path)
    return {
        "path": label,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "rows": rows,
    }


def _load_runner_report(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MsprofAnalysisError(
            f"cannot read incremental runner report {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise MsprofAnalysisError("incremental runner report must be an object")
    return payload


def _validate_runner_report(
    report: Mapping[str, Any],
) -> tuple[
    dict[int, str],
    dict[str, int],
    list[dict[str, int]],
    bool,
]:
    if (
        report.get("status") != "PASS"
        or report.get("runner_id") != INCREMENTAL_RUNNER_ID
        or report.get("cpu_fallback") is not False
    ):
        raise MsprofAnalysisError(
            "msprof analysis requires a passing real-device incremental report"
        )
    if int(report.get("schema_version", 0)) < 4:
        raise MsprofAnalysisError(
            "runner report predates model IDs/profile execution tracing"
        )
    protocol = report.get("protocol")
    if not isinstance(protocol, Mapping) or (
        protocol.get("kind") != "profile"
        or protocol.get("formal_latency_evidence") is not False
        or protocol.get("profile_model_execution_trace_enabled") is not True
    ):
        raise MsprofAnalysisError(
            "runner report must use diagnostic profile protocol with tracing"
        )
    if int(report.get("schema_version", 0)) >= 8 and protocol.get(
        "device_memory_allocation_policy"
    ) not in {"normal-only", "huge-first"}:
        raise MsprofAnalysisError(
            "runner device memory allocation policy is invalid"
        )
    requested_sync_window = protocol.get("dflash_sync_window", 1)
    if int(report.get("schema_version", 0)) >= 9 and (
        isinstance(requested_sync_window, bool)
        or not isinstance(requested_sync_window, int)
        or not 1 <= requested_sync_window <= 8
    ):
        raise MsprofAnalysisError("runner DFlash sync window is invalid")
    models = report.get("models")
    if not isinstance(models, list) or not models:
        raise MsprofAnalysisError("runner report omitted resident models")
    roles = tuple(item.get("role") for item in models if isinstance(item, Mapping))
    if roles == BASELINE_ROLES:
        unified = False
    elif roles == UNIFIED_ROLES:
        unified = True
    else:
        raise MsprofAnalysisError("runner report has an unknown model role order")
    model_to_role: dict[int, str] = {}
    role_to_model: dict[str, int] = {}
    for item in models:
        if not isinstance(item, Mapping):
            raise MsprofAnalysisError("runner model entry must be an object")
        role = str(item.get("role", ""))
        model_id = item.get("model_id")
        if isinstance(model_id, bool) or not isinstance(model_id, int) or model_id < 0:
            raise MsprofAnalysisError(f"runner {role} model_id is invalid")
        if model_id in model_to_role or role in role_to_model:
            raise MsprofAnalysisError("runner model IDs and roles must be unique")
        model_to_role[model_id] = role
        role_to_model[role] = model_id

    counters = report.get("execution_io_counters")
    if not isinstance(counters, Mapping):
        raise MsprofAnalysisError("runner report omitted execution counters")
    if int(report.get("schema_version", 0)) >= 5:
        verify_transactions = counters.get(
            "target_verify_commit_executions"
        )
        speculative_windows = counters.get("speculative_sync_windows")
        speculative_elided = counters.get(
            "speculative_synchronizations_elided"
        )
        speculative_d2h_elided = counters.get(
            "speculative_d2h_operations_elided"
        )
        speculative_d2h_padding = counters.get(
            "speculative_d2h_padding_bytes"
        )
        compact_slot_bytes = counters.get("compact_slot_bytes")
        compact_ordinary_bytes = counters.get("compact_ordinary_result_bytes")
        compact_verify_bytes = counters.get("compact_verify_result_bytes")
        if int(report.get("schema_version", 0)) >= 9:
            speculative_staging_operations = counters.get(
                "speculative_window_staging_operations"
            )
            speculative_staging_bytes = counters.get(
                "speculative_window_staging_bytes"
            )
            speculative_staging_device_bytes = counters.get(
                "speculative_window_staging_device_bytes"
            )
            speculative_staging_host_bytes = counters.get(
                "speculative_window_staging_pinned_host_bytes"
            )
        else:
            speculative_staging_operations = 0
            speculative_staging_bytes = 0
            speculative_staging_device_bytes = 0
            speculative_staging_host_bytes = 0
        prefill_completions = counters.get(
            "prefill_completion_synchronizations"
        )
        decode_transactions = counters.get("target_decode1_executions")
        stream_synchronizations = counters.get("stream_synchronizations")
        device_to_host_operations = counters.get("device_to_host_operations")
        if int(report.get("schema_version", 0)) >= 7:
            if protocol.get("prefill_completion_policy") not in {
                "separate",
                "coalesce-first-verify",
            }:
                raise MsprofAnalysisError(
                    "runner prefill completion policy is invalid"
                )
            prefill_verify_windows = counters.get(
                "prefill_verify_coalesced_windows"
            )
            prefill_verify_syncs_elided = counters.get(
                "prefill_verify_synchronizations_elided"
            )
            prefill_verify_d2h_elided = counters.get(
                "prefill_verify_d2h_operations_elided"
            )
            prefill_verify_padding = counters.get(
                "prefill_verify_d2h_padding_bytes"
            )
            prefill_verify_slot0 = counters.get(
                "prefill_verify_prefill_slot0_windows"
            )
            prefill_verify_slot1 = counters.get(
                "prefill_verify_prefill_slot1_windows"
            )
        else:
            prefill_verify_windows = 0
            prefill_verify_syncs_elided = 0
            prefill_verify_d2h_elided = 0
            prefill_verify_padding = 0
            prefill_verify_slot0 = 0
            prefill_verify_slot1 = 0
        values = (
            verify_transactions,
            speculative_windows,
            speculative_elided,
            speculative_d2h_elided,
            speculative_d2h_padding,
            compact_slot_bytes,
            compact_ordinary_bytes,
            compact_verify_bytes,
            prefill_completions,
            decode_transactions,
            stream_synchronizations,
            device_to_host_operations,
            prefill_verify_windows,
            prefill_verify_syncs_elided,
            prefill_verify_d2h_elided,
            prefill_verify_padding,
            prefill_verify_slot0,
            prefill_verify_slot1,
            speculative_staging_operations,
            speculative_staging_bytes,
            speculative_staging_device_bytes,
            speculative_staging_host_bytes,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in values
        ):
            raise MsprofAnalysisError(
                "runner speculative synchronization counters are invalid"
            )
        if (
            speculative_windows
            + speculative_elided
            + prefill_verify_windows
            != verify_transactions
            or stream_synchronizations
            != prefill_completions
            + decode_transactions
            + speculative_windows
            or device_to_host_operations
            + speculative_d2h_elided
            + prefill_verify_d2h_elided
            != prefill_completions
            + decode_transactions
            + verify_transactions
            or speculative_d2h_elided != speculative_elided
            or prefill_verify_syncs_elided != prefill_verify_windows
            or prefill_verify_d2h_elided != prefill_verify_windows
            or prefill_verify_slot0 + prefill_verify_slot1
            != prefill_verify_windows
            or compact_slot_bytes
            < max(compact_ordinary_bytes, compact_verify_bytes)
            or speculative_d2h_padding
            != speculative_d2h_elided
            * (compact_slot_bytes - compact_verify_bytes)
            or speculative_staging_bytes
            != speculative_staging_operations * compact_verify_bytes
            or speculative_staging_operations > verify_transactions
            or (
                int(report.get("schema_version", 0)) >= 9
                and requested_sync_window <= 2
                and speculative_staging_operations != 0
            )
            or (
                int(report.get("schema_version", 0)) >= 9
                and (
                    protocol.get("maximum_supported_dflash_sync_window") != 8
                    or speculative_staging_device_bytes
                    != 8 * compact_slot_bytes
                    or speculative_staging_host_bytes
                    != speculative_staging_device_bytes
                )
            )
            or prefill_verify_padding
            != prefill_verify_slot0
            * (compact_slot_bytes - compact_ordinary_bytes)
            + prefill_verify_slot1
            * (compact_slot_bytes - compact_verify_bytes)
        ):
            raise MsprofAnalysisError(
                "runner speculative synchronization counters do not close"
            )
    counter_names = {
        "target-prefill": "target_prefill_executions",
        "target-prefill-head": "target_prefill_head_executions",
        "target-decode1": "target_decode1_executions",
        "draft-propose": "draft_propose_executions",
        "target-verify-commit": "target_verify_commit_executions",
    }
    expected_by_role: dict[str, int] = {}
    for role in roles:
        value = counters.get(counter_names[role])
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise MsprofAnalysisError(f"runner {role} execution count is invalid")
        expected_by_role[role] = value
    if unified:
        expected_by_role["target-verify-commit"] += int(
            counters["target_decode1_executions"]
        )

    trace = report.get("profile_model_execution_trace")
    if not isinstance(trace, list):
        raise MsprofAnalysisError("runner report omitted profile execution trace")
    model_executions = counters.get("model_executions")
    if (
        isinstance(model_executions, bool)
        or not isinstance(model_executions, int)
        or model_executions <= 0
        or len(trace) != model_executions
        or sum(expected_by_role.values()) != model_executions
    ):
        raise MsprofAnalysisError("runner model execution totals do not close")

    normalized_trace: list[dict[str, int]] = []
    trace_counts: Counter[str] = Counter()
    rows_by_role: defaultdict[str, list[int]] = defaultdict(list)
    for index, event in enumerate(trace):
        if not isinstance(event, Mapping):
            raise MsprofAnalysisError("runner trace event must be an object")
        ordinal = event.get("ordinal")
        model_id = event.get("model_id")
        rows = event.get("physical_rows")
        if ordinal != index:
            raise MsprofAnalysisError("runner trace ordinals are not contiguous")
        if (
            isinstance(model_id, bool)
            or not isinstance(model_id, int)
            or model_id not in model_to_role
            or isinstance(rows, bool)
            or not isinstance(rows, int)
            or rows <= 0
        ):
            raise MsprofAnalysisError("runner trace model ID/rows are invalid")
        role = model_to_role[model_id]
        trace_counts[role] += 1
        rows_by_role[role].append(rows)
        normalized_trace.append(
            {"ordinal": ordinal, "model_id": model_id, "physical_rows": rows}
        )
    if {
        role: trace_counts[role] for role in expected_by_role
    } != expected_by_role:
        raise MsprofAnalysisError("runner trace role counts do not close")

    abi = report.get("abi")
    if not isinstance(abi, Mapping):
        raise MsprofAnalysisError("runner report omitted incremental ABI")
    prefill_width = int(abi.get("prefill_width", 0))
    verify_width = int(abi.get("verify_width", 0))
    if prefill_width <= 0 or verify_width <= 1:
        raise MsprofAnalysisError("runner prefill/verify widths are invalid")
    if any(row != prefill_width for row in rows_by_role["target-prefill"]):
        raise MsprofAnalysisError("prefill trace rows differ from its physical width")
    for role in ("target-prefill-head", "target-decode1"):
        if any(row != 1 for row in rows_by_role[role]):
            raise MsprofAnalysisError(f"{role} trace contains a non-T1 execution")
    target_rows = rows_by_role["target-verify-commit"]
    if unified:
        decode_count = int(counters["target_decode1_executions"])
        verify_count = int(counters["target_verify_commit_executions"])
        if (
            any(row < 1 or row > verify_width for row in target_rows)
            or sum(row == 1 for row in target_rows) != decode_count
            or sum(row > 1 for row in target_rows) != verify_count
        ):
            raise MsprofAnalysisError("unified Target-step gear trace does not close")
    elif any(row != verify_width for row in target_rows):
        raise MsprofAnalysisError("baseline verify trace is not fixed-width")
    if int(report.get("schema_version", 0)) >= 6:
        draft_policy = protocol.get("draft_feature_policy")
        if draft_policy not in {"fixed-16", "committed-prefix"}:
            raise MsprofAnalysisError("runner Draft feature policy is invalid")
        draft_counter_names = (
            "prefill_draft_propose_executions",
            "prefill_feature_rows_batched",
            "draft_verify_feature_input_rows",
            "draft_verify_full_width_equivalent_rows",
            "draft_verify_feature_rows_elided",
            "draft_verify_fixed_width_executions",
            "draft_verify_committed_prefix_executions",
            "draft_verify_pending_upper_bound_executions",
            "draft_dynamic_gear_count",
            "draft_verify_dynamic_gear_count",
            "draft_prefill_dynamic_gear_count",
        )
        draft_values = [counters.get(name) for name in draft_counter_names]
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in draft_values
        ):
            raise MsprofAnalysisError("runner Draft feature counters are invalid")
        (
            prefill_draft_count,
            prefill_feature_rows,
            verify_feature_rows,
            verify_full_rows,
            verify_elided_rows,
            fixed_count,
            prefix_count,
            pending_count,
            dynamic_gears,
            verify_gears,
            prefill_gears,
        ) = (int(value) for value in draft_values)
        total_draft_count = int(counters["draft_propose_executions"])
        verify_draft_count = total_draft_count - prefill_draft_count
        draft_rows = rows_by_role["draft-propose"]
        prefill_trace_rows = [row for row in draft_rows if row > verify_width]
        verify_trace_rows = [row for row in draft_rows if row <= verify_width]
        if (
            verify_draft_count < 0
            or len(prefill_trace_rows) != prefill_draft_count
            or len(verify_trace_rows) != verify_draft_count
            or any(row % prefill_width for row in prefill_trace_rows)
            or sum(prefill_trace_rows) != prefill_feature_rows
            or sum(verify_trace_rows) != verify_feature_rows
            or verify_full_rows != verify_draft_count * verify_width
            or verify_feature_rows + verify_elided_rows != verify_full_rows
            or fixed_count + prefix_count + pending_count != verify_draft_count
            or verify_gears != verify_width
            or dynamic_gears != verify_gears + prefill_gears
        ):
            raise MsprofAnalysisError(
                "runner Draft feature trace/counters do not close"
            )
        if draft_policy == "fixed-16":
            if (
                any(row != verify_width for row in verify_trace_rows)
                or fixed_count != verify_draft_count
                or prefix_count != 0
                or pending_count != 0
                or verify_elided_rows != 0
            ):
                raise MsprofAnalysisError("fixed-16 Draft trace differs")
        elif (
            fixed_count != 0
            or prefix_count + pending_count != verify_draft_count
        ):
            raise MsprofAnalysisError("committed-prefix Draft trace differs")
    return model_to_role, expected_by_role, normalized_trace, unified


def _read_op_summaries(
    files: Sequence[Path],
) -> tuple[dict[tuple[int, int], dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[int, int], dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    common_root = Path(os.path.commonpath([str(path.parent) for path in files]))
    for path in files:
        file_identity = _filename_model_infer(path)
        row_count = 0
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise MsprofAnalysisError(f"{path}: op_summary CSV has no header")
            for raw in reader:
                if not raw or not any((value or "").strip() for value in raw.values()):
                    continue
                row_count += 1
                row = _normalized_row(raw)
                model_value = _value(row, ("modelid",), required=False)
                infer_value = _value(
                    row, ("inferid", "iterationid"), required=False
                )
                if model_value is None or infer_value is None:
                    if file_identity is None:
                        raise MsprofAnalysisError(
                            f"{path}: op_summary needs Model ID and Infer ID columns "
                            "or an op_summary_<device>_<model>_<iteration>.csv name"
                        )
                    model_id, infer_id = file_identity
                else:
                    model_id = _integer(
                        model_value, field="Model ID", source=path
                    )
                    infer_id = _integer(
                        infer_value, field="Infer ID", source=path
                    )
                duration_value = _value(
                    row,
                    ("taskdurationus", "taskduration", "tasktimeus"),
                )
                duration_us = _number(
                    str(duration_value), field="Task Duration(us)", source=path
                )
                op_name = str(
                    _value(row, ("opname", "operatorname", "kernelname"))
                )
                op_type = str(
                    _value(
                        row,
                        ("optype", "operatortype", "kerneltype", "tasktype"),
                    )
                )
                start_us: float | None = None
                start_value = _value(
                    row, ("taskstarttimeus",), required=False
                )
                if start_value is not None:
                    start_us = _number(
                        start_value, field="Task Start Time(us)", source=path
                    )
                else:
                    start_ns = _value(
                        row, ("taskstarttimens",), required=False
                    )
                    if start_ns is not None:
                        start_us = _number(
                            start_ns, field="Task Start Time(ns)", source=path
                        ) / 1000.0
                key = (model_id, infer_id)
                group = groups.get(key)
                if group is None:
                    group = {
                        "model_id": model_id,
                        "infer_id": infer_id,
                        "source": path,
                        "rows": [],
                    }
                    groups[key] = group
                elif group["source"] != path:
                    raise MsprofAnalysisError(
                        "duplicate model/infer group appears in multiple op_summary "
                        f"files: model={model_id}, infer={infer_id}"
                    )
                group["rows"].append(
                    {
                        "duration_us": duration_us,
                        "start_us": start_us,
                        "op_name": op_name,
                        "op_type": op_type,
                    }
                )
        if row_count == 0:
            raise MsprofAnalysisError(f"{path}: op_summary CSV is empty")
        records.append(_record(path, relative_to=common_root, rows=row_count))
    return groups, records


def _read_api_statistics(
    files: Sequence[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    common_root = Path(os.path.commonpath([str(path.parent) for path in files]))
    aggregates: defaultdict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"time_us": 0.0, "count": 0}
    )
    seen_rows: set[tuple[Any, ...]] = set()
    records: list[dict[str, Any]] = []
    for path in files:
        row_count = 0
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise MsprofAnalysisError(f"{path}: api_statistic CSV has no header")
            for raw in reader:
                if not raw or not any((value or "").strip() for value in raw.values()):
                    continue
                row_count += 1
                row = _normalized_row(raw)
                api_name = str(_value(row, ("apiname", "api")))
                level = str(_value(row, ("level",), required=False) or "unknown")
                device = str(
                    _value(row, ("deviceid",), required=False) or "unknown"
                )
                time_us = _number(
                    str(_value(row, ("timeus", "time"))),
                    field="API Time(us)",
                    source=path,
                )
                count = _integer(
                    str(_value(row, ("count",))), field="API Count", source=path
                )
                identity = (device, level, api_name, time_us, count)
                if identity in seen_rows:
                    continue
                seen_rows.add(identity)
                aggregate = aggregates[(level, api_name)]
                aggregate["time_us"] += time_us
                aggregate["count"] += count
        if row_count == 0:
            raise MsprofAnalysisError(f"{path}: api_statistic CSV is empty")
        records.append(_record(path, relative_to=common_root, rows=row_count))
    values = [
        {
            "level": level,
            "api_name": api_name,
            "time_us": value["time_us"],
            "count": value["count"],
            "average_us": (
                value["time_us"] / value["count"] if value["count"] else 0.0
            ),
        }
        for (level, api_name), value in aggregates.items()
    ]
    values.sort(key=lambda item: (-float(item["time_us"]), str(item["api_name"])))
    return values, records


def _distribution(values: Iterable[float]) -> dict[str, float | int]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise MsprofAnalysisError("cannot summarize an empty duration set")
    p90_index = max(0, math.ceil(0.9 * len(ordered)) - 1)
    total = sum(ordered)
    return {
        "count": len(ordered),
        "total_us": total,
        "mean_us": total / len(ordered),
        "median_us": statistics.median(ordered),
        "p90_us": ordered[p90_index],
        "min_us": ordered[0],
        "max_us": ordered[-1],
    }


def _find_api(
    api_rows: Sequence[Mapping[str, Any]], name: str
) -> Mapping[str, Any]:
    matches = [
        item
        for item in api_rows
        if str(item.get("api_name", "")).lower() == name.lower()
    ]
    if len(matches) != 1:
        raise MsprofAnalysisError(
            f"api_statistic must contain exactly one {name} aggregate; "
            f"found {len(matches)}"
        )
    return matches[0]


def analyze_incremental_msprof(
    *,
    profile_dir: str | Path,
    runner_report: str | Path,
) -> dict[str, Any]:
    """Map every profiled model invocation back to its OM role and gear."""

    profile_root = Path(profile_dir).expanduser().resolve()
    runner_path = Path(runner_report).expanduser().resolve()
    if not profile_root.is_dir():
        raise MsprofAnalysisError(f"profile directory does not exist: {profile_root}")
    if not runner_path.is_file():
        raise MsprofAnalysisError(f"runner report does not exist: {runner_path}")
    report = _load_runner_report(runner_path)
    model_to_role, expected_by_role, trace, unified = _validate_runner_report(
        report
    )

    op_files = sorted(profile_root.rglob("op_summary_*.csv"))
    if not op_files:
        raise MsprofAnalysisError(
            "profile contains no op_summary CSV; export every model/iteration"
        )
    api_files = sorted(profile_root.rglob("api_statistic_*.csv"))
    if not api_files:
        raise MsprofAnalysisError(
            "profile contains no api_statistic CSV; collect AscendCL/runtime API data"
        )
    groups, op_records = _read_op_summaries(op_files)
    api_rows, api_records = _read_api_statistics(api_files)

    unexpected_models = sorted(
        {model_id for model_id, _infer_id in groups} - set(model_to_role)
    )
    if unexpected_models:
        raise MsprofAnalysisError(
            f"op_summary contains unknown model IDs: {unexpected_models}"
        )
    trace_by_model: defaultdict[int, list[dict[str, int]]] = defaultdict(list)
    for event in trace:
        trace_by_model[event["model_id"]].append(event)
    observed_by_model: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for group in groups.values():
        observed_by_model[group["model_id"]].append(group)
    for values in observed_by_model.values():
        values.sort(key=lambda item: int(item["infer_id"]))

    invocations: list[dict[str, Any]] = []
    operator_totals: defaultdict[tuple[str, int, str, str], dict[str, float | int]] = (
        defaultdict(lambda: {"task_count": 0, "task_duration_us": 0.0})
    )
    for model_id, role in model_to_role.items():
        expected_events = trace_by_model[model_id]
        observed_groups = observed_by_model[model_id]
        if len(observed_groups) != len(expected_events):
            raise MsprofAnalysisError(
                f"{role} profile coverage differs: expected "
                f"{len(expected_events)}, observed {len(observed_groups)}; "
                "export every model ID and iteration"
            )
        for event, group in zip(expected_events, observed_groups, strict=True):
            rows = group["rows"]
            durations = [float(item["duration_us"]) for item in rows]
            total_us = sum(durations)
            starts = [item["start_us"] for item in rows]
            wall_span_us: float | None = None
            if all(value is not None for value in starts):
                typed_starts = [float(value) for value in starts]
                wall_span_us = max(
                    start + duration
                    for start, duration in zip(typed_starts, durations, strict=True)
                ) - min(typed_starts)
            invocation = {
                "trace_ordinal": event["ordinal"],
                "model_id": model_id,
                "role": role,
                "infer_id": int(group["infer_id"]),
                "physical_rows": event["physical_rows"],
                "task_count": len(rows),
                "task_duration_us": total_us,
                "task_wall_span_us": wall_span_us,
            }
            invocations.append(invocation)
            for item in rows:
                key = (
                    role,
                    event["physical_rows"],
                    str(item["op_type"]),
                    str(item["op_name"]),
                )
                operator_totals[key]["task_count"] = int(
                    operator_totals[key]["task_count"]
                ) + 1
                operator_totals[key]["task_duration_us"] = float(
                    operator_totals[key]["task_duration_us"]
                ) + float(item["duration_us"])
    invocations.sort(key=lambda item: int(item["trace_ordinal"]))

    by_role: dict[str, Any] = {}
    by_role_and_rows: dict[str, Any] = {}
    for role in expected_by_role:
        role_invocations = [item for item in invocations if item["role"] == role]
        role_summary: dict[str, Any] = {
            "model_id": next(
                model_id
                for model_id, mapped_role in model_to_role.items()
                if mapped_role == role
            ),
            "executions": len(role_invocations),
            "wall_span_available": all(
                item["task_wall_span_us"] is not None for item in role_invocations
            ) if role_invocations else False,
        }
        if not role_invocations:
            by_role[role] = role_summary
            continue
        role_summary["invocation_task_duration"] = _distribution(
            float(item["task_duration_us"]) for item in role_invocations
        )
        if role_summary["wall_span_available"]:
            role_summary["invocation_task_wall_span"] = _distribution(
                float(item["task_wall_span_us"]) for item in role_invocations
            )
        by_role[role] = role_summary
        for physical_rows in sorted(
            {int(item["physical_rows"]) for item in role_invocations}
        ):
            subset = [
                item
                for item in role_invocations
                if int(item["physical_rows"]) == physical_rows
            ]
            by_role_and_rows[f"{role}:T={physical_rows}"] = {
                "role": role,
                "physical_rows": physical_rows,
                "invocation_task_duration": _distribution(
                    float(item["task_duration_us"]) for item in subset
                ),
            }

    total_device_task_us = sum(
        float(item["task_duration_us"]) for item in invocations
    )
    top_operators = [
        {
            "role": role,
            "physical_rows": rows,
            "op_type": op_type,
            "op_name": op_name,
            "task_count": int(value["task_count"]),
            "task_duration_us": float(value["task_duration_us"]),
            "device_task_share": (
                float(value["task_duration_us"]) / total_device_task_us
                if total_device_task_us
                else 0.0
            ),
        }
        for (role, rows, op_type, op_name), value in operator_totals.items()
    ]
    top_operators.sort(
        key=lambda item: (-float(item["task_duration_us"]), str(item["op_name"]))
    )

    counters = report["execution_io_counters"]
    verify_transactions = int(counters["target_verify_commit_executions"])
    speculative_windows = int(
        counters.get("speculative_sync_windows", verify_transactions)
    )
    speculative_elided = int(
        counters.get(
            "speculative_synchronizations_elided",
            verify_transactions - speculative_windows,
        )
    )
    speculative_d2h_elided = int(
        counters.get("speculative_d2h_operations_elided", 0)
    )
    speculative_d2h_padding = int(
        counters.get("speculative_d2h_padding_bytes", 0)
    )
    speculative_staging_operations = int(
        counters.get("speculative_window_staging_operations", 0)
    )
    speculative_staging_bytes = int(
        counters.get("speculative_window_staging_bytes", 0)
    )
    prefill_verify_windows = int(
        counters.get("prefill_verify_coalesced_windows", 0)
    )
    prefill_verify_syncs_elided = int(
        counters.get("prefill_verify_synchronizations_elided", 0)
    )
    prefill_verify_d2h_elided = int(
        counters.get("prefill_verify_d2h_operations_elided", 0)
    )
    prefill_verify_d2h_padding = int(
        counters.get("prefill_verify_d2h_padding_bytes", 0)
    )
    expected_api_counts = {
        "aclmdlExecuteAsync": int(counters["model_executions"]),
        "aclrtMemcpyAsync": (
            int(counters["host_to_device_operations"])
            + int(counters["device_to_host_operations"])
            + int(counters["decode_id_device_compaction_operations"])
            + speculative_staging_operations
        ),
        "aclrtMemsetAsync": (
            int(counters["state_memset_operations"])
            + int(counters["state_initialization_memset_operations"])
        ),
        "aclrtSynchronizeStream": (
            int(counters["stream_synchronizations"])
            + int(counters["state_initialization_stream_synchronizations"])
        ),
    }
    api_count_gates: dict[str, Any] = {}
    for name, expected in expected_api_counts.items():
        if expected == 0:
            matches = [
                item
                for item in api_rows
                if str(item.get("api_name", "")).lower() == name.lower()
            ]
            observed = sum(int(item["count"]) for item in matches)
        else:
            observed = int(_find_api(api_rows, name)["count"])
        if observed != expected:
            raise MsprofAnalysisError(
                f"{name} profile count differs: expected {expected}, observed {observed}"
            )
        api_count_gates[name] = {
            "status": "PASS",
            "expected": expected,
            "observed": observed,
        }

    role_totals = {
        role: float(summary["invocation_task_duration"]["total_us"])
        for role, summary in by_role.items()
        if "invocation_task_duration" in summary
    }
    if not role_totals:
        raise MsprofAnalysisError("profile contains no mapped model task durations")
    dominant_role = max(role_totals, key=role_totals.get)
    return {
        "schema_version": 1,
        "status": "PASS",
        "scope": "incremental C++ runner msprof role/gear diagnostic",
        "formal_latency_evidence": False,
        "topology": report["abi"]["physical_topology"],
        "unified_target_step": unified,
        "runner_report": {
            "path": str(runner_path),
            "sha256": _sha256(runner_path),
            "runner_version": report.get("runner_version"),
            "device_id": report.get("device_id"),
            "device_memory_allocation_policy": report["protocol"].get(
                "device_memory_allocation_policy", "legacy-not-recorded"
            ),
        },
        "profile_root": str(profile_root),
        "input_files": {
            "op_summary": op_records,
            "api_statistic": api_records,
        },
        "coverage": {
            "status": "PASS",
            "expected_model_executions": len(trace),
            "observed_model_executions": len(invocations),
            "expected_by_role": expected_by_role,
            "observed_by_role": dict(Counter(item["role"] for item in invocations)),
        },
        "model_id_to_role": {
            str(model_id): role for model_id, role in model_to_role.items()
        },
        "invocations": invocations,
        "by_role": by_role,
        "by_role_and_physical_rows": by_role_and_rows,
        "device_task_summary": {
            "summed_task_duration_us": total_device_task_us,
            "dominant_role": dominant_role,
            "dominant_role_task_duration_us": role_totals[dominant_role],
            "dominant_role_share": (
                role_totals[dominant_role] / total_device_task_us
                if total_device_task_us
                else 0.0
            ),
        },
        "top_operators": top_operators[:50],
        "api_statistics": api_rows,
        "api_count_gates": api_count_gates,
        "expected_memcpy_signature": {
            "host_to_device": {
                "operations": counters["host_to_device_operations"],
                "bytes": counters["host_to_device_bytes"],
            },
            "device_to_host": {
                "operations": counters["device_to_host_operations"],
                "bytes": counters["device_to_host_bytes"],
            },
            "device_to_device_decode_compaction": {
                "operations": counters["decode_id_device_compaction_operations"],
                "bytes": counters["decode_id_device_compaction_bytes"],
            },
            "device_to_device_speculative_staging": {
                "operations": speculative_staging_operations,
                "bytes": speculative_staging_bytes,
            },
            "manual_timeline_gate": (
                "Match these direction/count/byte totals against MemcpyAsync "
                "events; api_statistic has duration/count but no copy size."
            ),
        },
        "expected_draft_feature_signature": (
            {
                "policy": report["protocol"]["draft_feature_policy"],
                "physical_verify_rows": counters[
                    "draft_verify_feature_input_rows"
                ],
                "full_width_equivalent_rows": counters[
                    "draft_verify_full_width_equivalent_rows"
                ],
                "elided_rows": counters["draft_verify_feature_rows_elided"],
                "fixed_width_executions": counters[
                    "draft_verify_fixed_width_executions"
                ],
                "committed_prefix_executions": counters[
                    "draft_verify_committed_prefix_executions"
                ],
                "pending_upper_bound_executions": counters[
                    "draft_verify_pending_upper_bound_executions"
                ],
                "trace_gate": (
                    "sum draft-propose T<=16 physical_rows equals "
                    "draft_verify_feature_input_rows"
                ),
            }
            if int(report.get("schema_version", 0)) >= 6
            else {"status": "NOT_AVAILABLE_LEGACY_REPORT"}
        ),
        "expected_synchronization_signature": {
            "prefill_completion_policy": report["protocol"].get(
                "prefill_completion_policy", "legacy-separate"
            ),
            "stream_synchronizations": counters["stream_synchronizations"],
            "speculative_transactions": verify_transactions,
            "speculative_sync_windows": speculative_windows,
            "speculative_synchronizations_elided": speculative_elided,
            "speculative_d2h_operations_elided": speculative_d2h_elided,
            "speculative_d2h_padding_bytes": speculative_d2h_padding,
            "speculative_window_staging_operations": (
                speculative_staging_operations
            ),
            "speculative_window_staging_bytes": speculative_staging_bytes,
            "prefill_verify_coalesced_windows": prefill_verify_windows,
            "prefill_verify_synchronizations_elided": (
                prefill_verify_syncs_elided
            ),
            "prefill_verify_d2h_operations_elided": (
                prefill_verify_d2h_elided
            ),
            "prefill_verify_d2h_padding_bytes": (
                prefill_verify_d2h_padding
            ),
            "closure": (
                "speculative_sync_windows + "
                "speculative_synchronizations_elided + "
                "prefill_verify_coalesced_windows == "
                "speculative_transactions"
            ),
            "device_to_host_closure": (
                "device_to_host_operations + "
                "speculative_d2h_operations_elided + "
                "prefill_verify_d2h_operations_elided == "
                "prefill_completion_synchronizations + "
                "target_decode1_executions + speculative_transactions"
            ),
        },
        "claim_boundary": (
            "msprof is collector-perturbed diagnostic evidence. Do not use "
            "these durations for promotion or closed-runtime comparison; rerun "
            "the same hashes with profiling disabled and retain exactly ten "
            "measurements after three warmups."
        ),
    }


__all__ = [
    "MsprofAnalysisError",
    "analyze_incremental_msprof",
]
