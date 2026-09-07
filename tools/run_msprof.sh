#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_msprof.sh --label NAME --output-dir DIR [options] -- COMMAND [ARG ...]

Required:
  --label NAME              Stable case label.
  --output-dir DIR          Evidence root outside this copied source tree.

Options:
  --python PATH             Python used for preflight/manifest (default: python3).
  --msprof-bin PATH         msprof executable (default: MSPROF_BIN or msprof).
  --aic-metrics NAME        AI Core metrics (default: PipeUtilization).
  --task-time LEVEL         msprof task-time value (default: on).
  --profile-stage STAGE     Collect one prefill or draft-verify in run_npu/run_rollback.
  --profile-warmup N        Unprofiled fresh-state warmups for a stage (default: 1).
  --msprof-arg ARG          Append one safe msprof option; repeat as needed.
  --msproftx                Opt in to msproftx and benchmark MSTX ranges.
  --no-msproftx             Keep MSTX disabled (default; compatibility option).
  -h, --help                Show this help.

The wrapper requires a real torch_npu device and rejects CPU/operator fallback.
Profile data, logs, and the invocation manifest are written below --output-dir.
EOF
}

fail() {
  printf 'run_msprof.sh: %s\n' "$*" >&2
  exit 2
}

label=""
output_dir=""
python_bin="${PYTHON_BIN:-python3}"
msprof_bin="${MSPROF_BIN:-msprof}"
aic_metrics="PipeUtilization"
task_time="on"
profile_stage=""
profile_warmup=1
profile_warmup_set="false"
stage_report=""
msproftx="off"
extra_msprof_args=()

while (($#)); do
  case "$1" in
    --label)
      (($# >= 2)) || fail "--label requires a value"
      label="$2"
      shift 2
      ;;
    --output-dir)
      (($# >= 2)) || fail "--output-dir requires a value"
      output_dir="$2"
      shift 2
      ;;
    --python)
      (($# >= 2)) || fail "--python requires a value"
      python_bin="$2"
      shift 2
      ;;
    --msprof-bin)
      (($# >= 2)) || fail "--msprof-bin requires a value"
      msprof_bin="$2"
      shift 2
      ;;
    --aic-metrics)
      (($# >= 2)) || fail "--aic-metrics requires a value"
      aic_metrics="$2"
      shift 2
      ;;
    --task-time)
      (($# >= 2)) || fail "--task-time requires a value"
      task_time="$2"
      shift 2
      ;;
    --profile-stage)
      (($# >= 2)) || fail "--profile-stage requires prefill or draft-verify"
      profile_stage="$2"
      case "$profile_stage" in
        prefill|draft-verify) ;;
        *) fail "--profile-stage must be prefill or draft-verify" ;;
      esac
      shift 2
      ;;
    --profile-warmup)
      (($# >= 2)) || fail "--profile-warmup requires a count"
      [[ "$2" =~ ^[0-9]+$ ]] || fail "--profile-warmup must be non-negative"
      profile_warmup="$2"
      profile_warmup_set="true"
      shift 2
      ;;
    --msprof-arg)
      (($# >= 2)) || fail "--msprof-arg requires a value"
      case "$2" in
        --output*|--application*|--pid*|--dynamic*)
          fail "--msprof-arg may not override process/output ownership: $2"
          ;;
      esac
      extra_msprof_args+=("$2")
      shift 2
      ;;
    --msproftx)
      msproftx="on"
      shift
      ;;
    --no-msproftx)
      msproftx="off"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      fail "unknown option: $1"
      ;;
  esac
done

[[ -n "$label" ]] || fail "--label is required"
[[ "$label" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail "invalid label: $label"
[[ -n "$output_dir" ]] || fail "--output-dir is required"
(($# > 0)) || fail "a target command is required after --"
[[ "${ASCEND310P_SIMULATION_ONLY:-0}" != "1" ]] || \
  fail "simulation-only target profiles cannot produce msprof evidence"

application=("$@")
stage_runner="false"
for ((index=0; index<${#application[@]}; index++)); do
  case "${application[index]}" in
    --profile-stage|--profile-stage=*|--profile-output|--profile-output=*|--profile-warmup|--profile-warmup=*|--profile-aic-metrics|--profile-aic-metrics=*)
      fail "put --profile-stage/--profile-warmup on run_msprof.sh before --"
      ;;
    -m)
      case "${application[index+1]:-}" in
        models.dflash_v1.run_npu|models.dflash_v1.run_rollback) stage_runner="true" ;;
      esac
      ;;
    --report) stage_report="${application[index+1]:-}" ;;
    --report=*) stage_report="${application[index]#--report=}" ;;
  esac
done
if [[ -n "$profile_stage" ]]; then
  [[ "$stage_runner" == "true" ]] || \
    fail "--profile-stage requires python -m models.dflash_v1.run_npu (or run_rollback)"
  [[ "$msproftx" == "off" ]] || fail "stage API capture does not use --msproftx"
  [[ "$task_time" == "on" ]] || fail "stage API capture requires --task-time on"
  ((${#extra_msprof_args[@]} == 0)) || fail "stage API capture does not accept --msprof-arg"
  case "$aic_metrics" in
    PipeUtilization|Memory|MemoryUB) ;;
    *) fail "stage metrics must be PipeUtilization, Memory or MemoryUB" ;;
  esac
elif [[ "$profile_warmup_set" == "true" ]]; then
  fail "--profile-warmup requires --profile-stage"
fi
expect_device_value="false"
requested_device="npu:0"
for argument in "${application[@]}"; do
  if [[ "$expect_device_value" == "true" ]]; then
    [[ "$argument" == npu || "$argument" == npu:* ]] || \
      fail "explicit non-NPU device is forbidden: $argument"
    requested_device="$argument"
    expect_device_value="false"
    continue
  fi
  case "$argument" in
    --allow-op-fallback)
      fail "--allow-op-fallback is forbidden for target profiling"
      ;;
    --device)
      expect_device_value="true"
      ;;
    --device=*)
      device_value="${argument#--device=}"
      [[ "$device_value" == npu || "$device_value" == npu:* ]] || \
        fail "explicit non-NPU device is forbidden: $device_value"
      requested_device="$device_value"
      ;;
  esac
done
[[ "$expect_device_value" == "false" ]] || fail "--device requires a value"

if [[ "$python_bin" == */* ]]; then
  [[ -x "$python_bin" ]] || fail "Python is not executable: $python_bin"
else
  python_bin="$(command -v "$python_bin")" || fail "Python was not found"
fi
if [[ "$msprof_bin" == */* ]]; then
  [[ -x "$msprof_bin" ]] || fail "msprof is not executable: $msprof_bin"
else
  msprof_bin="$(command -v "$msprof_bin")" || fail "msprof was not found in PATH"
fi
command -v npu-smi >/dev/null 2>&1 || fail "npu-smi was not found in PATH"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source_root="$(cd -- "$script_dir/.." && pwd -P)"
output_root="$($python_bin -B -c \
  'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' \
  "$output_dir")"
if [[ "$output_root" == "$source_root" || "$output_root" == "$source_root/"* ]]; then
  fail "--output-dir must be outside the copied source tree: $source_root"
fi

profile_dir="$output_root/profile/msprof/$label"
log_dir="$output_root/log"
manifest_dir="$output_root/manifest"
manifest_path="$manifest_dir/$label.json"
runtime_log="$log_dir/msprof-$label.log"
preflight_log="$log_dir/preflight-$label.log"
device_log="$log_dir/device-$label.log"

[[ ! -e "$profile_dir" ]] || fail "profile output already exists: $profile_dir"
[[ ! -e "$manifest_path" ]] || fail "manifest already exists: $manifest_path"
mkdir -p "$output_root/profile/msprof" "$log_dir" "$manifest_dir"

if [[ -n "$profile_stage" ]]; then
  unset DFLASH_MSPROF_PROCESS_CAPTURE || true
  application+=(
    --profile-stage "$profile_stage" --profile-output "$profile_dir"
    --profile-warmup "$profile_warmup" --profile-aic-metrics "$aic_metrics"
  )
  if [[ -z "$stage_report" ]]; then
    stage_report="$output_root/$label-stage-report.json"
    application+=(--report "$stage_report")
  fi
else
  export DFLASH_MSPROF_PROCESS_CAPTURE=1
fi

if [[ "$msproftx" == "on" ]]; then
  export DFLASH_BENCHMARK_MSTX=1
else
  unset DFLASH_BENCHMARK_MSTX || true
fi
export DFLASH_MSPROF_DEVICE="$requested_device"

set +e
"$python_bin" -B - >"$preflight_log" 2>&1 <<'PY'
import json
import os
import torch

try:
    import torch_npu
except ImportError as error:
    raise SystemExit(f"torch_npu import failed: {error}")

npu = getattr(torch, "npu", None)
if npu is None or not callable(getattr(npu, "is_available", None)):
    raise SystemExit("torch.npu.is_available is unavailable")
if not npu.is_available():
    raise SystemExit("no NPU device is available")
requested = os.environ.get("DFLASH_MSPROF_DEVICE", "npu:0")
npu.set_device(requested)
current = int(npu.current_device())
if os.environ.get("DFLASH_BENCHMARK_MSTX") == "1":
    try:
        import mstx  # noqa: F401
    except ImportError as error:
        raise SystemExit(f"mstx import failed: {error}")
print(json.dumps({
    "status": "PASS",
    "torch_version": torch.__version__,
    "torch_npu_version": getattr(torch_npu, "__version__", None),
    "requested_device": requested,
    "device_index": current,
    "device_name": str(npu.get_device_name(current)),
}, sort_keys=True))
PY
preflight_status=$?
set -e
if ((preflight_status != 0)); then
  printf 'NPU preflight failed; see %s\n' "$preflight_log" >&2
  exit "$preflight_status"
fi

npu-smi info >"$device_log" 2>&1 || {
  status=$?
  printf 'npu-smi info failed; see %s\n' "$device_log" >&2
  exit "$status"
}

msprof_version="$($msprof_bin --version 2>&1 || true)"
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

msprof_args=(
  "--output=$profile_dir"
  "--ascendcl=on"
  "--runtime-api=on"
  "--task-time=$task_time"
  "--aicpu=on"
  "--ai-core=on"
  "--aic-mode=task-based"
  "--aic-metrics=$aic_metrics"
)
if [[ "$msproftx" == "on" ]]; then
  msprof_args+=("--msproftx=on")
fi
msprof_args+=("${extra_msprof_args[@]}")
if [[ -n "$profile_stage" ]]; then
  # The application owns collection. msprof is used only after stop/finalize.
  msprof_args=("--export=on" "--output=$profile_dir" "--summary-format=csv")
fi

write_manifest() {
  local run_status="$1"
  local exit_code="$2"
  local finished_at="$3"
  "$python_bin" -B - \
    "$manifest_path" "$run_status" "$exit_code" "$started_at" "$finished_at" \
    "$label" "$profile_dir" "$runtime_log" "$preflight_log" "$device_log" \
    "$msprof_bin" "$msprof_version" "$source_root" "$aic_metrics" "$task_time" \
    "$msproftx" "$requested_device" "$profile_stage" "$profile_warmup" "$stage_report" \
    "${#msprof_args[@]}" "${msprof_args[@]}" \
    "${#application[@]}" "${application[@]}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

values = sys.argv[1:]
(
    manifest_path, run_status, exit_code, started_at, finished_at, label,
    profile_dir, runtime_log, preflight_log, device_log, msprof_bin,
    msprof_version, source_root, aic_metrics, task_time, msproftx,
    requested_device, profile_stage, profile_warmup, stage_report,
) = values[:20]
cursor = 20
msprof_count = int(values[cursor])
cursor += 1
msprof_args = values[cursor:cursor + msprof_count]
cursor += msprof_count
application_count = int(values[cursor])
cursor += 1
application = values[cursor:cursor + application_count]

redacted_application = list(application)
for index, argument in enumerate(redacted_application[:-1]):
    if argument == "--prompt":
        redacted_application[index + 1] = "<redacted-inline-prompt>"
for index, argument in enumerate(redacted_application):
    if argument.startswith("--prompt="):
        redacted_application[index] = "--prompt=<redacted-inline-prompt>"

root = Path(source_root)
source_hasher = hashlib.sha256()
source_files = 0
source_paths = [
    root / "framework",
    root / "models" / "dflash_v1",
    root / "models" / "internal_dflash_bridge.py",
    root / "models" / "modeling_qwen3_5_hiai_nd_dflash_rollback.py",
    root / "models" / "export_model_wrapper_qwen3_5_dflash_rollback.py",
    root / "tools" / "run_msprof.sh",
    root / "docs" / "DFLASH_RUN_AND_VALIDATE.md",
    root / "docs" / "QUANT_AIR_OM_FRAMEWORK.md",
    root / "config" / "npu_benchmark_v1.json",
]
expanded = []
for source in source_paths:
    if source.is_dir():
        expanded.extend(path for path in source.rglob("*") if path.is_file())
    elif source.is_file():
        expanded.append(source)
for path in sorted(set(expanded)):
    relative = path.relative_to(root)
    if "__pycache__" in relative.parts or path.suffix == ".pyc":
        continue
    source_hasher.update(str(relative).encode("utf-8"))
    source_hasher.update(b"\0")
    source_hasher.update(hashlib.sha256(path.read_bytes()).digest())
    source_files += 1

payload = {
    "schema_version": 4,
    "status": run_status,
    "exit_code": int(exit_code),
    "label": label,
    "started_at": started_at,
    "finished_at": finished_at or None,
    "source": {
        "source_root": source_root,
        "identity_method": "content_hash_without_vcs_metadata",
        "source_tree_sha256": source_hasher.hexdigest(),
        "source_files": source_files,
    },
    "target": {
        "device_required": "Ascend NPU",
        "requested_device": requested_device,
        "cpu_fallback_allowed": False,
        "preflight_log": preflight_log,
        "device_log": device_log,
    },
    "msprof": {
        "executable": msprof_bin,
        "version": msprof_version,
        "arguments": msprof_args,
        "aic_metrics": aic_metrics,
        "task_time": task_time,
        "msproftx": msproftx,
        "collector": "pyACL stage API" if profile_stage else "msprof process",
        "profile_stage": profile_stage or None,
        "profile_warmup": int(profile_warmup) if profile_stage else None,
    },
    "application": redacted_application,
    "artifacts": {
        "profile_dir": profile_dir, "runtime_log": runtime_log,
        "stage_report": stage_report if profile_stage else None,
    },
    "claim_boundary": (
        "msprof is diagnostic evidence, not the latency baseline; retain "
        "separate unprofiled 3-warmup/10-measurement ordinary and DFlash runs"
    ),
}
Path(manifest_path).write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}

write_manifest "RUNNING" 0 ""
set +e
if [[ -n "$profile_stage" ]]; then
  "${application[@]}" 2>&1 | tee "$runtime_log"
  msprof_status=${PIPESTATUS[0]}
  if ((msprof_status == 0)); then
    "$msprof_bin" "${msprof_args[@]}" 2>&1 | tee -a "$runtime_log"
    msprof_status=${PIPESTATUS[0]}
  fi
  if ((msprof_status == 0)); then
    "$python_bin" -B - "$profile_dir" "$stage_report" "$profile_stage" <<'PY' 2>&1 | tee -a "$runtime_log"
import csv
import json
from pathlib import Path
import sys

root, report_path, stage = sys.argv[1:]
report = json.loads(Path(report_path).read_text(encoding="utf-8"))
expected = {
    "prefill": int(stage == "prefill"),
    "draft": int(stage == "draft-verify"),
    "target_verify": int(stage == "draft-verify"),
}
if (report.get("status") != "PASS_CAPTURE"
        or report.get("profile_stage") != stage
        or report.get("capture_windows") != 1
        or report.get("captured_calls") != expected
        or report.get("operator_fallback_enabled") is not False
        or Path(report.get("profile_output", "")).resolve() != Path(root).resolve()):
    raise SystemExit("stage report does not prove one requested capture window")
rows = 0
for path in Path(root).rglob("op_summary*.csv"):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows += sum(1 for row in csv.DictReader(stream) if any(row.values()))
if not rows:
    raise SystemExit("msprof export produced no operator rows; inspect the raw PROF_* data")
print(f"PASS: one {stage} capture, {rows} exported operator rows")
PY
    msprof_status=${PIPESTATUS[0]}
  fi
else
  "$msprof_bin" "${msprof_args[@]}" "${application[@]}" 2>&1 | tee "$runtime_log"
  msprof_status=${PIPESTATUS[0]}
fi
set -e
finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if ((msprof_status == 0)); then
  write_manifest "PASS" "$msprof_status" "$finished_at"
else
  write_manifest "FAIL" "$msprof_status" "$finished_at"
fi
exit "$msprof_status"
