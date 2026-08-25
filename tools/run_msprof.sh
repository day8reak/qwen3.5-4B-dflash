#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_msprof.sh --label NAME --output-dir DIR [options] -- COMMAND [ARG ...]

Required:
  --label NAME              Stable case label.
  --output-dir DIR          Evidence root outside this Git repository.

Options:
  --python PATH             Python used for preflight/manifest (default: python3).
  --msprof-bin PATH         msprof executable (default: MSPROF_BIN or msprof).
  --aic-metrics NAME        AI Core metrics (default: PipeUtilization).
  --task-time LEVEL         msprof task-time value (default: on).
  --msprof-arg ARG          Append one msprof option; repeat as needed.
  --no-msproftx             Disable msproftx collection and benchmark MSTX ranges.
  -h, --help                Show this help.

The wrapper requires a real torch_npu device and rejects CPU fallback.  Profile
data, logs, and the invocation manifest are written below --output-dir only.
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
msproftx="on"
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
repo_root="$(git -C "$script_dir/.." rev-parse --show-toplevel)"
output_root="$($python_bin -B -c \
  'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' \
  "$output_dir")"
if [[ "$output_root" == "$repo_root" || "$output_root" == "$repo_root/"* ]]; then
  fail "--output-dir must be outside the source repository: $repo_root"
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
name = str(npu.get_device_name(current))
if os.environ.get("DFLASH_BENCHMARK_MSTX") == "1":
    try:
        import mstx  # noqa: F401
    except ImportError as error:
        raise SystemExit(
            f"mstx import failed: {error}; use --no-msproftx only when ranges "
            "are intentionally disabled"
        )
print(json.dumps({
    "status": "PASS",
    "torch_version": torch.__version__,
    "torch_npu_version": getattr(torch_npu, "__version__", None),
    "requested_device": requested,
    "device_index": current,
    "device_name": name,
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
git_commit="$(git -C "$repo_root" rev-parse HEAD)"
git_branch="$(git -C "$repo_root" branch --show-current)"
git_dirty="false"
if [[ -n "$(git -C "$repo_root" status --porcelain)" ]]; then
  git_dirty="true"
fi

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

write_manifest() {
  local run_status="$1"
  local exit_code="$2"
  local finished_at="$3"
  "$python_bin" -B - \
    "$manifest_path" "$run_status" "$exit_code" "$started_at" "$finished_at" \
    "$label" "$profile_dir" "$runtime_log" "$preflight_log" "$device_log" \
    "$msprof_bin" "$msprof_version" "$repo_root" "$git_commit" "$git_branch" \
    "$git_dirty" "$aic_metrics" "$task_time" "$msproftx" "$requested_device" \
    "${#msprof_args[@]}" "${msprof_args[@]}" \
    "${#application[@]}" "${application[@]}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

values = sys.argv[1:]
(
    manifest_path,
    run_status,
    exit_code,
    started_at,
    finished_at,
    label,
    profile_dir,
    runtime_log,
    preflight_log,
    device_log,
    msprof_bin,
    msprof_version,
    repo_root,
    git_commit,
    git_branch,
    git_dirty,
    aic_metrics,
    task_time,
    msproftx,
    requested_device,
) = values[:20]
cursor = 20
msprof_count = int(values[cursor])
cursor += 1
msprof_args = values[cursor : cursor + msprof_count]
cursor += msprof_count
application_count = int(values[cursor])
cursor += 1
application = values[cursor : cursor + application_count]

redacted_application = list(application)
for index, argument in enumerate(redacted_application[:-1]):
    if argument == "--prompt":
        redacted_application[index + 1] = "<redacted-inline-prompt>"
for index, argument in enumerate(redacted_application):
    if argument.startswith("--prompt="):
        redacted_application[index] = "--prompt=<redacted-inline-prompt>"

root = Path(repo_root)
source_hasher = hashlib.sha256()
source_files = 0
source_paths = [
    root / "models" / "dflash_v1",
    root / "models" / "internal_dflash_bridge.py",
    root / "models" / "modeling_qwen3_5_hiai_nd.py",
    root / "tools" / "run_msprof.sh",
    root / "docs" / "NPU_BENCHMARK.md",
    root / "config" / "npu_benchmark_v1.json",
    root / "SOURCE_LOCK.json",
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
    "schema_version": 1,
    "status": run_status,
    "exit_code": int(exit_code),
    "label": label,
    "started_at": started_at,
    "finished_at": finished_at or None,
    "source": {
        "repository": repo_root,
        "git_commit": git_commit,
        "git_branch": git_branch,
        "git_dirty": git_dirty == "true",
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
    },
    "application": redacted_application,
    "artifacts": {
        "profile_dir": profile_dir,
        "runtime_log": runtime_log,
    },
    "claim_boundary": (
        "msprof is diagnostic evidence and not the latency baseline; retain "
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
"$msprof_bin" "${msprof_args[@]}" "${application[@]}" 2>&1 | tee "$runtime_log"
msprof_status=${PIPESTATUS[0]}
set -e
finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if ((msprof_status == 0)); then
  write_manifest "PASS" "$msprof_status" "$finished_at"
else
  write_manifest "FAIL" "$msprof_status" "$finished_at"
fi
exit "$msprof_status"
