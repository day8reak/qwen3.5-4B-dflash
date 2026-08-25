#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_msprof.sh --label NAME [options] -- <target command> [arguments...]

Options:
  --label NAME              Stable case label (required).
  --aic-metrics NAME        AI Core metric set (default: PipeUtilization).
  --task-time LEVEL         msprof task-time level (default: on).
  --msprof-bin PATH         msprof executable (default: MSPROF_BIN or msprof).
  --msprof-arg ARG          Append one extra msprof option; repeat as needed.
  --no-msproftx             Do not enable MSTX/msproftx collection.
  -h, --help                Show this help.

The active target profile must be a real Ascend 310P profile. Profile data,
logs, and the invocation manifest are written only below AI_RUN_DIR.
EOF
}

fail() {
  printf 'run_msprof.sh: %s\n' "$*" >&2
  exit 2
}

label=""
aic_metrics="PipeUtilization"
task_time="on"
msprof_bin="${MSPROF_BIN:-msprof}"
msproftx="on"
extra_msprof_args=()

while (($#)); do
  case "$1" in
    --label)
      (($# >= 2)) || fail "--label requires a value"
      label="$2"
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
    --msprof-bin)
      (($# >= 2)) || fail "--msprof-bin requires a value"
      msprof_bin="$2"
      shift 2
      ;;
    --msprof-arg)
      (($# >= 2)) || fail "--msprof-arg requires a value"
      case "$2" in
        --output*|--application*|--pid*|--dynamic*)
          fail "--msprof-arg may not override process or output ownership: $2"
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
(($# > 0)) || fail "a target command is required after --"
[[ -n "${AI_RUN_DIR:-}" ]] || fail "AI_RUN_DIR is unset; start a workspace session"
[[ -n "${AI_MODEL_ROOT:-}" ]] || fail "AI_MODEL_ROOT is unset; activate the model environment"
[[ -x "${AI_MODEL_PYTHON:-}" ]] || fail "AI_MODEL_PYTHON is not executable"
[[ -n "${AI_TARGET_PROFILE:-}" ]] || fail "AI_TARGET_PROFILE is unset"

case "$AI_TARGET_PROFILE" in
  */simulation|*/simulation/*)
    fail "the active target profile is simulation-only: $AI_TARGET_PROFILE"
    ;;
esac

for argument in "$@"; do
  [[ "$argument" != "--allow-op-fallback" ]] || \
    fail "--allow-op-fallback is forbidden for target profiling"
done

if [[ "$msprof_bin" == */* ]]; then
  [[ -x "$msprof_bin" ]] || fail "msprof is not executable: $msprof_bin"
else
  msprof_bin="$(command -v "$msprof_bin")" || fail "msprof was not found in PATH"
fi

profile_parent="$AI_RUN_DIR/profile/msprof"
profile_dir="$profile_parent/$label"
log_dir="$AI_RUN_DIR/log"
manifest_dir="$AI_RUN_DIR/out/performance/msprof"
manifest_path="$manifest_dir/$label.manifest.json"
runtime_log="$log_dir/msprof-$label.log"
preflight_log="$log_dir/msprof-$label-preflight.log"
device_log="$log_dir/msprof-$label-device.log"

[[ ! -e "$profile_dir" ]] || fail "profile output already exists: $profile_dir"
mkdir -p "$profile_parent" "$log_dir" "$manifest_dir"

[[ -n "${AI_TARGET_PREFLIGHT:-}" && -x "$AI_TARGET_PREFLIGHT" ]] || \
  fail "AI_TARGET_PREFLIGHT is unavailable"
"$AI_TARGET_PREFLIGHT" >"$preflight_log" 2>&1 || {
  status=$?
  printf 'target preflight failed; see %s\n' "$preflight_log" >&2
  exit "$status"
}

if command -v npu-smi >/dev/null 2>&1; then
  npu-smi info >"$device_log" 2>&1 || true
else
  printf 'npu-smi is unavailable\n' >"$device_log"
fi

msprof_version="$($msprof_bin --version 2>&1 || true)"
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
git_commit="$(git -C "$AI_MODEL_ROOT" rev-parse HEAD)"
git_branch="$(git -C "$AI_MODEL_ROOT" branch --show-current)"
git_root="$(git -C "$AI_MODEL_ROOT" rev-parse --show-toplevel)"
model_relative_path="${AI_MODEL_ROOT#"$git_root"/}"
model_git_tracked="false"
if git -C "$git_root" ls-files --error-unmatch \
  "$model_relative_path/project.yaml" >/dev/null 2>&1; then
  model_git_tracked="true"
fi
application=("$@")
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
  "$AI_MODEL_PYTHON" - \
    "$manifest_path" "$run_status" "$exit_code" "$started_at" "$finished_at" \
    "$label" "$profile_dir" "$runtime_log" "$preflight_log" "$device_log" \
    "$msprof_bin" "$msprof_version" "$git_commit" "$git_branch" \
    "$git_root" "$AI_MODEL_ROOT" "$model_git_tracked" \
    "$AI_TARGET_PROFILE" "${AI_TARGET_PROFILE_ID:-unknown}" \
    "$aic_metrics" "$task_time" "$msproftx" \
    "${#msprof_args[@]}" "${msprof_args[@]}" \
    "${#application[@]}" "${application[@]}" <<'PY'
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
    git_commit,
    git_branch,
    git_root,
    model_root,
    model_git_tracked,
    target_profile,
    target_profile_id,
    aic_metrics,
    task_time,
    msproftx,
) = values[:22]
cursor = 22
msprof_count = int(values[cursor])
cursor += 1
msprof_args = values[cursor : cursor + msprof_count]
cursor += msprof_count
application_count = int(values[cursor])
cursor += 1
application = values[cursor : cursor + application_count]

import hashlib

source_hasher = hashlib.sha256()
source_files = 0
source_roots = [
    "model",
    "specs",
    "targets/ascend310p/abi",
    "targets/ascend310p/runtime",
    "targets/ascend310p/scripts",
]
model_path = Path(model_root)
for relative_root in source_roots:
    root = model_path / relative_root
    if not root.exists():
        continue
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(model_path)
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
    "finished_at": finished_at,
    "workspace_source": {
        "git_root": git_root,
        "git_commit": git_commit,
        "git_branch": git_branch,
    },
    "model_source": {
        "root": model_root,
        "tracked_by_workspace_git": model_git_tracked == "true",
        "source_tree_sha256": source_hasher.hexdigest(),
        "source_files": source_files,
    },
    "target": {
        "profile": target_profile,
        "profile_id": target_profile_id,
        "cpu_fallback_allowed": False,
    },
    "msprof": {
        "executable": msprof_bin,
        "version": msprof_version,
        "arguments": msprof_args,
        "aic_metrics": aic_metrics,
        "task_time": task_time,
        "msproftx": msproftx,
    },
    "application": application,
    "artifacts": {
        "profile_dir": profile_dir,
        "runtime_log": runtime_log,
        "preflight_log": preflight_log,
        "device_log": device_log,
    },
    "claim_boundary": (
        "msprof data diagnoses the target run; final performance promotion "
        "requires an unprofiled 10/10 accuracy-stable device distribution"
    ),
}
Path(manifest_path).write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
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
