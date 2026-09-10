#!/usr/bin/env python3
"""Profile ordinary or DFlash OM stages through the shared msprof CLI controller."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "models/dflash_v1"))
from msprof_summary import summarize  # also available to standalone CSV consumers
from msprof_cli import stages_for_mode


def token_csv(value: object) -> str:
    if isinstance(value, str):
        parts = value.strip().split(",")
        if not all(part.strip().isdigit() for part in parts):
            raise ValueError("token IDs must be non-negative comma-separated integers")
        value = [int(part) for part in parts]
    if (not isinstance(value, list) or not value
            or any(type(token) is not int or token < 0 for token in value)):
        raise ValueError("token IDs must be a nonempty list of non-negative integers")
    return ",".join(str(token) for token in value)


def prompt_and_eos(args: argparse.Namespace, run: Path) -> tuple[str, str, str]:
    eos = args.eos_token_ids
    if args.prompt_token_ids is not None:
        prompt, source = args.prompt_token_ids, "--prompt-token-ids"
    else:
        report_path = (args.prompt_report or run / "reports/cpp-paired.json").expanduser()
        if args.prompt_report is not None or report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            prompt = report["prompt_token_ids"]
            if eos is None:
                eos = report.get("eos_token_ids")
            source = str(report_path)
        else:
            prompt_path = run / "prompt-ids.csv"
            prompt = prompt_path.read_text(encoding="utf-8")
            source = str(prompt_path)
    return token_csv(prompt), token_csv(eos if eos is not None else [248044]), source


def parser(default_stage="all") -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--profile-mode", "--mode", choices=("ordinary", "dflash"), default="dflash",
                        help="OM route; ordinary loads two OMs, DFlash loads three")
    result.add_argument("--profile-stage", "--stage", choices=("prefill", "decode", "draft", "verify", "all"),
                        default=default_stage, help="one selected stage or separate first-call windows for all stages")
    result.add_argument("--run-dir", type=Path, default=os.environ.get("AI_RUN_DIR"),
                        help="Existing run directory (default: AI_RUN_DIR)")
    result.add_argument("--runner", type=Path, default=os.environ.get("CPP_RUNNER"),
                        help="Matching AscendCL runner (default: CPP_RUNNER)")
    result.add_argument("--deployment-manifest", type=Path,
                        help="Default: RUN_DIR/artifacts/deployment-manifest.json")
    prompt = result.add_mutually_exclusive_group()
    prompt.add_argument("--prompt-report", type=Path,
                        help="Read prompt_token_ids and EOS from this infer-cpp report; "
                             "default: RUN_DIR/reports/cpp-paired.json, then prompt-ids.csv")
    prompt.add_argument("--prompt-token-ids", help="Explicit comma-separated prompt IDs")
    result.add_argument("--eos-token-ids", help="Override report EOS; without a report, default: 248044")
    result.add_argument("--msprof-bin", default=os.environ.get("MSPROF_BIN") or "msprof")
    result.add_argument("--device-id", type=int, default=0)
    result.add_argument("--max-new-tokens", type=int, default=32)
    result.add_argument("--max-draft-tokens", type=int, default=15)
    result.add_argument("--profile-warmup", type=int, default=1)
    result.add_argument("--profile-audit-draft-inputs", action="store_true",
                        help="For DFlash draft/all, hash features, KV and controls before capture (diagnostic only)")
    result.add_argument("--profile-timeout", type=int, default=600)
    result.add_argument("--aic-metrics", choices=("PipeUtilization", "Memory", "MemoryUB"),
                        default="PipeUtilization")
    return result


def run_profile(args: argparse.Namespace) -> Path:
    mode, stage = args.profile_mode, args.profile_stage
    available = stages_for_mode(mode, "cpp")
    if stage != "all" and stage not in available:
        raise ValueError(f"{mode} stages: {', '.join((*available, 'all'))}")
    if args.profile_audit_draft_inputs and (mode != "dflash" or stage not in {"draft", "all"}):
        raise ValueError("--profile-audit-draft-inputs requires dflash draft/all")
    if args.max_new_tokens < (1 if stage == "prefill" else 2):
        raise ValueError("prefill needs max-new-tokens >= 1; decode/Draft/verify need >= 2")
    if args.run_dir is None or args.runner is None:
        raise ValueError("Set AI_RUN_DIR and CPP_RUNNER, or pass --run-dir and --runner")
    run = args.run_dir.expanduser().resolve()
    runner = args.runner.expanduser().resolve()
    manifest = (args.deployment_manifest or run / "artifacts/deployment-manifest.json").expanduser().resolve()
    if not run.is_dir() or run.is_relative_to(REPOSITORY):
        raise ValueError("--run-dir must exist outside the source repository")
    if not runner.is_file() or not os.access(runner, os.X_OK):
        raise ValueError(f"C++ runner is missing or not executable: {runner}")
    if not manifest.is_file():
        raise ValueError(f"Deployment manifest not found: {manifest}")
    if (args.device_id < 0 or args.profile_warmup < 0 or args.profile_timeout <= 0
            or not 1 <= args.max_draft_tokens <= 15):
        raise ValueError("profiling needs device-id/warmup >= 0, timeout > 0, "
                         "and 1..15 max-draft-tokens")
    prompt, eos, prompt_source = prompt_and_eos(args, run)
    msprof = shutil.which(os.path.expanduser(args.msprof_bin))
    if msprof is None:
        raise ValueError("msprof not found; load the CANN environment or set MSPROF_BIN")
    if os.environ.get("ASCEND310P_SIMULATION_ONLY") == "1":
        raise ValueError("simulation-only target cannot produce msprof device measurements")
    parent = run / "msprof"
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=f"{mode}-{stage}-", dir=parent))
    plan, capture = output / "plan.txt", output / "capture"
    environment = dict(os.environ)
    environment["AI_RUN_DIR"] = str(run)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(REPOSITORY / "framework/python"), str(REPOSITORY), environment.get("PYTHONPATH"),
    )))
    print(f"Output: {output}\nMode/stage: {mode}/{stage}\nPrompt source: {prompt_source}\nEOS IDs: {eos}", flush=True)
    request = {
        "profile_mode": mode, "profile_stage": stage,
        "stages": list(available) if stage == "all" else [stage],
        "deployment_manifest": {"path": str(manifest), "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()},
        "runner": {"path": str(runner), "sha256": hashlib.sha256(runner.read_bytes()).hexdigest()},
        "prompt_source": prompt_source,
        "prompt_token_ids": [int(value) for value in prompt.split(",")],
        "eos_token_ids": [int(value) for value in eos.split(",")],
        "max_new_tokens": args.max_new_tokens, "max_draft_tokens": args.max_draft_tokens,
        "profile_warmup": args.profile_warmup, "aic_metrics": args.aic_metrics,
        "profile_audit_draft_inputs": args.profile_audit_draft_inputs,
    }
    (output / "profile-request.json").write_text(json.dumps(request, indent=2) + "\n")
    subprocess.run([
        sys.executable, "-B", "-m", "qwen35_dflash.ascend310p", "prepare-chunk-plan",
        "--deployment-manifest", str(manifest), "--mode", mode, "--output", str(plan),
    ], check=True, env=environment, cwd=REPOSITORY)
    digest = hashlib.sha256(plan.read_bytes()).hexdigest()
    print("Loading selected OMs once. Each stage gets fresh-state warmup and one capture window; "
          "prefill includes all prompt chunks. Model loading can take several minutes.", flush=True)
    print(f"Per-iteration input/output trace: {capture / 'profile/msprof' / (stage + '.iterations.jsonl')}",
          flush=True)
    subprocess.run([
        "bash", str(REPOSITORY / "tools/run_msprof.sh"),
        "--label", stage, "--output-dir", str(capture), "--python", sys.executable,
        "--msprof-bin", msprof, "--profile-backend", "cpp", "--profile-mode", mode,
        "--profile-stage", stage, "--profile-warmup", str(args.profile_warmup),
        "--profile-timeout", str(args.profile_timeout), "--aic-metrics", args.aic_metrics,
        "--", str(runner), "--model-kind", "chunk", "--model", str(plan),
        "--model-sha256", digest, "--prompt-token-ids", prompt, "--eos-token-ids", eos,
        "--device-id", str(args.device_id), "--max-new-tokens", str(args.max_new_tokens),
        "--max-draft-tokens", str(args.max_draft_tokens),
        *(["--profile-audit-draft-inputs", "true"] if args.profile_audit_draft_inputs else []),
    ], check=True, env=environment, cwd=REPOSITORY)
    print(f"\nSynchronized stage timing: {capture / (stage + '-stage-summary.csv')}")
    print(f"Per-stage operator timings: {capture / (stage + '-operator-types.csv')}")
    print(f"Per-stage hotspots: {capture / (stage + '-hotspots.txt')}")
    return output


def main(argv: list[str] | None = None, *, default_stage="all") -> int:
    args = parser(default_stage).parse_args(argv)
    try:
        run_profile(args)
    except subprocess.CalledProcessError as error:
        print(f"profile_om: command failed (exit {error.returncode}); "
              "inspect capture/log and capture/profile/msprof/*.iterations.jsonl "
              "under the Output directory printed above", file=sys.stderr)
        return error.returncode if error.returncode > 0 else 1
    except (OSError, ValueError, KeyError) as error:
        print(f"profile_om: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
