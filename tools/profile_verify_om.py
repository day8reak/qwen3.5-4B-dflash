#!/usr/bin/env python3
"""Profile one target_verify OM through the msprof CLI and rank its task times."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


REPOSITORY = Path(__file__).resolve().parents[1]


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


def summarize(capture: Path, output: Path, top: int = 20) -> str:
    """Keep each export separate: overlapping streams are not wall-clock time."""
    files = sorted(capture.rglob("op_summary*.csv"))
    if not files:
        raise ValueError(f"No op_summary CSV found below {capture}; inspect capture/log")
    type_rows, task_rows, lines = [], [], []
    for path in files:
        tasks, skipped = [], 0
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            names = {"".join(c for c in key.lower() if c.isalnum()): key
                     for key in reader.fieldnames or []}
            if "taskdurationus" not in names:
                raise ValueError(f"Missing Task Duration(us) column in {path}: {reader.fieldnames}")
            for row in reader:
                try:
                    duration_us = float(row[names["taskdurationus"]])
                except (ValueError, TypeError):
                    skipped += 1
                    continue
                if not math.isfinite(duration_us) or duration_us < 0:
                    skipped += 1
                    continue
                def field(key: str) -> str:
                    return row.get(names.get(key, "")) or "N/A"
                tasks.append({
                    "source_csv": str(path.relative_to(capture)),
                    "op_name": field("opname"), "op_type": field("optype"),
                    "task_type": field("tasktype"), "op_state": field("opstate"),
                    "duration_ms": duration_us / 1000,
                    "input_shapes": field("inputshapes"),
                    "input_data_types": field("inputdatatypes"),
                })
        if not tasks:
            raise ValueError(f"No finite, non-negative task durations in {path}")
        tasks.sort(key=lambda item: item["duration_ms"], reverse=True)
        task_rows.extend(tasks)
        groups: dict[tuple, list[float]] = {}
        for task in tasks:
            key = (task["op_type"], task["task_type"], task["op_state"])
            groups.setdefault(key, []).append(task["duration_ms"])
        types = [{
            "source_csv": str(path.relative_to(capture)),
            "op_type": key[0], "task_type": key[1], "op_state": key[2],
            "count": len(times), "total_ms": math.fsum(times),
            "mean_ms": math.fsum(times) / len(times), "max_ms": max(times),
        } for key, times in groups.items()]
        types.sort(key=lambda item: item["total_ms"], reverse=True)
        type_rows.extend(types)
        lines.extend([
            f"\nCSV: {path}", f"Valid tasks: {len(tasks)}; skipped invalid durations: {skipped}",
            "Operator types: count  total_ms  mean_ms  max_ms  OP Type / Task Type / OP State",
        ])
        for item in types[:top]:
            lines.append(f"{item['count']:6d} {item['total_ms']:11.3f} {item['mean_ms']:10.3f} "
                         f"{item['max_ms']:10.3f}  {item['op_type']} / "
                         f"{item['task_type']} / {item['op_state']}")
        lines.append("Individual tasks: duration_ms  Op Name / OP Type / Task Type / OP State / Input Shapes")
        for item in tasks[:top]:
            lines.append(f"{item['duration_ms']:11.3f}  {item['op_name']} / {item['op_type']} / "
                         f"{item['task_type']} / {item['op_state']} / {item['input_shapes']}")
    lines.append("\nTask-duration sums are not stage wall time: streams can overlap. "
                 "Each CSV is ranked separately; exports are not added together.")
    text = "\n".join(lines) + "\n"
    for name, rows in (("operator-types.csv", type_rows), ("operator-tasks.csv", task_rows)):
        with (output / name).open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with (output / "hotspots.txt").open("x", encoding="utf-8") as stream:
        stream.write(text)
    return text


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
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
    result.add_argument("--profile-timeout", type=int, default=600)
    result.add_argument("--aic-metrics", choices=("PipeUtilization", "Memory", "MemoryUB"),
                        default="PipeUtilization")
    return result


def run_profile(args: argparse.Namespace) -> Path:
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
            or args.max_new_tokens < 2 or not 1 <= args.max_draft_tokens <= 15):
        raise ValueError("verify needs device-id/warmup >= 0, timeout > 0, "
                         "max-new-tokens >= 2 and 1..15 max-draft-tokens")
    prompt, eos, prompt_source = prompt_and_eos(args, run)
    msprof = shutil.which(os.path.expanduser(args.msprof_bin))
    if msprof is None:
        raise ValueError("msprof not found; load the CANN environment or set MSPROF_BIN")
    if os.environ.get("ASCEND310P_SIMULATION_ONLY") == "1":
        raise ValueError("simulation-only target cannot produce msprof device measurements")
    parent = run / "msprof"
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="verify-", dir=parent))
    plan, capture = output / "plan.txt", output / "capture"
    environment = dict(os.environ)
    environment["AI_RUN_DIR"] = str(run)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(REPOSITORY / "framework/python"), str(REPOSITORY), environment.get("PYTHONPATH"),
    )))
    print(f"Output: {output}\nPrompt source: {prompt_source}\nEOS IDs: {eos}", flush=True)
    subprocess.run([
        sys.executable, "-B", "-m", "qwen35_dflash.ascend310p", "prepare-chunk-plan",
        "--deployment-manifest", str(manifest), "--mode", "dflash", "--output", str(plan),
    ], check=True, env=environment, cwd=REPOSITORY)
    digest = hashlib.sha256(plan.read_bytes()).hexdigest()
    print("Loading OMs, preparing prefill/draft and warming up outside capture; "
          "only ONE target_verify OM call is recorded. Model loading can take several minutes.", flush=True)
    subprocess.run([
        "bash", str(REPOSITORY / "tools/run_msprof.sh"),
        "--label", "verify", "--output-dir", str(capture), "--python", sys.executable,
        "--msprof-bin", msprof, "--profile-backend", "cpp", "--profile-mode", "dflash",
        "--profile-stage", "verify", "--profile-warmup", str(args.profile_warmup),
        "--profile-timeout", str(args.profile_timeout), "--aic-metrics", args.aic_metrics,
        "--", str(runner), "--model-kind", "chunk", "--model", str(plan),
        "--model-sha256", digest, "--prompt-token-ids", prompt, "--eos-token-ids", eos,
        "--device-id", str(args.device_id), "--max-new-tokens", str(args.max_new_tokens),
        "--max-draft-tokens", str(args.max_draft_tokens),
    ], check=True, env=environment, cwd=REPOSITORY)
    print(summarize(capture, output), end="", flush=True)
    print(f"\nSynchronized stage timing: {capture / 'verify-stage-summary.csv'}")
    print(f"Hotspots: {output / 'hotspots.txt'}\nAll operator types/tasks: {output / 'operator-types.csv'}, "
          f"{output / 'operator-tasks.csv'}")
    return output


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        run_profile(args)
    except subprocess.CalledProcessError as error:
        print(f"profile_verify_om: command failed (exit {error.returncode}); "
              "inspect the Output directory printed above, including capture/log", file=sys.stderr)
        return error.returncode if error.returncode > 0 else 1
    except (OSError, ValueError, KeyError) as error:
        print(f"profile_verify_om: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
