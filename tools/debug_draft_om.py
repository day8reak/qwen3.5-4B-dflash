#!/usr/bin/env python3
"""Temporary frozen-input Draft OM replay; no export, ATC or msprof."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from profile_om import prompt_and_eos

REPOSITORY = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=os.environ.get("AI_RUN_DIR"))
    p.add_argument("--runner", type=Path, default=os.environ.get("CPP_RUNNER"))
    p.add_argument("--deployment-manifest", type=Path)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--prompt-report", type=Path)
    group.add_argument("--prompt-token-ids")
    p.add_argument("--eos-token-ids", help="Read with the prompt; replay does not stop on EOS")
    p.add_argument("--pad-token-id", type=int, default=0)
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--max-draft-tokens", type=int, default=15)
    p.add_argument("--repetitions", type=int, default=20, help="Calls per phase and workspace policy")
    p.add_argument("--workspace", choices=("shared", "private", "both"), default="both")
    p.add_argument("--inputs", type=Path, help="Reuse an earlier report's input_directory")
    return p


def replay(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.run_dir is None or args.runner is None:
        raise ValueError("set AI_RUN_DIR/CPP_RUNNER or pass --run-dir/--runner")
    run, runner = args.run_dir.expanduser().resolve(), args.runner.expanduser().resolve()
    manifest = (args.deployment_manifest or run / "artifacts/deployment-manifest.json").expanduser().resolve()
    if not run.is_dir() or run.is_relative_to(REPOSITORY):
        raise ValueError("--run-dir must exist outside the source repository")
    if not runner.is_file() or not os.access(runner, os.X_OK):
        raise ValueError(f"C++ runner is missing or not executable: {runner}")
    if not manifest.is_file():
        raise ValueError(f"deployment manifest not found: {manifest}")
    if (not 1 <= args.repetitions <= 1000 or not 1 <= args.max_draft_tokens <= 15
            or args.device_id < 0 or args.pad_token_id < 0):
        raise ValueError("need repetitions 1..1000, proposals 1..15 and non-negative device/pad")
    if os.environ.get("PROFILING_MODE") not in (None, "", "false"):
        raise ValueError("run outside msprof; unset PROFILING_MODE for this diagnostic")
    prompt, eos, prompt_source = prompt_and_eos(args, run)
    inputs = args.inputs.expanduser().resolve() if args.inputs else None
    if inputs and not (inputs / "sha256.txt").is_file():
        raise ValueError("--inputs must be a completed frozen-input snapshot directory")
    output = Path(tempfile.mkdtemp(prefix="debug-draft-", dir=run))
    plan = output / "plan.txt"
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["AI_RUN_DIR"] = str(run)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(REPOSITORY / "framework/python"), str(REPOSITORY), env.get("PYTHONPATH"),
    )))
    request = {
        "schema_version": 1, "scope": "Draft OM stability diagnostic, not performance evidence",
        "runner": {"path": str(runner), "sha256": sha256(runner)},
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "deployment_manifest": {"path": str(manifest), "sha256": sha256(manifest)},
        "prompt_source": prompt_source, "prompt_token_ids": [int(x) for x in prompt.split(",")],
        "eos_token_ids": [int(x) for x in eos.split(",")],
        "pad_token_id": args.pad_token_id, "device_id": args.device_id,
        "proposal_count": args.max_draft_tokens, "repetitions_per_phase": args.repetitions,
        "workspace": args.workspace, "imported_inputs": str(inputs) if inputs else None,
    }
    (output / "request.json").write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
    print(f"Output: {output}\nExisting OMs are reused; rebuild the C++ runner before this diagnostic.",
          flush=True)
    with (output / "prepare-plan.log").open("w") as log:
        subprocess.run([
            sys.executable, "-B", "-m", "qwen35_dflash.ascend310p", "prepare-chunk-plan",
            "--deployment-manifest", str(manifest), "--mode", "dflash", "--output", str(plan),
        ], check=True, cwd=REPOSITORY, env=env, stdout=log, stderr=subprocess.STDOUT)

    policies = ("shared", "private") if args.workspace == "both" else (args.workspace,)
    results = []
    ok = True
    for policy in policies:
        report = output / f"{policy}.json"
        command = [
            str(runner), "--model-kind", "chunk", "--mode", "dflash",
            "--model", str(plan), "--model-sha256", sha256(plan),
            "--output", str(report), "--prompt-token-ids", prompt, "--eos-token-ids", eos,
            "--pad-token-id", str(args.pad_token_id), "--device-id", str(args.device_id),
            "--max-draft-tokens", str(args.max_draft_tokens),
            "--debug-draft-replay", str(args.repetitions),
            "--debug-draft-workspace", policy,
        ]
        if inputs:
            command.extend(("--debug-draft-inputs", str(inputs)))
        print(f"Workspace={policy}: loading three OMs; log={output / (policy + '.log')}", flush=True)
        with (output / f"{policy}.log").open("w") as log:
            proc = subprocess.run(command, cwd=REPOSITORY, env=env,
                                  stdout=log, stderr=subprocess.STDOUT)
        row = {"requested_workspace": policy, "exit_code": proc.returncode, "report": str(report)}
        if report.is_file():
            data = json.loads(report.read_text())
            row.update({key: data[key] for key in (
                "status", "actual_workspace_policy", "snapshot_sha256", "phases", "fake_acl",
                "reference_token_ids", "reference_output_sha256",
            )})
            if inputs is None:
                inputs = Path(data["input_directory"])
            print(f"{policy}: {data['status']}; trace={data['trace']}", flush=True)
        else:
            row["status"] = "ERROR"
            print(f"{policy}: runner failed before report; inspect {output / (policy + '.log')}", flush=True)
        results.append(row)
        ok = ok and proc.returncode == 0 and row["status"] == "PASS_REPLAY_CHECKS"
        # A failed replay still yields a valid snapshot; an aborted load may not.
        # Never silently regenerate different inputs for the second A/B process.
        if inputs is None:
            break
    paired = len(results) == 2 and all("snapshot_sha256" in row for row in results)
    identical_inputs = results[0]["snapshot_sha256"] == results[1]["snapshot_sha256"] if paired else None
    distinct_policies = (results[0]["actual_workspace_policy"] == "shared_serial"
                         and results[1]["actual_workspace_policy"] == "per_model") if paired else None
    same_reference_tokens = (results[0]["reference_token_ids"] == results[1]["reference_token_ids"]
                             if paired else None)
    same_reference_outputs = (results[0]["reference_output_sha256"] == results[1]["reference_output_sha256"]
                              if paired else None)
    if args.workspace == "both":
        ok = ok and paired and identical_inputs and distinct_policies and same_reference_tokens
    summary = {
        "schema_version": 1, "status": "PASS_REPLAY_CHECKS" if ok else "FAIL_OR_INCOMPLETE",
        "formal_latency_evidence": False, "ordinary_parity": "NOT_RUN", "runs": results,
        "same_frozen_inputs_across_processes": identical_inputs,
        "distinct_workspace_policies_exercised": distinct_policies,
        "reference_tokens_match_across_processes": same_reference_tokens,
        "reference_full_output_bytes_match_across_processes": same_reference_outputs,
        "note": "Input readback/synchronization changes timing. PASS does not rule out the original defect. "
                "Full output hashes include padding and are reported separately from valid tokens.",
    }
    (output / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Comparison: {output / 'comparison.json'}", flush=True)
    return output, bool(ok)


def main(argv: list[str] | None = None) -> int:
    try:
        _, ok = replay(parser().parse_args(argv))
        return 0 if ok else 1
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"debug_draft_om: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
