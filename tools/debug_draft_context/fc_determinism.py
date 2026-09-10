#!/usr/bin/env python3
"""Temporary FC determinism A/B using an existing context probe's AIR and inputs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from tools.debug_draft_context import run as probe


def prepare(args):
    from qwen35_dflash.ascend310p.utils import require_run_output
    run = args.run_dir.expanduser().resolve()
    os.environ["AI_RUN_DIR"] = str(run)
    if (not run.is_dir() or run == REPO or REPO in run.parents
            or os.environ.get("PROFILING_MODE")):
        raise ValueError("use an existing external run directory, outside msprof")
    if not 2 <= args.repetitions <= 1000 or args.device_id < 0:
        raise ValueError("invalid device or repetitions (2..1000)")
    previous = args.probe_dir.expanduser().resolve()
    air_path, request_path = previous / "air.json", previous / "request.json"
    air = json.loads(air_path.read_text())
    original_request = json.loads(request_path.read_text())
    graph, valid = air["graphs"]["fc"], air["valid_rows"]
    config = air["checkpoint"]["config"]
    values = probe.read_array(previous, graph["input"])
    if (type(valid) is not int or not 1 <= valid <= 64
            or values.shape != (1, 64, config["feature_size"])):
        raise ValueError("invalid FC input shape or valid rows")
    output_width = config["hidden_size"]
    expected = {"name": "fc_output", "dtype": "float16", "shape": [1, 64, output_width],
                "bytes": 128 * output_width, "valid_bytes": 2 * valid * output_width}
    if graph["outputs"] != [expected]:
        raise ValueError("expected one FP16 FC output matching the checkpoint")
    original_air = (previous / graph["air"]).resolve()
    if not original_air.is_relative_to(previous) or probe.digest(original_air) != graph["sha256"]:
        raise ValueError("source FC AIR path/hash mismatch")
    root = require_run_output(Path(tempfile.mkdtemp(prefix="debug-fc-determinism-", dir=run)))
    (root / "air").mkdir()
    copied_air = root / "air/fc.air"
    shutil.copyfile(original_air, copied_air)
    if probe.digest(copied_air) != graph["sha256"]:
        raise ValueError("FC AIR changed while copying")
    inp = probe.save_array(root, "inputs/features.bin", values)
    if inp["sha256"] != graph["input"]["sha256"]:
        raise ValueError("frozen FC inputs changed while copying")
    request = {"schema_version": 1, "device_id": args.device_id, "repetitions": args.repetitions,
        "draft_dir": original_request["draft_dir"], "valid_rows": valid,
        "expected_checkpoint_sha256": air["checkpoint"]["model_sha256"],
        "graph": {"air": "air/fc.air", "sha256": graph["sha256"], "input": inp, "outputs": [expected]},
        "source": {"directory": str(previous), "air_manifest_sha256": probe.digest(air_path),
                   "request_sha256": probe.digest(request_path),
                   "original_source_sha256": original_request.get("source_sha256", {})},
        "source_sha256": {str(p.relative_to(REPO)): probe.digest(p) for p in
            (Path(__file__), HERE / "run.py", REPO / "tools/debug_gdr/runner.cpp")},
        "inherited_environment": {k: os.environ.get(k) for k in
            ("ASCEND_CUSTOM_OPP_PATH", "ASCEND_LAUNCH_BLOCKING", "CLOSE_MATMUL_K_SHIFT")},
        "native_policy": "fresh process per mode; configure once before NPU work; warn_only=False",
        "om_policy": "same AIR and inputs, separate ATC --deterministic=0/1 and ACL processes",
        "formal_latency_evidence": False}
    probe.save_json(root / "request.json", request)
    print(f"Output: {root}", flush=True)
    return root, request


def configure_native(torch, mode):
    # Must run once, before the first NPU operation, in each fresh child process.
    torch.use_deterministic_algorithms(bool(mode), warn_only=False)
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    if enabled != bool(mode) or warn_only:
        raise RuntimeError("native determinism setting was not applied")
    return {"requested": mode, "algorithms_enabled": enabled, "warn_only": warn_only,
            "pid": os.getpid()}


def native(root, mode):
    if os.environ.get("ASCEND310P_SIMULATION_ONLY") == "1" or os.environ.get("PROFILING_MODE"):
        raise RuntimeError("FC replay requires real Ascend310P, outside msprof")
    request = json.loads((root / "request.json").read_text())
    import torch
    import torch_npu
    from safetensors import safe_open
    from models.dflash_v1.dflash_weights import require_official_dflash_checkpoint

    settings = configure_native(torch, mode)
    if not torch.npu.is_available():
        raise RuntimeError("Ascend NPU is unavailable")
    device = f"npu:{request['device_id']}"
    torch.npu.set_device(device)
    name = torch.npu.get_device_name(request["device_id"])
    if "310P" not in name:
        raise RuntimeError(f"expected Ascend310P, got {name}")
    audit = require_official_dflash_checkpoint(request["draft_dir"], verify_model_hash=True)
    if audit["model_sha256"] != request["expected_checkpoint_sha256"]:
        raise ValueError("checkpoint differs from the original FC probe")
    with safe_open(str(Path(request["draft_dir"]) / "model.safetensors"), framework="pt", device="cpu") as ckpt:
        weight = ckpt.get_tensor("fc.weight").to(device=device, dtype=torch.float16)
    values = probe.read_array(root, request["graph"]["input"])
    if tuple(weight.shape) != (request["graph"]["outputs"][0]["shape"][-1], values.shape[-1]):
        raise ValueError("FC weight/input ABI differs")

    class FC(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(weight, requires_grad=False)

        def forward(self, x):
            return (torch.nn.functional.linear(x, self.weight),)

    case = f"fc_det{mode}"
    with torch.inference_mode():
        replay = probe.native_replay(torch, FC().eval(), values, ("fc_output",), request["valid_rows"],
                                      request["repetitions"], device, root, case)
    probe.save_json(root / f"native-det{mode}.json", {"replay": replay, "determinism": settings,
        "device": name, "device_id": request["device_id"], "cpu_fallback": False,
        "torch_version": str(torch.__version__), "torch_npu_version": str(torch_npu.__version__),
        "checkpoint_sha256": audit["model_sha256"], "selected_kernel_identity": "NOT_CAPTURED"})


def verdict(results):
    def stable(result):
        item = result["outputs"]["fc_output"]
        return (item["mismatch_iterations"] == 0 and item["native_mismatch_iterations"] == 0
                and item["native_vs_om_reference"]["actual_nonfinite_elements"] == 0)
    if not stable(results[1]):
        return "VARIATION_WITH_DETERMINISTIC"
    return "BASELINE_NOT_REPRODUCED" if stable(results[0]) else "STABLE_WITH_DETERMINISTIC"


def run(args):
    from qwen35_dflash.ascend310p.compiler import resolve_atc_executable
    root, request = prepare(args)
    # Separate processes matter: some NPU versions cache the first effective setting.
    for mode in (0, 1):
        probe.call([sys.executable, "-B", __file__, "native", "--work-dir", root, "--mode", mode],
                   root / "logs" / f"native-det{mode}.log")
    atc = resolve_atc_executable(args.atc)
    graph = request["graph"]
    source = root / graph["air"]
    if probe.digest(source) != graph["sha256"]:
        raise ValueError("frozen FC AIR changed")
    (root / "om").mkdir()
    compiled = {}
    for mode in (0, 1):
        prefix = root / "om" / f"fc_det{mode}"
        command = [atc, "--mode=0", "--framework=1", f"--model={source}", f"--output={prefix}",
                   "--soc_version=Ascend310P3", "--precision_mode=must_keep_origin_dtype",
                   f"--deterministic={mode}"]
        probe.call(command, root / "logs" / f"atc-det{mode}.log")
        om = prefix.with_suffix(".om")
        compiled[str(mode)] = {"path": str(om), "sha256": probe.digest(om), "command": list(map(str, command))}
    probe.save_json(root / "om.json", compiled)
    build = root / "build"
    command = ["cmake", "-S", REPO / "tools/debug_gdr", "-B", build, "-DCMAKE_BUILD_TYPE=Release"]
    if args.ascendcl_root:
        command.append(f"-DASCENDCL_ROOT={args.ascendcl_root}")
    probe.call(command, root / "logs/cmake.log")
    probe.call(["cmake", "--build", build, "--parallel", "2"], root / "logs/build.log")
    probe.save_json(root / "tools.json", {
        "atc": {"path": str(atc), "sha256": probe.digest(atc)},
        "runner": {"path": str(build / "gdr_debug_runner"), "sha256": probe.digest(build / "gdr_debug_runner")}})
    results = []
    for mode in (0, 1):
        case = f"fc_det{mode}"
        om = Path(compiled[str(mode)]["path"])
        if probe.digest(om) != compiled[str(mode)]["sha256"]:
            raise ValueError("FC OM changed")
        path, _ = probe.plan(root, case, graph, om, request)
        probe.call([build / "gdr_debug_runner", path], root / "logs" / f"run-det{mode}.log", allowed=(0, 2))
        host = json.loads((root / f"native-det{mode}.json").read_text())
        if host["determinism"]["algorithms_enabled"] != bool(mode) or host["determinism"]["warn_only"]:
            raise ValueError("native mode differs from the requested case")
        result = probe.summarize_case(root, case, graph, request["valid_rows"], host["replay"], request["repetitions"])
        if result["cpu_fallback"] or host["cpu_fallback"]:
            raise RuntimeError("fake ACL/CPU results cannot produce an NPU determinism report")
        result["native_determinism"] = host["determinism"]
        result["atc_deterministic"] = mode
        results.append(result)
        item = result["outputs"]["fc_output"]
        print(f"[fc-determinism] mode={mode}: native={item['native_mismatch_iterations']}, "
              f"OM={item['mismatch_iterations']} changed calls", flush=True)
    if results[0]["native_determinism"]["pid"] == results[1]["native_determinism"]["pid"]:
        raise ValueError("native modes did not run in distinct processes")
    shape = graph["outputs"][0]["shape"]
    references = [np.fromfile(root / "om-results" / f"fc_det{mode}" / "fc_output-reference.bin", dtype="<f2")
                  .reshape(shape) for mode in (0, 1)]
    status = verdict(results)
    probe.save_json(root / "summary.json", {"schema_version": 1, "status": status, "cases": results,
        "same_air_sha256": graph["sha256"], "same_input_sha256": graph["input"]["sha256"],
        "om_det0_vs_det1_reference": probe.array_comparison(*references, request["valid_rows"]),
        "ordinary_parity": "NOT_RUN", "formal_latency_evidence": False,
        "note": "FC-only experiment. A stable deterministic case does not establish a particular split-K/atomic kernel "
                "or validate the complete Draft model. Native settings are requested/getter state, not kernel identity. "
                "Cross-mode rounding differences are separate from run-to-run changes. "
                "Existing AIR/OM artifacts, main-model settings and equality gates were not modified."})
    print(f"Summary: {root / 'summary.json'}", flush=True)
    return 1 if status == "VARIATION_WITH_DETERMINISTIC" else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    child = commands.add_parser("native", help="internal fresh-process native replay")
    child.add_argument("--work-dir", type=Path, required=True)
    child.add_argument("--mode", type=int, choices=(0, 1), required=True)
    all_cases = commands.add_parser("all", help="reuse FC AIR and frozen inputs for deterministic 0/1 A/B")
    all_cases.add_argument("--run-dir", type=Path, required=True)
    all_cases.add_argument("--probe-dir", type=Path, required=True)
    all_cases.add_argument("--repetitions", type=int, default=20)
    all_cases.add_argument("--device-id", type=int, default=0)
    all_cases.add_argument("--atc")
    all_cases.add_argument("--ascendcl-root", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "native":
            from qwen35_dflash.ascend310p.utils import require_run_output
            native(require_run_output(args.work_dir).resolve(), args.mode)
            return 0
        return run(args)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(2, f"fc-determinism: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
