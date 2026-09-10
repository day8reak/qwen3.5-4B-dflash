#!/usr/bin/env python3
"""Temporary frozen FC / RMSNorm / V-projection probes; no full-model claim."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "framework/python"), str(REPO)]

CASES = ("fc", "norm_adn", "norm_tensor", "vproj", "chain_adn", "chain_tensor")
WEIGHTS = ("fc.weight", "hidden_norm.weight", "layers.0.self_attn.v_proj.weight")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def save_array(root, name, array):
    array = np.ascontiguousarray(array)
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(array.tobytes())
    return {"path": name, "dtype": str(array.dtype), "shape": list(array.shape),
            "bytes": array.nbytes, "sha256": digest(path)}


def read_array(root, record):
    path = (root / record["path"]).resolve()
    if not path.is_relative_to(root.resolve()) or digest(path) != record["sha256"]:
        raise ValueError("probe artifact path/hash mismatch")
    raw = path.read_bytes()
    shape = record["shape"]
    if (record["dtype"] != "float16" or len(shape) != 3 or shape[:2] != [1, 64]
            or type(shape[2]) is not int or not 1 <= shape[2] <= 32768
            or len(raw) != 128 * shape[2] or len(raw) != record["bytes"]):
        raise ValueError("probe expects a complete FP16 [1,64,F] array")
    return np.frombuffer(raw, dtype="<f2").reshape(shape).copy()


def frozen_features(report_path, feature_width):
    report_path = Path(report_path).resolve()
    report = json.loads(report_path.read_text())
    if not all(key in report for key in ("input_directory", "snapshot_sha256", "draft_om_sha256")):
        raise ValueError("pass the replay's private.json/shared.json, not its analysis or comparison JSON")
    directory = Path(report["input_directory"])
    raw = {}
    for name in ("features", "valid_rows", "start_position"):
        path = directory / (name + ".bin")
        raw[name] = path.read_bytes()
        if hashlib.sha256(raw[name]).hexdigest() != report["snapshot_sha256"][name]:
            raise ValueError(f"frozen replay input hash mismatch: {name}")
    if len(raw["valid_rows"]) != 2 or len(raw["start_position"]) != 8:
        raise ValueError("bad replay scalar ABI")
    valid = int(np.frombuffer(raw["valid_rows"], dtype="<i2")[0])
    if not 1 <= valid <= 64 or len(raw["features"]) != 128 * feature_width:
        raise ValueError("bad replay feature shape or valid_rows")
    features = np.frombuffer(raw["features"], dtype="<f2").reshape(1, 64, feature_width).copy()
    # Same mask as DraftGraph.forward, before FC. No synthetic random inputs.
    features[:, valid:, :] = 0
    if not np.isfinite(features).all():
        raise ValueError("nonfinite valid features")
    source = {"report": str(report_path), "report_sha256": digest(report_path),
              "draft_om_sha256": report["draft_om_sha256"],
              "snapshot_sha256": report["snapshot_sha256"],
              "start_position": int(np.frombuffer(raw["start_position"], dtype="<i8")[0])}
    return features, valid, source


def module(torch, weights, eps, case):
    """Use the production linear and the actual old/new normalization formulas."""
    if case not in CASES:
        raise ValueError("unknown probe case")
    from models.dflash_v1.dflash_ops import TorchDFlashOps
    from models.dflash_v1.dflash_ascend310p_ops import _adn_rms_norm

    class Probe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Parameter(weights[WEIGHTS[0]], requires_grad=False)
            self.gamma = torch.nn.Parameter(weights[WEIGHTS[1]], requires_grad=False)
            self.v = torch.nn.Parameter(weights[WEIGHTS[2]], requires_grad=False)

        def forward(self, value):
            if case == "fc":
                return (torch.nn.functional.linear(value, self.fc),)
            if case == "vproj":
                return (torch.nn.functional.linear(value, self.v),)
            chain = case.startswith("chain_")
            projected = torch.nn.functional.linear(value, self.fc) if chain else value
            if case.endswith("_adn"):
                normalized = _adn_rms_norm(projected, self.gamma, eps)
            else:
                normalized = TorchDFlashOps().rms_norm(projected, self.gamma, eps)
            if not chain:
                return (normalized,)
            return projected, normalized, torch.nn.functional.linear(normalized, self.v)

    return Probe().eval()


def output_names(case):
    if case == "fc":
        return ("fc_output",)
    if case == "vproj":
        return ("v_output",)
    if case.startswith("norm_"):
        return ("norm_output",)
    return "fc_output", "norm_output", "v_output"


def input_name(case):
    return "projected" if case.startswith("norm_") else "normalized" if case == "vproj" else "features"


def array_comparison(reference, actual, valid):
    from tools.analyze_draft_replay import metrics
    result = metrics(reference[:, None], actual[:, None], 0, valid)
    first = result["first_difference"]
    if first:
        b, _, row, channel = first.pop("coordinate_bhsd")
        first["coordinate_bsf"] = [b, row, channel]
    return result


def native_replay(torch, model, array, names, valid, count, device, root, case):
    fixed = torch.from_numpy(array.copy()).to(device)
    current = fixed.clone()
    references, samples, records, differences = {}, [], {}, {}
    for iteration in range(count):
        current.copy_(fixed)
        output = model(current)
        torch.npu.synchronize()
        if current.cpu().numpy().tobytes() != array.tobytes():
            raise RuntimeError("native probe modified its read-only input")
        hashes = {}
        for name, tensor in zip(names, output, strict=True):
            result = tensor.detach().cpu().numpy().copy()
            if result.dtype != np.float16 or not np.isfinite(result).all():
                raise RuntimeError("native probe produced nonfinite/non-FP16 output")
            hashes[name] = hashlib.sha256(result[:, :valid].tobytes()).hexdigest()
            if iteration == 0:
                references[name] = hashes[name]
                records[name] = save_array(root, f"native/{case}/{name}-reference.bin", result)
            elif hashes[name] != references[name] and name not in differences:
                differences[name] = save_array(root, f"native/{case}/{name}-first-diff.bin", result)
        samples.append({"iteration": iteration, "valid_output_sha256": hashes})
    return {"samples": samples, "reference": records, "first_difference": differences,
            "input_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            "mismatch_iterations": {name: sum(s["valid_output_sha256"][name] != references[name]
                                               for s in samples) for name in names}}


def prepare_export(root):
    request = json.loads((root / "request.json").read_text())
    if os.environ.get("ASCEND310P_SIMULATION_ONLY") == "1":
        raise RuntimeError("probe requires an actual Ascend310P; no CPU fallback")
    import torch
    import torch_npu
    from safetensors import safe_open
    from models.dflash_v1.dflash_weights import require_official_dflash_checkpoint
    from qwen35_dflash.ascend310p.custom_op_export import prepare_custom_op_export, audit_custom_op_export
    from qwen35_dflash.ascend310p.contracts import CustomOpExportSpec
    from qwen35_dflash.ascend310p.runtime_input_export import canonical_runtime_input_abi
    from qwen35_dflash.ascend310p.utils import count_ge_ir_nodes

    device = f"npu:{request['device_id']}"
    if not torch.npu.is_available():
        raise RuntimeError("Ascend NPU is unavailable")
    torch.npu.set_device(device)
    device_name = torch.npu.get_device_name(request["device_id"])
    if "310P" not in device_name:
        raise RuntimeError(f"expected Ascend310P, got {device_name}")
    audit = require_official_dflash_checkpoint(request["draft_dir"], verify_model_hash=True)
    config = audit["config"]
    features, valid, source = frozen_features(request["replay_report"], config["feature_size"])
    # Only three checkpoint tensors are moved to the NPU; Target is never loaded.
    with safe_open(str(Path(request["draft_dir"]) / "model.safetensors"), framework="pt", device="cpu") as ckpt:
        weights = {name: ckpt.get_tensor(name).to(device=device, dtype=torch.float16) for name in WEIGHTS}
    eps = config["rms_norm_eps"]
    torchair = importlib.import_module("torchair")
    with torch.inference_mode():
        x = torch.from_numpy(features).to(device)
        projected = module(torch, weights, eps, "fc")(x)[0]
        normalized = module(torch, weights, eps, "norm_tensor")(projected)[0]
        torch.npu.synchronize()
        arrays = {"features": features, "projected": projected.cpu().numpy().copy(),
                  "normalized": normalized.cpu().numpy().copy()}
        inputs = {name: save_array(root, f"inputs/{name}.bin", value) for name, value in arrays.items()}
        native, graphs = {}, {}
        # Persist native checks before exporting, so an exporter failure retains them.
        for case in CASES:
            model = module(torch, weights, eps, case)
            native[case] = native_replay(torch, model, arrays[input_name(case)], output_names(case), valid,
                                         request["repetitions"], device, root, case)
        save_json(root / "native.json", native)
        for case in CASES:
            graph_dir = root / "air" / case
            graph_dir.mkdir(parents=True, exist_ok=False)
            model = module(torch, weights, eps, case)
            data = (torch.from_numpy(arrays[input_name(case)].copy()).to(device),)
            sessions = []
            if case.endswith("_adn"):
                sessions = [prepare_custom_op_export(CustomOpExportSpec(
                    "npu::adn_rms_norm", request["rms_ge_op_type"]), torchair)]
            previous = Path.cwd()
            try:
                os.chdir(graph_dir)
                with canonical_runtime_input_abi(torchair, public_inputs=data, public_names=("input",),
                                                  require_static_shapes=True) as abi:
                    torchair.dynamo_export(*data, model=model, export_path=str(graph_dir),
                                           export_name=f"draft_probe_{case}", dynamic=False)
            finally:
                os.chdir(previous)
            counts = count_ge_ir_nodes(graph_dir.rglob("dynamo.pbtxt"))
            norm_count = sum(counts.get(name, 0) for name in ("RmsNorm", "AdnRmsNorm"))
            if norm_count != int(case.endswith("_adn")):
                raise RuntimeError(f"unexpected exported norm nodes for {case}: {counts}")
            custom_audit = audit_custom_op_export(sessions, graph_dir, relative_to=root) if sessions else []
            files = list(graph_dir.glob("*.air"))
            if len(files) != 1:
                raise RuntimeError(f"expected one AIR for {case}")
            specs = []
            for name, value in zip(output_names(case), model(*data), strict=True):
                specs.append({"name": name, "dtype": "float16", "shape": list(value.shape),
                              "bytes": value.numel() * 2, "valid_bytes": valid * value.shape[-1] * 2})
            graphs[case] = {"air": str(files[0].relative_to(root)), "sha256": digest(files[0]),
                            "input": inputs[input_name(case)], "outputs": specs,
                            "public_input_abi": abi, "custom_op_audit": custom_audit, "ge_nodes": counts}
            print(f"[draft-probe] exported {case}", flush=True)
            torch._dynamo.reset()
    save_json(root / "air.json", {"graphs": graphs, "valid_rows": valid, "source": source,
        "checkpoint": audit, "inputs": inputs, "device": device_name,
        "torch_version": str(torch.__version__), "torch_npu_version": str(torch_npu.__version__),
        "rms_ge_op_type": request["rms_ge_op_type"],
        "custom_opp_path": os.environ.get("ASCEND_CUSTOM_OPP_PATH", ""),
        "selected_kernel_identity": "NOT_CAPTURED", "formal_latency_evidence": False})


def call(command, log, allowed=(0,)):
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[draft-probe] {log.stem}; log={log}", flush=True)
    with log.open("x") as stream:
        result = subprocess.run(list(map(str, command)), cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode not in allowed:
        raise RuntimeError(f"command failed ({result.returncode}): {log}\n{log.read_text(errors='replace')[-4000:]}")
    return result.returncode


def plan(root, case, graph, om, request):
    from tools.debug_gdr.run import plan_quote
    inp = graph["input"]
    read_array(root, inp)  # Check every reused input before launching ACL.
    out = root / "om-results" / case
    lines = ["DRAFT_CONTEXT_PROBE_V1", " ".join((plan_quote(om), plan_quote(out),
        str(request["device_id"]), "0", str(request["repetitions"]), '""', '"PipeUtilization"')), "1",
        f'"input" 1 {inp["bytes"]} 3 ' + " ".join(map(str, inp["shape"])) + " " + plan_quote(root / inp["path"]),
        str(len(graph["outputs"]))]
    for spec in graph["outputs"]:
        lines.append(plan_quote(spec["name"]) + f' 1 {spec["bytes"]} 3 ' +
                     " ".join(map(str, spec["shape"])) + f' {spec["valid_bytes"]}')
    dest = root / "plans" / (case + ".txt")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    return dest, out


def summarize_case(root, case, graph, valid, native, repetitions):
    directory = root / "om-results" / case
    report = json.loads((directory / "report.json").read_text())
    if (len(report["samples"]) != repetitions or len(native["samples"]) != repetitions
            or repetitions < 2 or report["profiled"] or report["warmup"] != 0):
        raise ValueError("probe did not record all unprofiled replay calls")
    if graph["input"]["sha256"] != native["input_sha256"]:
        raise ValueError("native/OM inputs differ")
    tensors = {}
    for spec in graph["outputs"]:
        name = spec["name"]
        reference_path = directory / (name + "-reference.bin")
        reference = np.fromfile(reference_path, dtype="<f2").reshape(spec["shape"])
        expected = report["samples"][0]["valid_output_sha256"][name]
        if hashlib.sha256(reference[:, :valid].tobytes()).hexdigest() != expected:
            raise ValueError("probe reference output changed after execution")
        item = {"mismatch_iterations": sum(s["valid_output_sha256"][name] != expected
                                           for s in report["samples"]),
                "native_mismatch_iterations": native["mismatch_iterations"][name],
                "native_vs_om_reference": array_comparison(read_array(root, native["reference"][name]), reference, valid),
                "reference_file": {"path": str(reference_path), "sha256": digest(reference_path)}}
        if name in native["first_difference"]:
            item["native_first_changed_example"] = array_comparison(
                read_array(root, native["reference"][name]),
                read_array(root, native["first_difference"][name]), valid)
        changed = directory / (name + "-first-diff.bin")
        if changed.exists():
            actual = np.fromfile(changed, dtype="<f2").reshape(spec["shape"])
            hashed = hashlib.sha256(actual[:, :valid].tobytes()).hexdigest()
            if hashed == expected or not any(s["valid_output_sha256"][name] == hashed for s in report["samples"]):
                raise ValueError("probe difference does not match a recorded call")
            item["first_changed_example"] = array_comparison(reference, actual, valid)
            item["difference_file"] = {"path": str(changed), "sha256": digest(changed)}
        tensors[name] = item
    return {"case": case, "cpu_fallback": report["cpu_fallback"], "outputs": tensors,
            "input_sha256": graph["input"]["sha256"], "om_sha256": report["model_sha256"],
            "repetitions": len(report["samples"])}


def run_all(args):
    from qwen35_dflash.ascend310p.compiler import resolve_atc_executable
    from qwen35_dflash.ascend310p.utils import require_run_output
    run = args.run_dir.expanduser().resolve()
    os.environ["AI_RUN_DIR"] = str(run)
    if run == REPO or REPO in run.parents or os.environ.get("PROFILING_MODE"):
        raise ValueError("use an external run directory and run outside msprof")
    factory_path = args.factory_config or run / "factory.json"
    if args.factory_config and not factory_path.is_file():
        raise ValueError(f"factory config does not exist: {factory_path}")
    config = json.loads(factory_path.read_text()) if factory_path.is_file() else {}
    draft_dir = args.draft_dir or config.get("draft_dir")
    if not draft_dir:
        raise ValueError("provide --draft-dir or the existing --factory-config")
    rms_type = args.rms_ge_op_type or config.get("adn_rms_norm_ge_op_type", "RmsNorm")
    if rms_type not in ("RmsNorm", "AdnRmsNorm") or not 2 <= args.repetitions <= 1000:
        raise ValueError("invalid norm GE type or repetitions (2..1000)")
    if not run.is_dir():
        raise ValueError(f"run directory does not exist: {run}")
    root = require_run_output(Path(tempfile.mkdtemp(prefix="debug-draft-context-", dir=run)))
    request = {"draft_dir": str(Path(draft_dir).expanduser().resolve()),
        "replay_report": str(args.replay_report.expanduser().resolve()),
        "device_id": args.device_id, "repetitions": args.repetitions, "rms_ge_op_type": rms_type,
        "source_sha256": {str(p.relative_to(REPO)): digest(p) for p in (
            HERE / "run.py", REPO / "tools/debug_gdr/runner.cpp",
            REPO / "models/dflash_v1/dflash_ops.py", REPO / "models/dflash_v1/dflash_ascend310p_ops.py")}}
    save_json(root / "request.json", request)
    print(f"Output: {root}", flush=True)
    # Native/export Python exits before ACL loads a probe OM.
    call([sys.executable, "-B", __file__, "export", "--work-dir", root], root / "logs/export.log")
    air = json.loads((root / "air.json").read_text())
    atc = resolve_atc_executable(args.atc)
    (root / "om").mkdir()
    compiled = {}
    for case, graph in air["graphs"].items():
        source = root / graph["air"]
        if digest(source) != graph["sha256"]:
            raise ValueError("probe AIR changed")
        prefix = root / "om" / case
        command = [atc, "--mode=0", "--framework=1", f"--model={source}", f"--output={prefix}",
                   "--soc_version=Ascend310P3", "--precision_mode=must_keep_origin_dtype"]
        call(command, root / "logs" / ("atc-" + case + ".log"))
        compiled[case] = {"path": str(prefix.with_suffix(".om")), "sha256": digest(prefix.with_suffix(".om")),
                          "command": list(map(str, command))}
    save_json(root / "om.json", compiled)
    build = root / "build"
    command = ["cmake", "-S", REPO / "tools/debug_gdr", "-B", build, "-DCMAKE_BUILD_TYPE=Release"]
    if args.ascendcl_root:
        command.append(f"-DASCENDCL_ROOT={args.ascendcl_root}")
    call(command, root / "logs/cmake.log")
    call(["cmake", "--build", build, "--parallel", "2"], root / "logs/build.log")
    native = json.loads((root / "native.json").read_text())
    results = []
    for case, graph in air["graphs"].items():
        om = Path(compiled[case]["path"])
        if digest(om) != compiled[case]["sha256"]:
            raise ValueError("probe OM changed")
        plan_path, _ = plan(root, case, graph, om, request)
        call([build / "gdr_debug_runner", plan_path], root / "logs" / ("run-" + case + ".log"), allowed=(0, 2))
        result = summarize_case(root, case, graph, air["valid_rows"], native[case], args.repetitions)
        if result["cpu_fallback"]:
            raise RuntimeError("host/fake ACL cannot produce an NPU probe result")
        results.append(result)
        print(f"[draft-probe] {case}: " + ", ".join(
            f"{n}: native={v['native_mismatch_iterations']}, OM={v['mismatch_iterations']} changed calls"
            for n, v in result["outputs"].items()), flush=True)
    stable = all(not v["mismatch_iterations"] and not v["native_mismatch_iterations"] and
                 v["native_vs_om_reference"]["actual_nonfinite_elements"] == 0
                 for r in results for v in r["outputs"].values())
    # Different norm formulas may produce different stable FP16 rounding.
    # Compare intra-case stability before interpreting cross-formula differences.
    norm_refs = [np.fromfile(root / "om-results" / c / "norm_output-reference.bin", dtype="<f2")
                 .reshape(air["graphs"][c]["outputs"][0]["shape"]) for c in ("norm_tensor", "norm_adn")]
    save_json(root / "summary.json", {"schema_version": 1,
        "status": "STABLE_PROBE_OUTPUTS" if stable else "OBSERVED_VARIATION", "cases": results,
        "same_frozen_norm_input": air["graphs"]["norm_adn"]["input"] == air["graphs"]["norm_tensor"]["input"],
        "norm_tensor_vs_adn_om_reference": array_comparison(*norm_refs, air["valid_rows"]),
        "ordinary_parity": "NOT_RUN", "formal_latency_evidence": False,
        "note": "Probe graph boundaries differ from the original Draft OM. Stable probes do not clear the original defect. "
                "The vproj input is one frozen Tensor-formula norm result; norm cases share one frozen FC result. "
                "ATC may fuse Tensor-formula nodes; inspect compiled kernels before attributing a kernel bug. "
                "No whole-model weights, operator selection or accuracy gate was changed."})
    print(f"Summary: {root / 'summary.json'}", flush=True)
    return 0 if stable else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="internal native check / AIR export phase")
    export.add_argument("--work-dir", type=Path, required=True)
    run = sub.add_parser("all", help="replay FC, both norm formulas, V projection and both small chains")
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--replay-report", type=Path, required=True)
    run.add_argument("--factory-config", type=Path)
    run.add_argument("--draft-dir", type=Path)
    run.add_argument("--rms-ge-op-type", choices=("RmsNorm", "AdnRmsNorm"))
    run.add_argument("--device-id", type=int, default=0)
    run.add_argument("--repetitions", type=int, default=20)
    run.add_argument("--atc")
    run.add_argument("--ascendcl-root", type=Path)
    args = p.parse_args()
    try:
        if args.command == "export":
            from qwen35_dflash.ascend310p.utils import require_run_output
            prepare_export(require_run_output(args.work_dir).resolve())
            return 0
        return run_all(args)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        p.exit(2, f"draft-context-probe: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
