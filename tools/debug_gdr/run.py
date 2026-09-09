#!/usr/bin/env python3
"""Temporary, weight-free GDR diagnostics. Delete tools/debug_gdr after diagnosis."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import runpy
import shlex
import shutil
import statistics
import subprocess
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "framework/python"), str(REPO)]
NAMES = ("query", "key", "value", "g", "beta", "initial_state", "effective_length")
DTYPES = ("float16", "float16", "float16", "float32", "float16", "float32", "int16")
SHAPES = ((1, 16, 32, 128),) * 3 + ((1, 16, 32),) * 2 + ((1, 32, 128, 128), (1,))
ACL_TYPES = {"float16": 1, "float32": 0, "int16": 6}
BASE_VARIANTS = ("both", "core", "state")
VARIANTS = (*BASE_VARIANTS, "core_no_state")
ATTRS = dict(chunk_size=64, output_final_state=True, use_qk_l2norm_in_kernel=True)


def attributes(variant):
    if variant not in VARIANTS:
        raise ValueError("invalid output variant")
    return {**ATTRS, "output_final_state": variant != "core_no_state"}


def cases(root):
    graphs = json.loads((root / "om.json").read_text())["graphs"]
    native_cases = [("native", "both")]
    if "core_no_state" in graphs:
        native_cases.append(("native", "core_no_state"))
    return native_cases + [("om", v) for v in VARIANTS if v in graphs]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan_quote(value):
    value = str(value)
    if any(ord(c) < 32 for c in value):
        raise ValueError("control characters are not allowed in plan paths")
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def output_dir(value, *, new=False):
    from qwen35_dflash.ascend310p.utils import require_run_output
    root = require_run_output(value)
    if root == REPO or REPO in root.parents:
        raise ValueError("debug artifacts must be outside the source checkout")
    if new:
        root.mkdir(parents=True, exist_ok=False)
    return root


def fresh(path):
    Path(path).mkdir(parents=True, exist_ok=False)
    return Path(path)


def load_manifest(root):
    m = json.loads((root / "inputs.json").read_text())
    if m.get("schema_version") != 1 or m.get("attributes") != ATTRS:
        raise ValueError("unexpected GDR input contract")
    if m.get("lengths") != list(dict.fromkeys(m.get("lengths", []))) or not m["lengths"]:
        raise ValueError("invalid or duplicate effective lengths")
    for length in m["lengths"]:
        if type(length) is not int or not 1 <= length <= 16:
            raise ValueError("effective length must be 1..16")
    for name, dtype, shape in zip(NAMES[:6], DTYPES[:6], SHAPES[:6]):
        rec = m["tensors"][name]
        if rec["dtype"] != dtype or rec["shape"] != list(shape):
            raise ValueError(f"wrong tensor contract: {name}")
        path = root / "inputs" / (name + ".bin")
        if path.stat().st_size != int(np.prod(shape)) * np.dtype(dtype).itemsize or digest(path) != rec["sha256"]:
            raise ValueError(f"changed input bytes: {name}")
    for length in m["lengths"]:
        path = root / "inputs" / f"length-{length}.bin"
        if path.read_bytes() != np.asarray([length], dtype="<i2").tobytes():
            raise ValueError(f"changed effective length: {path}")
    return m


def arrays(root, length):
    m = load_manifest(root)
    if length not in m["lengths"]:
        raise ValueError("length is absent from the frozen inputs")
    data = [np.fromfile(root / "inputs" / (n + ".bin"), dtype=d).reshape(s)
            for n, d, s in zip(NAMES[:6], DTYPES[:6], SHAPES[:6])]
    return (*data, np.array([length], dtype=np.int16))


def validate_arrays(data):
    if set(data) != set(NAMES):
        raise ValueError(f"input NPZ must contain exactly {NAMES}")
    result = {}
    for name, dtype, shape in zip(NAMES, DTYPES, SHAPES):
        value = np.asarray(data[name])
        if value.shape != shape or value.dtype != np.dtype(dtype):
            raise ValueError(f"{name}: expected {dtype}{shape}, got {value.dtype}{value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"nonfinite input: {name}")
        result[name] = np.ascontiguousarray(value)
    if not 1 <= int(result["effective_length"][0]) <= 16:
        raise ValueError("captured effective length must be 1..16")
    return result


def prepare(args):
    root = output_dir(args.work_dir, new=True)
    fresh(root / "inputs")
    if args.inputs:
        with np.load(args.inputs, allow_pickle=False) as archive:
            data = validate_arrays(dict(archive))
        source = {"kind": "supplied_npz", "path": str(Path(args.inputs).resolve()),
                  "sha256": digest(args.inputs), "original_effective_length": int(data["effective_length"][0])}
    else:
        rng = np.random.default_rng(args.seed)
        data = {name: rng.normal(0, .1, shape).astype(dtype)
                for name, dtype, shape in zip(NAMES, DTYPES, SHAPES)}
        data["g"] = -rng.uniform(.01, .5, SHAPES[3]).astype(np.float32)
        data["beta"] = rng.uniform(.05, .95, SHAPES[4]).astype(np.float16)
        data["effective_length"] = np.array([16], dtype=np.int16)
        source = {"kind": "synthetic", "seed": args.seed,
                  "limitation": "A fast synthetic case does not exonerate the original OM graph."}
    data = validate_arrays(data)
    records = {}
    for name in NAMES[:6]:
        path = root / "inputs" / (name + ".bin")
        data[name].tofile(path)
        records[name] = {"dtype": str(data[name].dtype), "shape": list(data[name].shape),
                         "bytes": path.stat().st_size, "sha256": digest(path)}
    for length in args.lengths:
        np.array([length], dtype=np.int16).tofile(root / "inputs" / f"length-{length}.bin")
    write_json(root / "inputs.json", {"schema_version": 1, "source": source, "lengths": args.lengths,
               "attributes": ATTRS, "input_order": NAMES, "tensors": records,
               "physical_rows": 16, "state_policy": "same nonzero round-start state; never feed outputs back"})
    print(f"[gdr-debug] frozen inputs: {root / 'inputs.json'}", flush=True)


def require_device(device):
    if os.environ.get("ASCEND310P_SIMULATION_ONLY") == "1":
        raise RuntimeError("simulation-only profile cannot produce device measurements")
    import torch
    import torch_npu
    if not torch.npu.is_available():
        raise RuntimeError("a real torch_npu device is required; CPU fallback is unavailable")
    torch.npu.set_device(device)
    name = torch.npu.get_device_name(device)
    if "310P" not in name:
        raise RuntimeError(f"expected Ascend310P, got {name}")
    # Fail at startup rather than using a substitute implementation.
    operation = torch.ops.npu.npu_chunk_gated_delta_rule.default
    return torch, torch_npu, operation


def gdr_module(torch, operation, variant):
    final_state = attributes(variant)["output_final_state"]
    class Gdr(torch.nn.Module):
        def forward(self, query, key, value, g, beta, initial_state, effective_length):
            core, state = operation(query, key, value, g=g, beta=beta,
                                    initial_state=initial_state, effective_length=effective_length,
                                    chunk_size=64, output_final_state=final_state, use_qk_l2norm_in_kernel=True)
            # Normalize the receiver's flattened core output without changing numerics.
            core = core.reshape(1, 16, 32, 128)
            if variant in ("core", "core_no_state"):
                return (core,)
            if variant == "state":
                return (state,)
            return core, state
    if variant not in VARIANTS:
        raise ValueError("invalid output variant")
    return Gdr().eval()


def output_specs(variant, length):
    specs = [("core_attn", "float16", SHAPES[0], length * 32 * 128 * 2),
             ("last_recurrent_state", "float32", SHAPES[5], 32 * 128 * 128 * 4)]
    return specs if variant == "both" else [specs[0 if variant in ("core", "core_no_state") else 1]]


def package_identity():
    # Inventory only declared vendor roots; this does not identify the selected kernel.
    found = {}
    for entry in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep):
        if not entry:
            continue
        base = Path(entry)
        roots = [base] + (list(base.glob("*/")) if base.name == "vendors" else [])
        for root in roots:
            for path in sorted(root.glob("op_impl/ai_core/tbe/kernel/**/chunk_gated_delta_rule/*")):
                if path.is_file() and path.suffix in {".o", ".json"}:
                    found[str(path.resolve())] = digest(path)
    return {"vendor_files": found, "selected_kernel_and_tiling": "NOT_CAPTURED_BY_THIS_INVENTORY",
            "ASCEND_CUSTOM_OPP_PATH": os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")}


def export(args):
    root = output_dir(args.work_dir)
    data = arrays(root, load_manifest(root)["lengths"][0])
    torch, torch_npu, operation = require_device(args.device_id)
    from qwen35_dflash.ascend310p.custom_op_export import (
        prepare_custom_op_export, audit_custom_op_export, validate_gdr_ge_prototype_environment)
    from qwen35_dflash.ascend310p.contracts import CustomOpExportSpec
    from qwen35_dflash.ascend310p.runtime_input_export import canonical_runtime_input_abi
    torchair = importlib.import_module("torchair")
    proto = validate_gdr_ge_prototype_environment()
    fresh(root / "air")
    graphs = {}
    for variant in getattr(args, "variants", BASE_VARIANTS):
        graph_dir = fresh(root / "air" / variant)
        # Distinct storages allow canonical_runtime_input_abi to retain all seven Data inputs.
        inputs = tuple(torch.from_numpy(a.copy()).to(f"npu:{args.device_id}") for a in data)
        if variant == "core_no_state":
            from tools.debug_gdr.no_state_frontend import register, serialized_audit
            frontend = register(torch, torch_npu, torchair)
            model = frontend.model
            serialized_context = serialized_audit(frontend.audit)
        else:
            session = prepare_custom_op_export(CustomOpExportSpec(
                "npu::npu_chunk_gated_delta_rule", "ChunkGatedDeltaRule"), torchair)
            model = gdr_module(torch, operation, variant)
            serialized_context = nullcontext()
        previous = Path.cwd()
        try:
            os.chdir(graph_dir)
            with torch.inference_mode(), canonical_runtime_input_abi(
                    torchair, public_inputs=inputs, public_names=NAMES, require_static_shapes=True) as abi, serialized_context:
                torchair.dynamo_export(*inputs, model=model,
                    export_path=str(graph_dir), export_name=f"gdr_{variant}", dynamic=False)
        finally:
            os.chdir(previous)
        if variant == "core_no_state":
            audit = frontend.audit
            if audit["converter_calls"] != 1:
                raise RuntimeError("expected one explicit output_final_state=False converter call")
        else:
            audit = audit_custom_op_export([session], graph_dir, relative_to=root)
        from qwen35_dflash.ascend310p.utils import count_ge_ir_nodes
        counts = count_ge_ir_nodes(graph_dir.rglob("dynamo.pbtxt"))
        if counts.get("ChunkGatedDeltaRule") != 1:
            raise RuntimeError(f"expected exactly one GDR node: {counts}")
        files = list(graph_dir.glob("*.air"))
        if len(files) != 1:
            raise RuntimeError(f"expected one AIR for {variant}")
        graphs[variant] = {"air": str(files[0].relative_to(root)), "sha256": digest(files[0]),
                           "public_input_abi": abi, "custom_op_audit": audit, "ge_nodes": counts,
                           "attributes": attributes(variant)}
        print(f"[gdr-debug] exported {variant}: one GDR, seven runtime inputs", flush=True)
        torch._dynamo.reset()
    write_json(root / "air.json", {"graphs": graphs, "prototype": proto,
               "torch": str(torch.__version__), "torch_npu": str(torch_npu.__version__),
               "device": torch.npu.get_device_name(args.device_id), "packages": package_identity()})


def execute(cmd, log, *, env=None):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    print("[gdr-debug] " + shlex.join(map(str, cmd)), flush=True)
    with log.open("x") as stream:
        with subprocess.Popen(list(map(str, cmd)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              cwd=REPO, env=env, text=True, errors="replace") as result:
            for line in result.stdout:
                stream.write(line)
                stream.flush()
                if "[stage-profile]" in line or "[gdr-debug]" in line:
                    print(line, end="", flush=True)
            result.wait()
    if result.returncode:
        tail = log.read_text(errors="replace")[-6000:]
        raise RuntimeError(f"command failed ({result.returncode}); log={log}\n{tail}")


def compile_models(args):
    root = output_dir(args.work_dir)
    manifest = json.loads((root / "air.json").read_text())
    fresh(root / "om")
    graphs = {}
    for variant in manifest["graphs"]:
        rec = manifest["graphs"][variant]
        air = root / rec["air"]
        if digest(air) != rec["sha256"]:
            raise ValueError("AIR changed since export")
        prefix = root / "om" / f"gdr_{variant}"
        cmd = [args.atc, "--mode=0", "--framework=1", f"--model={air}", f"--output={prefix}",
               f"--soc_version={args.soc_version}", "--precision_mode=must_keep_origin_dtype"]
        execute(cmd, root / "logs" / f"atc-{variant}.log")
        om = prefix.with_suffix(".om")
        graphs[variant] = {"path": str(om.relative_to(root)), "sha256": digest(om), "command": cmd,
                           "attributes": attributes(variant)}
    write_json(root / "om.json", {"graphs": graphs, "soc_version": args.soc_version,
                                   "packages": package_identity()})
    cmd = ["cmake", "-S", str(HERE), "-B", str(root / "build"), "-DCMAKE_BUILD_TYPE=Release"]
    if args.ascendcl_root:
        cmd.append(f"-DASCENDCL_ROOT={args.ascendcl_root}")
    execute(cmd, root / "logs" / "cmake.log")
    execute(["cmake", "--build", root / "build", "--parallel", "2"], root / "logs" / "build.log")


def native(args):
    root = output_dir(args.work_dir)
    data = arrays(root, args.length)
    out = output_dir(args.result_dir, new=True)
    torch, torch_npu, operation = require_device(args.device_id)
    module = gdr_module(torch, torch_npu.npu_chunk_gated_delta_rule, args.variant)
    identity = {"torch": str(torch.__version__), "torch_npu": str(torch_npu.__version__),
                "device": torch.npu.get_device_name(args.device_id), "device_id": args.device_id,
                "schema": str(operation._schema), "packages": package_identity()}
    samples, reference = [], None
    specs = output_specs(args.variant, args.length)
    np_dtype = {name: dtype for name, dtype, _, _ in specs}
    profile = None
    if args.profile_output:
        from models.dflash_v1.msprof_cli import MsprofStageProfiler
        profile = MsprofStageProfiler(args.profile_output, args.device_id, args.metrics,
                   lambda: torch.npu.synchronize(args.device_id), stage="verify")
    with torch.inference_mode(), (profile if profile else nullcontext()):
        for index in range(args.warmup + (1 if profile else args.repetitions)):
            tensors = tuple(torch.from_numpy(a.copy()).to(f"npu:{args.device_id}") for a in data)
            torch.npu.synchronize(args.device_id)
            measured = index >= args.warmup
            with profile.capture() if profile and measured else nullcontext():
                start = time.perf_counter_ns()
                outputs = module(*tensors)
                torch.npu.synchronize(args.device_id)
                elapsed = (time.perf_counter_ns() - start) / 1e6
            actual = [x.cpu().numpy() for x in outputs]
            hashes, stable = {}, True
            for (name, dtype, shape, valid_bytes), value in zip(specs, actual):
                if value.dtype != np.dtype(dtype) or value.size != int(np.prod(shape)):
                    raise ValueError(f"native output contract mismatch: {name}")
                raw = value.tobytes()
                compared = raw[:valid_bytes]
                if not np.isfinite(np.frombuffer(compared, dtype=np_dtype[name])).all():
                    raise ValueError(f"nonfinite native output: {name}")
                hashes[name] = hashlib.sha256(compared).hexdigest()
                value.reshape(shape).tofile(out / (name + ".bin"))
            # Inspect the input ABI after execution; repeated calls never feed state back.
            mutated = [NAMES[i] for i, value in enumerate(tensors)
                       if value.cpu().numpy().tobytes() != data[i].tobytes()]
            if mutated:
                raise RuntimeError(f"GDR modified declared read-only inputs: {mutated}")
            if measured:
                stable = reference is None or hashes == reference
                reference = hashes if reference is None else reference
                samples.append({"elapsed_ms": elapsed, "valid_output_sha256": hashes, "stable": stable})
    write_json(out / "report.json", {"backend": "torch_npu", "cpu_fallback": False, "variant": args.variant,
        "effective_length": args.length, "attributes": attributes(args.variant), "warmup": args.warmup,
        "profiled": bool(profile), "samples": samples, "identity": identity,
        "stable": all(x["stable"] for x in samples),
        "timing_scope": "operator dispatch plus device synchronization; output allocation included; transfers/checks excluded"})


def make_plan(root, out, variant, length, args):
    arrays(root, length)  # validate frozen bytes before constructing ACL inputs
    models = json.loads((root / "om.json").read_text())["graphs"]
    om = root / models[variant]["path"]
    if digest(om) != models[variant]["sha256"]:
        raise ValueError("OM changed since compilation")
    specs = output_specs(variant, length)
    lines = ["GDR_DEBUG_V1", plan_quote(om), plan_quote(out),
             f"{args.device_id} {args.warmup} {args.repetitions}",
             plan_quote(args.profile_output or ""), plan_quote(args.metrics), "7"]
    for name, dtype, shape in zip(NAMES, DTYPES, SHAPES):
        path = root / "inputs" / (f"length-{length}.bin" if name == "effective_length" else name + ".bin")
        count = int(np.prod(shape)) * np.dtype(dtype).itemsize
        lines.append(f'{json.dumps(name)} {ACL_TYPES[dtype]} {count} {len(shape)} '
                     + " ".join(map(str, shape)) + " " + plan_quote(path))
    lines.append(str(len(specs)))
    for name, dtype, shape, valid_bytes in specs:
        count = int(np.prod(shape)) * np.dtype(dtype).itemsize
        lines.append(f'{json.dumps(name)} {ACL_TYPES[dtype]} {count} {valid_bytes}')
    plan = out.parent / (out.name + ".plan.txt")
    with plan.open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    return plan


def child_command(root, args, backend, variant, length, out):
    if backend == "native":
        cmd = [sys.executable, "-B", str(HERE / "run.py"), "_native", "--work-dir", str(root),
               "--result-dir", str(out), "--variant", variant, "--length", str(length),
               "--device-id", str(args.device_id), "--warmup", str(args.warmup),
               "--repetitions", str(args.repetitions), "--metrics", args.metrics]
        if args.profile_output:
            cmd += ["--profile-output", args.profile_output]
        return cmd
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    return [str(root / "build/gdr_debug_runner"), str(make_plan(root, out, variant, length, args))]


def benchmark(args):
    root = output_dir(args.work_dir)
    lengths = load_manifest(root)["lengths"]
    fresh(root / "measurements")
    # Alternate native and OM observations by effective length. Same immutable inputs.
    for length in lengths:
        for backend, variant in cases(root):
            label = f"{backend}-{variant}-L{length}"
            out = root / "measurements" / label
            execute(child_command(root, args, backend, variant, length, out), root / "logs" / (label + ".log"))
    summarize(args)


def compare_output(reference, candidate, dtype, valid_bytes):
    a = np.frombuffer(reference[:valid_bytes], dtype=dtype).astype(np.float64)
    b = np.frombuffer(candidate[:valid_bytes], dtype=dtype).astype(np.float64)
    if len(reference) != len(candidate) or a.shape != b.shape:
        raise ValueError("output size mismatch")
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    return {"finite": finite, "exact_equal": bool(np.array_equal(a, b)),
            "max_abs_diff": float(np.max(np.abs(a - b))) if finite else None,
            "rmse": float(np.sqrt(np.mean((a - b) ** 2))) if finite else None,
            "compared_bytes": valid_bytes, "reference_sha256": hashlib.sha256(reference).hexdigest(),
            "candidate_sha256": hashlib.sha256(candidate).hexdigest()}


def summarize(args):
    root = output_dir(args.work_dir)
    m = load_manifest(root)
    rows = []
    for length in m["lengths"]:
        refdir = root / "measurements" / f"native-both-L{length}"
        for backend, variant in cases(root):
            label = f"{backend}-{variant}-L{length}"
            directory = root / "measurements" / label
            report = json.loads((directory / "report.json").read_text())
            samples = [r["elapsed_ms"] for r in report["samples"]]
            comparisons = {}
            for name, dtype, _, valid_bytes in output_specs(variant, length):
                comparisons[name] = compare_output((refdir / (name + ".bin")).read_bytes(),
                                        (directory / (name + ".bin")).read_bytes(), dtype, valid_bytes)
            rows.append({"case": label, "backend": backend, "variant": variant, "length": length,
                         "attributes": attributes(variant),
                         "median_ms": statistics.median(samples), "min_ms": min(samples),
                         "max_ms": max(samples), "measurements": len(samples), "stable": report["stable"],
                         "comparison_to_native": comparisons})
    payload = {"schema_version": 1, "status": "DIAGNOSTIC_OBSERVATIONS", "input_source": m["source"],
               "rows": rows, "numerical_policy": "report exact equality and errors; no relaxed acceptance gate",
               "core_tail_policy": "only first effective_length rows compared; unused tail may be undefined",
               "scope": "single GDR; a fast standalone case does not prove the full OM is correct or fast"}
    write_json(root / "summary.json", payload)
    lines = ["| Case | Median ms | Min ms | Max ms | Stable | Exact vs native |",
             "|---|---:|---:|---:|---|---|"]
    for row in rows:
        exact = all(v["exact_equal"] for v in row["comparison_to_native"].values())
        lines.append(f'| {row["case"]} | {row["median_ms"]:.6f} | {row["min_ms"]:.6f} | '
                     f'{row["max_ms"]:.6f} | {row["stable"]} | {exact} |')
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def no_state(args):
    """Append a False-attribute experiment using the previous frozen inputs/OMs."""
    parent = output_dir(args.work_dir)
    load_manifest(parent)
    previous = json.loads((parent / "om.json").read_text())
    reused = {}
    for variant in BASE_VARIANTS:
        record = previous["graphs"][variant]
        om = (parent / record["path"]).resolve()
        if digest(om) != record["sha256"]:
            raise ValueError(f"previous OM changed: {variant}")
        reused[variant] = {**record, "path": str(om), "attributes": attributes(variant),
                           "reused_from_manifest": str(parent / "om.json")}
    root = fresh(parent / ("outstate0-" + time.strftime("%Y%m%dT%H%M%S") + f"-{os.getpid()}"))
    shutil.copytree(parent / "inputs", root / "inputs")
    shutil.copyfile(parent / "inputs.json", root / "inputs.json")
    manifest = load_manifest(root)
    write_json(root / "experiment.json", {
        "comparison": "core output only: output_final_state=True versus False",
        "changed_attribute": {"output_final_state": {"baseline": True, "experiment": False}},
        "fixed_attributes": {"chunk_size": 64, "use_qk_l2norm_in_kernel": True},
        "parent": str(parent), "input_manifest_sha256": digest(parent / "inputs.json"),
        "reused_om_manifest_sha256": digest(parent / "om.json"),
        "lengths": manifest["lengths"],
        "unused_state_policy": "False state output is never read, downloaded, hashed or compared"})
    print(f"[gdr-debug] outstate=0 experiment: {root}", flush=True)
    execute([sys.executable, "-B", str(HERE / "run.py"), "export", "--work-dir", str(root),
             "--device-id", str(args.device_id), "--variants", "core_no_state"], root / "logs/export.log")
    child = argparse.Namespace(**{**vars(args), "work_dir": str(root)})
    compile_models(child)  # Only the new core_no_state AIR is compiled.
    compiled = json.loads((root / "om.json").read_text())
    compiled["graphs"] = {**reused, **compiled["graphs"]}
    write_json(root / "om.json", compiled)
    benchmark(child)
    if args.profile:
        for length in manifest["lengths"]:
            for backend in ("native", "om"):
                profile(argparse.Namespace(**{**vars(child), "backend": backend,
                                             "variant": "core_no_state", "length": length}))
    print(f"[gdr-debug] outstate=0 summary: {root / 'summary.md'}", flush=True)


def profile(args):
    root = output_dir(args.work_dir)
    if args.length not in load_manifest(root)["lengths"]:
        raise ValueError("length is absent from frozen inputs")
    base = fresh(root / "profiles" / (f"{args.backend}-{args.variant}-L{args.length}-{args.metrics}-"
                                     + time.strftime("%Y%m%dT%H%M%S") + f"-{os.getpid()}"))
    args.profile_output = str(base / "capture")
    args.repetitions = 1
    command = child_command(root, args, args.backend, args.variant, args.length, base / "result")
    controller = [sys.executable, "-B", "-m", "models.dflash_v1.msprof_cli", "--msprof-bin", args.msprof_bin,
                  "--output", args.profile_output, "--stage", "verify", "--backend",
                  "cpp" if args.backend == "om" else "python", "--metrics", args.metrics,
                  "--timeout", str(args.timeout), "--control-report", str(base / "control.json"), "--", *command]
    execute(controller, base / "controller.log")
    control = json.loads((base / "control.json").read_text())
    if (control.get("status") != "PASS_CONTROL" or
            any(control.get(k) is not True for k in ("start_acknowledged", "stop_acknowledged",
                                                       "quit_acknowledged", "capture_completed"))):
        raise RuntimeError("incomplete msprof start/stop/quit acknowledgement")
    result = json.loads((base / "result/report.json").read_text())
    if len(result["samples"]) != 1 or result["profiled"] is not True:
        raise RuntimeError("profile application did not execute exactly one measured call")
    execute([args.msprof_bin, "--export=on", f"--output={args.profile_output}", "--summary-format=csv"],
            base / "export.log")
    from tools.profile_verify_om import summarize as summarize_profile
    summarize_profile(base / "capture", fresh(base / "operators"))
    extract_gdr_rows(base / "capture", base / "gdr-rows.json")
    comparisons = {}
    reference = root / "measurements" / f"native-both-L{args.length}"
    if reference.is_dir():
        for name, dtype, _, valid_bytes in output_specs(args.variant, args.length):
            comparisons[name] = compare_output((reference / (name + ".bin")).read_bytes(),
                                      (base / "result" / (name + ".bin")).read_bytes(), dtype, valid_bytes)
    write_json(base / "case.json", {"backend": args.backend, "variant": args.variant,
        "effective_length": args.length, "attributes": attributes(args.variant), "expected_gdr_calls": 1,
        "comparison_to_unprofiled_native": comparisons,
        "scope": "one standalone GDR; protocol stage name verify is a controller routing label"})
    print(f"[gdr-debug] profile: {base}", flush=True)


def extract_gdr_rows(capture, output):
    exports = []
    for path in sorted(Path(capture).rglob("op_summary*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            keys = {"".join(c for c in name.lower() if c.isalnum()): name for name in reader.fieldnames or []}
            rows = [row for row in reader if "chunkgateddeltarule" in
                    "".join(c for c in row.get(keys.get("optype", ""), "").lower() if c.isalnum())]
        if len(rows) != 1:
            raise ValueError(f"expected exactly one GDR task per CSV, got {len(rows)}: {path}")
        duration = float(rows[0][keys["taskdurationus"]])
        if not math.isfinite(duration) or duration < 0:
            raise ValueError(f"invalid GDR task duration: {path}")
        exports.append({"csv": str(path), "gdr_duration_us": duration, "raw_row": rows[0]})
    if not exports:
        raise ValueError("msprof produced no op_summary CSV")
    write_json(output, {"status": "ONE_GDR_PER_EXPORT", "exports": exports,
                        "note": "Each CSV is separate; repeated exports are not added together."})


def capture(args):
    root = output_dir(args.work_dir, new=True)
    torch, torch_npu, operation = require_device(args.device_id)
    original = torch_npu.npu_chunk_gated_delta_rule
    captured = False
    seen = 0

    class CaptureComplete(BaseException):
        pass

    def wrapped(*positional, **keywords):
        nonlocal captured, seen
        values = {arg.name: arg.default_value for arg in operation._schema.arguments}
        values.update({arg.name: value for arg, value in zip(operation._schema.arguments, positional)})
        values.update(keywords)
        for canonical, alias in (("query", "q"), ("key", "k"), ("value", "v")):
            if canonical not in values and alias in values:
                values[canonical] = values[alias]
        query = values.get("query", values.get("q"))
        if not captured and query is not None and tuple(query.shape) == SHAPES[0]:
            selected = seen == args.capture_index
            seen += 1
            if not selected:
                return original(*positional, **keywords)
            attributes = {k: values[k] for k in ATTRS}
            if attributes != ATTRS:
                raise ValueError(f"unexpected captured GDR attributes: {attributes}")
            data = {}
            for name in NAMES:
                value = values.get(name)
                if value is None:
                    raise ValueError(f"missing captured input: {name}")
                data[name] = value.detach().cpu().contiguous().numpy().copy()
            data = validate_arrays(data)
            np.savez(root / "inputs.npz", **data)
            write_json(root / "capture.json", {"status": "CAPTURED_INPUTS", "source": "torch_npu eager",
                "attributes": attributes, "schema": str(operation._schema), "module": args.module,
                "effective_length": int(data["effective_length"][0]), "sha256": digest(root / "inputs.npz"),
                "matching_call_index": args.capture_index,
                "caller_stack": traceback.format_stack(limit=12),
                "note": "Captures native inputs, not the OM's internal addresses or tiling."})
            captured = True
            print(f"[gdr-debug] captured {root / 'inputs.npz'}", flush=True)
            if args.stop_after_capture:
                raise CaptureComplete()
        return original(*positional, **keywords)

    torch_npu.npu_chunk_gated_delta_rule = wrapped
    previous = sys.argv
    try:
        sys.argv = [args.module, *(args.application[1:] if args.application[:1] == ["--"] else args.application)]
        try:
            runpy.run_module(args.module, run_name="__main__", alter_sys=True)
        except CaptureComplete:
            pass
        except SystemExit as error:
            if error.code not in (None, 0):
                raise
    finally:
        sys.argv = previous
        torch_npu.npu_chunk_gated_delta_rule = original
    if not captured:
        raise RuntimeError(f"no requested [1,16,32,128] GDR call captured; matching calls seen={seen}")


def length_list(value):
    result = [int(x) for x in value.split(",")]
    if not result or len(result) != len(set(result)) or any(not 1 <= x <= 16 for x in result):
        raise argparse.ArgumentTypeError("lengths must be unique integers in 1..16")
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="command", required=True)
    for command in ("all", "no-state", "prepare", "export", "compile", "benchmark", "summarize", "profile", "capture", "_native"):
        s = subs.add_parser(command)
        s.add_argument("--work-dir", required=True, help="artifact directory below AI_RUN_DIR, outside source")
        s.add_argument("--device-id", type=int, default=0)
        s.add_argument("--warmup", type=int, default=3)
        s.add_argument("--repetitions", type=int, default=10)
        s.add_argument("--metrics", choices=("PipeUtilization", "Memory", "MemoryUB"), default="PipeUtilization")
        s.set_defaults(profile_output=None)
        if command in ("all", "prepare"):
            s.add_argument("--inputs", help="NPZ captured by the capture subcommand, or an equivalent seven-tensor NPZ")
            s.add_argument("--seed", type=int, default=20260909)
            s.add_argument("--lengths", type=length_list, default=[16, 8])
        if command == "export":
            s.add_argument("--variants", nargs="+", choices=VARIANTS, default=BASE_VARIANTS)
        if command in ("all", "compile", "no-state"):
            s.add_argument("--atc", default="atc")
            s.add_argument("--soc-version", default="Ascend310P3")
            s.add_argument("--ascendcl-root", default="")
        if command in ("all", "profile", "no-state"):
            s.add_argument("--msprof-bin", default="msprof")
            s.add_argument("--timeout", type=float, default=600)
        if command in ("all", "no-state"):
            s.add_argument("--profile", action="store_true", help="also capture one warmed GDR per case (no-state: new False cases only)")
        if command in ("profile", "_native"):
            s.add_argument("--variant", choices=VARIANTS, default="both")
            s.add_argument("--length", type=int, default=16)
        if command == "profile":
            s.add_argument("--backend", choices=("native", "om"), default="om")
        if command == "_native":
            s.add_argument("--result-dir", required=True)
            s.add_argument("--profile-output")
        if command == "capture":
            s.add_argument("--module", default="models.dflash_v1.run_npu")
            s.add_argument("--stop-after-capture", action="store_true")
            s.add_argument("--capture-index", type=int, default=0, help="zero-based index among matching T=16 GDR calls")
            s.add_argument("application", nargs=argparse.REMAINDER)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.device_id < 0 or args.warmup < 0 or args.repetitions < 1:
        raise ValueError("invalid device, warmup, or repetitions")
    if hasattr(args, "timeout") and (not math.isfinite(args.timeout) or args.timeout <= 0):
        raise ValueError("timeout must be finite and positive")
    if getattr(args, "capture_index", 0) < 0:
        raise ValueError("capture index must be non-negative")
    if args.command == "all":
        prepare(args)
        root = output_dir(args.work_dir)
        execute([sys.executable, "-B", str(HERE / "run.py"), "export", "--work-dir", str(root),
                 "--device-id", str(args.device_id)], root / "logs/export.log")
        compile_models(args)
        benchmark(args)
        if args.profile:
            for length in args.lengths:
                for backend, variant in cases(root):
                    profile(argparse.Namespace(**{**vars(args), "backend": backend, "variant": variant, "length": length}))
        return 0
    {"no-state": no_state, "prepare": prepare, "export": export, "compile": compile_models, "benchmark": benchmark,
     "summarize": summarize, "profile": profile, "capture": capture, "_native": native}[args.command](args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[gdr-debug] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
