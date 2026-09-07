from __future__ import annotations

import ctypes as C
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "framework/scripts/inspect_om_io.py"
SPEC = importlib.util.spec_from_file_location("om_io_inspector", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
inspector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inspector)


def manifest_fixture(root: Path) -> Path:
    model = root / "fused-speculative-step_linux_aarch64.om"
    model.write_bytes(b"host-test-OM-not-a-device-artifact")
    manifest = root / "deployment-manifest.json"
    manifest.write_text(json.dumps({
        "artifact_kind": "qwen35-dflash-ascend310p-om-bundle",
        "graphs": [{
            "name": "fused-speculative-step",
            "dynamic": True,
            "input_names": ["target_feature_tail"],
            "om": {
                "path": model.name,
                "bytes": model.stat().st_size,
                "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            },
        }],
    }), encoding="utf-8")
    return manifest


class FakeAcl:
    """A Shape-like metadata fixture, not a fake successful model execution."""

    def __init__(self, fail_load: bool = False, fail_cleanup: bool = False):
        self.calls: list[str] = []
        self.fail_load = fail_load
        self.fail_cleanup = fail_cleanup

    def aclInit(self, config):
        self.calls.append("init")
        return 0

    def aclrtSetDevice(self, device):
        self.calls.append("set_device")
        return 0

    def aclrtCreateContext(self, pointer, device):
        self.calls.append("create_context")
        C.cast(pointer, C.POINTER(C.c_void_p))[0] = 123
        return 0

    def aclmdlLoadFromFile(self, path, pointer):
        self.calls.append("load")
        C.cast(pointer, C.POINTER(C.c_uint32))[0] = 7
        return 100001 if self.fail_load else 0

    def aclmdlCreateDesc(self):
        self.calls.append("create_desc")
        return 456

    def aclmdlGetDesc(self, desc, model_id):
        self.calls.append("get_desc")
        return 0

    def aclmdlGetNumInputs(self, desc):
        return 3

    def aclmdlGetNumOutputs(self, desc):
        return 1

    def aclmdlGetInputDims(self, desc, index, pointer):
        if index == 2:
            return 100002
        dims = C.cast(pointer, C.POINTER(inspector.IODims)).contents
        dims.name = b"shape_symbol" if index == 0 else b"target_feature_tail"
        dims.dimCount = 0 if index == 0 else 3
        dims.dims[:3] = (1, -1, 8)
        return 0

    def aclmdlGetOutputDims(self, desc, index, pointer):
        dims = C.cast(pointer, C.POINTER(inspector.IODims)).contents
        dims.name = b"output"
        dims.dimCount = 1
        dims.dims[0] = -1
        return 0

    def aclmdlGetInputDataType(self, desc, index):
        return 9 if index == 0 else 1

    def aclmdlGetOutputDataType(self, desc, index):
        return 1

    def aclmdlGetInputSizeByIndex(self, desc, index):
        return 8 if index == 0 else 0

    def aclmdlGetOutputSizeByIndex(self, desc, index):
        return 0

    def aclmdlGetInputIndexByName(self, desc, name, pointer):
        assert name == b"ascend_mbatch_shape_data"
        return 100000

    def aclmdlGetInputDynamicGearCount(self, desc, index, pointer):
        assert index == C.c_size_t(-1).value
        return 100000

    def aclmdlDestroyDesc(self, desc):
        self.calls.append("destroy_desc")
        return 1 if self.fail_cleanup else 0

    def aclmdlUnload(self, model_id):
        self.calls.append("unload")
        return 0

    def aclrtDestroyContext(self, context):
        self.calls.append("destroy_context")
        return 0

    def aclrtResetDevice(self, device):
        self.calls.append("reset_device")
        return 0

    def aclFinalize(self):
        self.calls.append("finalize")
        return 0


def test_query_keeps_scalars_dynamic_sizes_and_query_errors():
    acl = FakeAcl()
    records = inspector.describe_io(acl, 456, "Input")
    assert records[0]["name"] == "shape_symbol"
    assert records[0]["dim_count"] == 0
    assert records[0]["shape"] == []
    assert records[0]["bytes"] == 8
    assert records[1]["shape"] == [1, -1, 8]
    assert records[1]["bytes"] == 0
    assert records[2] == {"index": 2, "dims_status": 100002, "dtype": 1, "bytes": 0}
    assert inspector.describe_io(acl, 456, "Output")[0]["bytes"] == 0


def test_missing_control_is_evidence_not_a_static_model_classification():
    result = inspector.describe_dynamic(FakeAcl(), 456)
    assert result == {
        "control_name": "ascend_mbatch_shape_data",
        "control_lookup_status": 100000,
        "control_index": None,
        "gear_count_status": 100000,
        "gear_count": None,
    }


def test_gears_preserve_full_flattened_dimensions():
    class GearAcl(FakeAcl):
        def aclmdlGetInputIndexByName(self, desc, name, pointer):
            C.cast(pointer, C.POINTER(C.c_size_t))[0] = 3
            return 0

        def aclmdlGetInputDynamicGearCount(self, desc, index, pointer):
            C.cast(pointer, C.POINTER(C.c_size_t))[0] = 2
            return 0

        def aclmdlGetInputDynamicDims(self, desc, index, dims, count):
            assert count == 2
            for gear, value in zip(dims, (16, 64)):
                gear.dimCount = 3
                gear.dims[:3] = (1, value, 8)
            return 0

    result = inspector.describe_dynamic(GearAcl(), 456)
    assert result["control_index"] == 3
    assert [gear["shape"] for gear in result["gears"]] == [[1, 16, 8], [1, 64, 8]]


def test_malformed_rank_is_bounded_and_preserved():
    dims = inspector.IODims()
    dims.dimCount = 129
    record = inspector.dims_record(dims)
    assert record["dim_count"] == 129
    assert record["rank_within_acl_capacity"] is False
    assert len(record["shape"]) == 128


def test_inspection_loads_one_model_without_inference_and_cleans_up(tmp_path):
    acl = FakeAcl()
    report = {}
    inspector.inspect_model(acl, tmp_path / "model.om", 0, report)
    assert report["status"] == "OBSERVED"
    assert report["model_id"] == 7
    assert acl.calls == [
        "init", "set_device", "create_context", "load", "create_desc", "get_desc",
        "destroy_desc", "unload", "destroy_context", "reset_device", "finalize",
    ]
    assert all(item["status"] == 0 for item in report["cleanup"])


def test_load_failure_releases_context_without_unloading_unknown_model(tmp_path):
    acl = FakeAcl(fail_load=True)
    report = {}
    with pytest.raises(RuntimeError, match="aclmdlLoadFromFile.*100001"):
        inspector.inspect_model(acl, tmp_path / "model.om", 0, report)
    assert acl.calls == [
        "init", "set_device", "create_context", "load",
        "destroy_context", "reset_device", "finalize",
    ]


def test_cleanup_failure_is_not_success_and_does_not_skip_remaining_cleanup(tmp_path):
    acl = FakeAcl(fail_cleanup=True)
    report = {}
    inspector.inspect_model(acl, tmp_path / "model.om", 0, report)
    assert report["status"] == "FAILED"
    assert report["cleanup_failed"] is True
    assert acl.calls[-1] == "finalize"


def test_manifest_locks_selected_model_without_predicting_its_actual_abi(tmp_path):
    manifest = manifest_fixture(tmp_path)
    model, metadata = inspector.resolve_model(manifest, "fused-speculative-step")
    assert model.name == "fused-speculative-step_linux_aarch64.om"
    assert metadata["declared_dynamic"] is True
    assert metadata["logical_input_names"] == ["target_feature_tail"]
    assert "inputs" not in metadata


@pytest.mark.parametrize("damage", ["size", "sha256", "duplicate", "escape"])
def test_invalid_manifest_is_rejected_before_loading(tmp_path, damage):
    manifest = manifest_fixture(tmp_path)
    data = json.loads(manifest.read_text())
    graph = data["graphs"][0]
    if damage == "size":
        graph["om"]["bytes"] += 1
    elif damage == "sha256":
        graph["om"]["sha256"] = "0" * 64
    elif damage == "duplicate":
        data["graphs"].append(graph.copy())
    else:
        graph["om"]["path"] = "../outside.om"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        inspector.resolve_model(manifest, "fused-speculative-step")


def test_output_requires_active_run_containment_and_no_overwrite(tmp_path, monkeypatch):
    destination = tmp_path / "reports/io.json"
    monkeypatch.delenv("AI_RUN_DIR", raising=False)
    with pytest.raises(RuntimeError, match="AI_RUN_DIR"):
        inspector.output_path(destination)
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="below AI_RUN_DIR"):
        inspector.output_path(tmp_path.parent / "outside.json")
    assert inspector.output_path(destination) == destination
    destination.write_text("retained evidence")
    with pytest.raises(FileExistsError):
        inspector.output_path(destination)
    assert destination.read_text() == "retained evidence"


def test_plain_python_cli_without_torch_retains_library_failure(tmp_path):
    manifest = manifest_fixture(tmp_path)
    report = tmp_path / "reports/io.json"
    environment = dict(os.environ, AI_RUN_DIR=str(tmp_path))
    completed = subprocess.run(
        [sys.executable, "-I", "-S", str(SCRIPT),
         "--deployment-manifest", str(manifest), "--output", str(report),
         "--library", str(tmp_path / "missing-libascendcl.so")],
        env=environment, capture_output=True, text=True,
    )
    assert completed.returncode == 1
    data = json.loads(report.read_text())
    assert data["status"] == "FAILED"
    assert "OSError" in data["error"]
    assert "missing-libascendcl.so" in data["error"]
    assert "Report:" in completed.stdout
