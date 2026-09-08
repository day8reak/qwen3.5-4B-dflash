"""Cross-language fake ACL lifecycle and report gates, never device evidence."""
import copy
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from qwen35_dflash.ascend310p.cpp_runtime import validate_incremental_cpp_runner_report
from qwen35_dflash.ascend310p.utils import sha256_file


@pytest.fixture(scope="module")
def split_reports(tmp_path_factory):
    if not shutil.which("cmake") or not shutil.which("c++"):
        pytest.skip("host C++ toolchain required for fake ACL integration")
    source = Path(__file__).resolve().parents[1] / "framework/runtime/cpp"
    build = tmp_path_factory.mktemp("static-split-cpp")
    for command in (
        ["cmake", "-S", str(source), "-B", str(build),
         "-DQWEN35_DFLASH_BUILD_ACL_RUNNER=OFF", "-DCMAKE_BUILD_TYPE=Release"],
        ["cmake", "--build", str(build), "--target", "qwen35_dflash_incremental_acl_runner_fake", "--parallel", "4"],
        ["ctest", "--test-dir", str(build), "-R", "^qwen35_dflash_fake_static_split_runner$", "--output-on-failure"],
    ):
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
    paths = sorted(build.glob("fake-static-split-report-*.json"))
    assert len(paths) == 31
    return [json.loads(path.read_text()) for path in paths]


def _options(report):
    protocol = report["protocol"]
    return dict(
        prompt_token_ids=report["prompt_token_ids"],
        om_sha256_by_role={m["role"]: sha256_file(m["path"]) for m in report["models"]},
        device_id=0, max_new_tokens=report["limits"]["max_new_tokens"],
        max_draft_tokens=report["limits"]["max_draft_tokens"],
        state_reset_policy=protocol["state_reset_policy"],
        decode_carrier_policy=protocol["decode_carrier_policy"],
        draft_feature_policy=protocol["draft_feature_policy"],
        dflash_sync_window=protocol["dflash_sync_window"],
        prefill_completion_policy=protocol["prefill_completion_policy"],
        zero_accept_fallback_policy=protocol["zero_accept_fallback_policy"],
        draft_static_feature_rows=64,
        model_residency_policy=report["model_residency"]["policy"],
    )


def test_cpp_reports_close_in_python_control_plane(split_reports):
    for report in split_reports:
        validate_incremental_cpp_runner_report(report, **_options(report))


@pytest.mark.parametrize("scope,key", [
    ("model_memory_query", "allocated_weight_bytes"),
    ("execution_io_counters", "target_prefill_head_executions"),
    ("execution_io_counters", "draft_static_padding_rows"),
    ("execution_io_counters", "draft_dynamic_gear_count"),
    ("model_residency", "split_group_loads"),
    ("model_residency", "model_unloads"),
])
def test_corrupted_split_reports_fail_closed(split_reports, scope, key):
    for report in split_reports:
        damaged = copy.deepcopy(report)
        damaged[scope][key] += 1
        with pytest.raises((RuntimeError, ValueError)):
            validate_incremental_cpp_runner_report(damaged, **_options(report))
