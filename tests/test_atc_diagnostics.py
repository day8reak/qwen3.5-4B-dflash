"""ATC diagnostics are compiler evidence, not a target execution test."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/python"))

from qwen35_dflash.ascend310p.compiler import _atc_failure_detail


GDR_ERROR = (
    "[DEBUG] GE ParseJsonFormatString:add error_code EZ9999 success\n"
    "[ERROR] GE No supported engine found\n"
    "[PID: 1571259] Unsupported_Operator(EZ3002): Optype [ChunkGatedDeltaRule] "
    "of Ops kernel [AIcoreEngine] is unsupported. Reason: "
    "[tbe-custom] type ChunkGatedDeltaRule is not found in this op store. "
    "[Dynamic shape check]: data type DT_FLOAT of output [core_attn] is not supported. "
    "All supported data type: {DT_FLOAT16}\n"
)


def test_error_surfaces_type_mismatch_not_just_exit_or_shutdown():
    detail = _atc_failure_detail(GDR_ERROR + "[INFO] shutdown\n" * 30, {})
    assert "Unsupported_Operator(EZ3002)" in detail
    assert "DT_FLOAT of output [core_attn]" in detail
    assert "ParseJsonFormatString" not in detail
    assert "does not establish a missing kernel" in detail
    assert "no pre-save GDR dtype audit" in detail


def test_error_distinguishes_a_passing_python_descriptor_from_atc_inference():
    graph = {"runtime_input_abi": {"gdr_output_dtypes": {"status": "PASS", "node_count": 24}}}
    detail = _atc_failure_detail(GDR_ERROR, graph)
    assert "passed the pre-save dtype check" in detail
    assert "InferShape/InferDataType" in detail
    assert "no pre-save GDR dtype audit" not in detail


def test_error_excerpt_is_bounded_and_retains_other_operator_diagnostics():
    detail = _atc_failure_detail("Unsupported_Operator(EZ3002): OtherOp " + "x" * 20000, {})
    assert "OtherOp" in detail
    assert "GDR output contract" not in detail
    assert len(detail) < 6100
    assert _atc_failure_detail("", {}) == ""
    assert "fatal: test compiler error" in _atc_failure_detail("fatal: test compiler error", {})
