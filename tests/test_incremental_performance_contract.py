from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "framework" / "abi" / "incremental-performance-v2.json"
DOCUMENT_PATH = ROOT / "docs" / "INCREMENTAL_OM_PERFORMANCE.md"
FRAMEWORK_LOCK_PATH = ROOT / "framework" / "FRAMEWORK_LOCK.json"
DEPLOYMENT_PATH = ROOT / "framework" / "abi" / "dflash-deployment-v1.json"
PERFORMANCE_PATH = ROOT / "framework" / "abi" / "performance-v1.json"


def _contract() -> dict[str, object]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_incremental_contract_is_a_non_active_exact_proposal() -> None:
    contract = _contract()
    assert contract["status"] == "PROPOSED_NOT_ACTIVE"
    correctness = contract["non_negotiable_correctness"]
    assert correctness["ordinary_target_is_authoritative"] is True
    assert correctness["allowed_token_id_mismatches"] == 0
    assert correctness["allowed_eos_mismatches"] == 0
    assert correctness["approximation_allowed"] is False
    assert contract["activation_gate"]["requires_explicit_approval"] is True


def test_incremental_state_budget_matches_locked_qwen35_shapes() -> None:
    contract = _contract()
    symbols = contract["symbols"]
    budget = contract["bytes_at_batch1_kv_max_len_2048_verify_t16"]

    linear_layers = symbols["linear_attention_layers"]
    full_layers = symbols["full_attention_layers"]
    draft_layers = symbols["draft_layers"]
    verify_rows = symbols["maximum_verify_rows"]

    target_conv = (
        linear_layers
        * symbols["gdr_channels"]
        * symbols["conv_window"]
        * 2
    )
    target_recurrent = (
        linear_layers
        * symbols["gdr_value_heads"]
        * symbols["gdr_key_dim"]
        * symbols["gdr_value_dim"]
        * 2
    )
    target_kv = (
        full_layers
        * 2
        * 2048
        * symbols["target_kv_heads"]
        * symbols["target_head_dim"]
        * 2
    )
    draft_kv = (
        draft_layers
        * 2
        * 2048
        * symbols["draft_kv_heads"]
        * symbols["draft_head_dim"]
        * 2
    )
    recurrent_bank_fp32 = target_recurrent * verify_rows * 2
    conv_bank_fp16 = target_conv * verify_rows

    assert budget["target_scalar_conv_fp16"] == target_conv
    assert budget["target_scalar_recurrent_fp16"] == target_recurrent
    assert budget["target_full_attention_kv_fp16"] == target_kv
    assert budget["draft_persistent_kv_fp16"] == draft_kv
    assert budget["target_persistent_state_total"] == (
        target_conv + target_recurrent + target_kv
    )
    assert budget["persistent_state_subtotal"] == (
        target_conv + target_recurrent + target_kv + draft_kv
    )
    assert budget["transient_target_recurrent_bank_fp32"] == recurrent_bank_fp32
    assert budget["transient_target_conv_bank_fp16"] == conv_bank_fp16
    assert budget["transient_verify_bank_subtotal"] == (
        recurrent_bank_fp32 + conv_bank_fp16
    )


def test_topology_is_selected_by_evidence_not_file_count() -> None:
    contract = _contract()
    candidates = contract["physical_topology_candidates"]
    assert {candidate["id"] for candidate in candidates} == {
        "two-dynamic",
        "three-resident",
        "four-resident",
        "fused-speculative-step",
    }
    selection = contract["topology_selection_gates"]
    assert "sum queried weightSize" in selection["weight_budget"]
    assert "cross-artifact weight sharing must not be assumed" in selection["weight_budget"]
    assert "zero-mismatch" in selection["winner"]
    assert "not the candidate with the most or fewest OM files" in selection["winner"]


def test_hot_loop_keeps_large_state_and_proposals_on_device() -> None:
    contract = _contract()
    state = contract["state_ownership"]
    assert "target GDR state banks" in state["host_forbidden_payloads_per_round"]
    assert "target or Draft KV cache" in state["host_forbidden_payloads_per_round"]
    assert contract["hot_loop"]["draft_to_verify"].startswith(
        "proposal IDs stay on device"
    )
    assert contract["hot_loop"]["synchronization"].startswith(
        "one host-visible synchronization"
    )
    transaction = contract["strict_greedy_transaction"]
    assert transaction["selected_target_state_slot"].startswith("a because")
    assert transaction["committed_target_input_rows"] == "1 + a"


def test_document_contains_memory_inspector_and_claim_boundary() -> None:
    document = DOCUMENT_PATH.read_text(encoding="utf-8")
    assert "qwen35_dflash_om_inspect" in document
    assert "--model target-prefill=" in document
    assert "--model target-verify-commit=" in document
    assert "--state-bytes" in document
    assert "PROPOSED_NOT_ACTIVE" in document
    assert "不能宣称" in document


def test_current_integrated_runner_freezes_exact_ranged_io_evidence() -> None:
    framework_lock = json.loads(FRAMEWORK_LOCK_PATH.read_text(encoding="utf-8"))
    deployment = json.loads(DEPLOYMENT_PATH.read_text(encoding="utf-8"))
    performance = json.loads(PERFORMANCE_PATH.read_text(encoding="utf-8"))

    assert framework_lock["schema_version"] == 10
    runtime = framework_lock["runtime"]
    assert "input device mirrors" in runtime["memory"]
    assert "last K+1 rows" in runtime["memory"]
    assert "actual/full-equivalent H2D and D2H bytes" in (
        runtime["execution_io_evidence"]
    )

    cpp_runtime = deployment["cpp_runtime"]
    assert "changed contiguous range" in cpp_runtime["memory"]
    assert "last K+1 Target rows" in cpp_runtime["execution"]
    assert "actual versus full-equivalent transfer bytes" in (
        cpp_runtime["execution_io_report"]
    )

    runner = performance["runner_contract"]
    assert runner["current_integrated_om_io"].startswith(
        "persistent input device mirror"
    )
    assert "maximum_target_elements_per_call" in runner["required_io_counters"]
