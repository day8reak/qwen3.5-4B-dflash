from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "framework" / "abi" / "incremental-performance-v2.json"
APPROVAL_PATH = (
    ROOT
    / "framework"
    / "abi"
    / "approvals"
    / "incremental-performance-v2.json"
)
DOCUMENT_PATH = ROOT / "docs" / "INCREMENTAL_OM_PERFORMANCE.md"
FRAMEWORK_LOCK_PATH = ROOT / "framework" / "FRAMEWORK_LOCK.json"
DEPLOYMENT_PATH = ROOT / "framework" / "abi" / "dflash-deployment-v1.json"
PERFORMANCE_PATH = ROOT / "framework" / "abi" / "performance-v1.json"
BATCHED_CACHE_UPDATE_PATH = (
    ROOT / "framework" / "abi" / "batched-cache-update-v1.json"
)


def _contract() -> dict[str, object]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _batched_cache_update_proposal() -> dict[str, object]:
    return json.loads(BATCHED_CACHE_UPDATE_PATH.read_text(encoding="utf-8"))


def test_incremental_contract_has_exact_approval_but_is_not_active() -> None:
    contract = _contract()
    approval = json.loads(APPROVAL_PATH.read_text(encoding="utf-8"))
    assert contract["status"] == "APPROVED_IN_IMPLEMENTATION_NOT_ACTIVE"
    assert approval["status"] == "APPROVED"
    assert approval["approval_statement"] == "批准多OM状态图"
    assert approval["proposal"]["sha256_before_approval"] == (
        contract["approval"]["approved_proposal_sha256"]
    )
    assert approval["proposal"]["git_commit"] == (
        contract["approval"]["approved_base_commit"]
    )
    correctness = contract["non_negotiable_correctness"]
    assert correctness["ordinary_target_is_authoritative"] is True
    assert correctness["allowed_token_id_mismatches"] == 0
    assert correctness["allowed_eos_mismatches"] == 0
    assert correctness["approximation_allowed"] is False
    assert contract["activation_gate"]["requires_explicit_approval"] is False
    assert contract["activation_gate"]["explicit_approval_status"] == (
        "PASS_RECORDED"
    )
    assert approval["constraints"]["approximation_allowed"] is False


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
    target_recurrent_fp32 = (
        linear_layers
        * symbols["gdr_value_heads"]
        * symbols["gdr_key_dim"]
        * symbols["gdr_value_dim"]
        * 4
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
    recurrent_bank_fp32 = target_recurrent_fp32 * verify_rows
    conv_bank_fp16 = target_conv * verify_rows

    assert budget["target_scalar_conv_fp16"] == target_conv
    assert budget["target_scalar_recurrent_fp32"] == target_recurrent_fp32
    assert budget["target_full_attention_kv_fp16"] == target_kv
    assert budget["draft_persistent_kv_fp16"] == draft_kv
    assert budget["target_persistent_state_total"] == (
        target_conv + target_recurrent_fp32 + target_kv
    )
    assert budget["persistent_state_subtotal"] == (
        target_conv + target_recurrent_fp32 + target_kv + draft_kv
    )
    assert budget["transient_target_recurrent_bank_fp32"] == recurrent_bank_fp32
    assert budget["transient_target_conv_bank_fp16"] == conv_bank_fp16
    assert budget["transient_verify_bank_subtotal"] == (
        recurrent_bank_fp32 + conv_bank_fp16
    )
    assert budget["legacy_graph_entry_seed_live_set"] == (
        recurrent_bank_fp32 + conv_bank_fp16
    )
    per_layer_seed = recurrent_bank_fp32 // linear_layers
    assert budget["per_linear_layer_jit_recurrent_seed_max"] == per_layer_seed
    assert budget["source_graph_seed_live_set_reduction_candidate"] == (
        budget["legacy_graph_entry_seed_live_set"] - per_layer_seed
    )
    assert budget["conv_input_bank_materialization_bytes_eliminated_per_verify"] == (
        conv_bank_fp16
    )
    assert budget["conv_input_bank_gathers_eliminated_per_verify"] == linear_layers
    legacy_index_nodes = full_layers * 2 * verify_rows * 2
    assert budget["legacy_cache_index_div_or_remainder_nodes_per_verify"] == (
        legacy_index_nodes
    )
    assert budget["current_cache_index_div_or_remainder_nodes_per_verify"] == 2
    assert budget["cache_index_div_or_remainder_nodes_eliminated_per_verify"] == (
        legacy_index_nodes - 2
    )
    assert budget["legacy_cache_index_cast_nodes_per_verify"] == legacy_index_nodes
    assert budget["current_cache_index_cast_nodes_per_verify"] == 2
    assert budget["cache_index_cast_nodes_eliminated_per_verify"] == (
        legacy_index_nodes - 2
    )
    assert budget["full_attention_mask_casts_eliminated_per_target_call"] == (
        full_layers
    )
    assert budget["cache_update_model_nodes_per_verify_unchanged"] == (
        full_layers * 2 * verify_rows
    )
    assert budget["seed_policy"] == "per-linear-layer-jit-v1"


def test_batched_cache_update_is_an_exact_unapproved_boundary() -> None:
    proposal = _batched_cache_update_proposal()
    assert proposal["status"] == "AWAITING_EXPLICIT_APPROVAL"
    assert proposal["classification"] == (
        "exact_graph_and_operator_boundary_change"
    )
    assert proposal["correctness_contract"]["approximation_allowed"] is False
    assert proposal["correctness_contract"][
        "external_om_bindings_unchanged"
    ] is True
    approval = proposal["approval"]
    assert approval["required"] is True
    assert approval["required_statement"] == "批准 batched-cache-update-v1"
    assert approval["record_present"] is False
    assert approval["existing_multi_om_approval_covers_this_change"] is False
    assert not (ROOT / approval["record_path_after_approval"]).exists()


def test_batched_cache_update_proposal_counts_match_locked_topology() -> None:
    contract = _contract()
    proposal = _batched_cache_update_proposal()
    symbols = contract["symbols"]
    baseline = proposal["baseline"]
    candidate = proposal["candidate"]
    current = (
        symbols["full_attention_layers"]
        * 2
        * symbols["maximum_verify_rows"]
    )
    batched = symbols["full_attention_layers"] * 2
    assert baseline["current_model_nodes_at_t16"] == current == 256
    assert candidate["model_nodes_at_t16"] == batched == 16
    assert candidate["model_nodes_eliminated_at_t16"] == current - batched
    assert candidate["model_node_reduction_percent"] == 93.75
    assert proposal["rollback"]["safe_git_commit"] == (
        baseline["git_commit"]
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
    assert "stays on device" in contract["hot_loop"]["draft_to_verify"]
    assert "window 2" in contract["hot_loop"]["synchronization"]
    assert set(contract["hot_loop"]["dflash_sync_window_policies"]) == {
        "1",
        "2",
    }
    assert "K1 may be smaller" in contract["hot_loop"][
        "dflash_sync_window_budget_guard"
    ]
    transaction = contract["strict_greedy_transaction"]
    assert transaction["selected_target_state_slot"].startswith("a because")
    assert transaction["committed_target_input_rows"] == "1 + a"
    assert transaction["fixed_physical_verify"].startswith(
        "the Target always executes 16 causal rows"
    )


def test_tensor_abi_persists_only_scalar_target_state() -> None:
    contract = _contract()
    tensor_abi = contract["tensor_abi"]
    target = {item["name"]: item for item in tensor_abi["target_persistent_state"]}
    assert target["target_conv_state"]["shape"][:2] == [24, 1]
    assert target["target_recurrent_state"]["dtype"] == "float32"
    assert target["target_key_cache"]["shape"][0] == 8
    assert target["target_value_cache"]["shape"][0] == 8
    assert tensor_abi["external_transient_banks_forbidden"] == [
        "target_conv_state_bank [24,1,16,8192,4]",
        "target_recurrent_state_bank [24,1,16,32,128,128]",
    ]
    assert tensor_abi["scalar_state_seed_policy"] == (
        "per-linear-layer-jit-v1"
    )
    assert tensor_abi["verify_cache_index_policy"] == "once-per-verify-v1"
    carriers = {item["name"]: item for item in tensor_abi["round_carriers"]}
    assert carriers["verify_input_ids"]["shape"] == [1, 16]
    assert carriers["logical_proposal_count"]["range"] == [1, 15]


def test_document_contains_memory_inspector_and_claim_boundary() -> None:
    document = DOCUMENT_PATH.read_text(encoding="utf-8")
    assert "qwen35_dflash_om_inspect" in document
    assert "--model target-prefill=" in document
    assert "--model target-verify-commit=" in document
    assert "--state-bytes" in document
    assert "APPROVED_IN_IMPLEMENTATION_NOT_ACTIVE" in document
    assert "--decode-carrier-policy" in document
    assert "analyze-msprof" in document
    assert "profile_model_execution_trace" in document
    assert "expected/observed" in document
    assert "one-token-h2d" in document
    assert "last-token-d2d" in document
    assert "--dflash-sync-window" in document
    assert "K0=15,K1=14" in document
    assert "不能宣称" in document


def test_current_integrated_runner_freezes_exact_ranged_io_evidence() -> None:
    framework_lock = json.loads(FRAMEWORK_LOCK_PATH.read_text(encoding="utf-8"))
    deployment = json.loads(DEPLOYMENT_PATH.read_text(encoding="utf-8"))
    performance = json.loads(PERFORMANCE_PATH.read_text(encoding="utf-8"))

    assert framework_lock["schema_version"] == 17
    assert "per-linear-layer-jit-v1" in framework_lock["runtime"][
        "incremental_verify_scalar_state_seed"
    ]
    assert "once-per-verify-v1" in framework_lock["runtime"][
        "incremental_verify_cache_indices"
    ]
    assert "ping-pong" in framework_lock["runtime"][
        "incremental_decode_device_carrier"
    ]
    runtime = framework_lock["runtime"]
    assert "input device mirrors" in runtime["memory"]
    assert "last K+1 rows" in runtime["memory"]
    assert "actual/full-equivalent H2D and D2H bytes" in (
        runtime["execution_io_evidence"]
    )
    assert "model ID" in runtime["incremental_profile_attribution"]
    assert "formal unprofiled 3+10" in runtime[
        "incremental_profile_attribution"
    ]

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
    assert "last-token-d2d" in runner[
        "incremental_five_om_io"
    ]
    assert "D2D" in runner["incremental_five_om_io"]
    assert "same-binary" in runner["required_decode_carrier_policy"]
    assert any(
        "full/base/count/proposal" in item
        for item in runner["required_io_counters"]
    )
    assert "maximum_target_elements_per_call" in runner["required_io_counters"]


def test_decode_device_carrier_contract_closes_frozen_fake_acl_work() -> None:
    contract = _contract()
    hot_loop = contract["hot_loop"]
    evidence = hot_loop["fake_acl_70_token_paired_3_plus_10"]

    assert "D2D" in hot_loop["last_committed_token_to_decode"]
    assert "8-byte" in hot_loop["decode_fallback"]
    assert set(hot_loop["decode_carrier_policies"]) == {
        "one-token-h2d",
        "last-token-d2d",
    }
    selection_gate = hot_loop["decode_carrier_selection_gate"]
    assert "same runner binary" in selection_gate
    assert "3 warmups plus 10 measurements" in selection_gate
    assert "zero token/EOS mismatch" in selection_gate
    assert "measurement noise" in selection_gate
    assert "64-byte" in hot_loop["rejected_unaligned_multi_row_binding"]
    assert "exactly one H2D per chunk" in hot_loop[
        "prefill_control_prefix_policy"
    ]
    assert evidence["target_decode1_executions"] == (
        evidence["decode_id_device_carrier_hits"]
        + evidence["decode_id_upload_operations"]
    )
    assert evidence[
        "decode_id_h2d_operations_eliminated_vs_packed_prefill_baseline"
    ] == (
        evidence["decode_id_device_carrier_hits"]
    )
    assert evidence["decode_id_device_carrier_hits"] == 78
    assert evidence["decode_id_multi_token_carrier_hits"] == 13
    assert evidence["decode_id_device_compaction_operations"] == 13
    assert evidence["decode_id_device_compaction_bytes"] == 104
    assert evidence["total_h2d_operations_packed_prefill_baseline"] == 130
    assert evidence["total_h2d_operations_one_token_carrier"] == 65
    assert evidence["total_h2d_operations_current"] == 52
    assert evidence["total_h2d_bytes_packed_prefill_baseline"] == 47216
    assert evidence["total_h2d_bytes_one_token_carrier"] == 46696
    assert evidence["total_h2d_bytes_last_token_before_prefix_liveness"] == 46592
    assert evidence["total_h2d_bytes_one_token_current"] == 31400
    assert evidence["total_h2d_bytes_current"] == 31296
    assert evidence["prefill_control_upload_operations"] == 52
    assert evidence["prefill_control_full_upload_operations"] == 1
    assert evidence["prefill_control_base_upload_operations"] == 38
    assert evidence["prefill_control_count_upload_operations"] == 12
    assert evidence["prefill_control_proposal_upload_operations"] == 1
    assert evidence["prefill_control_full_bytes"] == 896
    assert evidence["prefill_control_base_bytes"] == 578
    assert evidence["prefill_control_count_bytes"] == 644
    assert evidence["prefill_control_proposal_bytes"] == 708
    assert evidence[
        "prefill_control_h2d_bytes_elided_vs_full_uploads"
    ] == 15296
    assert evidence["copy_api_operations_one_token_carrier"] == 65
    assert evidence["copy_api_operations_current_h2d_plus_d2d"] == 65
    assert evidence["compact_ping_pong_device_bytes"] == 1024
    assert evidence["additional_compact_device_bytes_vs_previous_runner"] == 512
    assert evidence["device_to_host_operations_unchanged"] == 117
    assert evidence["device_to_host_bytes_unchanged"] == 32604


def test_two_transaction_window_contract_closes_matched_fake_acl_work() -> None:
    hot_loop = _contract()["hot_loop"]
    evidence = hot_loop[
        "fake_acl_adaptive_k_sync_window_70_token_32_output_paired_3_plus_10"
    ]

    assert evidence["first_and_second_proposal_counts"] == [15, 14]
    assert evidence["token_id_mismatches"] == 0
    assert evidence["eos_mismatches"] == 0
    for field in (
        "model_executions",
        "host_to_device_operations",
        "host_to_device_bytes",
        "proposal_count_upload_operations",
    ):
        assert evidence[f"{field}_both"] > 0
    assert (
        evidence["window_1_device_to_host_operations"]
        - evidence["window_2_device_to_host_operations"]
        == evidence["window_2_speculative_d2h_operations_elided"]
    )
    assert (
        evidence["window_2_device_to_host_bytes"]
        - evidence["window_1_device_to_host_bytes"]
        == evidence["window_2_speculative_d2h_padding_bytes"]
    )
    assert evidence["window_2_speculative_d2h_padding_bytes"] == (
        evidence["window_2_speculative_d2h_operations_elided"]
        * (
            evidence["compact_slot_bytes_both"]
            - evidence["compact_verify_result_bytes_both"]
        )
    )
    assert evidence["window_1_speculative_sync_windows"] == (
        evidence["window_2_speculative_sync_windows"]
        + evidence["window_2_speculative_synchronizations_elided"]
    )
    assert (
        evidence["window_1_stream_synchronizations"]
        - evidence["window_2_stream_synchronizations"]
        == evidence["stream_synchronizations_elided"]
        == evidence["window_2_speculative_synchronizations_elided"]
        == evidence["window_2_speculative_d2h_operations_elided"]
    )
    assert evidence["per_dflash_measurement_transactions_both"] == 2
    assert evidence["per_dflash_measurement_windows_before"] == 2
    assert evidence["per_dflash_measurement_windows_after"] == 1
    assert evidence["proposal_count_staging_pinned_host_bytes_both"] == 8


def test_two_transaction_window_short_case_records_coalesced_d2h() -> None:
    evidence = _contract()["hot_loop"][
        "fake_acl_two_transaction_sync_window_70_token_paired_3_plus_10"
    ]

    assert evidence["stable_generated_token_ids_both"] == list(range(11, 21))
    assert evidence["token_id_mismatches"] == 0
    assert evidence["eos_mismatches"] == 0
    assert (
        evidence["window_1_device_to_host_operations"]
        - evidence["window_2_device_to_host_operations"]
        == evidence["window_2_speculative_d2h_operations_elided"]
        == 13
    )
    assert (
        evidence["window_2_device_to_host_bytes"]
        - evidence["window_1_device_to_host_bytes"]
        == evidence["window_2_speculative_d2h_padding_bytes"]
        == 780
    )
    assert evidence["window_2_speculative_d2h_padding_bytes"] == (
        evidence["window_2_speculative_d2h_operations_elided"]
        * (
            evidence["compact_slot_bytes_both"]
            - evidence["compact_verify_result_bytes_both"]
        )
    )
