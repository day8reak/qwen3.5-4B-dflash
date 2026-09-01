from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_PYTHON = ROOT / "framework" / "python"
if str(FRAMEWORK_PYTHON) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_PYTHON))

from qwen35_dflash.ascend310p.cpp_runtime import (  # noqa: E402
    ASYNC_MEMSET_STATE_RESET_POLICY,
    COALESCE_FIRST_VERIFY_PREFILL_COMPLETION_POLICY,
    COMMITTED_PREFIX_DRAFT_FEATURE_POLICY,
    FIXED_VERIFY_WIDTH_DRAFT_FEATURE_POLICY,
    HUGE_FIRST_DEVICE_MEMORY_POLICY,
    IMMUTABLE_ZERO_STATE_RESET_POLICY,
    INCREMENTAL_CPP_RUNNER_ID,
    INCREMENTAL_STATE_POLICY,
    LAST_TOKEN_D2D_DECODE_CARRIER_POLICY,
    MAX_DFLASH_SYNC_WINDOW,
    ONE_TOKEN_H2D_DECODE_CARRIER_POLICY,
    SEPARATE_PREFILL_COMPLETION_POLICY,
    _INCREMENTAL_GRAPH_ABI,
    _UNIFIED_TARGET_STEP_GRAPH_ABI,
    _resolve_incremental_oms,
    build_cpp_runner,
    validate_cpp_runner_options,
    validate_incremental_cpp_runner_report,
)
from qwen35_dflash.ascend310p import cpp_runtime  # noqa: E402


def _mode(
    generation_mode: str,
    *,
    prefill_speculative_windows: int = 0,
) -> dict[str, object]:
    measurement = {
        "generated_token_ids": [11, 12, 13, 14, 15, 16],
        "stop_reason": "length",
        "counters": {
            "speculative_transactions": prefill_speculative_windows,
            "prefill_speculative_windows": prefill_speculative_windows,
        },
    }
    return {
        "status": "PASS",
        "generation_mode": generation_mode,
        "warmup": 3,
        "repetitions": 10,
        "stable_generated_token_ids": [11, 12, 13, 14, 15, 16],
        "stable_stop_reason": "length",
        "measurements": [copy.deepcopy(measurement) for _ in range(10)],
    }


def _hashes() -> dict[str, str]:
    return {
        role: hashlib.sha256(role.encode("utf-8")).hexdigest()
        for role in _INCREMENTAL_GRAPH_ABI
    }


def _report(
    state_reset_policy: str = ASYNC_MEMSET_STATE_RESET_POLICY,
    prompt_token_ids: list[int] | None = None,
    decode_carrier_policy: str = LAST_TOKEN_D2D_DECODE_CARRIER_POLICY,
    draft_feature_policy: str = FIXED_VERIFY_WIDTH_DRAFT_FEATURE_POLICY,
    dflash_sync_window: int = 1,
    prefill_completion_policy: str = SEPARATE_PREFILL_COMPLETION_POLICY,
) -> dict[str, object]:
    prompt = [10] if prompt_token_ids is None else list(prompt_token_ids)
    request_count = 26
    prompt_chunks = (len(prompt) + 63) // 64
    deferred_prefill = request_count * (prompt_chunks - 1)
    target_prefill_executions = request_count * prompt_chunks
    dflash_request_count = request_count // 2
    draft_propose_executions = 39
    prefill_draft_executions = dflash_request_count
    verify_draft_executions = (
        draft_propose_executions - prefill_draft_executions
    )
    committed_prefix = (
        draft_feature_policy == COMMITTED_PREFIX_DRAFT_FEATURE_POLICY
    )
    draft_verify_feature_rows = (
        4 * verify_draft_executions
        if committed_prefix
        else 16 * verify_draft_executions
    )
    draft_verify_full_rows = 16 * verify_draft_executions
    draft_verify_prefix_executions = (
        verify_draft_executions
        if committed_prefix and dflash_sync_window == 1
        else 0
    )
    draft_verify_pending_executions = (
        verify_draft_executions
        if committed_prefix and dflash_sync_window > 1
        else 0
    )
    prefill_draft_elided = dflash_request_count * (prompt_chunks - 1)
    prefill_feature_rows = dflash_request_count * prompt_chunks * 64
    prefill_control_bytes = 896
    prefill_base_control_bytes = 578
    prefill_count_control_bytes = 644
    prefill_proposal_control_bytes = 708
    prefill_persistent_control_tail_bytes = (
        prefill_control_bytes - prefill_proposal_control_bytes
    )
    prefill_control_full_uploads = 1
    prefill_control_proposal_uploads = 1
    prefill_control_count_uploads = dflash_request_count - 1
    prefill_control_base_uploads = (
        target_prefill_executions
        - prefill_control_full_uploads
        - prefill_control_count_uploads
        - prefill_control_proposal_uploads
    )
    prefill_control_upload_bytes = (
        prefill_control_full_uploads * prefill_control_bytes
        + prefill_control_base_uploads * prefill_base_control_bytes
        + prefill_control_count_uploads * prefill_count_control_bytes
        + prefill_control_proposal_uploads
        * prefill_proposal_control_bytes
    )
    prefill_control_h2d_bytes_elided = (
        target_prefill_executions * prefill_control_bytes
        - prefill_control_upload_bytes
    )
    decode_executions = 65
    prefill_verify_windows = (
        dflash_request_count
        if prefill_completion_policy
        == COALESCE_FIRST_VERIFY_PREFILL_COMPLETION_POLICY
        else 0
    )
    remaining_verify_transactions = 26 - prefill_verify_windows
    speculative_windows = (
        remaining_verify_transactions + dflash_sync_window - 1
    ) // dflash_sync_window
    speculative_syncs_elided = (
        remaining_verify_transactions - speculative_windows
    )
    complete_windows, final_window = divmod(
        remaining_verify_transactions, dflash_sync_window
    )
    speculative_staging_operations = (
        complete_windows * dflash_sync_window
        + (final_window if final_window > 2 else 0)
        if dflash_sync_window > 2
        else 0
    )
    last_token_d2d = (
        decode_carrier_policy == LAST_TOKEN_D2D_DECODE_CARRIER_POLICY
    )
    decode_upload_operations = 0 if last_token_d2d else 13
    decode_carrier_hits = decode_executions - decode_upload_operations
    decode_multi_token_carrier_hits = 13 if last_token_d2d else 0
    decode_device_compactions = decode_multi_token_carrier_hits
    proposal_upload_operations = 2
    immutable_zero = (
        state_reset_policy == IMMUTABLE_ZERO_STATE_RESET_POLICY
    )
    working_state_bytes = 2048
    reset_bytes = 1024
    zero_state_bytes = reset_bytes if immutable_zero else 0
    state_bytes = working_state_bytes + zero_state_bytes
    hashes = _hashes()
    models = [
        {
            "role": role,
            "model_id": model_id,
            "sha256": hashes[role],
            "work_bytes": 64,
            "weight_bytes": 64 if role == "target-prefill-head" else 256,
        }
        for model_id, role in enumerate(_INCREMENTAL_GRAPH_ABI, start=1)
    ]
    return {
        "schema_version": 9,
        "status": "PASS",
        "runner_id": INCREMENTAL_CPP_RUNNER_ID,
        "candidate_status": "APPROVED_IN_IMPLEMENTATION_NOT_ACTIVE",
        "cpu_fallback": False,
        "device_id": 0,
        "models": models,
        "prompt_token_ids": prompt,
        "limits": {"max_new_tokens": 6, "max_draft_tokens": 3},
        "protocol": {
            "warmup": 3,
            "repetitions": 10,
            "kind": "evidence",
            "formal_latency_evidence": True,
            "profile_model_execution_trace_enabled": False,
            "device_memory_allocation_policy": "normal-only",
            "dflash_sync_window": dflash_sync_window,
            "maximum_supported_dflash_sync_window": MAX_DFLASH_SYNC_WINDOW,
            "decode_iteration_scope": (
                "one host-visible synchronization window; a DFlash window "
                "may contain one to eight complete speculative transactions"
            ),
            "prefill_completion_policy": prefill_completion_policy,
            "prefill_completion_policy_description": (
                "intermediate prompt chunks stay queued; on eligible DFlash "
                "requests the final prefill and first verify share one compact "
                "D2H and stream synchronization; first-token host visibility "
                "is delayed until that verify completes"
                if prefill_verify_windows
                else "intermediate prompt chunks stay queued; final chunk "
                "performs the only prefill compact D2H and stream "
                "synchronization before decode"
            ),
            "prefill_control_policy": (
                "each chunk uploads one prefix ending after IDs/effective "
                "length, final-Draft total count, a changed proposal count, "
                "or a changed process-resident EOS table/count; all device "
                "subsegments start at 64-byte boundaries"
            ),
            "prefill_draft_policy": (
                "Target feature slabs stay device-resident; non-final prompt "
                "chunks execute no Draft OM; final prompt completion executes "
                "one prebound dynamic-gear Draft OM"
            ),
            "prefill_feature_arena_policy": (
                "contiguous 64-row FP16 slabs with 64-byte-aligned starts and "
                "one terminal guard; no D2D compaction"
            ),
            "prefill_target_lm_head_policy": (
                "target-prefill body contains no LM head; target-prefill-head "
                "executes exactly once after the final physical prompt chunk"
            ),
            "device_suballocation_policy": (
                "64-byte segment starts; ALIGN_UP(payload,32)+32 reserved span"
            ),
            "decode_input_policy": (
                "the last committed token from any compact Target result stays "
                "on device; row zero binds directly and later rows use an "
                "8-byte D2D copy into the aligned decode scalar; caller "
                "overrides use the pinned-host H2D fallback"
                if last_token_d2d
                else "one-token compact Target results bind row zero "
                "directly; multi-token commits and caller overrides use the "
                "pinned-host 8-byte H2D fallback"
            ),
            "target_step_zero_count_policy": (
                "not applicable; target-decode1 is a separate static OM"
            ),
            "state_reset_policy": state_reset_policy,
            "decode_carrier_policy": decode_carrier_policy,
            "draft_feature_policy": draft_feature_policy,
            "draft_feature_policy_description": (
                "after a synchronized verify, Draft binds exactly accepted+1 "
                "leading Target feature rows; each later unsynchronized "
                "transaction binds its predecessor's causal K+1 upper bound; "
                "masked suffix "
                "cache writes are scratch and overwritten before becoming "
                "visible"
                if committed_prefix
                else "verify-source Draft binds the original physical N=16; "
                "this is the rollback and matched-baseline route"
            ),
            "state_reset_only_barriers": 0,
            "state_reset_device_work_included_by_prefill_barrier": (
                not immutable_zero
            ),
            "state_zero_initialization_included_in_startup": immutable_zero,
        },
        "abi": {
            "id": (
                "qwen35-4b-dflash-ascend310p-incremental-performance-v2"
            ),
            "physical_topology": "split-prefill-head-five-resident-v1",
            "state_policy": "explicit device-resident ping-pong",
            "sequence_capacity": 128,
            "prefill_width": 64,
            "proposal_width": 15,
            "verify_width": 16,
            "eos_table_width": 4,
        },
        "model_memory_query": {
            "source": "aclmdlQuerySize",
            "sum_work_bytes": 320,
            "max_work_bytes": 64,
            "sum_weight_bytes": 1088,
            "state_device_bytes": state_bytes,
            "working_state_device_bytes": working_state_bytes,
            "immutable_zero_state_device_bytes": zero_state_bytes,
            "state_reset_bytes_per_request": reset_bytes,
            "carrier_device_bytes": 8192,
            "compact_ping_pong_device_bytes": 1024,
            "speculative_window_staging_device_bytes": 4096,
            "speculative_window_staging_pinned_host_bytes": 4096,
            "compact_slot_bytes": 512,
            "compact_ordinary_result_bytes": 257,
            "compact_verify_result_bytes": 452,
            "prefill_control_bytes_per_slot": prefill_control_bytes,
            "prefill_base_control_bytes_per_slot": (
                prefill_base_control_bytes
            ),
            "prefill_count_control_bytes_per_slot": (
                prefill_count_control_bytes
            ),
            "prefill_proposal_control_bytes_per_slot": (
                prefill_proposal_control_bytes
            ),
            "prefill_persistent_control_tail_bytes_per_slot": (
                prefill_persistent_control_tail_bytes
            ),
            "prefill_staging_pinned_host_bytes": 1792,
            "proposal_count_staging_pinned_host_bytes": 32,
            "prefill_feature_slab_bytes": 1024,
            "prefill_feature_arena_bytes": 2112,
            "draft_dynamic_gear_count": 18,
            "draft_verify_dynamic_gear_count": 16,
            "draft_prefill_dynamic_gear_count": 2,
            "target_step_zero_count_device_bytes": 0,
            "explicit_allocated_device_bytes_excluding_runtime": (
                64 + 1088 + state_bytes + 8192
            ),
            "load_policy": (
                "five aclmdlLoadFromFileWithMem sessions; one max-sized serial "
                "workspace; separate per-artifact weights; no cross-OM weight "
                "sharing assumed"
            ),
        },
        "execution_io_counters": {
            "model_executions": (
                target_prefill_executions
                + request_count
                + decode_executions
                + draft_propose_executions
                + 26
            ),
            "target_prefill_executions": target_prefill_executions,
            "target_prefill_head_executions": request_count,
            "target_prefill_head_executions_elided": deferred_prefill,
            "target_decode1_executions": decode_executions,
            "draft_propose_executions": draft_propose_executions,
            "target_verify_commit_executions": 26,
            "stream_synchronizations": (
                request_count + decode_executions + speculative_windows
            ),
            "speculative_sync_windows": speculative_windows,
            "speculative_synchronizations_elided": (
                speculative_syncs_elided
            ),
            "speculative_d2h_operations_elided": (
                speculative_syncs_elided
            ),
            "speculative_d2h_padding_bytes": (
                speculative_syncs_elided * 60
            ),
            "speculative_window_staging_operations": (
                speculative_staging_operations
            ),
            "speculative_window_staging_bytes": (
                speculative_staging_operations * 452
            ),
            "speculative_window_staging_device_bytes": 4096,
            "speculative_window_staging_pinned_host_bytes": 4096,
            "prefill_verify_coalesced_windows": prefill_verify_windows,
            "prefill_verify_synchronizations_elided": prefill_verify_windows,
            "prefill_verify_d2h_operations_elided": prefill_verify_windows,
            "prefill_verify_d2h_padding_bytes": prefill_verify_windows * 255,
            "prefill_verify_prefill_slot0_windows": prefill_verify_windows,
            "prefill_verify_prefill_slot1_windows": 0,
            "prefill_completion_synchronizations": request_count,
            "deferred_prefill_chunks": deferred_prefill,
            "prefill_synchronizations_elided": deferred_prefill,
            "prefill_compact_downloads_elided": deferred_prefill,
            "prefill_draft_propose_executions": prefill_draft_executions,
            "prefill_draft_propose_executions_elided": prefill_draft_elided,
            "prefill_feature_rows_batched": prefill_feature_rows,
            "draft_verify_feature_input_rows": draft_verify_feature_rows,
            "draft_verify_full_width_equivalent_rows": (
                draft_verify_full_rows
            ),
            "draft_verify_feature_rows_elided": (
                draft_verify_full_rows - draft_verify_feature_rows
            ),
            "draft_verify_fixed_width_executions": (
                0 if committed_prefix else verify_draft_executions
            ),
            "draft_verify_committed_prefix_executions": (
                draft_verify_prefix_executions
            ),
            "draft_verify_pending_upper_bound_executions": (
                draft_verify_pending_executions
            ),
            "prefill_control_upload_operations": target_prefill_executions,
            "prefill_control_upload_bytes": prefill_control_upload_bytes,
            "prefill_control_full_upload_operations": (
                prefill_control_full_uploads
            ),
            "prefill_control_base_upload_operations": (
                prefill_control_base_uploads
            ),
            "prefill_control_count_upload_operations": (
                prefill_control_count_uploads
            ),
            "prefill_control_proposal_upload_operations": (
                prefill_control_proposal_uploads
            ),
            "prefill_control_h2d_bytes_elided": (
                prefill_control_h2d_bytes_elided
            ),
            "prefill_h2d_operations_elided": target_prefill_executions,
            "decode_id_upload_operations": decode_upload_operations,
            "decode_id_upload_bytes": decode_upload_operations * 8,
            "decode_id_device_carrier_hits": decode_carrier_hits,
            "decode_id_multi_token_carrier_hits": (
                decode_multi_token_carrier_hits
            ),
            "decode_id_h2d_operations_elided": decode_carrier_hits,
            "decode_id_device_compaction_operations": (
                decode_device_compactions
            ),
            "decode_id_device_compaction_bytes": (
                decode_device_compactions * 8
            ),
            "proposal_count_upload_operations": proposal_upload_operations,
            "proposal_count_upload_bytes": proposal_upload_operations * 4,
            "proposal_count_staging_pinned_host_bytes": 32,
            "state_resets": 26,
            "state_memset_operations": 0 if immutable_zero else 52,
            "state_memset_bytes": 0 if immutable_zero else 26 * reset_bytes,
            "state_initialization_memset_operations": 2 if immutable_zero else 0,
            "state_initialization_memset_bytes": zero_state_bytes,
            "state_initialization_stream_synchronizations": (
                1 if immutable_zero else 0
            ),
            "host_to_device_operations": (
                target_prefill_executions
                + decode_upload_operations
                + proposal_upload_operations
            ),
            "host_to_device_bytes": (
                prefill_control_upload_bytes
                + decode_upload_operations * 8
                + proposal_upload_operations * 4
            ),
            "device_to_host_operations": (
                117 - speculative_syncs_elided - prefill_verify_windows
            ),
            "device_to_host_bytes": (
                8192 + speculative_syncs_elided * 60
            ),
            "state_device_bytes": state_bytes,
            "working_state_device_bytes": working_state_bytes,
            "immutable_zero_state_device_bytes": zero_state_bytes,
            "state_reset_bytes_per_request": reset_bytes,
            "carrier_device_bytes": 8192,
            "compact_ping_pong_device_bytes": 1024,
            "compact_slot_bytes": 512,
            "compact_ordinary_result_bytes": 257,
            "compact_verify_result_bytes": 452,
            "prefill_staging_slots": 2,
            "prefill_control_bytes_per_slot": prefill_control_bytes,
            "prefill_base_control_bytes_per_slot": (
                prefill_base_control_bytes
            ),
            "prefill_count_control_bytes_per_slot": (
                prefill_count_control_bytes
            ),
            "prefill_proposal_control_bytes_per_slot": (
                prefill_proposal_control_bytes
            ),
            "prefill_persistent_control_tail_bytes_per_slot": (
                prefill_persistent_control_tail_bytes
            ),
            "prefill_staging_pinned_host_bytes": 1792,
            "proposal_count_staging_pinned_host_bytes": 32,
            "prefill_feature_slab_bytes": 1024,
            "prefill_feature_arena_bytes": 2112,
            "draft_dynamic_gear_count": 18,
            "draft_verify_dynamic_gear_count": 16,
            "draft_prefill_dynamic_gear_count": 2,
            "target_step_zero_count_device_bytes": 0,
            "target_step_zero_count_bindings": 0,
        },
        "ordinary": _mode("ordinary-greedy"),
        "dflash": _mode(
            "dflash-strict-greedy",
            prefill_speculative_windows=(1 if prefill_verify_windows else 0),
        ),
        "profile_model_execution_trace": [],
        "ordinary_parity": {
            "status": "PASS",
            "token_id_mismatches": 0,
            "eos_mismatches": 0,
        },
    }


def _unified_report(
    state_reset_policy: str = ASYNC_MEMSET_STATE_RESET_POLICY,
    prompt_token_ids: list[int] | None = None,
    decode_carrier_policy: str = LAST_TOKEN_D2D_DECODE_CARRIER_POLICY,
) -> dict[str, object]:
    report = _report(
        state_reset_policy,
        prompt_token_ids,
        decode_carrier_policy,
    )
    report["models"] = [
        item
        for item in report["models"]
        if item["role"] != "target-decode1"
    ]
    report["protocol"]["target_step_zero_count_policy"] = (
        "T=1 datasets bind a process-resident aligned INT32 zero; positive K "
        "stays in the mutable proposal carrier"
    )
    report["abi"]["physical_topology"] = (
        "split-prefill-head-four-resident-unified-target-step-v1"
    )

    memory = report["model_memory_query"]
    memory["sum_work_bytes"] = sum(
        int(item["work_bytes"]) for item in report["models"]
    )
    memory["sum_weight_bytes"] = sum(
        int(item["weight_bytes"]) for item in report["models"]
    )
    memory["carrier_device_bytes"] = int(memory["carrier_device_bytes"]) + 64
    memory["prefill_control_bytes_per_slot"] = 960
    memory["prefill_persistent_control_tail_bytes_per_slot"] = 252
    memory["prefill_staging_pinned_host_bytes"] = 1920
    memory["target_step_dynamic_gear_count"] = 16
    memory["target_step_zero_count_device_bytes"] = 4
    memory["load_policy"] = (
        "four aclmdlLoadFromFileWithMem sessions; one max-sized serial "
        "workspace; separate per-artifact weights; no cross-OM weight "
        "sharing assumed"
    )
    memory["explicit_allocated_device_bytes_excluding_runtime"] = (
        int(memory["max_work_bytes"])
        + int(memory["sum_weight_bytes"])
        + int(memory["state_device_bytes"])
        + int(memory["carrier_device_bytes"])
    )

    execution = report["execution_io_counters"]
    prefill = int(execution["target_prefill_executions"])
    full = int(execution["prefill_control_full_upload_operations"])
    base = int(execution["prefill_control_base_upload_operations"])
    count = int(execution["prefill_control_count_upload_operations"])
    proposal = int(execution["prefill_control_proposal_upload_operations"])
    prefill_bytes = full * 960 + base * 578 + count * 644 + proposal * 708
    execution["prefill_control_upload_bytes"] = prefill_bytes
    execution["prefill_control_h2d_bytes_elided"] = prefill * 960 - prefill_bytes
    execution["carrier_device_bytes"] = memory["carrier_device_bytes"]
    execution["prefill_control_bytes_per_slot"] = 960
    execution["prefill_persistent_control_tail_bytes_per_slot"] = 252
    execution["prefill_staging_pinned_host_bytes"] = 1920
    execution["host_to_device_bytes"] = (
        prefill_bytes
        + int(execution["decode_id_upload_bytes"])
        + int(execution["proposal_count_upload_bytes"])
    )
    execution["target_step_dynamic_gear_count"] = 16
    decode = int(execution["target_decode1_executions"])
    verify = int(execution["target_verify_commit_executions"])
    target_rows = decode + 3 * verify
    execution["target_step_input_rows"] = target_rows
    execution["target_step_padded_rows_elided"] = (
        16 * (decode + verify) - target_rows
    )
    execution["target_step_zero_count_device_bytes"] = 4
    execution["target_step_zero_count_bindings"] = decode
    return report


def _validate(
    report: dict[str, object],
    state_reset_policy: str = ASYNC_MEMSET_STATE_RESET_POLICY,
    prompt_token_ids: list[int] | None = None,
    decode_carrier_policy: str = LAST_TOKEN_D2D_DECODE_CARRIER_POLICY,
    draft_feature_policy: str = FIXED_VERIFY_WIDTH_DRAFT_FEATURE_POLICY,
    dflash_sync_window: int = 1,
    prefill_completion_policy: str = SEPARATE_PREFILL_COMPLETION_POLICY,
    unified_target_step: bool = False,
) -> None:
    prompt = [10] if prompt_token_ids is None else list(prompt_token_ids)
    hashes = _hashes()
    if unified_target_step:
        hashes = {
            role: hashes[role]
            for role in _UNIFIED_TARGET_STEP_GRAPH_ABI
        }
    validate_incremental_cpp_runner_report(
        report,
        prompt_token_ids=prompt,
        om_sha256_by_role=hashes,
        device_id=0,
        max_new_tokens=6,
        max_draft_tokens=3,
        state_reset_policy=state_reset_policy,
        decode_carrier_policy=decode_carrier_policy,
        draft_feature_policy=draft_feature_policy,
        dflash_sync_window=dflash_sync_window,
        prefill_completion_policy=prefill_completion_policy,
    )


@pytest.mark.parametrize(
    "state_reset_policy",
    [
        ASYNC_MEMSET_STATE_RESET_POLICY,
        IMMUTABLE_ZERO_STATE_RESET_POLICY,
    ],
)
@pytest.mark.parametrize(
    "decode_carrier_policy",
    [
        LAST_TOKEN_D2D_DECODE_CARRIER_POLICY,
        ONE_TOKEN_H2D_DECODE_CARRIER_POLICY,
    ],
)
def test_incremental_runner_report_closes_state_and_transaction_counters(
    state_reset_policy: str,
    decode_carrier_policy: str,
) -> None:
    _validate(
        _report(
            state_reset_policy,
            decode_carrier_policy=decode_carrier_policy,
        ),
        state_reset_policy,
        decode_carrier_policy=decode_carrier_policy,
    )


def test_incremental_runner_report_closes_multi_chunk_prefill() -> None:
    prompt = [1] * 69 + [10]
    _validate(_report(prompt_token_ids=prompt), prompt_token_ids=prompt)


def test_incremental_runner_accepts_huge_first_build_identity() -> None:
    report = _report()
    report["protocol"]["device_memory_allocation_policy"] = (
        HUGE_FIRST_DEVICE_MEMORY_POLICY
    )
    _validate(report)


@pytest.mark.parametrize("value", [None, "unknown"])
def test_incremental_runner_rejects_unknown_device_memory_policy(
    value: object,
) -> None:
    report = _report()
    report["protocol"]["device_memory_allocation_policy"] = value
    with pytest.raises(RuntimeError, match="device memory policy"):
        _validate(report)


def test_build_cpp_runner_rejects_unknown_device_memory_policy(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="device_memory_policy"):
        build_cpp_runner(
            build_dir=tmp_path / "build",
            output=tmp_path / "build.json",
            device_memory_policy="unknown",
        )


def test_build_cpp_runner_keeps_policy_specific_logs_with_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build = tmp_path / "build-huge-first"
    report = tmp_path / "reports" / "build-huge-first.json"
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    monkeypatch.setattr(cpp_runtime, "preflight_cpp_runner", lambda _: None)
    captured: list[list[str]] = []

    def fake_run(
        command: list[str], **_: object
    ) -> subprocess.CompletedProcess[str]:
        captured.append(command)
        if "--build" in command:
            for name in (
                "qwen35_dflash_acl_runner",
                "qwen35_dflash_incremental_acl_runner",
            ):
                runner = build / name
                runner.write_text("fake runner", encoding="utf-8")
                runner.chmod(0o755)
        return subprocess.CompletedProcess(command, 0, stdout="PASS\n")

    monkeypatch.setattr(cpp_runtime.subprocess, "run", fake_run)
    payload = build_cpp_runner(
        build_dir=build,
        output=report,
        cmake="/usr/bin/cmake",
        device_memory_policy=HUGE_FIRST_DEVICE_MEMORY_POLICY,
    )

    assert any(
        "-DQWEN35_DFLASH_DEVICE_MEMORY_POLICY=huge-first" in command
        for command in captured
    )
    assert payload["device_memory_allocation_policy"] == "huge-first"
    for record in payload["logs"].values():
        assert record["path"].startswith(
            "build-huge-first/qwen35_dflash_build_logs/"
        )


def test_unified_incremental_runner_report_closes_resident_zero_count() -> None:
    _validate(_unified_report(), unified_target_step=True)


def test_unified_incremental_runner_rejects_missing_zero_count_binding() -> None:
    report = _unified_report()
    report["execution_io_counters"]["target_step_zero_count_bindings"] -= 1
    with pytest.raises(RuntimeError, match="zero-count bindings"):
        _validate(report, unified_target_step=True)


def test_incremental_runner_rejects_decode_carrier_policy_mismatch() -> None:
    report = _report(
        decode_carrier_policy=ONE_TOKEN_H2D_DECODE_CARRIER_POLICY
    )
    with pytest.raises(RuntimeError, match="decode carrier policy differs"):
        _validate(report)


@pytest.mark.parametrize("dflash_sync_window", [1, 2, 8])
def test_incremental_runner_accepts_committed_prefix_draft_features(
    dflash_sync_window: int,
) -> None:
    _validate(
        _report(
            draft_feature_policy=COMMITTED_PREFIX_DRAFT_FEATURE_POLICY,
            dflash_sync_window=dflash_sync_window,
        ),
        draft_feature_policy=COMMITTED_PREFIX_DRAFT_FEATURE_POLICY,
        dflash_sync_window=dflash_sync_window,
    )


def test_incremental_runner_rejects_draft_feature_policy_mismatch() -> None:
    report = _report(
        draft_feature_policy=COMMITTED_PREFIX_DRAFT_FEATURE_POLICY
    )
    with pytest.raises(RuntimeError, match="Draft feature policy differs"):
        _validate(report)


def test_incremental_runner_accepts_prefill_first_verify_coalescing() -> None:
    policy = COALESCE_FIRST_VERIFY_PREFILL_COMPLETION_POLICY
    report = _report(prefill_completion_policy=policy)
    _validate(report, prefill_completion_policy=policy)
    counters = report["execution_io_counters"]
    assert counters["prefill_verify_coalesced_windows"] == 13
    assert counters["prefill_verify_synchronizations_elided"] == 13
    assert counters["prefill_verify_d2h_operations_elided"] == 13


def test_incremental_runner_rejects_prefill_completion_policy_mismatch() -> None:
    report = _report(
        prefill_completion_policy=(
            COALESCE_FIRST_VERIFY_PREFILL_COMPLETION_POLICY
        )
    )
    with pytest.raises(RuntimeError, match="prefill completion policy"):
        _validate(report)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stream_synchronizations", 118),
        ("speculative_sync_windows", 25),
        ("speculative_synchronizations_elided", 1),
        ("speculative_d2h_operations_elided", 1),
        ("speculative_d2h_padding_bytes", 1),
        ("prefill_verify_coalesced_windows", 1),
        ("prefill_verify_synchronizations_elided", 1),
        ("prefill_verify_d2h_operations_elided", 1),
        ("prefill_verify_d2h_padding_bytes", 1),
        ("prefill_verify_prefill_slot0_windows", 1),
        ("prefill_verify_prefill_slot1_windows", 1),
        ("device_to_host_operations", 118),
        ("state_resets", 25),
        ("model_executions", 157),
        ("prefill_control_upload_operations", 25),
        ("prefill_control_upload_bytes", 1),
        ("prefill_control_full_upload_operations", 2),
        ("prefill_control_base_upload_operations", 1),
        ("prefill_control_count_upload_operations", 1),
        ("prefill_control_proposal_upload_operations", 2),
        ("prefill_control_h2d_bytes_elided", 1),
        ("prefill_base_control_bytes_per_slot", 1),
        ("prefill_count_control_bytes_per_slot", 1),
        ("prefill_proposal_control_bytes_per_slot", 1),
        ("prefill_persistent_control_tail_bytes_per_slot", 1),
        ("host_to_device_operations", 1),
        ("decode_id_device_carrier_hits", 1),
        ("decode_id_multi_token_carrier_hits", 66),
        ("decode_id_h2d_operations_elided", 1),
        ("decode_id_device_compaction_operations", 12),
        ("decode_id_device_compaction_bytes", 1),
        ("compact_ping_pong_device_bytes", 1),
        ("compact_slot_bytes", 511),
        ("compact_ordinary_result_bytes", 256),
        ("compact_verify_result_bytes", 451),
        ("proposal_count_upload_bytes", 7),
        ("proposal_count_staging_pinned_host_bytes", 1),
        ("draft_verify_feature_input_rows", 415),
        ("draft_verify_full_width_equivalent_rows", 415),
        ("draft_verify_feature_rows_elided", 1),
        ("draft_verify_fixed_width_executions", 25),
        ("draft_verify_committed_prefix_executions", 1),
        ("draft_verify_pending_upper_bound_executions", 1),
    ],
)
def test_incremental_runner_rejects_inconsistent_execution_counters(
    field: str,
    value: int,
) -> None:
    report = _report()
    report["execution_io_counters"][field] = value
    with pytest.raises(RuntimeError):
        _validate(report)


def test_incremental_runner_rejects_profile_timing_as_formal_evidence() -> None:
    report = _report()
    report["protocol"]["kind"] = "profile"
    report["protocol"]["formal_latency_evidence"] = False
    with pytest.raises(RuntimeError, match="diagnostic-only"):
        _validate(report)


def test_incremental_runner_rejects_a_full_target_prefill_head_copy() -> None:
    report = _report()
    models = {item["role"]: item for item in report["models"]}
    models["target-prefill-head"]["weight_bytes"] = models[
        "target-prefill"
    ]["weight_bytes"]
    with pytest.raises(RuntimeError, match="prefill-head weight"):
        _validate(report)


def test_incremental_runner_config_is_explicit() -> None:
    identity = validate_cpp_runner_options(
        {
            "device_model": "Ascend310P3",
            "cann": "test-cann",
            "driver": "test-driver",
            "firmware": "test-firmware",
            "runtime": "AscendCL",
            "state_policy": INCREMENTAL_STATE_POLICY,
            "state_reset_policy": IMMUTABLE_ZERO_STATE_RESET_POLICY,
            "decode_carrier_policy": ONE_TOKEN_H2D_DECODE_CARRIER_POLICY,
            "draft_feature_policy": COMMITTED_PREFIX_DRAFT_FEATURE_POLICY,
            "dflash_sync_window": 2,
            "prefill_completion_policy": (
                COALESCE_FIRST_VERIFY_PREFILL_COMPLETION_POLICY
            ),
            "pad_token_id": 0,
        },
        0,
    )
    assert identity["state_policy"] == INCREMENTAL_STATE_POLICY
    assert (
        identity["state_reset_policy"]
        == IMMUTABLE_ZERO_STATE_RESET_POLICY
    )
    assert (
        identity["decode_carrier_policy"]
        == ONE_TOKEN_H2D_DECODE_CARRIER_POLICY
    )
    assert identity["dflash_sync_window"] == 2
    assert identity["prefill_completion_policy"] == (
        COALESCE_FIRST_VERIFY_PREFILL_COMPLETION_POLICY
    )
    assert (
        identity["draft_feature_policy"]
        == COMMITTED_PREFIX_DRAFT_FEATURE_POLICY
    )


def test_incremental_runner_rejects_unknown_state_reset_policy() -> None:
    with pytest.raises(ValueError, match="state_reset_policy"):
        validate_cpp_runner_options(
            {
                "device_model": "Ascend310P3",
                "cann": "test-cann",
                "driver": "test-driver",
                "firmware": "test-firmware",
                "runtime": "AscendCL",
                "state_policy": INCREMENTAL_STATE_POLICY,
                "state_reset_policy": "unknown",
            },
            0,
        )


def test_incremental_runner_rejects_unknown_decode_carrier_policy() -> None:
    with pytest.raises(ValueError, match="decode_carrier_policy"):
        validate_cpp_runner_options(
            {
                "device_model": "Ascend310P3",
                "cann": "test-cann",
                "driver": "test-driver",
                "firmware": "test-firmware",
                "runtime": "AscendCL",
                "state_policy": INCREMENTAL_STATE_POLICY,
                "decode_carrier_policy": "unknown",
            },
            0,
        )


def test_incremental_runner_rejects_unknown_draft_feature_policy() -> None:
    with pytest.raises(ValueError, match="draft_feature_policy"):
        validate_cpp_runner_options(
            {
                "device_model": "Ascend310P3",
                "cann": "test-cann",
                "driver": "test-driver",
                "firmware": "test-firmware",
                "runtime": "AscendCL",
                "state_policy": INCREMENTAL_STATE_POLICY,
                "draft_feature_policy": "unknown",
            },
            0,
        )


def test_incremental_runner_rejects_unknown_dflash_sync_window() -> None:
    with pytest.raises(ValueError, match="dflash_sync_window"):
        validate_cpp_runner_options(
            {
                "device_model": "Ascend310P3",
                "cann": "test-cann",
                "driver": "test-driver",
                "firmware": "test-firmware",
                "runtime": "AscendCL",
                "state_policy": INCREMENTAL_STATE_POLICY,
                "dflash_sync_window": 9,
            },
            0,
        )


def test_incremental_runner_rejects_unknown_prefill_completion_policy() -> None:
    with pytest.raises(ValueError, match="prefill_completion_policy"):
        validate_cpp_runner_options(
            {
                "device_model": "Ascend310P3",
                "cann": "test-cann",
                "driver": "test-driver",
                "firmware": "test-firmware",
                "runtime": "AscendCL",
                "state_policy": INCREMENTAL_STATE_POLICY,
                "prefill_completion_policy": "unknown",
            },
            0,
        )


def test_incremental_runner_rejects_dflash_sync_window_report_mismatch() -> None:
    with pytest.raises(RuntimeError, match="sync window"):
        _validate(_report(dflash_sync_window=1), dflash_sync_window=2)


def test_incremental_runner_accepts_closed_two_transaction_window() -> None:
    _validate(_report(dflash_sync_window=2), dflash_sync_window=2)


def test_incremental_runner_accepts_closed_eight_transaction_window() -> None:
    _validate(_report(dflash_sync_window=8), dflash_sync_window=8)


def test_resolve_incremental_oms_locks_all_five_abis_and_hashes(
    tmp_path: Path,
) -> None:
    graphs = []
    for role, (inputs, outputs) in _INCREMENTAL_GRAPH_ABI.items():
        om = tmp_path / f"{role}.om"
        om.write_bytes(role.encode("utf-8"))
        graphs.append(
            {
                "name": role,
                "role": role,
                "input_names": inputs,
                "output_names": outputs,
                "dynamic": role == "draft-propose",
                "input_dim_gears": (
                    {"0": {"1": list(range(1, 17)) + [64, 128]}}
                    if role == "draft-propose"
                    else {}
                ),
                "om": {
                    "path": om.name,
                    "sha256": hashlib.sha256(om.read_bytes()).hexdigest(),
                },
            }
        )
    manifest_path = tmp_path / "deployment.json"
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_kind": "qwen35-dflash-ascend310p-om-bundle",
                "status": "PASS",
                "graphs": graphs,
            }
        ),
        encoding="utf-8",
    )
    resolved, _ = _resolve_incremental_oms(manifest_path)
    assert list(resolved) == list(_INCREMENTAL_GRAPH_ABI)
    assert all(item[0].is_file() for item in resolved.values())

    graphs[2]["output_names"] = ["wrong"]
    manifest_path.write_text(json.dumps({
        "artifact_kind": "qwen35-dflash-ascend310p-om-bundle",
        "status": "PASS",
        "graphs": graphs,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="output order"):
        _resolve_incremental_oms(manifest_path)

    graphs[2]["output_names"] = _INCREMENTAL_GRAPH_ABI[
        "target-decode1"
    ][1]
    graphs[3]["input_dim_gears"] = {
        "0": {"1": list(range(2, 17)) + [64, 128]}
    }
    manifest_path.write_text(json.dumps({
        "artifact_kind": "qwen35-dflash-ascend310p-om-bundle",
        "status": "PASS",
        "graphs": graphs,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="N=1..16"):
        _resolve_incremental_oms(manifest_path)


def test_resolve_incremental_oms_accepts_only_complete_t1_to_t16_target_step(
    tmp_path: Path,
) -> None:
    graphs = []
    for role, (inputs, outputs) in _UNIFIED_TARGET_STEP_GRAPH_ABI.items():
        om = tmp_path / f"{role}.om"
        om.write_bytes(role.encode("utf-8"))
        graph = {
            "name": role,
            "role": role,
            "input_names": inputs,
            "output_names": outputs,
            "om": {
                "path": om.name,
                "sha256": hashlib.sha256(om.read_bytes()).hexdigest(),
            },
        }
        if role == "target-verify-commit":
            graph["dynamic"] = True
            graph["input_dim_gears"] = {
                "0": {"1": list(range(1, 17))}
            }
        elif role == "draft-propose":
            graph["dynamic"] = True
            graph["input_dim_gears"] = {
                "0": {"1": list(range(1, 17)) + [64, 128]}
            }
        graphs.append(graph)
    manifest_path = tmp_path / "deployment.json"
    manifest = {
        "artifact_kind": "qwen35-dflash-ascend310p-om-bundle",
        "status": "PASS",
        "graphs": graphs,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    resolved, _ = _resolve_incremental_oms(manifest_path)
    assert list(resolved) == list(_UNIFIED_TARGET_STEP_GRAPH_ABI)

    graphs[-1]["input_dim_gears"] = {"0": {"1": list(range(2, 17))}}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="T=1..16"):
        _resolve_incremental_oms(manifest_path)


def test_run_cpp_pair_routes_all_five_hash_locked_oms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    bundle = run_root / "bundle"
    bundle.mkdir(parents=True)
    monkeypatch.setenv("AI_RUN_DIR", str(run_root))
    air_manifest = bundle / "air-manifest.json"
    air_manifest.write_text("{}", encoding="utf-8")
    graphs = []
    for role, (inputs, outputs) in _INCREMENTAL_GRAPH_ABI.items():
        om = bundle / f"{role}.om"
        om.write_bytes(role.encode("utf-8"))
        graphs.append(
            {
                "name": role,
                "role": role,
                "input_names": inputs,
                "output_names": outputs,
                "dynamic": role == "draft-propose",
                "input_dim_gears": (
                    {"0": {"1": list(range(1, 17)) + [64, 128]}}
                    if role == "draft-propose"
                    else {}
                ),
                "om": {
                    "path": om.name,
                    "sha256": hashlib.sha256(om.read_bytes()).hexdigest(),
                },
            }
        )
    deployment = bundle / "deployment-manifest.json"
    deployment.write_text(
        json.dumps(
            {
                "artifact_kind": "qwen35-dflash-ascend310p-om-bundle",
                "status": "PASS",
                "air_manifest": {
                    "path": air_manifest.name,
                    "sha256": hashlib.sha256(
                        air_manifest.read_bytes()
                    ).hexdigest(),
                },
                "compiler": {"identity": "fake"},
                "target": {"soc_version": "Ascend310P3"},
                "graphs": graphs,
            }
        ),
        encoding="utf-8",
    )
    runner = run_root / "fake-incremental-runner"
    runner.write_text("fake", encoding="utf-8")
    runner.chmod(0o755)
    monkeypatch.setattr(cpp_runtime, "preflight_cpp_runner", lambda _: runner)
    captured: dict[str, list[str]] = {}

    def execute(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        output = Path(command[command.index("--output") + 1])
        output.write_text(
            json.dumps(_report(IMMUTABLE_ZERO_STATE_RESET_POLICY)),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="fake PASS\n")

    payload = cpp_runtime.run_cpp_pair(
        deployment_manifest=deployment,
        runner=runner,
        runner_options={
            "device_model": "Ascend310P3",
            "cann": "fake-cann",
            "driver": "fake-driver",
            "firmware": "fake-firmware",
            "runtime": "fake-AscendCL",
            "state_policy": INCREMENTAL_STATE_POLICY,
            "state_reset_policy": IMMUTABLE_ZERO_STATE_RESET_POLICY,
            "decode_carrier_policy": LAST_TOKEN_D2D_DECODE_CARRIER_POLICY,
            "pad_token_id": 0,
        },
        prompt_token_ids=[10],
        eos_token_ids=[],
        device_id=0,
        max_new_tokens=6,
        max_draft_tokens=3,
        raw_output=run_root / "out" / "raw.json",
        log_output=run_root / "log" / "runner.log",
        progress=False,
        execute=execute,
    )
    command = captured["command"]
    for role in _INCREMENTAL_GRAPH_ABI:
        assert f"--{role}" in command
        assert f"--{role}-sha256" in command
    assert "--model" not in command
    assert command[command.index("--state-reset-policy") + 1] == (
        IMMUTABLE_ZERO_STATE_RESET_POLICY
    )
    assert command[command.index("--decode-carrier-policy") + 1] == (
        LAST_TOKEN_D2D_DECODE_CARRIER_POLICY
    )
    assert command[command.index("--draft-feature-policy") + 1] == (
        FIXED_VERIFY_WIDTH_DRAFT_FEATURE_POLICY
    )
    assert command[command.index("--dflash-sync-window") + 1] == "1"
    assert command[command.index("--prefill-completion-policy") + 1] == (
        SEPARATE_PREFILL_COMPLETION_POLICY
    )
    assert command[command.index("--measurement-protocol") + 1] == "evidence"
    assert payload["backend_metadata"]["state_policy"] == (
        INCREMENTAL_STATE_POLICY
    )
    assert payload["backend_metadata"]["state_implementation"] == (
        "explicit Target/Draft device state"
    )
