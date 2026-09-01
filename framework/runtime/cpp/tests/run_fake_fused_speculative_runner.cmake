set(required_values
  RUNNER PREFILL PREFILL_SHA PREFILL_HEAD PREFILL_HEAD_SHA
  DECODE DECODE_SHA FUSED FUSED_SHA OUTPUT
)
foreach(required ${required_values})
  if(NOT DEFINED ${required})
    message(FATAL_ERROR "${required} is required")
  endif()
endforeach()

string(REPEAT "1," 69 PROMPT_PREFIX)
set(PROMPT_IDS "${PROMPT_PREFIX}10")
file(REMOVE "${OUTPUT}" "${OUTPUT}.tmp")

execute_process(
  COMMAND "${RUNNER}"
    --target-prefill "${PREFILL}"
    --target-prefill-sha256 "${PREFILL_SHA}"
    --target-prefill-head "${PREFILL_HEAD}"
    --target-prefill-head-sha256 "${PREFILL_HEAD_SHA}"
    --target-decode1 "${DECODE}"
    --target-decode1-sha256 "${DECODE_SHA}"
    --fused-speculative-step "${FUSED}"
    --fused-speculative-step-sha256 "${FUSED_SHA}"
    --output "${OUTPUT}"
    --prompt-token-ids "${PROMPT_IDS}"
    --eos-token-ids 999
    --pad-token-id 0
    --max-new-tokens 10
    --max-draft-tokens 3
    --dflash-sync-window 2
    --warmup 1
    --repetitions 1
    --measurement-protocol profile
    --state-reset-policy immutable-zero
    --decode-carrier-policy last-token-d2d
    --draft-feature-policy committed-prefix
  RESULT_VARIABLE result
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr
)
if(NOT result EQUAL 0)
  message(FATAL_ERROR
    "fake fused speculative runner failed: ${result}\n${stdout}\n${stderr}"
  )
endif()
if(NOT stderr MATCHES "stage=validate-four-fused-om-start" OR
   NOT stderr MATCHES "stage=model-load-done role=fused-speculative-step" OR
   NOT stderr MATCHES "stage=model-dataset-plan-build-start role=incremental-runtime" OR
   NOT stderr MATCHES "stage=model-dataset-plan-build-done role=incremental-runtime" OR
   NOT stderr MATCHES "stage=write-report-done status=PASS")
  message(FATAL_ERROR "fake fused runner omitted live progress:\n${stderr}")
endif()

file(READ "${OUTPUT}" report)
string(JSON schema_version GET "${report}" schema_version)
string(JSON status GET "${report}" status)
string(JSON topology GET "${report}" abi physical_topology)
string(JSON model_count LENGTH "${report}" models)
string(JSON parity_mismatch GET "${report}" ordinary_parity token_id_mismatches)
string(JSON eos_mismatch GET "${report}" ordinary_parity eos_mismatches)
string(JSON trace_enabled GET "${report}" protocol profile_model_execution_trace_enabled)
string(JSON sync_window GET "${report}" protocol dflash_sync_window)
string(JSON reset_policy GET "${report}" protocol state_reset_policy)
string(JSON feature_policy GET "${report}" protocol draft_feature_policy)
string(JSON model_executions GET "${report}" execution_io_counters model_executions)
string(JSON prefill_executions GET "${report}" execution_io_counters target_prefill_executions)
string(JSON head_executions GET "${report}" execution_io_counters target_prefill_head_executions)
string(JSON decode_executions GET "${report}" execution_io_counters target_decode1_executions)
string(JSON draft_executions GET "${report}" execution_io_counters draft_propose_executions)
string(JSON verify_executions GET "${report}" execution_io_counters target_verify_commit_executions)
string(JSON fused_executions GET "${report}" execution_io_counters fused_speculative_step_executions)
string(JSON launches_elided GET "${report}" execution_io_counters draft_to_verify_model_launches_elided)
string(JSON prefill_draft_executions GET "${report}" execution_io_counters prefill_draft_propose_executions)
string(JSON verify_feature_rows GET "${report}" execution_io_counters draft_verify_feature_input_rows)
string(JSON verify_full_rows GET "${report}" execution_io_counters draft_verify_full_width_equivalent_rows)
string(JSON verify_elided_rows GET "${report}" execution_io_counters draft_verify_feature_rows_elided)
string(JSON pending_routes GET "${report}" execution_io_counters draft_verify_pending_upper_bound_executions)
string(JSON synchronizations GET "${report}" execution_io_counters stream_synchronizations)
string(JSON speculative_windows GET "${report}" execution_io_counters speculative_sync_windows)
string(JSON speculative_syncs_elided GET "${report}" execution_io_counters speculative_synchronizations_elided)
string(JSON d2h_operations GET "${report}" execution_io_counters device_to_host_operations)
string(JSON speculative_d2h_elided GET "${report}" execution_io_counters speculative_d2h_operations_elided)
string(JSON prefill_completions GET "${report}" execution_io_counters prefill_completion_synchronizations)
string(JSON state_memsets GET "${report}" execution_io_counters state_memset_operations)
string(JSON init_memsets GET "${report}" execution_io_counters state_initialization_memset_operations)
string(JSON init_syncs GET "${report}" execution_io_counters state_initialization_stream_synchronizations)
string(JSON trace_length LENGTH "${report}" profile_model_execution_trace)

set(fused_model_id "")
foreach(model_index RANGE 0 3)
  string(JSON role GET "${report}" models ${model_index} role)
  if(role STREQUAL "fused-speculative-step")
    string(JSON fused_model_id GET "${report}" models ${model_index} model_id)
  endif()
endforeach()
if(fused_model_id STREQUAL "")
  message(FATAL_ERROR "fused model role is absent: ${report}")
endif()

set(fused_trace_count 0)
set(fused_prefill_trace_count 0)
set(fused_verify_trace_count 0)
set(fused_verify_trace_rows 0)
math(EXPR last_trace_index "${trace_length} - 1")
foreach(trace_index RANGE 0 ${last_trace_index})
  string(JSON trace_model_id GET
    "${report}" profile_model_execution_trace ${trace_index} model_id
  )
  if(trace_model_id EQUAL fused_model_id)
    string(JSON physical_rows GET
      "${report}" profile_model_execution_trace ${trace_index} physical_rows
    )
    math(EXPR fused_trace_count "${fused_trace_count} + 1")
    if(physical_rows GREATER 16)
      math(EXPR fused_prefill_trace_count "${fused_prefill_trace_count} + 1")
    else()
      math(EXPR fused_verify_trace_count "${fused_verify_trace_count} + 1")
      math(EXPR fused_verify_trace_rows
        "${fused_verify_trace_rows} + ${physical_rows}"
      )
    endif()
  endif()
endforeach()

math(EXPR physical_role_total
  "${prefill_executions} + ${head_executions} + ${decode_executions} + ${fused_executions}"
)
math(EXPR logical_role_total
  "${prefill_executions} + ${head_executions} + ${decode_executions} + ${draft_executions} + ${verify_executions}"
)
math(EXPR physical_plus_elided "${model_executions} + ${launches_elided}")
math(EXPR verify_trace_expected "${fused_executions} - ${prefill_draft_executions}")
math(EXPR verify_row_closure "${verify_feature_rows} + ${verify_elided_rows}")
math(EXPR expected_synchronizations
  "${prefill_completions} + ${decode_executions} + ${speculative_windows}"
)
math(EXPR closed_speculative_transactions
  "${speculative_windows} + ${speculative_syncs_elided}"
)
math(EXPR closed_d2h
  "${d2h_operations} + ${speculative_d2h_elided}"
)
math(EXPR logical_transactions
  "${prefill_completions} + ${decode_executions} + ${verify_executions}"
)

if(NOT schema_version EQUAL 12 OR
   NOT status STREQUAL "PASS" OR
   NOT topology STREQUAL "split-prefill-head-four-resident-fused-speculative-step-v1" OR
   NOT model_count EQUAL 4 OR
   NOT parity_mismatch EQUAL 0 OR NOT eos_mismatch EQUAL 0 OR
   NOT trace_enabled OR NOT trace_length EQUAL model_executions OR
   NOT sync_window EQUAL 2 OR
   NOT reset_policy STREQUAL "immutable-zero" OR
   NOT feature_policy STREQUAL "committed-prefix" OR
   NOT model_executions EQUAL physical_role_total OR
   NOT physical_plus_elided EQUAL logical_role_total OR
   NOT draft_executions EQUAL verify_executions OR
   NOT fused_executions EQUAL draft_executions OR
   NOT launches_elided EQUAL fused_executions OR
   NOT fused_trace_count EQUAL fused_executions OR
   NOT fused_prefill_trace_count EQUAL prefill_draft_executions OR
   NOT fused_verify_trace_count EQUAL verify_trace_expected OR
   NOT fused_verify_trace_rows EQUAL verify_feature_rows OR
   NOT verify_row_closure EQUAL verify_full_rows OR
   NOT pending_routes EQUAL fused_verify_trace_count OR
   NOT closed_speculative_transactions EQUAL verify_executions OR
   NOT synchronizations EQUAL expected_synchronizations OR
   NOT closed_d2h EQUAL logical_transactions OR
   NOT state_memsets EQUAL 0 OR NOT init_memsets EQUAL 2 OR
   NOT init_syncs EQUAL 1)
  message(FATAL_ERROR "fake fused speculative report gates failed: ${report}")
endif()
