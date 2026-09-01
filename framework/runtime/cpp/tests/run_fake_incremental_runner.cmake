set(required_values
  RUNNER PREFILL PREFILL_SHA DECODE DECODE_SHA DRAFT DRAFT_SHA
  VERIFY VERIFY_SHA OUTPUT
)
foreach(required ${required_values})
  if(NOT DEFINED ${required})
    message(FATAL_ERROR "${required} is required")
  endif()
endforeach()
if(NOT DEFINED RESET_POLICY)
  set(RESET_POLICY "async-memset")
endif()
if(NOT DEFINED MEASUREMENT_PROTOCOL)
  set(MEASUREMENT_PROTOCOL "evidence")
endif()
if(MEASUREMENT_PROTOCOL STREQUAL "evidence")
  set(WARMUP 3)
  set(REPETITIONS 10)
  set(EXPECTED_RESETS 26)
elseif(MEASUREMENT_PROTOCOL STREQUAL "profile")
  set(WARMUP 1)
  set(REPETITIONS 1)
  set(EXPECTED_RESETS 4)
else()
  message(FATAL_ERROR "unknown MEASUREMENT_PROTOCOL=${MEASUREMENT_PROTOCOL}")
endif()

string(REPEAT "1," 69 PROMPT_PREFIX)
set(PROMPT_IDS "${PROMPT_PREFIX}10")

file(REMOVE "${OUTPUT}" "${OUTPUT}.tmp")
execute_process(
  COMMAND "${RUNNER}"
    --target-prefill "${PREFILL}"
    --target-prefill-sha256 "${PREFILL_SHA}"
    --target-decode1 "${DECODE}"
    --target-decode1-sha256 "${DECODE_SHA}"
    --draft-propose "${DRAFT}"
    --draft-propose-sha256 "${DRAFT_SHA}"
    --target-verify-commit "${VERIFY}"
    --target-verify-commit-sha256 "${VERIFY_SHA}"
    --output "${OUTPUT}"
    --prompt-token-ids "${PROMPT_IDS}"
    --eos-token-ids 999
    --pad-token-id 0
    --max-new-tokens 6
    --max-draft-tokens 3
    --warmup "${WARMUP}"
    --repetitions "${REPETITIONS}"
    --device-id 0
    --measurement-protocol "${MEASUREMENT_PROTOCOL}"
    --state-reset-policy "${RESET_POLICY}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr
)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "fake incremental runner failed: ${result}\n${stdout}\n${stderr}")
endif()
if(NOT stderr MATCHES "stage=validate-four-om-start" OR
   NOT stderr MATCHES "stage=model-load-done role=target-prefill" OR
   NOT stderr MATCHES "phase=warmup" OR
   NOT stderr MATCHES "stage=decode-done" OR
   NOT stderr MATCHES "stage=write-report-done status=PASS")
  message(FATAL_ERROR "fake incremental runner omitted live progress:\n${stderr}")
endif()

file(READ "${OUTPUT}" report)
string(JSON status GET "${report}" status)
string(JSON runner_id GET "${report}" runner_id)
string(JSON mismatch GET "${report}" ordinary_parity token_id_mismatches)
string(JSON eos_mismatch GET "${report}" ordinary_parity eos_mismatches)
string(JSON repetitions GET "${report}" dflash repetitions)
string(JSON report_protocol GET "${report}" protocol kind)
string(JSON model_count LENGTH "${report}" models)
string(JSON model_executions GET "${report}" execution_io_counters model_executions)
string(JSON synchronizations GET "${report}" execution_io_counters stream_synchronizations)
string(JSON resets GET "${report}" execution_io_counters state_resets)
string(JSON report_reset_policy GET "${report}" protocol state_reset_policy)
string(JSON reset_only_barriers GET "${report}" protocol state_reset_only_barriers)
string(JSON state_memsets GET "${report}" execution_io_counters state_memset_operations)
string(JSON state_memset_bytes GET "${report}" execution_io_counters state_memset_bytes)
string(JSON init_memsets GET "${report}" execution_io_counters state_initialization_memset_operations)
string(JSON init_memset_bytes GET "${report}" execution_io_counters state_initialization_memset_bytes)
string(JSON init_syncs GET "${report}" execution_io_counters state_initialization_stream_synchronizations)
string(JSON state_bytes GET "${report}" execution_io_counters state_device_bytes)
string(JSON working_state_bytes GET "${report}" execution_io_counters working_state_device_bytes)
string(JSON zero_state_bytes GET "${report}" execution_io_counters immutable_zero_state_device_bytes)
string(JSON reset_bytes GET "${report}" execution_io_counters state_reset_bytes_per_request)
string(JSON d2h_operations GET "${report}" execution_io_counters device_to_host_operations)
string(JSON prefill_executions GET "${report}" execution_io_counters target_prefill_executions)
string(JSON prefill_completions GET "${report}" execution_io_counters prefill_completion_synchronizations)
string(JSON deferred_prefill GET "${report}" execution_io_counters deferred_prefill_chunks)
string(JSON prefill_syncs_elided GET "${report}" execution_io_counters prefill_synchronizations_elided)
string(JSON prefill_d2h_elided GET "${report}" execution_io_counters prefill_compact_downloads_elided)
string(JSON prefill_staging_slots GET "${report}" execution_io_counters prefill_staging_slots)
string(JSON prefill_control_slot_bytes GET "${report}" execution_io_counters prefill_control_bytes_per_slot)
string(JSON prefill_staging_bytes GET "${report}" execution_io_counters prefill_staging_pinned_host_bytes)
string(JSON prefill_control_uploads GET "${report}" execution_io_counters prefill_control_upload_operations)
string(JSON prefill_control_upload_bytes GET "${report}" execution_io_counters prefill_control_upload_bytes)
string(JSON prefill_h2d_elided GET "${report}" execution_io_counters prefill_h2d_operations_elided)
string(JSON decode_uploads GET "${report}" execution_io_counters decode_id_upload_operations)
string(JSON decode_upload_bytes GET "${report}" execution_io_counters decode_id_upload_bytes)
string(JSON proposal_uploads GET "${report}" execution_io_counters proposal_count_upload_operations)
string(JSON proposal_upload_bytes GET "${report}" execution_io_counters proposal_count_upload_bytes)
string(JSON h2d_operations GET "${report}" execution_io_counters host_to_device_operations)
string(JSON h2d_bytes GET "${report}" execution_io_counters host_to_device_bytes)
string(JSON decode_executions GET "${report}" execution_io_counters target_decode1_executions)
string(JSON draft_executions GET "${report}" execution_io_counters draft_propose_executions)
string(JSON verify_executions GET "${report}" execution_io_counters target_verify_commit_executions)
math(EXPR transactions "${prefill_completions} + ${decode_executions} + ${verify_executions}")
math(EXPR role_total
  "${prefill_executions} + ${decode_executions} + ${draft_executions} + ${verify_executions}"
)
math(EXPR closed_state_bytes "${working_state_bytes} + ${zero_state_bytes}")
math(EXPR expected_prefill_executions "2 * ${EXPECTED_RESETS}")
math(EXPR expected_prefill_control_bytes "${prefill_executions} * ${prefill_control_slot_bytes}")
math(EXPR expected_decode_upload_bytes "${decode_uploads} * 8")
math(EXPR expected_proposal_upload_bytes "${proposal_uploads} * 4")
math(EXPR closed_h2d_operations "${prefill_control_uploads} + ${decode_uploads} + ${proposal_uploads}")
math(EXPR closed_h2d_bytes "${prefill_control_upload_bytes} + ${decode_upload_bytes} + ${proposal_upload_bytes}")
if(RESET_POLICY STREQUAL "async-memset")
  math(EXPR expected_state_memsets "2 * ${resets}")
  math(EXPR expected_state_memset_bytes "${reset_bytes} * ${resets}")
  if(NOT state_memsets EQUAL expected_state_memsets OR
     NOT state_memset_bytes EQUAL expected_state_memset_bytes OR
     NOT zero_state_bytes EQUAL 0 OR NOT init_memsets EQUAL 0 OR
     NOT init_memset_bytes EQUAL 0 OR NOT init_syncs EQUAL 0)
    message(FATAL_ERROR "fake async reset counters failed: ${report}")
  endif()
elseif(RESET_POLICY STREQUAL "immutable-zero")
  if(NOT state_memsets EQUAL 0 OR NOT state_memset_bytes EQUAL 0 OR
     NOT zero_state_bytes EQUAL reset_bytes OR NOT init_memsets EQUAL 2 OR
     NOT init_memset_bytes EQUAL zero_state_bytes OR NOT init_syncs EQUAL 1)
    message(FATAL_ERROR "fake immutable-zero counters failed: ${report}")
  endif()
else()
  message(FATAL_ERROR "unknown RESET_POLICY=${RESET_POLICY}")
endif()
if(NOT status STREQUAL "PASS" OR
   NOT runner_id STREQUAL "qwen35-dflash-ascendcl-cpp-incremental-v2" OR
   NOT mismatch EQUAL 0 OR NOT eos_mismatch EQUAL 0 OR
   NOT repetitions EQUAL REPETITIONS OR NOT model_count EQUAL 4 OR
   NOT report_protocol STREQUAL MEASUREMENT_PROTOCOL OR
   NOT model_executions EQUAL role_total OR
   NOT synchronizations EQUAL transactions OR
   NOT d2h_operations EQUAL transactions OR
   NOT resets EQUAL EXPECTED_RESETS OR
   NOT prefill_executions EQUAL expected_prefill_executions OR
   NOT prefill_completions EQUAL EXPECTED_RESETS OR
   NOT deferred_prefill EQUAL EXPECTED_RESETS OR
   NOT prefill_syncs_elided EQUAL deferred_prefill OR
   NOT prefill_d2h_elided EQUAL deferred_prefill OR
   NOT prefill_staging_slots EQUAL 2 OR
   NOT prefill_control_slot_bytes EQUAL 832 OR
   NOT prefill_staging_bytes EQUAL 1664 OR
   NOT prefill_control_uploads EQUAL prefill_executions OR
   NOT prefill_control_upload_bytes EQUAL expected_prefill_control_bytes OR
   NOT prefill_h2d_elided EQUAL prefill_executions OR
   NOT decode_uploads EQUAL decode_executions OR
   NOT decode_upload_bytes EQUAL expected_decode_upload_bytes OR
   NOT proposal_upload_bytes EQUAL expected_proposal_upload_bytes OR
   NOT h2d_operations EQUAL closed_h2d_operations OR
   NOT h2d_bytes EQUAL closed_h2d_bytes OR
   NOT report_reset_policy STREQUAL RESET_POLICY OR
   NOT reset_only_barriers EQUAL 0 OR
   NOT state_bytes EQUAL closed_state_bytes OR
   NOT synchronizations LESS model_executions)
  message(FATAL_ERROR "fake incremental report gates failed: ${report}")
endif()
