set(required_values
  RUNNER PREFILL PREFILL_SHA PREFILL_HEAD PREFILL_HEAD_SHA DRAFT DRAFT_SHA
  VERIFY VERIFY_SHA OUTPUT
)
foreach(required ${required_values})
  if(NOT DEFINED ${required})
    message(FATAL_ERROR "${required} is required")
  endif()
endforeach()
if(UNIFIED_TARGET_STEP)
  set(EXPECTED_MODEL_COUNT 4)
  set(TOPOLOGY_COUNT "four")
  set(DECODE_ARGS)
else()
  foreach(required DECODE DECODE_SHA)
    if(NOT DEFINED ${required})
      message(FATAL_ERROR "${required} is required")
    endif()
  endforeach()
  set(EXPECTED_MODEL_COUNT 5)
  set(TOPOLOGY_COUNT "five")
  set(DECODE_ARGS
    --target-decode1 "${DECODE}"
    --target-decode1-sha256 "${DECODE_SHA}"
  )
endif()
if(NOT DEFINED RESET_POLICY)
  set(RESET_POLICY "async-memset")
endif()
if(NOT DEFINED DECODE_CARRIER_POLICY)
  set(DECODE_CARRIER_POLICY "last-token-d2d")
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
    --target-prefill-head "${PREFILL_HEAD}"
    --target-prefill-head-sha256 "${PREFILL_HEAD_SHA}"
    ${DECODE_ARGS}
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
    --decode-carrier-policy "${DECODE_CARRIER_POLICY}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr
)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "fake incremental runner failed: ${result}\n${stdout}\n${stderr}")
endif()
if(NOT stderr MATCHES "stage=validate-${TOPOLOGY_COUNT}-om-start" OR
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
string(JSON physical_topology GET "${report}" abi physical_topology)
string(JSON resident_model_load GET "${report}" startup_ms acl_and_resident_model_load)
string(JSON model_executions GET "${report}" execution_io_counters model_executions)
string(JSON synchronizations GET "${report}" execution_io_counters stream_synchronizations)
string(JSON resets GET "${report}" execution_io_counters state_resets)
string(JSON report_reset_policy GET "${report}" protocol state_reset_policy)
string(JSON report_decode_carrier_policy GET "${report}" protocol decode_carrier_policy)
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
string(JSON prefill_head_executions GET "${report}" execution_io_counters target_prefill_head_executions)
string(JSON prefill_head_elided GET "${report}" execution_io_counters target_prefill_head_executions_elided)
string(JSON prefill_completions GET "${report}" execution_io_counters prefill_completion_synchronizations)
string(JSON deferred_prefill GET "${report}" execution_io_counters deferred_prefill_chunks)
string(JSON prefill_syncs_elided GET "${report}" execution_io_counters prefill_synchronizations_elided)
string(JSON prefill_d2h_elided GET "${report}" execution_io_counters prefill_compact_downloads_elided)
string(JSON prefill_draft_executions GET "${report}" execution_io_counters prefill_draft_propose_executions)
string(JSON prefill_draft_elided GET "${report}" execution_io_counters prefill_draft_propose_executions_elided)
string(JSON prefill_feature_rows GET "${report}" execution_io_counters prefill_feature_rows_batched)
string(JSON prefill_staging_slots GET "${report}" execution_io_counters prefill_staging_slots)
string(JSON prefill_control_slot_bytes GET "${report}" execution_io_counters prefill_control_bytes_per_slot)
string(JSON prefill_base_control_slot_bytes GET "${report}" execution_io_counters prefill_base_control_bytes_per_slot)
string(JSON prefill_count_control_slot_bytes GET "${report}" execution_io_counters prefill_count_control_bytes_per_slot)
string(JSON prefill_proposal_control_slot_bytes GET "${report}" execution_io_counters prefill_proposal_control_bytes_per_slot)
string(JSON prefill_persistent_control_tail_bytes GET "${report}" execution_io_counters prefill_persistent_control_tail_bytes_per_slot)
string(JSON prefill_staging_bytes GET "${report}" execution_io_counters prefill_staging_pinned_host_bytes)
string(JSON prefill_feature_slab_bytes GET "${report}" execution_io_counters prefill_feature_slab_bytes)
string(JSON prefill_feature_arena_bytes GET "${report}" execution_io_counters prefill_feature_arena_bytes)
string(JSON draft_dynamic_gears GET "${report}" execution_io_counters draft_dynamic_gear_count)
string(JSON target_step_dynamic_gears GET "${report}" execution_io_counters target_step_dynamic_gear_count)
string(JSON target_step_input_rows GET "${report}" execution_io_counters target_step_input_rows)
string(JSON target_step_elided_rows GET "${report}" execution_io_counters target_step_padded_rows_elided)
string(JSON prefill_control_uploads GET "${report}" execution_io_counters prefill_control_upload_operations)
string(JSON prefill_control_upload_bytes GET "${report}" execution_io_counters prefill_control_upload_bytes)
string(JSON prefill_control_full_uploads GET "${report}" execution_io_counters prefill_control_full_upload_operations)
string(JSON prefill_control_base_uploads GET "${report}" execution_io_counters prefill_control_base_upload_operations)
string(JSON prefill_control_count_uploads GET "${report}" execution_io_counters prefill_control_count_upload_operations)
string(JSON prefill_control_proposal_uploads GET "${report}" execution_io_counters prefill_control_proposal_upload_operations)
string(JSON prefill_control_h2d_bytes_elided GET "${report}" execution_io_counters prefill_control_h2d_bytes_elided)
string(JSON prefill_h2d_elided GET "${report}" execution_io_counters prefill_h2d_operations_elided)
string(JSON decode_uploads GET "${report}" execution_io_counters decode_id_upload_operations)
string(JSON decode_upload_bytes GET "${report}" execution_io_counters decode_id_upload_bytes)
string(JSON decode_carrier_hits GET "${report}" execution_io_counters decode_id_device_carrier_hits)
string(JSON decode_multi_token_carrier_hits GET "${report}" execution_io_counters decode_id_multi_token_carrier_hits)
string(JSON decode_h2d_elided GET "${report}" execution_io_counters decode_id_h2d_operations_elided)
string(JSON decode_device_compactions GET "${report}" execution_io_counters decode_id_device_compaction_operations)
string(JSON decode_device_compaction_bytes GET "${report}" execution_io_counters decode_id_device_compaction_bytes)
string(JSON compact_ping_pong_bytes GET "${report}" execution_io_counters compact_ping_pong_device_bytes)
string(JSON proposal_uploads GET "${report}" execution_io_counters proposal_count_upload_operations)
string(JSON proposal_upload_bytes GET "${report}" execution_io_counters proposal_count_upload_bytes)
string(JSON h2d_operations GET "${report}" execution_io_counters host_to_device_operations)
string(JSON h2d_bytes GET "${report}" execution_io_counters host_to_device_bytes)
string(JSON decode_executions GET "${report}" execution_io_counters target_decode1_executions)
string(JSON draft_executions GET "${report}" execution_io_counters draft_propose_executions)
string(JSON verify_executions GET "${report}" execution_io_counters target_verify_commit_executions)
math(EXPR transactions "${prefill_completions} + ${decode_executions} + ${verify_executions}")
math(EXPR role_total
  "${prefill_executions} + ${prefill_head_executions} + ${decode_executions} + ${draft_executions} + ${verify_executions}"
)
math(EXPR target_step_transactions "${decode_executions} + ${verify_executions}")
math(EXPR target_step_fixed_rows "16 * ${target_step_transactions}")
math(EXPR target_step_closed_rows "${target_step_input_rows} + ${target_step_elided_rows}")
math(EXPR closed_state_bytes "${working_state_bytes} + ${zero_state_bytes}")
math(EXPR expected_prefill_executions "2 * ${EXPECTED_RESETS}")
math(EXPR expected_dflash_requests "${EXPECTED_RESETS} / 2")
math(EXPR expected_prefill_feature_rows "${expected_dflash_requests} * 128")
math(EXPR expected_prefill_control_full_uploads "1")
if(UNIFIED_TARGET_STEP)
  set(expected_prefill_control_proposal_uploads ${expected_dflash_requests})
  set(expected_prefill_control_count_uploads 0)
else()
  set(expected_prefill_control_proposal_uploads 1)
  math(EXPR expected_prefill_control_count_uploads "${expected_dflash_requests} - 1")
endif()
math(EXPR expected_prefill_control_base_uploads
  "${prefill_executions} - ${expected_prefill_control_full_uploads} - ${expected_prefill_control_proposal_uploads} - ${expected_prefill_control_count_uploads}"
)
math(EXPR expected_prefill_control_bytes
  "${expected_prefill_control_full_uploads} * ${prefill_control_slot_bytes} + ${expected_prefill_control_base_uploads} * ${prefill_base_control_slot_bytes} + ${expected_prefill_control_count_uploads} * ${prefill_count_control_slot_bytes} + ${expected_prefill_control_proposal_uploads} * ${prefill_proposal_control_slot_bytes}"
)
math(EXPR expected_prefill_control_h2d_bytes_elided
  "${prefill_executions} * ${prefill_control_slot_bytes} - ${expected_prefill_control_bytes}"
)
math(EXPR expected_decode_upload_bytes "${decode_uploads} * 8")
math(EXPR expected_decode_device_compaction_bytes "${decode_device_compactions} * 8")
math(EXPR closed_decode_routes "${decode_uploads} + ${decode_carrier_hits}")
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
if(UNIFIED_TARGET_STEP)
  set(EXPECTED_TOPOLOGY "split-prefill-head-four-resident-unified-target-step-v1")
  string(JSON topology_model_load GET "${report}" startup_ms acl_and_four_model_load)
  if(NOT target_step_dynamic_gears EQUAL 16 OR
     NOT target_step_closed_rows EQUAL target_step_fixed_rows OR
     NOT target_step_elided_rows GREATER 0)
    message(FATAL_ERROR "fake unified Target-step row gates failed: ${report}")
  endif()
else()
  set(EXPECTED_TOPOLOGY "split-prefill-head-five-resident-v1")
  string(JSON topology_model_load GET "${report}" startup_ms acl_and_five_model_load)
  if(NOT target_step_dynamic_gears EQUAL 0)
    message(FATAL_ERROR "fake baseline unexpectedly reported Target-step gears: ${report}")
  endif()
endif()
if(NOT physical_topology STREQUAL EXPECTED_TOPOLOGY OR
   NOT resident_model_load EQUAL topology_model_load)
  message(FATAL_ERROR "fake topology/startup identity differs: ${report}")
endif()
if(DECODE_CARRIER_POLICY STREQUAL "last-token-d2d")
  if(NOT report_decode_carrier_policy STREQUAL DECODE_CARRIER_POLICY OR
     NOT decode_uploads EQUAL 0 OR
     NOT decode_carrier_hits EQUAL decode_executions OR
     NOT decode_multi_token_carrier_hits GREATER 0 OR
     NOT decode_multi_token_carrier_hits LESS decode_carrier_hits OR
     NOT decode_device_compactions EQUAL decode_multi_token_carrier_hits)
    message(FATAL_ERROR "fake last-token D2D counters failed: ${report}")
  endif()
elseif(DECODE_CARRIER_POLICY STREQUAL "one-token-h2d")
  if(NOT report_decode_carrier_policy STREQUAL DECODE_CARRIER_POLICY OR
     NOT decode_uploads GREATER 0 OR
     NOT decode_carrier_hits GREATER 0 OR
     NOT decode_multi_token_carrier_hits EQUAL 0 OR
     NOT decode_device_compactions EQUAL 0 OR
     NOT decode_device_compaction_bytes EQUAL 0)
    message(FATAL_ERROR "fake one-token H2D counters failed: ${report}")
  endif()
else()
  message(FATAL_ERROR "unknown DECODE_CARRIER_POLICY=${DECODE_CARRIER_POLICY}")
endif()
if(NOT status STREQUAL "PASS" OR
   NOT runner_id STREQUAL "qwen35-dflash-ascendcl-cpp-incremental-v3" OR
   NOT mismatch EQUAL 0 OR NOT eos_mismatch EQUAL 0 OR
   NOT repetitions EQUAL REPETITIONS OR NOT model_count EQUAL EXPECTED_MODEL_COUNT OR
   NOT report_protocol STREQUAL MEASUREMENT_PROTOCOL OR
   NOT model_executions EQUAL role_total OR
   NOT synchronizations EQUAL transactions OR
   NOT d2h_operations EQUAL transactions OR
   NOT resets EQUAL EXPECTED_RESETS OR
   NOT prefill_executions EQUAL expected_prefill_executions OR
   NOT prefill_head_executions EQUAL EXPECTED_RESETS OR
   NOT prefill_head_elided EQUAL deferred_prefill OR
   NOT prefill_completions EQUAL EXPECTED_RESETS OR
   NOT deferred_prefill EQUAL EXPECTED_RESETS OR
   NOT prefill_syncs_elided EQUAL deferred_prefill OR
   NOT prefill_d2h_elided EQUAL deferred_prefill OR
   NOT prefill_draft_executions EQUAL expected_dflash_requests OR
   NOT prefill_draft_elided EQUAL expected_dflash_requests OR
   NOT prefill_feature_rows EQUAL expected_prefill_feature_rows OR
   NOT draft_executions EQUAL expected_dflash_requests OR
   NOT prefill_staging_slots EQUAL 2 OR
   NOT prefill_control_slot_bytes EQUAL 896 OR
   NOT prefill_base_control_slot_bytes EQUAL 578 OR
   NOT prefill_count_control_slot_bytes EQUAL 644 OR
   NOT prefill_proposal_control_slot_bytes EQUAL 708 OR
   NOT prefill_persistent_control_tail_bytes EQUAL 188 OR
   NOT prefill_staging_bytes EQUAL 1792 OR
   NOT prefill_feature_slab_bytes EQUAL 1024 OR
   NOT prefill_feature_arena_bytes EQUAL 2112 OR
   NOT draft_dynamic_gears EQUAL 3 OR
   NOT prefill_control_uploads EQUAL prefill_executions OR
   NOT prefill_control_full_uploads EQUAL expected_prefill_control_full_uploads OR
   NOT prefill_control_base_uploads EQUAL expected_prefill_control_base_uploads OR
   NOT prefill_control_count_uploads EQUAL expected_prefill_control_count_uploads OR
   NOT prefill_control_proposal_uploads EQUAL expected_prefill_control_proposal_uploads OR
   NOT prefill_control_upload_bytes EQUAL expected_prefill_control_bytes OR
   NOT prefill_control_h2d_bytes_elided EQUAL expected_prefill_control_h2d_bytes_elided OR
   NOT prefill_h2d_elided EQUAL prefill_executions OR
   NOT closed_decode_routes EQUAL decode_executions OR
   NOT decode_carrier_hits GREATER 0 OR
   NOT decode_h2d_elided EQUAL decode_carrier_hits OR
   NOT decode_device_compaction_bytes EQUAL expected_decode_device_compaction_bytes OR
   NOT decode_upload_bytes EQUAL expected_decode_upload_bytes OR
   NOT compact_ping_pong_bytes GREATER 0 OR
   NOT proposal_upload_bytes EQUAL expected_proposal_upload_bytes OR
   NOT h2d_operations EQUAL closed_h2d_operations OR
   NOT h2d_bytes EQUAL closed_h2d_bytes OR
   NOT report_reset_policy STREQUAL RESET_POLICY OR
   NOT reset_only_barriers EQUAL 0 OR
   NOT state_bytes EQUAL closed_state_bytes OR
   NOT synchronizations LESS model_executions)
  message(FATAL_ERROR "fake incremental report gates failed: ${report}")
endif()
