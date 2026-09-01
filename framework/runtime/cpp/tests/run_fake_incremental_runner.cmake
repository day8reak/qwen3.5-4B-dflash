set(required_values
  RUNNER PREFILL PREFILL_SHA DECODE DECODE_SHA DRAFT DRAFT_SHA
  VERIFY VERIFY_SHA OUTPUT
)
foreach(required ${required_values})
  if(NOT DEFINED ${required})
    message(FATAL_ERROR "${required} is required")
  endif()
endforeach()

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
    --prompt-token-ids 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,10
    --eos-token-ids 999
    --pad-token-id 0
    --max-new-tokens 6
    --max-draft-tokens 3
    --warmup 3
    --repetitions 10
    --device-id 0
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
string(JSON model_count LENGTH "${report}" models)
string(JSON model_executions GET "${report}" execution_io_counters model_executions)
string(JSON synchronizations GET "${report}" execution_io_counters stream_synchronizations)
string(JSON resets GET "${report}" execution_io_counters state_resets)
string(JSON d2h_operations GET "${report}" execution_io_counters device_to_host_operations)
string(JSON prefill_executions GET "${report}" execution_io_counters target_prefill_executions)
string(JSON decode_executions GET "${report}" execution_io_counters target_decode1_executions)
string(JSON draft_executions GET "${report}" execution_io_counters draft_propose_executions)
string(JSON verify_executions GET "${report}" execution_io_counters target_verify_commit_executions)
math(EXPR transactions "${prefill_executions} + ${decode_executions} + ${verify_executions}")
math(EXPR role_total
  "${prefill_executions} + ${decode_executions} + ${draft_executions} + ${verify_executions}"
)
if(NOT status STREQUAL "PASS" OR
   NOT runner_id STREQUAL "qwen35-dflash-ascendcl-cpp-incremental-v2" OR
   NOT mismatch EQUAL 0 OR NOT eos_mismatch EQUAL 0 OR
   NOT repetitions EQUAL 10 OR NOT model_count EQUAL 4 OR
   NOT model_executions EQUAL role_total OR
   NOT synchronizations EQUAL transactions OR
   NOT d2h_operations EQUAL transactions OR
   NOT resets EQUAL 26 OR
   NOT synchronizations LESS model_executions)
  message(FATAL_ERROR "fake incremental report gates failed: ${report}")
endif()
