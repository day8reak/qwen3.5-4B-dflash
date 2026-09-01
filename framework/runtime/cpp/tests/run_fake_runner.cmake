if(NOT DEFINED RUNNER OR NOT DEFINED MODEL OR NOT DEFINED MODEL_SHA OR NOT DEFINED OUTPUT)
  message(FATAL_ERROR "RUNNER, MODEL, MODEL_SHA and OUTPUT are required")
endif()

file(REMOVE "${OUTPUT}" "${OUTPUT}.tmp")
execute_process(
  COMMAND "${RUNNER}"
    --model "${MODEL}"
    --model-sha256 "${MODEL_SHA}"
    --output "${OUTPUT}"
    --prompt-token-ids 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,10
    --eos-token-ids 999
    --pad-token-id 0
    --max-new-tokens 6
    --max-draft-tokens 15
    --warmup 3
    --repetitions 10
    --device-id 0
  RESULT_VARIABLE result
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr
)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "fake ACL runner failed: ${result}\n${stdout}\n${stderr}")
endif()
if(NOT stderr MATCHES "stage=validate-om-start" OR
   NOT stderr MATCHES "phase=warmup" OR
   NOT stderr MATCHES "stage=decode-done" OR
   NOT stderr MATCHES "stage=write-report-done status=PASS")
  message(FATAL_ERROR "fake ACL runner omitted live progress:\n${stderr}")
endif()

file(READ "${OUTPUT}" report)
string(JSON status GET "${report}" status)
string(JSON mismatch GET "${report}" ordinary_parity token_id_mismatches)
string(JSON eos_mismatch GET "${report}" ordinary_parity eos_mismatches)
string(JSON repetitions GET "${report}" dflash repetitions)
string(JSON work_bytes GET "${report}" model_memory_query work_bytes)
string(JSON weight_bytes GET "${report}" model_memory_query weight_bytes)
string(JSON executions GET "${report}" execution_io_counters model_executions)
string(JSON synchronizations GET "${report}" execution_io_counters stream_synchronizations)
string(JSON h2d_bytes GET "${report}" execution_io_counters host_to_device_bytes)
string(JSON full_h2d_bytes GET "${report}" execution_io_counters full_host_to_device_bytes)
string(JSON d2h_bytes GET "${report}" execution_io_counters device_to_host_bytes)
string(JSON full_d2h_bytes GET "${report}" execution_io_counters full_device_to_host_bytes)
string(JSON maximum_target_elements GET "${report}" execution_io_counters maximum_target_elements_per_call)
if(NOT status STREQUAL "PASS" OR NOT mismatch EQUAL 0 OR
   NOT eos_mismatch EQUAL 0 OR NOT repetitions EQUAL 10 OR
   NOT work_bytes EQUAL 64 OR NOT weight_bytes EQUAL 256 OR
   NOT executions EQUAL 117 OR NOT synchronizations EQUAL executions OR
   NOT h2d_bytes LESS full_h2d_bytes OR NOT d2h_bytes LESS full_d2h_bytes OR
   NOT maximum_target_elements EQUAL 16)
  message(FATAL_ERROR "fake ACL report gates failed: ${report}")
endif()
