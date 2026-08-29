if(NOT DEFINED RUNNER OR NOT DEFINED MODEL OR NOT DEFINED MODEL_SHA OR NOT DEFINED OUTPUT)
  message(FATAL_ERROR "RUNNER, MODEL, MODEL_SHA and OUTPUT are required")
endif()

file(REMOVE "${OUTPUT}" "${OUTPUT}.tmp")
execute_process(
  COMMAND "${RUNNER}"
    --model "${MODEL}"
    --model-sha256 "${MODEL_SHA}"
    --output "${OUTPUT}"
    --prompt-token-ids 10
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

file(READ "${OUTPUT}" report)
string(JSON status GET "${report}" status)
string(JSON mismatch GET "${report}" ordinary_parity token_id_mismatches)
string(JSON eos_mismatch GET "${report}" ordinary_parity eos_mismatches)
string(JSON repetitions GET "${report}" dflash repetitions)
if(NOT status STREQUAL "PASS" OR NOT mismatch EQUAL 0 OR
   NOT eos_mismatch EQUAL 0 OR NOT repetitions EQUAL 10)
  message(FATAL_ERROR "fake ACL report gates failed: ${report}")
endif()
