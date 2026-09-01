file(REMOVE "${OUTPUT}" "${OUTPUT}.tmp")
execute_process(
  COMMAND "${INSPECTOR}"
    --model target-prefill=${MODEL}
    --model target-verify=${MODEL}
    --state-bytes 128
    --io-runtime-margin-bytes 64
    --device-budget-bytes 1024
    --output "${OUTPUT}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr
)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "fake OM inspector failed: ${stdout}\n${stderr}")
endif()
if(NOT EXISTS "${OUTPUT}")
  message(FATAL_ERROR "fake OM inspector did not write its report")
endif()
file(READ "${OUTPUT}" report)
string(JSON status GET "${report}" status)
string(JSON weight_sum GET "${report}" budget weight_bytes_sum)
string(JSON shared_workspace GET "${report}" budget serial_shared_workspace_bytes)
string(JSON state_bytes GET "${report}" budget state_bytes)
string(JSON margin_bytes GET "${report}" budget io_and_runtime_margin_bytes)
string(JSON minimum_resident GET "${report}" budget minimum_resident_bytes_excluding_unlisted_overhead)
string(JSON fits GET "${report}" budget fits_declared_budget)
if(NOT status STREQUAL "QUERIED" OR NOT weight_sum EQUAL 512 OR
   NOT shared_workspace EQUAL 64 OR NOT state_bytes EQUAL 128 OR
   NOT margin_bytes EQUAL 64 OR NOT minimum_resident EQUAL 768 OR
   NOT fits)
  message(FATAL_ERROR "fake OM inspector report gates failed: ${report}")
endif()
