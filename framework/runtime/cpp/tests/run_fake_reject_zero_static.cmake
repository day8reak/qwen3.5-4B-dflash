execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    QWEN35_DFLASH_FAKE_ZERO_STATIC_INPUT=1
    "${TEST_RUNNER}"
    "${PREFILL}"
    "${PREFILL_HEAD}"
    "${DECODE}"
    "${DRAFT}"
    "${VERIFY}"
    "${DYNAMIC_VERIFY}"
    "${FUSED}"
  RESULT_VARIABLE status
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr
)

if(status EQUAL 0)
  message(FATAL_ERROR "incremental runner accepted a zero-sized static input")
endif()

set(output "${stdout}${stderr}")
string(
  FIND
  "${output}"
  "target-prefill: OM input[0] name='input_ids' dtype=9 bytes=0 shape=[1,64] has an invalid zero byte size"
  diagnostic_offset
)
if(diagnostic_offset EQUAL -1)
  message(FATAL_ERROR "unexpected zero-static-input diagnostic: ${output}")
endif()
