execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    QWEN35_DFLASH_FAKE_ZERO_RANK_PUBLIC_INPUT=1
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
  message(FATAL_ERROR "incremental runner accepted a zero-rank public input")
endif()

set(output "${stdout}${stderr}")
string(
  FIND
  "${output}"
  "target-prefill: OM input[0] name='input_ids' has an invalid dimension count 0 (expected 1..128)"
  diagnostic_offset
)
if(diagnostic_offset EQUAL -1)
  message(FATAL_ERROR "unexpected zero-rank-input diagnostic: ${output}")
endif()
