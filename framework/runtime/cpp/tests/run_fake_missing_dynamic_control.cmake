execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    QWEN35_DFLASH_FAKE_MISSING_DYNAMIC_CONTROL=1
    "${TEST_RUNNER}"
    "${PREFILL}" "${PREFILL_HEAD}" "${DECODE}" "${DRAFT}"
    "${VERIFY}" "${DYNAMIC_VERIFY}" "${FUSED}"
  RESULT_VARIABLE status
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr
)

if(status EQUAL 0)
  message(FATAL_ERROR "runner silently accepted an unsupported Shape-like OM")
endif()

set(output "${stdout}${stderr}")
foreach(expected IN ITEMS
    "fused-speculative-step: dynamic execution contract mismatch"
    "aclmdlGetInputIndexByName returned 100000"
    "aclmdlSetDatasetTensorDesc"
    "input_count=48"
    "name='target_feature_tail' dimCount=3 dtype=1 bytes=0 shape=[1,-1,8]"
    "name='lifted_float_scalar' dimCount=0 dtype=11 bytes=8 shape=[]"
    "output_count=16")
  string(FIND "${output}" "${expected}" diagnostic_offset)
  if(diagnostic_offset EQUAL -1)
    message(FATAL_ERROR "missing diagnostic '${expected}': ${output}")
  endif()
endforeach()
