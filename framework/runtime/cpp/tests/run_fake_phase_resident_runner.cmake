# Host-only lifecycle simulation, never CANN/NPU memory or latency evidence.
cmake_policy(SET CMP0054 NEW)
set(RESIDENCY phase-resident)
include("${CMAKE_CURRENT_LIST_DIR}/run_fake_static_fused_runner.cmake")
set(ENV{QWEN35_DFLASH_FAKE_STATIC_FUSED} 1)
list(POP_BACK model_args _residency_value _residency_option)

# The same four files and exact request: separate weights exceed this budget;
# a single reusable arena fits. State, carriers and workspace still count.
set(ENV{QWEN35_DFLASH_FAKE_WEIGHT_SCALE} 4096)
set(ENV{QWEN35_DFLASH_FAKE_DEVICE_LIMIT} 2097152)
foreach(policy all-resident phase-resident)
  set(output "${OUTPUT}-budget-${policy}.json")
  file(REMOVE "${output}" "${output}.tmp")
  execute_process(COMMAND "${RUNNER}" ${model_args}
    --model-residency-policy "${policy}"
    --output "${output}" --prompt-token-ids "${prompt}"
    --fused-static-feature-rows 64 --max-new-tokens 32 --max-draft-tokens 3
    --warmup 3 --repetitions 10 --measurement-protocol evidence
    RESULT_VARIABLE result OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
  if(policy STREQUAL "all-resident")
    if(result EQUAL 0 OR EXISTS "${output}" OR
       NOT stderr MATCHES "aclrtMalloc owner=.*weights requested_bytes=.*device_free_bytes=")
      message(FATAL_ERROR "all-resident OOM did not fail with allocation context: ${stderr}")
    endif()
  else()
    if(NOT result EQUAL 0)
      message(FATAL_ERROR "phase-resident did not fit the same memory budget: ${stderr}")
    endif()
    file(READ "${output}" report)
    string(JSON allocated GET "${report}" model_memory_query allocated_weight_bytes)
    string(JSON saved GET "${report}" model_memory_query weight_bytes_elided)
    string(JSON peak GET "${report}" model_residency peak_resident_models)
    string(JSON excluded GET "${report}" protocol model_load_excluded_from_latency)
    string(JSON mismatch GET "${report}" ordinary_parity token_id_mismatches)
    if(NOT allocated EQUAL 1048576 OR NOT saved EQUAL 2359296 OR
       NOT peak EQUAL 1 OR excluded OR NOT mismatch EQUAL 0)
      message(FATAL_ERROR "phase memory/parity/timing contract differs: ${report}")
    endif()
  endif()
endforeach()
unset(ENV{QWEN35_DFLASH_FAKE_WEIGHT_SCALE})
unset(ENV{QWEN35_DFLASH_FAKE_DEVICE_LIMIT})
list(APPEND model_args --model-residency-policy phase-resident)

# Trace model identity by execution role, not the expired inspection model IDs.
# Repeated hot decode calls must not trigger reloads.
foreach(fallback disabled request-target-only)
  if(fallback STREQUAL "request-target-only")
    set(ENV{QWEN35_DFLASH_FAKE_ZERO_ACCEPT} 1)
  endif()
  set(output "${OUTPUT}-trace-${fallback}.json")
  file(REMOVE "${output}" "${output}.tmp")
  execute_process(COMMAND "${RUNNER}" ${model_args}
    --output "${output}" --prompt-token-ids "${prompt}"
    --fused-static-feature-rows 64 --max-new-tokens 32 --max-draft-tokens 3
    --warmup 1 --repetitions 1 --measurement-protocol profile
    --zero-accept-fallback-policy "${fallback}"
    RESULT_VARIABLE result OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "phase trace failed: ${stderr}")
  endif()
  file(READ "${output}" report)
  string(JSON size LENGTH "${report}" profile_model_execution_trace)
  math(EXPR last "${size} - 1")
  set(previous_role "")
  set(previous_id -1)
  set(transitions 0)
  set(hot_calls 0)
  foreach(index RANGE ${last})
    string(JSON role GET "${report}" profile_model_execution_trace ${index} role)
    string(JSON id GET "${report}" profile_model_execution_trace ${index} model_id)
    if(role STREQUAL previous_role)
      if(NOT id EQUAL previous_id)
        message(FATAL_ERROR "hot model reloaded: ${role}")
      endif()
      math(EXPR hot_calls "${hot_calls} + 1")
    else()
      math(EXPR transitions "${transitions} + 1")
    endif()
    set(previous_role "${role}")
    set(previous_id "${id}")
  endforeach()
  string(JSON switches GET "${report}" model_residency model_switches)
  string(JSON loads GET "${report}" model_residency model_loads)
  string(JSON unloads GET "${report}" model_residency model_unloads)
  math(EXPR expected_loads "${transitions} + 4")
  math(EXPR expected_unloads "${expected_loads} - 1")
  if(NOT switches EQUAL transitions OR NOT loads EQUAL expected_loads OR
     NOT unloads EQUAL expected_unloads OR NOT hot_calls GREATER 0)
    message(FATAL_ERROR "model residency lifetime accounting differs: ${report}")
  endif()
  unset(ENV{QWEN35_DFLASH_FAKE_ZERO_ACCEPT})
endforeach()

# The extra weight-switch barrier must also close with coalesced prefill and
# EOS in prefill, inside the first window, or in a later window.
foreach(eos 11 15 40)
  set(output "${OUTPUT}-coalesced-eos-${eos}.json")
  file(REMOVE "${output}" "${output}.tmp")
  execute_process(COMMAND "${RUNNER}" ${model_args}
    --output "${output}" --prompt-token-ids "${prompt}" --eos-token-ids "${eos}"
    --fused-static-feature-rows 64 --max-new-tokens 32 --max-draft-tokens 3
    --warmup 3 --repetitions 10 --measurement-protocol evidence
    --prefill-completion-policy coalesce-first-verify --dflash-sync-window 8
    RESULT_VARIABLE result OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "phase coalesced EOS failed: ${stderr}")
  endif()
  file(READ "${output}" report)
  string(JSON mismatch GET "${report}" ordinary_parity token_id_mismatches)
  string(JSON stop GET "${report}" dflash stable_stop_reason)
  if(NOT mismatch EQUAL 0 OR NOT stop STREQUAL "eos")
    message(FATAL_ERROR "phase coalesced EOS parity failed")
  endif()
endforeach()

# Unload/reload failures cannot reuse live weights or publish a PASS report.
foreach(fault FAIL_RELOAD FAIL_UNLOAD RELOAD_ABI_DRIFT FAIL_EXECUTE)
  set(ENV{QWEN35_DFLASH_FAKE_${fault}} 1)
  set(output "${OUTPUT}-${fault}.json")
  file(REMOVE "${output}" "${output}.tmp")
  execute_process(COMMAND "${RUNNER}" ${model_args}
    --output "${output}" --prompt-token-ids "${prompt}"
    --fused-static-feature-rows 64 --max-new-tokens 32 --max-draft-tokens 3
    RESULT_VARIABLE result OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
  if(result EQUAL 0 OR EXISTS "${output}" OR
     NOT stderr MATCHES "(aclmdlLoadFromFileWithMem|aclmdlUnload|reloaded static OM|aclmdlExecuteAsync)")
    message(FATAL_ERROR "phase fault ${fault} did not fail closed: ${stderr}")
  endif()
  if(stderr MATCHES "fake ACL cleanup left live")
    message(FATAL_ERROR "phase fault ${fault} left pending model resources: ${stderr}")
  endif()
  unset(ENV{QWEN35_DFLASH_FAKE_${fault}})
endforeach()
