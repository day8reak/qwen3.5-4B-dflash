set(model_args
  --target-prefill "${PREFILL}" --target-prefill-sha256 "${PREFILL_SHA}"
  --target-prefill-head "${PREFILL_HEAD}" --target-prefill-head-sha256 "${PREFILL_HEAD_SHA}"
  --target-decode1 "${DECODE}" --target-decode1-sha256 "${DECODE_SHA}"
  --fused-speculative-step "${FUSED}" --fused-speculative-step-sha256 "${FUSED_SHA}"
)
if(RESIDENCY)
  list(APPEND model_args --model-residency-policy "${RESIDENCY}")
endif()
string(REPEAT "1," 16 prefix)
set(prompt "${prefix}10")
set(ENV{QWEN35_DFLASH_FAKE_STATIC_FUSED} 1)
foreach(reset async-memset immutable-zero)
  foreach(feature fixed-16 committed-prefix)
    foreach(window 1 2 8)
      set(output "${OUTPUT}-${reset}-${feature}-${window}.json")
      # These files belong to this test's current build directory only.
      file(REMOVE "${output}" "${output}.tmp")
      execute_process(COMMAND "${RUNNER}" ${model_args}
        --output "${output}" --prompt-token-ids "${prompt}"
        --fused-static-feature-rows 64 --max-new-tokens 32 --max-draft-tokens 3
        --warmup 3 --repetitions 10 --measurement-protocol evidence
        --state-reset-policy "${reset}" --draft-feature-policy "${feature}"
        --dflash-sync-window "${window}"
        RESULT_VARIABLE result OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
      if(NOT result EQUAL 0)
        message(FATAL_ERROR "static fused ${reset}/${feature}/${window}: ${stderr}")
      endif()
      file(READ "${output}" report)
      string(JSON status GET "${report}" status)
      string(JSON mismatch GET "${report}" ordinary_parity token_id_mismatches)
      foreach(scope model_memory_query execution_io_counters)
        string(JSON rows GET "${report}" ${scope} fused_static_feature_rows)
        string(JSON shape GET "${report}" ${scope} draft_dynamic_shape)
        foreach(field draft_om_dynamic_gear_count draft_dynamic_gear_count
                      draft_prefill_dynamic_gear_count draft_verify_dynamic_gear_count)
          string(JSON gears GET "${report}" ${scope} ${field})
          if(NOT gears EQUAL 0)
            message(FATAL_ERROR "static OM fabricated gears: ${report}")
          endif()
        endforeach()
        if(NOT rows EQUAL 64 OR shape)
          message(FATAL_ERROR "static OM used dynamic shapes: ${report}")
        endif()
      endforeach()
      string(JSON calls GET "${report}" execution_io_counters fused_speculative_step_executions)
      string(JSON physical GET "${report}" execution_io_counters fused_static_physical_feature_rows)
      string(JSON source GET "${report}" execution_io_counters fused_static_source_feature_rows)
      string(JSON padding GET "${report}" execution_io_counters fused_static_padding_rows)
      string(JSON memsets GET "${report}" execution_io_counters fused_static_padding_operations)
      math(EXPR expected "${calls} * 64")
      math(EXPR actual "${source} + ${padding}")
      if(NOT status STREQUAL "PASS" OR NOT mismatch EQUAL 0 OR
         NOT physical EQUAL expected OR NOT actual EQUAL expected OR
         NOT memsets EQUAL calls OR NOT padding GREATER 0)
        message(FATAL_ERROR "static parity/padding accounting failed: ${report}")
      endif()
    endforeach()
  endforeach()
endforeach()

# One output token needs no Draft; two must perform eager terminal K=1.
foreach(new_tokens 1 2)
  set(output "${OUTPUT}-short-${new_tokens}.json")
  file(REMOVE "${output}" "${output}.tmp")
  execute_process(COMMAND "${RUNNER}" ${model_args}
    --output "${output}" --prompt-token-ids "${prompt}"
    --fused-static-feature-rows 64 --max-new-tokens "${new_tokens}" --max-draft-tokens 3
    --warmup 3 --repetitions 10 --measurement-protocol evidence
    RESULT_VARIABLE result OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "static short request failed: ${stderr}")
  endif()
  file(READ "${output}" report)
  foreach(field fused_speculative_step_executions fused_static_physical_feature_rows
                fused_static_source_feature_rows fused_static_padding_rows
                fused_static_padding_operations)
    string(JSON count GET "${report}" execution_io_counters ${field})
    if(new_tokens EQUAL 1 AND NOT count EQUAL 0)
      message(FATAL_ERROR "one-token request unexpectedly called Draft: ${report}")
    elseif(new_tokens EQUAL 2 AND NOT count GREATER 0)
      message(FATAL_ERROR "two-token request omitted terminal Draft K=1: ${report}")
    endif()
  endforeach()
endforeach()

foreach(fault missing-opt-in wrong-shape oversized-prompt capacity-budget dynamic-om)
  set(rows 64)
  set(tokens "${prompt}")
  set(new_tokens 32)
  set(ENV{QWEN35_DFLASH_FAKE_STATIC_FUSED} 1)
  if(fault STREQUAL "missing-opt-in")
    set(rows 0)
  elseif(fault STREQUAL "wrong-shape")
    set(rows 128)
  elseif(fault STREQUAL "oversized-prompt")
    string(REPEAT "1," 64 long_prefix)
    set(tokens "${long_prefix}10")
  elseif(fault STREQUAL "capacity-budget")
    set(new_tokens 48)
  elseif(fault STREQUAL "dynamic-om")
    unset(ENV{QWEN35_DFLASH_FAKE_STATIC_FUSED})
  endif()
  set(output "${OUTPUT}-${fault}.json")
  file(REMOVE "${output}" "${output}.tmp")
  execute_process(COMMAND "${RUNNER}" ${model_args}
    --output "${output}" --prompt-token-ids "${tokens}"
    --fused-static-feature-rows "${rows}" --max-new-tokens "${new_tokens}"
    RESULT_VARIABLE result OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
  if(result EQUAL 0 OR EXISTS "${output}" OR
     NOT stderr MATCHES "(static.*(carrier|shape|budget)|loaded fused OM shape|phase-resident requires.*static)")
    message(FATAL_ERROR "static admission did not reject ${fault}: ${stderr}")
  endif()
endforeach()
