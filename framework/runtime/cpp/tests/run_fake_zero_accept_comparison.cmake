set(required_values
  RUNNER PREFILL PREFILL_SHA PREFILL_HEAD PREFILL_HEAD_SHA DECODE DECODE_SHA
  DRAFT DRAFT_SHA VERIFY VERIFY_SHA OUTPUT_BASE OUTPUT_FALLBACK
)
foreach(required ${required_values})
  if(NOT DEFINED ${required})
    message(FATAL_ERROR "${required} is required")
  endif()
endforeach()

string(REPEAT "1," 69 PROMPT_PREFIX)
set(PROMPT_IDS "${PROMPT_PREFIX}10")
file(REMOVE
  "${OUTPUT_BASE}" "${OUTPUT_BASE}.tmp"
  "${OUTPUT_FALLBACK}" "${OUTPUT_FALLBACK}.tmp"
)
set(ENV{QWEN35_DFLASH_FAKE_ZERO_ACCEPT} "1")

foreach(policy disabled request-target-only)
  if(policy STREQUAL "disabled")
    set(output "${OUTPUT_BASE}")
  else()
    set(output "${OUTPUT_FALLBACK}")
  endif()
  execute_process(
    COMMAND "${RUNNER}"
      --target-prefill "${PREFILL}"
      --target-prefill-sha256 "${PREFILL_SHA}"
      --target-prefill-head "${PREFILL_HEAD}"
      --target-prefill-head-sha256 "${PREFILL_HEAD_SHA}"
      --target-decode1 "${DECODE}"
      --target-decode1-sha256 "${DECODE_SHA}"
      --draft-propose "${DRAFT}"
      --draft-propose-sha256 "${DRAFT_SHA}"
      --target-verify-commit "${VERIFY}"
      --target-verify-commit-sha256 "${VERIFY_SHA}"
      --output "${output}"
      --prompt-token-ids "${PROMPT_IDS}"
      --eos-token-ids 999
      --pad-token-id 0
      --max-new-tokens 10
      --max-draft-tokens 3
      --dflash-sync-window 1
      --prefill-completion-policy separate
      --zero-accept-fallback-policy "${policy}"
      --warmup 3
      --repetitions 10
      --device-id 0
      --measurement-protocol evidence
      --state-reset-policy async-memset
      --decode-carrier-policy last-token-d2d
      --draft-feature-policy fixed-16
      --progress false
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr
  )
  if(NOT result EQUAL 0)
    message(FATAL_ERROR
      "fake zero-accept ${policy} run failed: ${result}\n${stdout}\n${stderr}"
    )
  endif()
endforeach()
unset(ENV{QWEN35_DFLASH_FAKE_ZERO_ACCEPT})

file(READ "${OUTPUT_BASE}" baseline)
file(READ "${OUTPUT_FALLBACK}" fallback)
foreach(name baseline fallback)
  string(JSON ${name}_status GET "${${name}}" status)
  string(JSON ${name}_mismatch GET
    "${${name}}" ordinary_parity token_id_mismatches
  )
  string(JSON ${name}_eos_mismatch GET
    "${${name}}" ordinary_parity eos_mismatches
  )
  string(JSON ${name}_ordinary_tokens GET
    "${${name}}" ordinary stable_generated_token_ids
  )
  string(JSON ${name}_dflash_tokens GET
    "${${name}}" dflash stable_generated_token_ids
  )
  string(JSON ${name}_stop GET "${${name}}" dflash stable_stop_reason)
  string(JSON ${name}_ordinary_graph_calls GET
    "${${name}}" ordinary totals graph_calls
  )
  string(JSON ${name}_dflash_graph_calls GET
    "${${name}}" dflash totals graph_calls
  )
  string(JSON ${name}_zero_transactions GET
    "${${name}}" dflash totals zero_accept_transactions
  )
  string(JSON ${name}_activations GET
    "${${name}}" dflash totals zero_accept_fallback_activations
  )
  string(JSON ${name}_target_only_iterations GET
    "${${name}}" dflash totals target_only_fallback_iterations
  )
  string(JSON ${name}_model_executions GET
    "${${name}}" execution_io_counters model_executions
  )
  string(JSON ${name}_draft_executions GET
    "${${name}}" execution_io_counters draft_propose_executions
  )
  string(JSON ${name}_verify_executions GET
    "${${name}}" execution_io_counters target_verify_commit_executions
  )
endforeach()
string(JSON baseline_policy GET
  "${baseline}" protocol zero_accept_fallback_policy
)
string(JSON fallback_policy GET
  "${fallback}" protocol zero_accept_fallback_policy
)

if(NOT baseline_status STREQUAL "PASS" OR
   NOT fallback_status STREQUAL "PASS" OR
   NOT baseline_mismatch EQUAL 0 OR NOT fallback_mismatch EQUAL 0 OR
   NOT baseline_eos_mismatch EQUAL 0 OR NOT fallback_eos_mismatch EQUAL 0 OR
   NOT baseline_ordinary_tokens STREQUAL baseline_dflash_tokens OR
   NOT fallback_ordinary_tokens STREQUAL fallback_dflash_tokens OR
   NOT baseline_dflash_tokens STREQUAL fallback_dflash_tokens OR
   NOT baseline_stop STREQUAL fallback_stop OR
   NOT baseline_policy STREQUAL "disabled" OR
   NOT fallback_policy STREQUAL "request-target-only" OR
   NOT baseline_ordinary_graph_calls EQUAL fallback_ordinary_graph_calls OR
   NOT baseline_zero_transactions EQUAL 80 OR
   NOT baseline_activations EQUAL 0 OR
   NOT baseline_target_only_iterations EQUAL 0 OR
   NOT fallback_zero_transactions EQUAL 10 OR
   NOT fallback_activations EQUAL 10 OR
   NOT fallback_target_only_iterations EQUAL 70 OR
   NOT baseline_dflash_graph_calls EQUAL 200 OR
   NOT fallback_dflash_graph_calls EQUAL 130 OR
   NOT baseline_model_executions EQUAL 416 OR
   NOT fallback_model_executions EQUAL 325 OR
   NOT baseline_draft_executions EQUAL 104 OR
   NOT fallback_draft_executions EQUAL 13 OR
   NOT baseline_verify_executions EQUAL 104 OR
   NOT fallback_verify_executions EQUAL 13)
  message(FATAL_ERROR
    "fake zero-accept A/B did not preserve tokens or eliminate the expected "
    "Draft/Verify executions\nbaseline=${baseline}\nfallback=${fallback}"
  )
endif()
