# This test injects fake-ACL output faults. It is not model/device evidence.
set(ENV{QWEN35_DFLASH_FAKE_STATIC_SPLIT} 1)
set(model_args
  --target-prefill "${PREFILL}" --target-prefill-sha256 "${PREFILL_SHA}"
  --target-decode1 "${DECODE}" --target-decode1-sha256 "${DECODE_SHA}"
  --draft-propose "${DRAFT}" --draft-propose-sha256 "${DRAFT_SHA}"
  --target-verify-commit "${VERIFY}" --target-verify-commit-sha256 "${VERIFY_SHA}")
foreach(fault token eos)
  set(ENV{QWEN35_DFLASH_FAKE_PARITY_FAULT} "${fault}")
  set(output "${OUTPUT}-${fault}.json")
  file(REMOVE "${output}" "${output}.tmp"
              "${output}.failure.json" "${output}.failure.json.tmp")
  execute_process(COMMAND "${RUNNER}" ${model_args} --output "${output}"
    --prompt-token-ids 10 --max-new-tokens 8 --max-draft-tokens 3
    --eos-token-ids 99 --progress false
    RESULT_VARIABLE result OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
  if(NOT result EQUAL 1 OR EXISTS "${output}" OR NOT EXISTS "${output}.failure.json")
    message(FATAL_ERROR "failure did not close before PASS report: ${stderr}")
  endif()
  file(READ "${output}.failure.json" report)
  string(JSON status GET "${report}" status)
  string(JSON formal GET "${report}" formal_latency_evidence)
  string(JSON kind GET "${report}" comparison kind)
  string(JSON phase GET "${report}" comparison phase)
  string(JSON run GET "${report}" comparison run_index)
  string(JSON position GET "${report}" comparison first_mismatch_index)
  string(JSON expected GET "${report}" comparison expected token_at_mismatch)
  string(JSON actual GET "${report}" comparison actual token_at_mismatch)
  string(JSON original GET "${report}" comparison expected measurement generated_token_ids 1)
  string(JSON last_run GET "${report}" last_progress run_index)
  string(JSON path GET "${report}" comparison actual transactions 1 path)
  string(JSON anchor GET "${report}" comparison actual transactions 1 anchor_token_id)
  string(JSON begin GET "${report}" comparison actual transactions 1 generated_begin)
  string(JSON slot GET "${report}" comparison actual transactions 1 compact_slot)
  if(NOT status STREQUAL "FAIL" OR formal OR
     NOT kind STREQUAL "ordinary_dflash_parity" OR NOT phase STREQUAL "warmup" OR
     NOT run EQUAL 1 OR NOT last_run EQUAL 1 OR NOT position EQUAL 1 OR
     NOT expected EQUAL 12 OR NOT original EQUAL 12 OR
     NOT path STREQUAL "speculative-verify" OR NOT anchor EQUAL 11 OR
     NOT begin EQUAL 1 OR slot LESS 0 OR
     NOT stderr MATCHES "first_mismatch_index=1" OR
     NOT stderr MATCHES "stage=failure-report-done")
    message(FATAL_ERROR "missing actionable mismatch evidence: ${report} ${stderr}")
  endif()
  if(fault STREQUAL "eos")
    string(JSON stop GET "${report}" comparison actual measurement stop_reason)
    string(JSON stop_diff GET "${report}" comparison stop_reason_mismatch)
    if(NOT actual EQUAL 99 OR NOT stop STREQUAL "eos" OR NOT stop_diff)
      message(FATAL_ERROR "EOS/length failure lost: ${report}")
    endif()
  elseif(NOT actual EQUAL 112)
    message(FATAL_ERROR "wrong injected token: ${report}")
  endif()

  # Existing evidence is immutable: a retry must fail admission, not replace it.
  file(SHA256 "${output}.failure.json" original_hash)
  execute_process(COMMAND "${RUNNER}" ${model_args} --output "${output}"
    --prompt-token-ids 10 RESULT_VARIABLE retry ERROR_VARIABLE stderr OUTPUT_QUIET)
  file(SHA256 "${output}.failure.json" retry_hash)
  if(NOT retry EQUAL 1 OR NOT original_hash STREQUAL retry_hash OR
     NOT stderr MATCHES "already exists" OR stderr MATCHES "stage=load-")
    message(FATAL_ERROR "retry overwrote diagnostics or loaded models: ${stderr}")
  endif()
endforeach()

# An unwritable diagnostic destination must not replace the original mismatch.
# A test-owned regular file as the parent works even when CTest runs as root.
set(blocked_parent "${OUTPUT}-blocked-parent")
file(WRITE "${blocked_parent}" "diagnostic failure test sentinel\n")
file(SHA256 "${blocked_parent}" blocked_hash)
set(ENV{QWEN35_DFLASH_FAKE_PARITY_FAULT} token)
execute_process(COMMAND "${RUNNER}" ${model_args}
  --output "${blocked_parent}/report.json"
  --prompt-token-ids 10 --max-new-tokens 8 --max-draft-tokens 3
  --eos-token-ids 99 --progress false
  RESULT_VARIABLE result ERROR_VARIABLE stderr OUTPUT_QUIET)
file(SHA256 "${blocked_parent}" after_blocked_hash)
if(NOT result EQUAL 1 OR NOT blocked_hash STREQUAL after_blocked_hash OR
   NOT stderr MATCHES "first_mismatch_index=1" OR
   NOT stderr MATCHES "failure report unavailable" OR
   stderr MATCHES "stage=failure-report-done")
  message(FATAL_ERROR "diagnostic write error masked original failure: ${stderr}")
endif()
unset(ENV{QWEN35_DFLASH_FAKE_PARITY_FAULT})

# Launch errors must retain context, even though there is no completed pair.
set(output "${OUTPUT}-execute.json")
file(REMOVE "${output}" "${output}.tmp"
            "${output}.failure.json" "${output}.failure.json.tmp")
set(ENV{QWEN35_DFLASH_FAKE_FAIL_EXECUTE} 1)
execute_process(COMMAND "${RUNNER}" ${model_args} --output "${output}"
  --prompt-token-ids 10 RESULT_VARIABLE result ERROR_VARIABLE stderr OUTPUT_QUIET)
unset(ENV{QWEN35_DFLASH_FAKE_FAIL_EXECUTE})
if(NOT result EQUAL 1 OR NOT EXISTS "${output}.failure.json" OR EXISTS "${output}")
  message(FATAL_ERROR "launch error diagnostic missing: ${stderr}")
endif()
file(READ "${output}.failure.json" report)
string(JSON error GET "${report}" error)
string(JSON stage GET "${report}" stage)
string(JSON comparison_type TYPE "${report}" comparison)
if(NOT error MATCHES "ACL error.*model_id=.*physical_rows=.*target_state_slot=" OR
   NOT stage STREQUAL "benchmark" OR NOT comparison_type STREQUAL "NULL")
  message(FATAL_ERROR "launch failure misreported as token mismatch: ${report}")
endif()
