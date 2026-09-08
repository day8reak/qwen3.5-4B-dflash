#pragma once

#include "qwen35_dflash/incremental_acl_executor.hpp"

#include <functional>
#include <string>

namespace qwen35::dflash {

struct TargetParityDiagnostic {
  bool token_parity = true;
  bool cursor_parity = true;
  std::string first_token_mismatch_json = "null";
  std::string detail_json;
};

// Bounded DFlash capture followed by ordinary teacher-forced and same-state
// replays. It consumes/mutates the executor and cannot share a timing run.
TargetParityDiagnostic DiagnoseTargetParity(
    AclIncrementalExecutor& executor,
    const std::vector<std::int64_t>& prompt,
    const GenerationOptions& options,
    std::size_t max_transactions,
    const std::function<void(const std::string&)>& progress = {});

}  // namespace qwen35::dflash
