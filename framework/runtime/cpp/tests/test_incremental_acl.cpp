#include "qwen35_dflash/generation.hpp"
#include "qwen35_dflash/incremental_acl_executor.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void RunPolicy(
    const std::array<std::filesystem::path, 4>& model_paths,
    qwen35::dflash::IncrementalStateResetPolicy reset_policy) {
  qwen35::dflash::IncrementalOmPaths paths{
      model_paths[0], model_paths[1], model_paths[2], model_paths[3]};
  std::vector<std::string> loaded_roles;
  qwen35::dflash::AclIncrementalExecutor executor(
      std::move(paths),
      0,
      [&](const char* role, const char* stage, std::size_t, std::size_t) {
        if (std::string(stage) == "load-done") {
          loaded_roles.emplace_back(role);
        }
      },
      reset_policy);
  Require(loaded_roles.size() == 4, "not all fake OMs were loaded");
  Require(executor.sequence_length() == 64, "fake state capacity differs");
  Require(executor.prefill_width() == 64, "fake prefill gear differs");
  Require(executor.proposal_width() == 15, "fake proposal width differs");
  Require(executor.eos_table_width() == 4, "fake EOS width differs");
  Require(executor.state_reset_policy() == reset_policy, "reset policy differs");

  qwen35::dflash::GenerationOptions options;
  options.max_new_tokens = 6;
  options.max_draft_tokens = 3;
  const auto ordinary = qwen35::dflash::GenerateStatefulOnce(
      executor,
      {10},
      qwen35::dflash::GenerationMode::kOrdinary,
      options);
  const auto dflash = qwen35::dflash::GenerateStatefulOnce(
      executor,
      {10},
      qwen35::dflash::GenerationMode::kDFlash,
      options);
  const std::vector<std::int64_t> expected{11, 12, 13, 14, 15, 16};
  Require(ordinary.generated_token_ids == expected, "fake ACL ordinary differs");
  Require(dflash.generated_token_ids == expected, "fake ACL DFlash differs");
  Require(dflash.counters.accepted_draft_tokens == 3, "fake ACL acceptance differs");

  options.eos_token_ids = {13};
  const auto eos = qwen35::dflash::GenerateStatefulOnce(
      executor,
      {10},
      qwen35::dflash::GenerationMode::kDFlash,
      options);
  Require(
      eos.generated_token_ids == std::vector<std::int64_t>({11, 12, 13}),
      "fake ACL EOS tokens differ");
  Require(eos.stop_reason == "eos", "fake ACL EOS stop differs");

  const auto& stats = executor.execution_stats();
  Require(stats.target_prefill_executions > 0, "prefill OM was not executed");
  Require(stats.target_decode1_executions > 0, "decode OM was not executed");
  Require(stats.draft_propose_executions > 0, "Draft OM was not executed");
  Require(
      stats.target_verify_commit_executions > 0,
      "verify OM was not executed");
  Require(stats.state_resets == 3, "state reset count differs");
  Require(stats.working_state_device_bytes > 0, "working state is missing");
  Require(stats.state_reset_bytes_per_request > 0, "reset byte set is empty");
  Require(
      stats.state_device_bytes == stats.working_state_device_bytes +
          stats.immutable_zero_state_device_bytes,
      "state allocation total does not close");
  if (reset_policy ==
      qwen35::dflash::IncrementalStateResetPolicy::kAsyncMemset) {
    Require(
        stats.state_memset_operations == 2 * stats.state_resets,
        "async reset did not clear Target/Draft input arenas");
    Require(
        stats.state_memset_bytes ==
            stats.state_reset_bytes_per_request * stats.state_resets,
        "async reset byte counter differs");
    Require(
        stats.immutable_zero_state_device_bytes == 0 &&
            stats.state_initialization_memset_operations == 0 &&
            stats.state_initialization_stream_synchronizations == 0,
        "async reset unexpectedly allocated immutable zero state");
  } else {
    Require(
        stats.state_memset_operations == 0 && stats.state_memset_bytes == 0,
        "immutable zero performed a per-request state clear");
    Require(
        stats.immutable_zero_state_device_bytes ==
            stats.state_reset_bytes_per_request,
        "immutable zero allocation size differs");
    Require(
        stats.state_initialization_memset_operations == 2 &&
            stats.state_initialization_memset_bytes ==
                stats.immutable_zero_state_device_bytes &&
            stats.state_initialization_stream_synchronizations == 1,
        "immutable zero startup initialization counters differ");
  }
  Require(
      stats.stream_synchronizations ==
          stats.target_prefill_executions +
              stats.target_decode1_executions +
              stats.target_verify_commit_executions,
      "reset or Draft introduced an extra transaction barrier");
  Require(
      stats.device_to_host_operations == stats.stream_synchronizations,
      "each transaction must return exactly one compact host result");
  Require(stats.state_device_bytes > 0, "state arenas were not reported");
  Require(
      stats.device_to_host_operations <
          stats.target_prefill_executions +
              stats.target_decode1_executions +
              stats.draft_propose_executions +
              stats.target_verify_commit_executions,
      "Draft introduced an extra host-visible result copy");
  Require(executor.model_memory().size() == 4, "model memory set differs");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 5) {
      throw std::invalid_argument("expected four fake OM paths");
    }
    const std::array<std::filesystem::path, 4> paths{
        std::filesystem::path(argv[1]),
        std::filesystem::path(argv[2]),
        std::filesystem::path(argv[3]),
        std::filesystem::path(argv[4]),
    };
    RunPolicy(
        paths,
        qwen35::dflash::IncrementalStateResetPolicy::kAsyncMemset);
    RunPolicy(
        paths,
        qwen35::dflash::IncrementalStateResetPolicy::kImmutableZero);
    std::cout << "PASS: both reset policies preserve exact four-OM tokens, "
                 "device state routing and compact synchronization\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
