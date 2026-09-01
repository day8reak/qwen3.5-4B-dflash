#include "qwen35_dflash/generation.hpp"
#include "qwen35_dflash/incremental_acl_executor.hpp"

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 5) {
      throw std::invalid_argument("expected four fake OM paths");
    }
    qwen35::dflash::IncrementalOmPaths paths{
        std::filesystem::path(argv[1]),
        std::filesystem::path(argv[2]),
        std::filesystem::path(argv[3]),
        std::filesystem::path(argv[4]),
    };
    std::vector<std::string> loaded_roles;
    qwen35::dflash::AclIncrementalExecutor executor(
        std::move(paths),
        0,
        [&](const char* role, const char* stage, std::size_t, std::size_t) {
          if (std::string(stage) == "load-done") {
            loaded_roles.emplace_back(role);
          }
        });
    Require(loaded_roles.size() == 4, "not all fake OMs were loaded");
    Require(executor.sequence_length() == 64, "fake state capacity differs");
    Require(executor.prefill_width() == 64, "fake prefill gear differs");
    Require(executor.proposal_width() == 15, "fake proposal width differs");
    Require(executor.eos_table_width() == 4, "fake EOS width differs");

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
    Require(
        stats.state_memset_operations == 2 * stats.state_resets,
        "state reset did not clear exactly the current Target/Draft arenas");
    Require(
        stats.stream_synchronizations ==
            stats.target_prefill_executions +
                stats.target_decode1_executions +
                stats.target_verify_commit_executions,
        "reset or Draft introduced an extra stream barrier");
    Require(
        stats.device_to_host_operations == stats.stream_synchronizations,
        "each transaction must return exactly one compact host result");
    Require(stats.state_device_bytes > 0, "state arenas were not reported");
    Require(stats.device_to_host_operations <
                stats.target_prefill_executions +
                    stats.target_decode1_executions +
                    stats.draft_propose_executions +
                    stats.target_verify_commit_executions,
            "Draft introduced an extra host-visible result copy");
    Require(executor.model_memory().size() == 4, "model memory set differs");
    std::cout << "PASS: four resident fake OMs, device state routing, exact "
                 "tokens and compact synchronization\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
