#pragma once

#include "qwen35_dflash/generation.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace qwen35::dflash {

struct IncrementalOmPaths {
  std::filesystem::path target_prefill;
  std::filesystem::path target_decode1;
  std::filesystem::path draft_propose;
  std::filesystem::path target_verify_commit;
};

struct IncrementalModelMemory {
  std::string role;
  std::size_t work_bytes = 0;
  std::size_t weight_bytes = 0;
};

struct IncrementalAclExecutionStats {
  std::size_t target_prefill_executions = 0;
  std::size_t target_decode1_executions = 0;
  std::size_t draft_propose_executions = 0;
  std::size_t target_verify_commit_executions = 0;
  std::size_t stream_synchronizations = 0;
  std::size_t host_to_device_operations = 0;
  std::size_t host_to_device_bytes = 0;
  std::size_t device_to_host_operations = 0;
  std::size_t device_to_host_bytes = 0;
  std::size_t state_device_bytes = 0;
  std::size_t carrier_device_bytes = 0;
};

using IncrementalModelProgress = std::function<void(
    const char* role,
    const char* stage,
    std::size_t work_bytes,
    std::size_t weight_bytes)>;

// Four resident OM sessions implementing the approved exact state graph.
// Target and Draft states are ping-ponged in device arenas.  Proposal IDs,
// Target features and cursors never cross the host boundary.  A speculative
// method enqueues Draft -> Target verify/commit and performs one stream sync
// only after a compact transaction result has been queued for D2H.
class AclIncrementalExecutor final : public StatefulGraphExecutor {
 public:
  explicit AclIncrementalExecutor(
      IncrementalOmPaths model_paths,
      int device_id = 0,
      IncrementalModelProgress progress = {});
  ~AclIncrementalExecutor() override;

  AclIncrementalExecutor(const AclIncrementalExecutor&) = delete;
  AclIncrementalExecutor& operator=(const AclIncrementalExecutor&) = delete;
  AclIncrementalExecutor(AclIncrementalExecutor&&) noexcept;
  AclIncrementalExecutor& operator=(AclIncrementalExecutor&&) noexcept;

  std::size_t sequence_length() const noexcept override;
  std::size_t prefill_width() const noexcept override;
  std::size_t proposal_width() const noexcept override;
  std::size_t eos_table_width() const noexcept override;

  void Reset(
      std::int64_t pad_token_id,
      const std::vector<std::int64_t>& eos_token_ids) override;
  StatefulStep PrefillChunk(
      const std::vector<std::int64_t>& token_ids,
      bool prepare_draft,
      std::size_t logical_proposal_count) override;
  StatefulStep DecodeOne(std::int64_t input_token_id) override;
  StatefulStep SpeculativeStep(
      std::size_t logical_proposal_count) override;

  const std::vector<IncrementalModelMemory>& model_memory() const noexcept;
  const IncrementalAclExecutionStats& execution_stats() const noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace qwen35::dflash
