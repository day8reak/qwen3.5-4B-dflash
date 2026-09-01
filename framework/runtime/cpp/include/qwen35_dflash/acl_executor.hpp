#pragma once

#include "qwen35_dflash/generation.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>

namespace qwen35::dflash {

struct AclExecutionStats {
  std::size_t model_executions = 0;
  std::size_t stream_synchronizations = 0;
  std::size_t host_to_device_operations = 0;
  std::size_t host_to_device_bytes = 0;
  std::size_t full_host_to_device_bytes = 0;
  std::size_t host_to_device_copies_skipped = 0;
  std::size_t device_to_host_operations = 0;
  std::size_t device_to_host_bytes = 0;
  std::size_t full_device_to_host_bytes = 0;
  std::size_t target_elements_downloaded = 0;
  std::size_t maximum_target_elements_per_call = 0;
};

// Direct AscendCL executor for the ordered deployment ABI:
//   inputs:  input_ids [1,S], attention_mask [1,S]
//   outputs: target_top1 [1,S], draft_top1 [1,K]
// Every tensor is INT64. The implementation allocates pinned host memory,
// device buffers, datasets, context and stream once, then reuses them.
class AclExecutor final : public GraphExecutor {
 public:
  explicit AclExecutor(
      const std::filesystem::path& model_path,
      int device_id = 0);
  ~AclExecutor() override;

  AclExecutor(const AclExecutor&) = delete;
  AclExecutor& operator=(const AclExecutor&) = delete;
  AclExecutor(AclExecutor&&) noexcept;
  AclExecutor& operator=(AclExecutor&&) noexcept;

  std::size_t sequence_length() const noexcept override;
  std::size_t draft_width() const noexcept override;
  std::size_t model_work_bytes() const noexcept;
  std::size_t model_weight_bytes() const noexcept;
  const AclExecutionStats& execution_stats() const noexcept;
  const GraphOutputs& Execute(
      const std::vector<std::int64_t>& committed_prefix,
      std::int64_t pad_token_id) override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace qwen35::dflash
