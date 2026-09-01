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
  std::filesystem::path target_prefill_head;
  std::filesystem::path target_decode1;
  std::filesystem::path draft_propose;
  std::filesystem::path target_verify_commit;
};

struct IncrementalModelMemory {
  std::string role;
  std::size_t work_bytes = 0;
  std::size_t weight_bytes = 0;
};

enum class IncrementalStateResetPolicy {
  // Clear the first Target and Draft input arenas on the execution stream for
  // every request. The clear is consumed by the first prefill barrier.
  kAsyncMemset,
  // Keep one read-only zero Target/Draft arena initialized at process startup.
  // The first prefill reads it and writes into the ordinary ping-pong arenas.
  kImmutableZero,
};

const char* IncrementalStateResetPolicyName(
    IncrementalStateResetPolicy policy) noexcept;

enum class IncrementalDecodeCarrierPolicy {
  // Bind one-token compact row zero directly. A multi-token result falls back
  // to the original 8-byte pinned-host H2D path on the next decode.
  kOneTokenHostFallback,
  // Keep every last committed token on device. Multi-token rows compact D2D
  // into the existing aligned scalar before the next decode.
  kLastTokenDeviceCompact,
};

const char* IncrementalDecodeCarrierPolicyName(
    IncrementalDecodeCarrierPolicy policy) noexcept;

struct IncrementalAclExecutionStats {
  std::size_t target_prefill_executions = 0;
  std::size_t target_prefill_head_executions = 0;
  std::size_t target_prefill_head_executions_elided = 0;
  std::size_t target_decode1_executions = 0;
  std::size_t draft_propose_executions = 0;
  std::size_t target_verify_commit_executions = 0;
  std::size_t stream_synchronizations = 0;
  std::size_t prefill_completion_synchronizations = 0;
  std::size_t deferred_prefill_chunks = 0;
  std::size_t prefill_synchronizations_elided = 0;
  std::size_t prefill_compact_downloads_elided = 0;
  std::size_t prefill_draft_propose_executions = 0;
  std::size_t prefill_draft_propose_executions_elided = 0;
  std::size_t prefill_feature_rows_batched = 0;
  std::size_t prefill_control_upload_operations = 0;
  std::size_t prefill_control_upload_bytes = 0;
  std::size_t prefill_control_full_upload_operations = 0;
  std::size_t prefill_control_base_upload_operations = 0;
  std::size_t prefill_control_count_upload_operations = 0;
  std::size_t prefill_control_proposal_upload_operations = 0;
  std::size_t prefill_control_h2d_bytes_elided = 0;
  std::size_t prefill_h2d_operations_elided = 0;
  std::size_t decode_id_upload_operations = 0;
  std::size_t decode_id_upload_bytes = 0;
  std::size_t decode_id_device_carrier_hits = 0;
  std::size_t decode_id_multi_token_carrier_hits = 0;
  std::size_t decode_id_h2d_operations_elided = 0;
  std::size_t decode_id_device_compaction_operations = 0;
  std::size_t decode_id_device_compaction_bytes = 0;
  std::size_t proposal_count_upload_operations = 0;
  std::size_t proposal_count_upload_bytes = 0;
  std::size_t state_resets = 0;
  std::size_t state_memset_operations = 0;
  std::size_t state_memset_bytes = 0;
  std::size_t state_initialization_memset_operations = 0;
  std::size_t state_initialization_memset_bytes = 0;
  std::size_t state_initialization_stream_synchronizations = 0;
  std::size_t host_to_device_operations = 0;
  std::size_t host_to_device_bytes = 0;
  std::size_t device_to_host_operations = 0;
  std::size_t device_to_host_bytes = 0;
  std::size_t state_device_bytes = 0;
  std::size_t working_state_device_bytes = 0;
  std::size_t immutable_zero_state_device_bytes = 0;
  std::size_t state_reset_bytes_per_request = 0;
  std::size_t carrier_device_bytes = 0;
  std::size_t compact_ping_pong_device_bytes = 0;
  std::size_t prefill_staging_slots = 0;
  std::size_t prefill_control_bytes_per_slot = 0;
  std::size_t prefill_base_control_bytes_per_slot = 0;
  std::size_t prefill_count_control_bytes_per_slot = 0;
  std::size_t prefill_proposal_control_bytes_per_slot = 0;
  std::size_t prefill_persistent_control_tail_bytes_per_slot = 0;
  std::size_t prefill_staging_pinned_host_bytes = 0;
  std::size_t prefill_feature_slab_bytes = 0;
  std::size_t prefill_feature_arena_bytes = 0;
  std::size_t draft_dynamic_gear_count = 0;
};

using IncrementalModelProgress = std::function<void(
    const char* role,
    const char* stage,
    std::size_t work_bytes,
    std::size_t weight_bytes)>;

// Five resident OM sessions implementing the approved exact state graph.
// The prefill body excludes its QLinear LM head; a small head-only OM runs
// once after the final physical prompt chunk. This moves the prefill head
// weight instead of retaining a dead copy in the body artifact.
// Target/Draft states and compact Target results are ping-ponged in device
// arenas. The carrier policy either binds only one-token row zero and falls
// back to H2D after multi-token commits, or retains every last committed token,
// binding row zero directly and compacting later rows D2D into the aligned
// input. An explicit caller override retains the original H2D fallback.
// Per-chunk control uploads stop after the last field consumed by that chunk:
// base IDs/effective length, final-Draft count, changed proposal, or the full
// EOS tail. The EOS table/count stay resident and are refreshed only when
// Reset changes their identity, without adding a separate H2D operation.
// Proposal IDs, Target features and cursors never cross the host boundary. A speculative
// method enqueues Draft -> Target verify/commit and performs one stream sync
// only after a compact transaction result has been queued for D2H. The first
// prefill after Reset either enqueues state clears on the same stream or reads
// immutable zero state initialized at startup. Neither policy adds a reset-only
// synchronization to a request.
class AclIncrementalExecutor final : public StatefulGraphExecutor {
 public:
  explicit AclIncrementalExecutor(
      IncrementalOmPaths model_paths,
      int device_id = 0,
      IncrementalModelProgress progress = {},
      IncrementalStateResetPolicy state_reset_policy =
          IncrementalStateResetPolicy::kAsyncMemset,
      IncrementalDecodeCarrierPolicy decode_carrier_policy =
          IncrementalDecodeCarrierPolicy::kLastTokenDeviceCompact);
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
  std::size_t PrefillChunkDeferred(
      const std::vector<std::int64_t>& token_ids,
      bool prepare_draft,
      std::size_t logical_proposal_count) override;
  StatefulStep DecodeOne(std::int64_t input_token_id) override;
  StatefulStep SpeculativeStep(
      std::size_t logical_proposal_count) override;

  const std::vector<IncrementalModelMemory>& model_memory() const noexcept;
  const IncrementalAclExecutionStats& execution_stats() const noexcept;
  IncrementalStateResetPolicy state_reset_policy() const noexcept;
  IncrementalDecodeCarrierPolicy decode_carrier_policy() const noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace qwen35::dflash
