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
    const std::array<std::filesystem::path, 5>& model_paths,
    qwen35::dflash::IncrementalStateResetPolicy reset_policy,
    qwen35::dflash::IncrementalDecodeCarrierPolicy decode_carrier_policy) {
  qwen35::dflash::IncrementalOmPaths paths{
      model_paths[0], model_paths[1], model_paths[2], model_paths[3],
      model_paths[4]};
  std::vector<std::string> loaded_roles;
  qwen35::dflash::AclIncrementalExecutor executor(
      std::move(paths),
      0,
      [&](const char* role, const char* stage, std::size_t, std::size_t) {
        if (std::string(stage) == "load-done") {
          loaded_roles.emplace_back(role);
        }
      },
      reset_policy,
      decode_carrier_policy);
  Require(loaded_roles.size() == 5, "not all fake OMs were loaded");
  Require(executor.sequence_length() == 128, "fake state capacity differs");
  Require(executor.prefill_width() == 64, "fake prefill gear differs");
  Require(executor.proposal_width() == 15, "fake proposal width differs");
  Require(executor.eos_table_width() == 4, "fake EOS width differs");
  Require(
      executor.max_speculative_sync_window() == 8,
      "fake executor sync-window capability differs");
  Require(executor.state_reset_policy() == reset_policy, "reset policy differs");
  Require(
      executor.decode_carrier_policy() == decode_carrier_policy,
      "decode carrier policy differs");
  Require(
      executor.draft_feature_policy() ==
          qwen35::dflash::IncrementalDraftFeaturePolicy::kFixedVerifyWidth,
      "default Draft feature policy differs");

  qwen35::dflash::GenerationOptions options;
  options.max_new_tokens = 6;
  options.max_draft_tokens = 3;
  const auto ordinary = qwen35::dflash::GenerateStatefulOnce(
      executor,
      {10},
      qwen35::dflash::GenerationMode::kOrdinary,
      options);
  const auto ordinary_stats = executor.execution_stats();
  Require(
      ordinary_stats.proposal_count_upload_operations == 0,
      "unified ordinary decode uploaded a mutable zero proposal count");
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

  options.eos_token_ids.clear();
  std::vector<std::int64_t> long_prompt(70, 1);
  long_prompt.back() = 10;
  const auto long_ordinary = qwen35::dflash::GenerateStatefulOnce(
      executor,
      long_prompt,
      qwen35::dflash::GenerationMode::kOrdinary,
      options);
  const auto long_dflash = qwen35::dflash::GenerateStatefulOnce(
      executor,
      long_prompt,
      qwen35::dflash::GenerationMode::kDFlash,
      options);
  Require(
      long_ordinary.generated_token_ids == expected,
      "multi-chunk fake ACL ordinary differs");
  Require(
      long_dflash.generated_token_ids == expected,
      "multi-chunk fake ACL DFlash differs");

  const auto& stats = executor.execution_stats();
  Require(stats.target_prefill_executions > 0, "prefill OM was not executed");
  Require(
      stats.target_prefill_head_executions == stats.state_resets,
      "prefill head did not execute exactly once per request");
  Require(
      stats.target_prefill_head_executions_elided ==
          stats.deferred_prefill_chunks,
      "prefill head was not elided for every intermediate chunk");
  Require(stats.target_decode1_executions > 0, "decode OM was not executed");
  Require(stats.draft_propose_executions > 0, "Draft OM was not executed");
  Require(
      stats.target_verify_commit_executions > 0,
      "verify OM was not executed");
  Require(stats.state_resets == 5, "state reset count differs");
  Require(stats.target_prefill_executions == 7, "prefill chunk count differs");
  Require(
      stats.prefill_completion_synchronizations == stats.state_resets,
      "prefill completion count differs from request count");
  Require(stats.deferred_prefill_chunks == 2, "deferred prefill count differs");
  Require(
      stats.prefill_synchronizations_elided ==
              stats.deferred_prefill_chunks &&
          stats.prefill_compact_downloads_elided ==
              stats.deferred_prefill_chunks,
      "multi-chunk prefill did not elide one sync/D2H per intermediate chunk");
  Require(
      stats.prefill_staging_slots == 2 &&
          stats.prefill_control_bytes_per_slot == 896 &&
          stats.prefill_base_control_bytes_per_slot == 578 &&
          stats.prefill_count_control_bytes_per_slot == 644 &&
          stats.prefill_proposal_control_bytes_per_slot == 708 &&
          stats.prefill_persistent_control_tail_bytes_per_slot == 188 &&
          stats.prefill_staging_pinned_host_bytes == 1792 &&
          stats.proposal_count_staging_pinned_host_bytes == 32,
      "prefill pinned-host staging ring differs");
  Require(
      stats.prefill_draft_propose_executions == 3 &&
          stats.prefill_draft_propose_executions_elided == 1 &&
          stats.prefill_feature_rows_batched == 256 &&
          stats.draft_propose_executions == 3,
      "prefill did not batch exactly one Draft execution per DFlash request");
  Require(
      stats.prefill_feature_slab_bytes == 1024 &&
          stats.prefill_feature_arena_bytes == 2112 &&
          stats.draft_dynamic_gear_count == 18 &&
          stats.draft_verify_dynamic_gear_count == 16 &&
          stats.draft_prefill_dynamic_gear_count == 2,
      "prefill feature arena or dynamic gear set differs");
  Require(
      stats.draft_verify_feature_input_rows == 0 &&
          stats.draft_verify_full_width_equivalent_rows == 0 &&
          stats.draft_verify_feature_rows_elided == 0 &&
          stats.draft_verify_fixed_width_executions == 0 &&
          stats.draft_verify_committed_prefix_executions == 0 &&
          stats.draft_verify_pending_upper_bound_executions == 0,
      "request set unexpectedly executed a verify-source Draft route");
  Require(
      stats.prefill_control_upload_operations ==
              stats.target_prefill_executions &&
          stats.prefill_control_full_upload_operations == 3 &&
          stats.prefill_control_base_upload_operations == 2 &&
          stats.prefill_control_count_upload_operations == 1 &&
          stats.prefill_control_proposal_upload_operations == 1 &&
          stats.prefill_control_full_upload_operations +
                  stats.prefill_control_base_upload_operations +
                  stats.prefill_control_count_upload_operations +
                  stats.prefill_control_proposal_upload_operations ==
              stats.prefill_control_upload_operations &&
          stats.prefill_h2d_operations_elided ==
              stats.target_prefill_executions &&
          stats.prefill_control_upload_bytes ==
              stats.prefill_control_full_upload_operations *
                      stats.prefill_control_bytes_per_slot +
                  stats.prefill_control_base_upload_operations *
                      stats.prefill_base_control_bytes_per_slot +
                  stats.prefill_control_count_upload_operations *
                      stats.prefill_count_control_bytes_per_slot +
                  stats.prefill_control_proposal_upload_operations *
                      stats.prefill_proposal_control_bytes_per_slot &&
          stats.prefill_control_h2d_bytes_elided ==
              stats.prefill_control_upload_operations *
                      stats.prefill_control_bytes_per_slot -
                  stats.prefill_control_upload_bytes,
      "prefill variable/persistent control upload counters do not close");
  Require(
      stats.decode_id_upload_operations +
                  stats.decode_id_device_carrier_hits ==
              stats.target_decode1_executions &&
          stats.decode_id_h2d_operations_elided ==
              stats.decode_id_device_carrier_hits &&
          stats.decode_id_device_compaction_bytes ==
              stats.decode_id_device_compaction_operations *
                  sizeof(std::int64_t) &&
          stats.decode_id_upload_bytes ==
              stats.decode_id_upload_operations * sizeof(std::int64_t) &&
          stats.proposal_count_upload_bytes ==
              stats.proposal_count_upload_operations *
                  sizeof(std::int32_t) &&
          stats.host_to_device_operations ==
              stats.prefill_control_upload_operations +
                  stats.decode_id_upload_operations +
                  stats.proposal_count_upload_operations &&
          stats.host_to_device_bytes ==
              stats.prefill_control_upload_bytes +
                  stats.decode_id_upload_bytes +
                  stats.proposal_count_upload_bytes,
      "decode carrier route or packed H2D counters do not close");
  if (decode_carrier_policy ==
      qwen35::dflash::IncrementalDecodeCarrierPolicy::
          kLastTokenDeviceCompact) {
    Require(
        stats.decode_id_upload_operations == 0 &&
            stats.decode_id_device_carrier_hits ==
                stats.target_decode1_executions &&
            stats.decode_id_multi_token_carrier_hits > 0 &&
            stats.decode_id_multi_token_carrier_hits <
                stats.decode_id_device_carrier_hits &&
            stats.decode_id_device_compaction_operations ==
                stats.decode_id_multi_token_carrier_hits,
        "last-token D2D decode carrier counters differ");
  } else {
    Require(
        stats.decode_id_upload_operations > 0 &&
            stats.decode_id_device_carrier_hits > 0 &&
            stats.decode_id_multi_token_carrier_hits == 0 &&
            stats.decode_id_device_compaction_operations == 0 &&
            stats.decode_id_device_compaction_bytes == 0,
        "one-token H2D fallback decode carrier counters differ");
  }
  Require(stats.working_state_device_bytes > 0, "working state is missing");
  Require(
      stats.compact_ping_pong_device_bytes > 0 &&
          stats.compact_ping_pong_device_bytes ==
              2 * stats.compact_slot_bytes &&
          stats.compact_slot_bytes == 512 &&
          stats.compact_ordinary_result_bytes == 257 &&
          stats.compact_verify_result_bytes == 452 &&
          stats.speculative_window_staging_device_bytes ==
              8 * stats.compact_slot_bytes &&
          stats.speculative_window_staging_pinned_host_bytes ==
              stats.speculative_window_staging_device_bytes &&
          stats.speculative_window_staging_operations == 0 &&
          stats.speculative_window_staging_bytes == 0 &&
          stats.carrier_device_bytes >
              stats.compact_ping_pong_device_bytes +
                  stats.speculative_window_staging_device_bytes,
      "compact result ping-pong allocation was not reported");
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
      stats.speculative_sync_windows +
              stats.speculative_synchronizations_elided ==
          stats.target_verify_commit_executions &&
          stats.speculative_d2h_operations_elided ==
              stats.speculative_synchronizations_elided &&
          stats.speculative_d2h_padding_bytes ==
              stats.speculative_d2h_operations_elided *
                  (stats.compact_slot_bytes -
                   stats.compact_verify_result_bytes) &&
          stats.stream_synchronizations ==
              stats.prefill_completion_synchronizations +
                  stats.target_decode1_executions +
                  stats.speculative_sync_windows,
      "reset or Draft introduced an extra transaction barrier");
  Require(
      stats.device_to_host_operations +
              stats.speculative_d2h_operations_elided ==
          stats.prefill_completion_synchronizations +
              stats.target_decode1_executions +
              stats.target_verify_commit_executions,
      "each transaction must return exactly one compact host result");
  Require(stats.state_device_bytes > 0, "state arenas were not reported");
  Require(
      stats.device_to_host_operations <
          stats.target_prefill_executions +
              stats.target_prefill_head_executions +
              stats.target_decode1_executions +
              stats.draft_propose_executions +
              stats.target_verify_commit_executions,
      "Draft introduced an extra host-visible result copy");
  Require(executor.model_memory().size() == 5, "model memory set differs");
}

void TestExplicitDecodeOverride(
    const std::array<std::filesystem::path, 5>& model_paths) {
  qwen35::dflash::IncrementalOmPaths paths{
      model_paths[0], model_paths[1], model_paths[2], model_paths[3],
      model_paths[4]};
  qwen35::dflash::AclIncrementalExecutor executor(
      std::move(paths),
      0,
      {},
      qwen35::dflash::IncrementalStateResetPolicy::kAsyncMemset);

  executor.Reset(0, {});
  const auto prefill = executor.PrefillChunk({10}, false, 0);
  Require(
      prefill.token_ids == std::vector<std::int64_t>({11}),
      "override test prefill token differs");

  const auto before_hit = executor.execution_stats();
  const auto carrier_decode = executor.DecodeOne(11);
  Require(
      carrier_decode.token_ids == std::vector<std::int64_t>({12}),
      "device-carried decode token differs");
  const auto after_hit = executor.execution_stats();
  Require(
      after_hit.decode_id_device_carrier_hits ==
              before_hit.decode_id_device_carrier_hits + 1 &&
          after_hit.decode_id_upload_operations ==
              before_hit.decode_id_upload_operations,
      "matching decode ID did not use the device carrier");

  const auto override_decode = executor.DecodeOne(99);
  Require(
      override_decode.token_ids == std::vector<std::int64_t>({100}),
      "explicit decode override was not honored");
  const auto& after_override = executor.execution_stats();
  Require(
      after_override.decode_id_device_carrier_hits ==
              after_hit.decode_id_device_carrier_hits &&
          after_override.decode_id_upload_operations ==
              after_hit.decode_id_upload_operations + 1 &&
          after_override.decode_id_upload_bytes ==
              after_hit.decode_id_upload_bytes + sizeof(std::int64_t),
      "explicit decode override did not use the exact H2D fallback");
}

void TestUnifiedTargetStep(
    const std::array<std::filesystem::path, 4>& model_paths) {
  qwen35::dflash::IncrementalOmPaths paths{
      model_paths[0], model_paths[1], {}, model_paths[2], model_paths[3]};
  std::vector<std::string> loaded_roles;
  qwen35::dflash::AclIncrementalExecutor executor(
      std::move(paths),
      0,
      [&](const char* role, const char* stage, std::size_t, std::size_t) {
        if (std::string(stage) == "load-done") {
          loaded_roles.emplace_back(role);
        }
      },
      qwen35::dflash::IncrementalStateResetPolicy::kAsyncMemset,
      qwen35::dflash::IncrementalDecodeCarrierPolicy::kLastTokenDeviceCompact,
      true);
  Require(executor.unified_target_step(), "unified Target-step was not selected");
  Require(loaded_roles.size() == 4, "unified route did not load four OMs");
  Require(executor.model_memory().size() == 4, "unified memory set differs");
  const auto& memory = executor.model_memory();
  Require(
      memory[0].model_id != memory[1].model_id &&
          memory[0].model_id != memory[2].model_id &&
          memory[0].model_id != memory[3].model_id &&
          memory[1].model_id != memory[2].model_id &&
          memory[1].model_id != memory[3].model_id &&
          memory[2].model_id != memory[3].model_id,
      "unified runtime model IDs are not unique");

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
  Require(ordinary.generated_token_ids == expected, "unified ordinary differs");
  Require(dflash.generated_token_ids == expected, "unified DFlash differs");
  const auto& stats = executor.execution_stats();
  Require(
      stats.target_step_dynamic_gear_count == 16,
      "unified Target-step gears differ");
  Require(
      stats.draft_dynamic_gear_count == 18 &&
          stats.draft_verify_dynamic_gear_count == 16 &&
          stats.draft_prefill_dynamic_gear_count == 2,
      "unified Draft dynamic gears differ");
  Require(
      stats.target_step_zero_count_device_bytes == sizeof(std::int32_t) &&
          stats.target_step_zero_count_bindings ==
              stats.target_decode1_executions,
      "unified Target-step resident zero-count bindings differ");
  Require(
      stats.prefill_control_bytes_per_slot == 960 &&
          stats.prefill_persistent_control_tail_bytes_per_slot == 252 &&
          stats.prefill_staging_pinned_host_bytes == 1920,
      "unified Target-step control carrier layout differs");
  Require(
      stats.target_step_input_rows + stats.target_step_padded_rows_elided ==
          16 * (stats.target_decode1_executions +
                stats.target_verify_commit_executions) &&
          stats.target_step_padded_rows_elided > 0,
      "unified Target-step physical/scratch rows do not close");
  const auto& trace = executor.model_execution_trace();
  const std::size_t model_executions =
      stats.target_prefill_executions +
      stats.target_prefill_head_executions +
      stats.target_decode1_executions + stats.draft_propose_executions +
      stats.target_verify_commit_executions;
  Require(
      trace.size() == model_executions,
      "unified profile execution trace does not close");
  std::size_t target_step_t1 = 0;
  for (std::size_t index = 0; index < trace.size(); ++index) {
    Require(trace[index].ordinal == index, "profile trace ordinal differs");
    if (trace[index].model_id == memory[3].model_id &&
        trace[index].physical_rows == 1) {
      ++target_step_t1;
    }
  }
  Require(
      target_step_t1 == stats.target_decode1_executions,
      "unified profile trace T1 count differs");
}

void TestCommittedDraftFeaturePrefix(
    const std::array<std::filesystem::path, 5>& model_paths,
    std::size_t sync_window) {
  qwen35::dflash::IncrementalOmPaths paths{
      model_paths[0], model_paths[1], model_paths[2], model_paths[3],
      model_paths[4]};
  qwen35::dflash::AclIncrementalExecutor executor(
      std::move(paths),
      0,
      {},
      qwen35::dflash::IncrementalStateResetPolicy::kAsyncMemset,
      qwen35::dflash::IncrementalDecodeCarrierPolicy::kLastTokenDeviceCompact,
      false,
      qwen35::dflash::IncrementalDraftFeaturePolicy::kCommittedPrefix);
  Require(
      executor.draft_feature_policy() ==
          qwen35::dflash::IncrementalDraftFeaturePolicy::kCommittedPrefix,
      "committed-prefix Draft feature policy was not selected");

  qwen35::dflash::GenerationOptions options;
  options.max_new_tokens = sync_window > 2 ? 30 : 10;
  options.max_draft_tokens = sync_window > 2 ? 7 : 3;
  options.dflash_sync_window = sync_window;
  const auto generated = qwen35::dflash::GenerateStatefulOnce(
      executor,
      {10},
      qwen35::dflash::GenerationMode::kDFlash,
      options);
  std::vector<std::int64_t> expected;
  expected.reserve(options.max_new_tokens);
  for (std::size_t index = 0; index < options.max_new_tokens; ++index) {
    expected.push_back(11 + static_cast<std::int64_t>(index));
  }
  Require(
      generated.generated_token_ids == expected,
      "committed-prefix Draft route changed exact generated tokens");

  const auto& stats = executor.execution_stats();
  Require(
      stats.draft_propose_executions >=
          stats.prefill_draft_propose_executions,
      "verify-source Draft execution count underflowed");
  const std::size_t verify_draft_executions =
      stats.draft_propose_executions -
      stats.prefill_draft_propose_executions;
  Require(verify_draft_executions > 0, "committed-prefix route was not exercised");
  Require(
      stats.draft_dynamic_gear_count == 18 &&
          stats.draft_verify_dynamic_gear_count == 16 &&
          stats.draft_prefill_dynamic_gear_count == 2 &&
          stats.draft_verify_fixed_width_executions == 0 &&
          stats.draft_verify_committed_prefix_executions +
                  stats.draft_verify_pending_upper_bound_executions ==
              verify_draft_executions &&
          stats.draft_verify_full_width_equivalent_rows ==
              16 * verify_draft_executions &&
          stats.draft_verify_feature_input_rows +
                  stats.draft_verify_feature_rows_elided ==
              stats.draft_verify_full_width_equivalent_rows &&
          stats.draft_verify_feature_rows_elided > 0,
      "committed-prefix Draft row counters do not close");
  if (sync_window == 1) {
    Require(
        stats.draft_verify_committed_prefix_executions ==
                verify_draft_executions &&
            stats.draft_verify_pending_upper_bound_executions == 0,
        "window-one Draft did not use the host-visible committed prefix");
  } else {
    Require(
        stats.draft_verify_pending_upper_bound_executions > 0,
        "multi-transaction Draft did not use its causal pending upper bound");
  }
  if (sync_window > 2) {
    Require(
        stats.speculative_window_staging_operations > 0 &&
            stats.speculative_window_staging_bytes ==
                stats.speculative_window_staging_operations *
                    stats.compact_verify_result_bytes,
        "extended committed-prefix window did not stage compact results");
  }
}

void TestPrefillVerifyCoalescing(
    const std::array<std::filesystem::path, 5>& model_paths) {
  qwen35::dflash::IncrementalOmPaths paths{
      model_paths[0], model_paths[1], model_paths[2], model_paths[3],
      model_paths[4]};
  qwen35::dflash::AclIncrementalExecutor executor(std::move(paths));
  Require(
      executor.supports_prefill_verify_coalescing(),
      "ACL executor did not advertise prefill/verify coalescing");

  qwen35::dflash::GenerationOptions options;
  options.max_new_tokens = 6;
  options.max_draft_tokens = 3;
  options.coalesce_prefill_with_first_verify = true;
  const auto generated = qwen35::dflash::GenerateStatefulOnce(
      executor,
      {10},
      qwen35::dflash::GenerationMode::kDFlash,
      options);
  Require(
      generated.generated_token_ids ==
          std::vector<std::int64_t>({11, 12, 13, 14, 15, 16}),
      "prefill/verify coalescing changed exact generated tokens");
  Require(
      generated.counters.prefill_speculative_windows == 1 &&
          generated.counters.speculative_transactions == 1 &&
          generated.counters.decode_iterations == 1,
      "prefill/verify generation counters differ");

  const auto& stats = executor.execution_stats();
  Require(
      stats.prefill_verify_coalesced_windows == 1 &&
          stats.prefill_verify_synchronizations_elided == 1 &&
          stats.prefill_verify_d2h_operations_elided == 1 &&
          stats.prefill_verify_prefill_slot0_windows == 0 &&
          stats.prefill_verify_prefill_slot1_windows == 1,
      "prefill/verify coalescing event counters differ");
  Require(
      stats.prefill_verify_d2h_padding_bytes ==
          stats.compact_slot_bytes - stats.compact_verify_result_bytes,
      "prefill/verify coalesced D2H span differs");
  Require(
      stats.stream_synchronizations ==
          stats.prefill_completion_synchronizations +
              stats.target_decode1_executions +
              stats.speculative_sync_windows,
      "prefill/verify coalescing did not remove exactly one barrier");
  Require(
      stats.stream_synchronizations +
              stats.speculative_synchronizations_elided +
              stats.prefill_verify_synchronizations_elided ==
          stats.prefill_completion_synchronizations +
              stats.target_decode1_executions +
              stats.target_verify_commit_executions,
      "prefill/verify synchronization accounting does not close");
  Require(
      stats.device_to_host_operations +
              stats.speculative_d2h_operations_elided +
              stats.prefill_verify_d2h_operations_elided ==
          stats.prefill_completion_synchronizations +
              stats.target_decode1_executions +
              stats.target_verify_commit_executions,
      "prefill/verify D2H accounting does not close");
}

void TestExtendedSpeculativeWindow(
    const std::array<std::filesystem::path, 5>& model_paths) {
  qwen35::dflash::IncrementalOmPaths paths{
      model_paths[0], model_paths[1], model_paths[2], model_paths[3],
      model_paths[4]};
  qwen35::dflash::AclIncrementalExecutor executor(std::move(paths));
  qwen35::dflash::GenerationOptions options;
  options.max_new_tokens = 58;
  options.max_draft_tokens = 15;
  options.dflash_sync_window = 8;
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
  Require(
      ordinary.generated_token_ids == dflash.generated_token_ids &&
          dflash.generated_token_ids.size() == options.max_new_tokens,
      "extended speculative window changed exact generated tokens");
  const auto& stats = executor.execution_stats();
  Require(
      stats.speculative_sync_windows == 1 &&
          stats.speculative_synchronizations_elided == 3 &&
          stats.speculative_d2h_operations_elided == 3 &&
          stats.speculative_window_staging_operations == 4 &&
          stats.speculative_window_staging_bytes ==
              4 * stats.compact_verify_result_bytes,
      "extended speculative window did not close four transactions in one barrier");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 7) {
      throw std::invalid_argument("expected five baseline and one dynamic fake OM paths");
    }
    const std::array<std::filesystem::path, 5> paths{
        std::filesystem::path(argv[1]),
        std::filesystem::path(argv[2]),
        std::filesystem::path(argv[3]),
        std::filesystem::path(argv[4]),
        std::filesystem::path(argv[5]),
    };
    RunPolicy(
        paths,
        qwen35::dflash::IncrementalStateResetPolicy::kAsyncMemset,
        qwen35::dflash::IncrementalDecodeCarrierPolicy::
            kLastTokenDeviceCompact);
    RunPolicy(
        paths,
        qwen35::dflash::IncrementalStateResetPolicy::kImmutableZero,
        qwen35::dflash::IncrementalDecodeCarrierPolicy::
            kLastTokenDeviceCompact);
    RunPolicy(
        paths,
        qwen35::dflash::IncrementalStateResetPolicy::kAsyncMemset,
        qwen35::dflash::IncrementalDecodeCarrierPolicy::
            kOneTokenHostFallback);
    RunPolicy(
        paths,
        qwen35::dflash::IncrementalStateResetPolicy::kImmutableZero,
        qwen35::dflash::IncrementalDecodeCarrierPolicy::
            kOneTokenHostFallback);
    TestExplicitDecodeOverride(paths);
    TestCommittedDraftFeaturePrefix(paths, 1);
    TestCommittedDraftFeaturePrefix(paths, 2);
    TestCommittedDraftFeaturePrefix(paths, 8);
    TestPrefillVerifyCoalescing(paths);
    TestExtendedSpeculativeWindow(paths);
    TestUnifiedTargetStep(
        {paths[0], paths[1], paths[3], std::filesystem::path(argv[6])});
    std::cout << "PASS: reset, decode carrier and committed-prefix Draft "
                 "feature policies preserve exact tokens, device state "
                 "routing, compact synchronization and override fallback\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
