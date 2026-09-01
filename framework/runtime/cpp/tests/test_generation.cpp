#include "qwen35_dflash/generation.hpp"
#include "qwen35_dflash/sha256.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class FakeExecutor final : public qwen35::dflash::GraphExecutor {
 public:
  explicit FakeExecutor(bool corrupt_second_proposal = false)
      : corrupt_second_proposal_(corrupt_second_proposal) {
    outputs_.target_top1.resize(sequence_length_);
    outputs_.draft_top1.resize(draft_width_);
  }

  std::size_t sequence_length() const noexcept override {
    return sequence_length_;
  }

  std::size_t draft_width() const noexcept override { return draft_width_; }

  const qwen35::dflash::GraphOutputs& Execute(
      const std::vector<std::int64_t>& prefix,
      std::int64_t) override {
    std::fill(outputs_.target_top1.begin(), outputs_.target_top1.end(), 0);
    for (std::size_t index = 0; index < prefix.size(); ++index) {
      outputs_.target_top1[index] = prefix[index] + 1;
    }
    for (std::size_t index = 0; index < draft_width_; ++index) {
      outputs_.draft_top1[index] =
          prefix.back() + static_cast<std::int64_t>(index) + 1;
    }
    if (corrupt_second_proposal_ && draft_width_ > 1) {
      outputs_.draft_top1[1] += 100;
    }
    return outputs_;
  }

 private:
  static constexpr std::size_t sequence_length_ = 32;
  static constexpr std::size_t draft_width_ = 3;
  bool corrupt_second_proposal_ = false;
  qwen35::dflash::GraphOutputs outputs_;
};

class FakeStatefulExecutor final
    : public qwen35::dflash::StatefulGraphExecutor {
 public:
  explicit FakeStatefulExecutor(
      bool corrupt_second_proposal = false,
      bool corrupt_first_proposal = false)
      : corrupt_second_proposal_(corrupt_second_proposal),
        corrupt_first_proposal_(corrupt_first_proposal) {}

  std::size_t sequence_length() const noexcept override { return 256; }
  std::size_t prefill_width() const noexcept override { return 4; }
  std::size_t proposal_width() const noexcept override { return 15; }
  std::size_t eos_table_width() const noexcept override { return 4; }

  void Reset(
      std::int64_t pad_token_id,
      const std::vector<std::int64_t>& eos_token_ids) override {
    if (pad_token_id < 0 || eos_token_ids.size() > eos_table_width()) {
      throw std::invalid_argument("invalid fake state reset");
    }
    eos_ = eos_token_ids;
    anchor_ = 0;
    prepared_ = false;
    prepared_count_ = 0;
    speculative_windows_.clear();
    prefill_verify_windows_.clear();
  }

  qwen35::dflash::StatefulStep PrefillChunk(
      const std::vector<std::int64_t>& token_ids,
      bool prepare_draft,
      std::size_t logical_proposal_count) override {
    if (token_ids.empty() || token_ids.size() > prefill_width()) {
      throw std::invalid_argument("invalid fake prefill chunk");
    }
    anchor_ = token_ids.back() + 1;
    if (prepare_draft) {
      Prepare(logical_proposal_count);
    } else if (logical_proposal_count != 0) {
      throw std::invalid_argument("proposal count without Draft preparation");
    }
    return qwen35::dflash::StatefulStep{
        {anchor_},
        prepare_draft ? 2U : 1U,
        0,
        0,
        0,
        IsEos(anchor_),
    };
  }

  qwen35::dflash::StatefulStep DecodeOne(
      std::int64_t input_token_id) override {
    if (input_token_id < 0) {
      throw std::invalid_argument("negative fake decode input");
    }
    anchor_ = input_token_id + 1;
    prepared_ = false;
    return qwen35::dflash::StatefulStep{
        {anchor_}, 1, 0, 0, 0, IsEos(anchor_)};
  }

  qwen35::dflash::StatefulStep SpeculativeStep(
      std::size_t logical_proposal_count) override {
    if (logical_proposal_count == 0 ||
        logical_proposal_count > proposal_width()) {
      throw std::invalid_argument("invalid fake proposal count");
    }
    std::size_t executions = 1;
    if (!prepared_) {
      Prepare(logical_proposal_count);
      executions = 2;
    } else if (prepared_count_ != logical_proposal_count) {
      throw std::runtime_error("prepared fake proposal count changed");
    }

    std::size_t drafted = prepared_proposals_.size();
    std::size_t accepted = 0;
    while (accepted < drafted &&
           prepared_proposals_[accepted] ==
               anchor_ + static_cast<std::int64_t>(accepted) + 1) {
      ++accepted;
    }
    std::vector<std::int64_t> committed(
        prepared_proposals_.begin(),
        prepared_proposals_.begin() + static_cast<std::ptrdiff_t>(accepted));
    const bool accepted_eos =
        !committed.empty() && IsEos(committed.back());
    if (!accepted_eos) {
      committed.push_back(anchor_ + static_cast<std::int64_t>(accepted) + 1);
    }
    anchor_ = committed.back();
    prepared_ = false;
    return qwen35::dflash::StatefulStep{
        committed,
        executions,
        drafted,
        accepted,
        drafted - accepted,
        IsEos(committed.back()),
    };
  }

  bool supports_prefill_verify_coalescing() const noexcept override {
    return true;
  }

  std::vector<qwen35::dflash::StatefulStep>
  PrefillChunkAndSpeculative(
      const std::vector<std::int64_t>& token_ids,
      std::size_t logical_proposal_count) override {
    prefill_verify_windows_.push_back(logical_proposal_count);
    std::vector<qwen35::dflash::StatefulStep> result;
    result.reserve(2);
    result.push_back(PrefillChunk(
        token_ids, true, logical_proposal_count));
    result.push_back(SpeculativeStep(logical_proposal_count));
    return result;
  }

  std::size_t max_speculative_sync_window() const noexcept override {
    return 8;
  }

  std::vector<qwen35::dflash::StatefulStep> SpeculativeWindow(
      const std::vector<std::size_t>& logical_proposal_counts) override {
    if (logical_proposal_counts.empty() ||
        logical_proposal_counts.size() > max_speculative_sync_window()) {
      throw std::invalid_argument("invalid fake speculative window");
    }
    speculative_windows_.push_back(logical_proposal_counts);
    std::vector<qwen35::dflash::StatefulStep> result;
    result.reserve(logical_proposal_counts.size());
    for (const std::size_t count : logical_proposal_counts) {
      result.push_back(SpeculativeStep(count));
    }
    return result;
  }

  const std::vector<std::vector<std::size_t>>& speculative_windows()
      const noexcept {
    return speculative_windows_;
  }

  const std::vector<std::size_t>& prefill_verify_windows() const noexcept {
    return prefill_verify_windows_;
  }

 private:
  bool IsEos(std::int64_t token) const {
    return std::find(eos_.begin(), eos_.end(), token) != eos_.end();
  }

  void Prepare(std::size_t logical_proposal_count) {
    if (logical_proposal_count == 0 ||
        logical_proposal_count > proposal_width()) {
      throw std::invalid_argument("invalid fake Draft preparation count");
    }
    prepared_proposals_.clear();
    for (std::size_t index = 0; index < logical_proposal_count; ++index) {
      std::int64_t token =
          anchor_ + static_cast<std::int64_t>(index) + 1;
      if (corrupt_first_proposal_ && index == 0) {
        token += 100;
      }
      if (corrupt_second_proposal_ && index == 1) {
        token += 100;
      }
      prepared_proposals_.push_back(token);
      if (IsEos(token)) {
        break;
      }
    }
    prepared_count_ = logical_proposal_count;
    prepared_ = true;
  }

  bool corrupt_second_proposal_ = false;
  bool corrupt_first_proposal_ = false;
  std::vector<std::int64_t> eos_;
  std::int64_t anchor_ = 0;
  bool prepared_ = false;
  std::size_t prepared_count_ = 0;
  std::vector<std::int64_t> prepared_proposals_;
  std::vector<std::vector<std::size_t>> speculative_windows_;
  std::vector<std::size_t> prefill_verify_windows_;
};

void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Callable>
void RequireThrows(Callable&& callable, const std::string& message) {
  try {
    callable();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error(message);
}

qwen35::dflash::GenerationOptions Options() {
  qwen35::dflash::GenerationOptions options;
  options.pad_token_id = 0;
  options.max_new_tokens = 6;
  options.max_draft_tokens = 3;
  return options;
}

void TestOrdinaryAndFullAcceptanceMatch() {
  FakeExecutor executor;
  const auto ordinary = qwen35::dflash::GenerateOnce(
      executor, {10}, qwen35::dflash::GenerationMode::kOrdinary, Options());
  const auto dflash = qwen35::dflash::GenerateOnce(
      executor, {10}, qwen35::dflash::GenerationMode::kDFlash, Options());
  const std::vector<std::int64_t> expected{11, 12, 13, 14, 15, 16};
  Require(ordinary.generated_token_ids == expected, "ordinary tokens differ");
  Require(dflash.generated_token_ids == expected, "DFlash tokens differ");
  Require(dflash.counters.accepted_draft_tokens > 0, "no draft was accepted");
  Require(
      dflash.counters.rejected_draft_tokens == 0,
      "fully matching draft was rejected");
}

void TestCorrectionPreservesAuthority() {
  FakeExecutor executor(true);
  const auto ordinary = qwen35::dflash::GenerateOnce(
      executor, {10}, qwen35::dflash::GenerationMode::kOrdinary, Options());
  const auto dflash = qwen35::dflash::GenerateOnce(
      executor, {10}, qwen35::dflash::GenerationMode::kDFlash, Options());
  Require(
      ordinary.generated_token_ids == dflash.generated_token_ids,
      "correction did not preserve ordinary output");
  Require(dflash.counters.accepted_draft_tokens > 0, "first proposal was not accepted");
  Require(dflash.counters.rejected_draft_tokens > 0, "bad proposal was not rejected");
}

void TestEosStopsBothModesAtSameToken() {
  FakeExecutor executor;
  auto options = Options();
  options.eos_token_ids = {13};
  const auto ordinary = qwen35::dflash::GenerateOnce(
      executor, {10}, qwen35::dflash::GenerationMode::kOrdinary, options);
  const auto dflash = qwen35::dflash::GenerateOnce(
      executor, {10}, qwen35::dflash::GenerationMode::kDFlash, options);
  const std::vector<std::int64_t> expected{11, 12, 13};
  Require(ordinary.generated_token_ids == expected, "ordinary EOS tokens differ");
  Require(dflash.generated_token_ids == expected, "DFlash EOS tokens differ");
  Require(ordinary.stop_reason == "eos", "ordinary stop reason is not EOS");
  Require(dflash.stop_reason == "eos", "DFlash stop reason is not EOS");
}

void TestCapacityGate() {
  FakeExecutor executor;
  auto options = Options();
  options.max_new_tokens = 3;
  const std::vector<std::int64_t> prompt(31, 1);
  RequireThrows(
      [&] {
        static_cast<void>(qwen35::dflash::GenerateOnce(
            executor,
            prompt,
            qwen35::dflash::GenerationMode::kOrdinary,
            options));
      },
      "capacity overflow was accepted");
}

void TestPairedBenchmarkIsStableAndExact() {
  FakeExecutor executor;
  const auto result =
      qwen35::dflash::BenchmarkPair(executor, {10}, Options(), 1, 3);
  Require(result.token_id_mismatches == 0, "paired token mismatch");
  Require(result.eos_mismatches == 0, "paired EOS mismatch");
  Require(result.ordinary.repetitions == 3, "ordinary repetition count differs");
  Require(result.dflash.repetitions == 3, "DFlash repetition count differs");
  Require(
      result.ordinary.stable_generated_token_ids ==
          result.dflash.stable_generated_token_ids,
      "paired stable tokens differ");
}

void TestProgressReportsPhaseAndTokenMovement() {
  FakeExecutor executor;
  std::vector<std::string> stages;
  bool saw_warmup = false;
  bool saw_measurement = false;
  bool saw_generated_token = false;
  const auto progress = [&](const qwen35::dflash::ProgressEvent& event) {
    stages.emplace_back(event.stage);
    saw_warmup = saw_warmup || std::string(event.phase) == "warmup";
    saw_measurement =
        saw_measurement || std::string(event.phase) == "measurement";
    saw_generated_token =
        saw_generated_token || event.generated_tokens > 0;
  };
  static_cast<void>(qwen35::dflash::BenchmarkPair(
      executor, {10}, Options(), 1, 2, progress));
  Require(saw_warmup, "progress omitted the warmup phase");
  Require(saw_measurement, "progress omitted the measurement phase");
  Require(saw_generated_token, "progress never reported generated tokens");
  Require(
      std::find(stages.begin(), stages.end(), "prefill-start") != stages.end(),
      "progress omitted prefill start");
  Require(
      std::find(stages.begin(), stages.end(), "decode-done") != stages.end(),
      "progress omitted decode completion");
  Require(
      std::find(stages.begin(), stages.end(), "run-done") != stages.end(),
      "progress omitted run completion");
}

void TestStatefulOrdinaryAndDFlashMatchAcrossPromptChunks() {
  FakeStatefulExecutor executor;
  auto options = Options();
  options.max_new_tokens = 9;
  const std::vector<std::int64_t> prompt{1, 2, 3, 4, 5, 6, 7};
  const auto ordinary = qwen35::dflash::GenerateStatefulOnce(
      executor, prompt, qwen35::dflash::GenerationMode::kOrdinary, options);
  const auto dflash = qwen35::dflash::GenerateStatefulOnce(
      executor, prompt, qwen35::dflash::GenerationMode::kDFlash, options);
  const std::vector<std::int64_t> expected{8, 9, 10, 11, 12, 13, 14, 15, 16};
  Require(ordinary.generated_token_ids == expected, "stateful ordinary differs");
  Require(dflash.generated_token_ids == expected, "stateful DFlash differs");
  Require(dflash.counters.accepted_draft_tokens > 0, "stateful Draft unused");
  Require(
      dflash.counters.graph_calls < ordinary.counters.graph_calls,
      "stateful DFlash did not reduce OM calls under full acceptance");
}

void TestStatefulCorrectionAndEosRemainExact() {
  FakeStatefulExecutor executor(true);
  auto options = Options();
  options.max_new_tokens = 8;
  options.eos_token_ids = {16};
  const auto ordinary = qwen35::dflash::GenerateStatefulOnce(
      executor, {10}, qwen35::dflash::GenerationMode::kOrdinary, options);
  const auto dflash = qwen35::dflash::GenerateStatefulOnce(
      executor, {10}, qwen35::dflash::GenerationMode::kDFlash, options);
  Require(
      ordinary.generated_token_ids == dflash.generated_token_ids,
      "stateful correction changed the authoritative tokens");
  Require(ordinary.stop_reason == "eos", "stateful ordinary missed EOS");
  Require(dflash.stop_reason == "eos", "stateful DFlash missed EOS");
  Require(
      dflash.counters.rejected_draft_tokens > 0,
      "stateful bad proposal was not rejected");
}

void TestStatefulPairedBenchmarkAndProgress() {
  FakeStatefulExecutor executor;
  bool saw_reset = false;
  bool saw_decode = false;
  const auto progress = [&](const qwen35::dflash::ProgressEvent& event) {
    saw_reset = saw_reset || std::string(event.stage) == "state-reset-staged";
    saw_decode = saw_decode || std::string(event.stage) == "decode-done";
  };
  const auto result = qwen35::dflash::BenchmarkPairStateful(
      executor, {10}, Options(), 1, 3, progress);
  Require(result.token_id_mismatches == 0, "stateful paired token mismatch");
  Require(result.eos_mismatches == 0, "stateful paired EOS mismatch");
  Require(saw_reset, "stateful progress omitted reset");
  Require(saw_decode, "stateful progress omitted decode");
}

void TestTwoTransactionWindowUsesBudgetSafeSecondProposalCount() {
  FakeStatefulExecutor executor;
  auto options = Options();
  options.max_new_tokens = 32;
  options.max_draft_tokens = 15;
  options.dflash_sync_window = 2;
  const auto ordinary = qwen35::dflash::GenerateStatefulOnce(
      executor, {10}, qwen35::dflash::GenerationMode::kOrdinary, options);
  const auto dflash = qwen35::dflash::GenerateStatefulOnce(
      executor, {10}, qwen35::dflash::GenerationMode::kDFlash, options);
  Require(
      ordinary.generated_token_ids == dflash.generated_token_ids,
      "two-transaction window changed authoritative tokens");
  Require(
      executor.speculative_windows().size() == 1 &&
          executor.speculative_windows().front() ==
              std::vector<std::size_t>({15, 14}),
      "two-transaction window did not use the budget-safe K=15/K=14 pair");
  Require(
      dflash.counters.speculative_transactions == 2 &&
          dflash.counters.decode_iterations == 1,
      "two-transaction window counters differ");
}

void TestTwoTransactionWindowStopsAtFirstTransactionEos() {
  FakeStatefulExecutor executor;
  auto options = Options();
  options.max_new_tokens = 8;
  options.max_draft_tokens = 3;
  options.dflash_sync_window = 2;
  options.eos_token_ids = {13};
  const auto ordinary = qwen35::dflash::GenerateStatefulOnce(
      executor, {10}, qwen35::dflash::GenerationMode::kOrdinary, options);
  const auto dflash = qwen35::dflash::GenerateStatefulOnce(
      executor, {10}, qwen35::dflash::GenerationMode::kDFlash, options);
  const std::vector<std::int64_t> expected{11, 12, 13};
  Require(
      ordinary.generated_token_ids == expected &&
          dflash.generated_token_ids == expected,
      "two-transaction window committed output after first-transaction EOS");
  Require(
      ordinary.stop_reason == "eos" && dflash.stop_reason == "eos",
      "two-transaction window changed EOS stop reason");
  Require(
      executor.speculative_windows().size() == 1 &&
          executor.speculative_windows().front() ==
              std::vector<std::size_t>({3, 2}),
      "EOS case did not exercise a queued two-transaction window");
  Require(
      dflash.counters.speculative_transactions == 2 &&
          dflash.counters.decode_iterations == 1,
      "EOS case did not account for the queued second transaction");
}

void TestEightTransactionWindowUsesBudgetSafeProposalCounts() {
  FakeStatefulExecutor executor;
  auto options = Options();
  options.max_new_tokens = 128;
  options.max_draft_tokens = 15;
  options.dflash_sync_window = 8;
  const auto ordinary = qwen35::dflash::GenerateStatefulOnce(
      executor, {10}, qwen35::dflash::GenerationMode::kOrdinary, options);
  const auto dflash = qwen35::dflash::GenerateStatefulOnce(
      executor, {10}, qwen35::dflash::GenerationMode::kDFlash, options);
  Require(
      ordinary.generated_token_ids == dflash.generated_token_ids,
      "eight-transaction window changed authoritative tokens");
  Require(
      executor.speculative_windows().size() == 1 &&
          executor.speculative_windows().front() ==
              std::vector<std::size_t>(
                  {15, 15, 15, 15, 15, 15, 15, 14}),
      "eight-transaction window did not reserve the exact worst-case budget");
  Require(
      dflash.counters.speculative_transactions == 8 &&
          dflash.counters.decode_iterations == 1,
      "eight-transaction window counters differ");
}

void TestEightTransactionWindowStopsAtFirstTransactionEos() {
  FakeStatefulExecutor executor;
  auto options = Options();
  options.max_new_tokens = 32;
  options.max_draft_tokens = 3;
  options.dflash_sync_window = 8;
  options.eos_token_ids = {13};
  const auto ordinary = qwen35::dflash::GenerateStatefulOnce(
      executor, {10}, qwen35::dflash::GenerationMode::kOrdinary, options);
  const auto dflash = qwen35::dflash::GenerateStatefulOnce(
      executor, {10}, qwen35::dflash::GenerationMode::kDFlash, options);
  const std::vector<std::int64_t> expected{11, 12, 13};
  Require(
      ordinary.generated_token_ids == expected &&
          dflash.generated_token_ids == expected,
      "eight-transaction window committed output after first-window EOS");
  Require(
      executor.speculative_windows().size() == 1 &&
          executor.speculative_windows().front() ==
              std::vector<std::size_t>({3, 3, 3, 3, 3, 3, 3, 2}),
      "EOS case did not queue the budget-safe eight-transaction window");
  Require(
      dflash.counters.speculative_transactions == 8 &&
          dflash.counters.decode_iterations == 1,
      "EOS case did not account for all queued transactions");
}

void TestSmallGenerationBudgetsPrepareOnlyBudgetSafeDraft() {
  FakeStatefulExecutor executor;
  auto options = Options();
  options.max_draft_tokens = 15;

  options.max_new_tokens = 2;
  const auto two = qwen35::dflash::GenerateStatefulOnce(
      executor, {10}, qwen35::dflash::GenerationMode::kDFlash, options);
  Require(
      two.generated_token_ids == std::vector<std::int64_t>({11, 12}),
      "two-token budget changed authoritative output");
  Require(
      two.counters.speculative_transactions == 0 &&
          two.counters.drafted_tokens == 0,
      "two-token budget executed an unusable Draft");

  options.max_new_tokens = 3;
  const auto three = qwen35::dflash::GenerateStatefulOnce(
      executor, {10}, qwen35::dflash::GenerationMode::kDFlash, options);
  Require(
      three.generated_token_ids ==
          std::vector<std::int64_t>({11, 12, 13}),
      "three-token budget changed authoritative output");
  Require(
      three.counters.speculative_transactions == 1 &&
          three.counters.drafted_tokens == 1,
      "three-token budget did not use exact K=1");

  options.max_new_tokens = 4;
  const auto four = qwen35::dflash::GenerateStatefulOnce(
      executor, {10}, qwen35::dflash::GenerationMode::kDFlash, options);
  Require(
      four.generated_token_ids ==
          std::vector<std::int64_t>({11, 12, 13, 14}),
      "four-token budget changed authoritative output");
  Require(
      four.counters.speculative_transactions == 1 &&
          four.counters.drafted_tokens == 2,
      "four-token budget did not use exact K=2");
}

void TestPrefillFirstVerifyCoalescingIsExactAndAccounted() {
  FakeStatefulExecutor executor;
  auto options = Options();
  options.max_new_tokens = 9;
  options.max_draft_tokens = 3;
  options.coalesce_prefill_with_first_verify = true;
  const auto ordinary = qwen35::dflash::GenerateStatefulOnce(
      executor, {10}, qwen35::dflash::GenerationMode::kOrdinary, options);
  const auto dflash = qwen35::dflash::GenerateStatefulOnce(
      executor, {10}, qwen35::dflash::GenerationMode::kDFlash, options);
  Require(
      ordinary.generated_token_ids == dflash.generated_token_ids,
      "prefill/verify coalescing changed authoritative output");
  Require(
      executor.prefill_verify_windows() == std::vector<std::size_t>({3}),
      "prefill/verify coalescing did not use the prepared K=3");
  Require(
      dflash.counters.prefill_speculative_windows == 1 &&
          dflash.counters.speculative_transactions == 2 &&
          dflash.counters.decode_iterations == 1,
      "prefill/verify coalescing counters differ");
}

void TestPrefillFirstVerifyCoalescingDoesNotCommitAfterPrefillEos() {
  FakeStatefulExecutor executor;
  auto options = Options();
  options.max_new_tokens = 8;
  options.max_draft_tokens = 3;
  options.coalesce_prefill_with_first_verify = true;
  options.eos_token_ids = {11};
  const auto dflash = qwen35::dflash::GenerateStatefulOnce(
      executor, {10}, qwen35::dflash::GenerationMode::kDFlash, options);
  Require(
      dflash.generated_token_ids == std::vector<std::int64_t>({11}) &&
          dflash.stop_reason == "eos",
      "prefill/verify coalescing committed output after prefill EOS");
  Require(
      dflash.counters.prefill_speculative_windows == 1 &&
          dflash.counters.speculative_transactions == 1 &&
          dflash.counters.decode_iterations == 0,
      "prefill EOS did not account for the queued verify transaction");
}

void TestZeroAcceptFallbackSwitchesToExactTargetOnlyGeneration() {
  auto options = Options();
  options.max_new_tokens = 10;
  options.max_draft_tokens = 3;

  FakeStatefulExecutor ordinary_executor(false, true);
  const auto ordinary = qwen35::dflash::GenerateStatefulOnce(
      ordinary_executor,
      {10},
      qwen35::dflash::GenerationMode::kOrdinary,
      options);

  FakeStatefulExecutor disabled_executor(false, true);
  const auto disabled = qwen35::dflash::GenerateStatefulOnce(
      disabled_executor,
      {10},
      qwen35::dflash::GenerationMode::kDFlash,
      options);

  options.zero_accept_fallback_policy =
      qwen35::dflash::ZeroAcceptFallbackPolicy::kRequestTargetOnly;
  FakeStatefulExecutor fallback_executor(false, true);
  const auto fallback = qwen35::dflash::GenerateStatefulOnce(
      fallback_executor,
      {10},
      qwen35::dflash::GenerationMode::kDFlash,
      options);

  Require(
      ordinary.generated_token_ids == disabled.generated_token_ids &&
          ordinary.generated_token_ids == fallback.generated_token_ids,
      "zero-accept fallback changed authoritative tokens");
  Require(
      disabled.counters.zero_accept_transactions > 1 &&
          disabled.counters.zero_accept_fallback_activations == 0 &&
          disabled.counters.target_only_fallback_iterations == 0,
      "disabled zero-accept policy changed the rollback baseline");
  Require(
      fallback.counters.zero_accept_transactions == 1 &&
          fallback.counters.zero_accept_fallback_activations == 1 &&
          fallback.counters.target_only_fallback_iterations > 0,
      "zero-accept policy did not switch the request to Target-only");
  Require(
      fallback.counters.graph_calls < disabled.counters.graph_calls,
      "zero-accept Target-only route did not eliminate OM executions");
}

void TestZeroAcceptFallbackConsumesAQueuedWindowBeforeSwitching() {
  auto options = Options();
  options.max_new_tokens = 16;
  options.max_draft_tokens = 3;
  options.dflash_sync_window = 2;
  options.zero_accept_fallback_policy =
      qwen35::dflash::ZeroAcceptFallbackPolicy::kRequestTargetOnly;
  FakeStatefulExecutor executor(false, true);
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
      ordinary.generated_token_ids == dflash.generated_token_ids,
      "windowed zero-accept fallback changed authoritative tokens");
  Require(
      executor.speculative_windows().size() == 1 &&
          executor.speculative_windows().front() ==
              std::vector<std::size_t>({3, 3}),
      "zero-accept fallback did not consume exactly one queued window");
  Require(
      dflash.counters.speculative_transactions == 2 &&
          dflash.counters.zero_accept_transactions == 2 &&
          dflash.counters.zero_accept_fallback_activations == 1 &&
          dflash.counters.target_only_fallback_iterations > 0,
      "windowed zero-accept fallback counters differ");
}

void TestSha256KnownVector() {
  Require(
      qwen35::dflash::Sha256("abc") ==
          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
      "SHA-256 known vector differs");
}

}  // namespace

int main() {
  try {
    TestOrdinaryAndFullAcceptanceMatch();
    TestCorrectionPreservesAuthority();
    TestEosStopsBothModesAtSameToken();
    TestCapacityGate();
    TestPairedBenchmarkIsStableAndExact();
    TestProgressReportsPhaseAndTokenMovement();
    TestStatefulOrdinaryAndDFlashMatchAcrossPromptChunks();
    TestStatefulCorrectionAndEosRemainExact();
    TestStatefulPairedBenchmarkAndProgress();
    TestTwoTransactionWindowUsesBudgetSafeSecondProposalCount();
    TestTwoTransactionWindowStopsAtFirstTransactionEos();
    TestEightTransactionWindowUsesBudgetSafeProposalCounts();
    TestEightTransactionWindowStopsAtFirstTransactionEos();
    TestSmallGenerationBudgetsPrepareOnlyBudgetSafeDraft();
    TestPrefillFirstVerifyCoalescingIsExactAndAccounted();
    TestPrefillFirstVerifyCoalescingDoesNotCommitAfterPrefillEos();
    TestZeroAcceptFallbackSwitchesToExactTargetOnlyGeneration();
    TestZeroAcceptFallbackConsumesAQueuedWindowBeforeSwitching();
    TestSha256KnownVector();
    std::cout << "PASS: recompute/stateful C++ schedulers, parity, EOS, "
                 "capacity and SHA-256\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
