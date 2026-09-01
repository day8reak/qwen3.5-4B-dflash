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
  explicit FakeStatefulExecutor(bool corrupt_second_proposal = false)
      : corrupt_second_proposal_(corrupt_second_proposal) {}

  std::size_t sequence_length() const noexcept override { return 32; }
  std::size_t prefill_width() const noexcept override { return 4; }
  std::size_t proposal_width() const noexcept override { return 3; }
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
  std::vector<std::int64_t> eos_;
  std::int64_t anchor_ = 0;
  bool prepared_ = false;
  std::size_t prepared_count_ = 0;
  std::vector<std::int64_t> prepared_proposals_;
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
    saw_reset = saw_reset || std::string(event.stage) == "state-reset-done";
    saw_decode = saw_decode || std::string(event.stage) == "decode-done";
  };
  const auto result = qwen35::dflash::BenchmarkPairStateful(
      executor, {10}, Options(), 1, 3, progress);
  Require(result.token_id_mismatches == 0, "stateful paired token mismatch");
  Require(result.eos_mismatches == 0, "stateful paired EOS mismatch");
  Require(saw_reset, "stateful progress omitted reset");
  Require(saw_decode, "stateful progress omitted decode");
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
    TestSha256KnownVector();
    std::cout << "PASS: recompute/stateful C++ schedulers, parity, EOS, "
                 "capacity and SHA-256\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
