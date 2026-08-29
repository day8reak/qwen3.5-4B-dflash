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
    TestSha256KnownVector();
    std::cout << "PASS: C++ scheduler, parity, EOS, capacity and SHA-256\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
