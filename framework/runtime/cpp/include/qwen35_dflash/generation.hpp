#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qwen35::dflash {

enum class GenerationMode {
  kOrdinary,
  kDFlash,
};

struct GraphOutputs {
  std::vector<std::int64_t> target_top1;
  std::vector<std::int64_t> draft_top1;
};

// The implementation owns all input/output storage. Execute must not allocate
// ACL buffers or reload the model; callers invoke it in the decode hot path.
class GraphExecutor {
 public:
  virtual ~GraphExecutor() = default;

  virtual std::size_t sequence_length() const noexcept = 0;
  virtual std::size_t draft_width() const noexcept = 0;
  virtual const GraphOutputs& Execute(
      const std::vector<std::int64_t>& committed_prefix,
      std::int64_t pad_token_id) = 0;
};

struct GenerationOptions {
  std::int64_t pad_token_id = 0;
  std::size_t max_new_tokens = 32;
  std::size_t max_draft_tokens = 15;
  std::vector<std::int64_t> eos_token_ids;
  bool trace_rounds = false;
};

struct GenerationCounters {
  std::size_t graph_calls = 0;
  std::size_t drafted_tokens = 0;
  std::size_t accepted_draft_tokens = 0;
  std::size_t rejected_draft_tokens = 0;
  std::size_t decode_iterations = 0;
  std::size_t speculation_disable_events = 0;
  std::size_t target_only_fallback_rounds = 0;
};

struct GenerationRound {
  // Includes the current anchor after prefill, matching native ReplayRound.
  std::size_t committed_prefix_length = 0;
  std::string stage;
  std::vector<std::int64_t> proposed_token_ids;
  std::vector<std::int64_t> target_token_ids;
  std::vector<std::int64_t> accepted_draft_token_ids;
  std::vector<std::int64_t> emitted_token_ids;
  std::int64_t fallback_token_id = -1;
};

struct GenerationMeasurement {
  std::vector<std::int64_t> generated_token_ids;
  std::string stop_reason;
  GenerationCounters counters;
  double prefill_ms = 0.0;
  double request_reset_ms = 0.0;
  double decode_ms = 0.0;
  double model_total_ms = 0.0;
  std::vector<double> decode_iteration_ms;
  std::map<std::string, std::vector<double>> stage_ms;
  std::vector<GenerationRound> rounds;
};

struct Distribution {
  std::size_t count = 0;
  double min = 0.0;
  double max = 0.0;
  double mean = 0.0;
  double median = 0.0;
  double p90 = 0.0;
  double population_stdev = 0.0;
};

struct BenchmarkResult {
  GenerationMode mode = GenerationMode::kOrdinary;
  std::size_t warmup = 0;
  std::size_t repetitions = 0;
  std::vector<std::int64_t> stable_generated_token_ids;
  std::string stable_stop_reason;
  Distribution prefill_ms;
  Distribution decode_ms;
  Distribution model_total_ms;
  std::vector<GenerationMeasurement> measurements;
  std::size_t total_graph_calls = 0;
  std::size_t total_drafted_tokens = 0;
  std::size_t total_accepted_draft_tokens = 0;
  std::size_t total_rejected_draft_tokens = 0;
  double acceptance_rate = 0.0;
  double generated_tokens_per_second = 0.0;
};

struct PairedBenchmarkResult {
  BenchmarkResult ordinary;
  BenchmarkResult dflash;
  std::size_t token_id_mismatches = 0;
  std::size_t eos_mismatches = 0;
};

// Keep the failed measurements available to the report writer without making
// the strict parity gate optional. Callers must still return a failure status.
class PairedBenchmarkMismatch : public std::runtime_error {
 public:
  PairedBenchmarkMismatch(PairedBenchmarkResult result, const std::string& message)
      : std::runtime_error(message), result_(std::move(result)) {}
  PairedBenchmarkResult TakeResult() { return std::move(result_); }

 private:
  PairedBenchmarkResult result_;
};

GenerationMeasurement GenerateOnce(
    GraphExecutor& executor,
    const std::vector<std::int64_t>& prompt_token_ids,
    GenerationMode mode,
    const GenerationOptions& options);

BenchmarkResult Benchmark(
    GraphExecutor& executor,
    const std::vector<std::int64_t>& prompt_token_ids,
    GenerationMode mode,
    const GenerationOptions& options,
    std::size_t warmup,
    std::size_t repetitions);

// Runs both modes through the same loaded executor. Warmups and measurements
// alternate order to reduce thermal/order bias while preserving raw samples.
PairedBenchmarkResult BenchmarkPair(
    GraphExecutor& executor,
    const std::vector<std::int64_t>& prompt_token_ids,
    const GenerationOptions& options,
    std::size_t warmup,
    std::size_t repetitions);

// Apply the same strict token/EOS gate to independently measured mode groups.
PairedBenchmarkResult PairBenchmarks(BenchmarkResult ordinary, BenchmarkResult dflash);

Distribution Summarize(const std::vector<double>& values);
const char* ModeName(GenerationMode mode) noexcept;

}  // namespace qwen35::dflash
