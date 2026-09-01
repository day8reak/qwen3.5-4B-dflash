#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace qwen35::dflash {

enum class GenerationMode {
  kOrdinary,
  kDFlash,
};

struct ProgressEvent {
  const char* phase = "single";
  GenerationMode mode = GenerationMode::kOrdinary;
  std::size_t run_index = 1;
  std::size_t run_total = 1;
  const char* stage = "run-start";
  std::size_t generated_tokens = 0;
  std::size_t max_new_tokens = 0;
  std::size_t prefix_tokens = 0;
  std::size_t graph_calls = 0;
  std::size_t decode_iteration = 0;
  double elapsed_ms = 0.0;
};

using ProgressCallback = std::function<void(const ProgressEvent&)>;

struct GraphOutputs {
  std::vector<std::int64_t> target_top1;
  std::vector<std::int64_t> draft_top1;
};

// Compact host-visible result produced by one explicit-state Target graph.
// All large Target/Draft states and feature carriers remain owned by the
// concrete executor on device.  model_executions counts enqueued OM calls;
// one method call may execute Draft followed by Target while synchronizing
// only once.
struct StatefulStep {
  std::vector<std::int64_t> token_ids;
  std::size_t model_executions = 0;
  std::size_t drafted_tokens = 0;
  std::size_t accepted_draft_tokens = 0;
  std::size_t rejected_draft_tokens = 0;
  bool finished = false;
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

// Exact explicit-state graph suite used by the approved multi-OM route.
// Reset stages request metadata before model timing. The production ACL
// executor defers state initialization and EOS upload to the first
// PrefillChunk so their device work remains inside prefill latency.
// PrefillChunk and DecodeOne each expose one Target completion barrier.
// PrefillChunkDeferred may leave an intermediate prompt chunk queued without
// returning its unused compact result; the next completing executor call must
// preserve stream order. The default implementation remains synchronous so
// non-ACL executors do not need a specialized asynchronous path.
// SpeculativeStep enqueues Draft -> Target verify/commit and exposes exactly
// one completion barrier for the whole transaction.
class StatefulGraphExecutor {
 public:
  virtual ~StatefulGraphExecutor() = default;

  virtual std::size_t sequence_length() const noexcept = 0;
  virtual std::size_t prefill_width() const noexcept = 0;
  virtual std::size_t proposal_width() const noexcept = 0;
  virtual std::size_t eos_table_width() const noexcept = 0;

  virtual void Reset(
      std::int64_t pad_token_id,
      const std::vector<std::int64_t>& eos_token_ids) = 0;

  virtual StatefulStep PrefillChunk(
      const std::vector<std::int64_t>& token_ids,
      bool prepare_draft,
      std::size_t logical_proposal_count) = 0;

  virtual std::size_t PrefillChunkDeferred(
      const std::vector<std::int64_t>& token_ids,
      bool prepare_draft,
      std::size_t logical_proposal_count) {
    return PrefillChunk(
        token_ids, prepare_draft, logical_proposal_count).model_executions;
  }

  virtual StatefulStep DecodeOne(std::int64_t input_token_id) = 0;

  virtual StatefulStep SpeculativeStep(
      std::size_t logical_proposal_count) = 0;

  // A concrete executor may keep the final prefill completion and the first
  // Target verify in one ordered stream window. The returned steps are the
  // final prefill result followed by the speculative result. This changes
  // host visibility (and therefore TTFT attribution), but not token semantics.
  virtual bool supports_prefill_verify_coalescing() const noexcept;
  virtual std::vector<StatefulStep> PrefillChunkAndSpeculative(
      const std::vector<std::int64_t>& token_ids,
      std::size_t logical_proposal_count);

  // A concrete executor may enqueue more than one complete speculative
  // transaction before exposing a host barrier. The default preserves the
  // synchronous contract and stops before launching work after a reported
  // EOS. Implementations must return results in execution order.
  virtual std::size_t max_speculative_sync_window() const noexcept;
  virtual std::vector<StatefulStep> SpeculativeWindow(
      const std::vector<std::size_t>& logical_proposal_counts);
};

struct GenerationOptions {
  std::int64_t pad_token_id = 0;
  std::size_t max_new_tokens = 32;
  std::size_t max_draft_tokens = 15;
  std::size_t dflash_sync_window = 1;
  bool coalesce_prefill_with_first_verify = false;
  std::vector<std::int64_t> eos_token_ids;
};

struct GenerationCounters {
  std::size_t graph_calls = 0;
  std::size_t drafted_tokens = 0;
  std::size_t accepted_draft_tokens = 0;
  std::size_t rejected_draft_tokens = 0;
  std::size_t speculative_transactions = 0;
  // Physical speculative transactions completed inside the prefill timer.
  // They are not host-visible decode iterations.
  std::size_t prefill_speculative_windows = 0;
  // Host-visible decode windows. With dflash_sync_window=1 this is also the
  // transaction count; a larger exact window may contain multiple rounds.
  std::size_t decode_iterations = 0;
};

struct GenerationMeasurement {
  std::vector<std::int64_t> generated_token_ids;
  std::string stop_reason;
  GenerationCounters counters;
  double prefill_ms = 0.0;
  double decode_ms = 0.0;
  double model_total_ms = 0.0;
  std::vector<double> decode_iteration_ms;
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

GenerationMeasurement GenerateOnce(
    GraphExecutor& executor,
    const std::vector<std::int64_t>& prompt_token_ids,
    GenerationMode mode,
    const GenerationOptions& options,
    const ProgressCallback& progress = {});

BenchmarkResult Benchmark(
    GraphExecutor& executor,
    const std::vector<std::int64_t>& prompt_token_ids,
    GenerationMode mode,
    const GenerationOptions& options,
    std::size_t warmup,
    std::size_t repetitions,
    const ProgressCallback& progress = {});

// Runs both modes through the same loaded executor. Warmups and measurements
// alternate order to reduce thermal/order bias while preserving raw samples.
PairedBenchmarkResult BenchmarkPair(
    GraphExecutor& executor,
    const std::vector<std::int64_t>& prompt_token_ids,
    const GenerationOptions& options,
    std::size_t warmup,
    std::size_t repetitions,
    const ProgressCallback& progress = {});

GenerationMeasurement GenerateStatefulOnce(
    StatefulGraphExecutor& executor,
    const std::vector<std::int64_t>& prompt_token_ids,
    GenerationMode mode,
    const GenerationOptions& options,
    const ProgressCallback& progress = {});

BenchmarkResult BenchmarkStateful(
    StatefulGraphExecutor& executor,
    const std::vector<std::int64_t>& prompt_token_ids,
    GenerationMode mode,
    const GenerationOptions& options,
    std::size_t warmup,
    std::size_t repetitions,
    const ProgressCallback& progress = {});

PairedBenchmarkResult BenchmarkPairStateful(
    StatefulGraphExecutor& executor,
    const std::vector<std::int64_t>& prompt_token_ids,
    const GenerationOptions& options,
    std::size_t warmup,
    std::size_t repetitions,
    const ProgressCallback& progress = {});

Distribution Summarize(const std::vector<double>& values);
const char* ModeName(GenerationMode mode) noexcept;

}  // namespace qwen35::dflash
