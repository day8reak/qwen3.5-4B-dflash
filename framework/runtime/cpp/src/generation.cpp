#include "qwen35_dflash/generation.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace qwen35::dflash {
namespace {

using Clock = std::chrono::steady_clock;

double Milliseconds(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

void RequireNonNegative(
    const std::vector<std::int64_t>& values,
    const char* description) {
  if (std::any_of(values.begin(), values.end(), [](std::int64_t value) {
        return value < 0;
      })) {
    throw std::runtime_error(std::string(description) +
                             " contains a negative token ID");
  }
}

void ValidateOutputs(
    const GraphExecutor& executor,
    const GraphOutputs& outputs) {
  if (outputs.target_top1.size() != executor.sequence_length()) {
    throw std::runtime_error("target_top1 size differs from the fixed OM gear");
  }
  if (outputs.draft_top1.size() != executor.draft_width()) {
    throw std::runtime_error("draft_top1 size differs from the OM ABI");
  }
}

bool IsEos(
    std::int64_t token,
    const std::unordered_set<std::int64_t>& eos) {
  return eos.find(token) != eos.end();
}

void AppendCommitted(
    const std::vector<std::int64_t>& values,
    std::size_t remaining,
    const std::unordered_set<std::int64_t>& eos,
    std::vector<std::int64_t>* generated,
    std::vector<std::int64_t>* prefix,
    bool* finished) {
  if (values.empty()) {
    throw std::runtime_error("generation step committed no token");
  }
  if (values.size() > remaining) {
    throw std::runtime_error("generation step exceeded the remaining token budget");
  }
  for (std::size_t index = 0; index < values.size(); ++index) {
    const std::int64_t token = values[index];
    if (token < 0) {
      throw std::runtime_error("generation step returned a negative token ID");
    }
    generated->push_back(token);
    prefix->push_back(token);
    if (IsEos(token, eos)) {
      if (index + 1 != values.size()) {
        throw std::runtime_error("generation step returned tokens after EOS");
      }
      *finished = true;
    }
  }
}

void ValidateInputs(
    const GraphExecutor& executor,
    const std::vector<std::int64_t>& prompt,
    const GenerationOptions& options) {
  if (executor.sequence_length() <= 1) {
    throw std::invalid_argument("OM sequence gear must exceed one token");
  }
  if (executor.draft_width() == 0) {
    throw std::invalid_argument("OM draft width must be positive");
  }
  if (prompt.empty()) {
    throw std::invalid_argument("prompt token IDs must not be empty");
  }
  RequireNonNegative(prompt, "prompt");
  RequireNonNegative(options.eos_token_ids, "EOS set");
  if (options.pad_token_id < 0) {
    throw std::invalid_argument("pad token ID must be non-negative");
  }
  if (options.max_new_tokens == 0) {
    throw std::invalid_argument("max_new_tokens must be positive");
  }
  if (options.max_draft_tokens == 0) {
    throw std::invalid_argument("max_draft_tokens must be positive");
  }
  if (prompt.size() + options.max_new_tokens - 1 >
      executor.sequence_length()) {
    throw std::invalid_argument(
        "prompt plus requested generation exceeds the fixed OM gear");
  }
}

BenchmarkResult FinalizeBenchmark(
    GenerationMode mode,
    std::size_t warmup,
    std::vector<GenerationMeasurement> measurements) {
  if (measurements.empty()) {
    throw std::invalid_argument("benchmark repetitions must be positive");
  }
  const auto& reference_tokens = measurements.front().generated_token_ids;
  const auto& reference_stop = measurements.front().stop_reason;
  for (const auto& measurement : measurements) {
    if (measurement.generated_token_ids != reference_tokens) {
      throw std::runtime_error(
          "measured repetitions produced different token IDs");
    }
    if (measurement.stop_reason != reference_stop) {
      throw std::runtime_error(
          "measured repetitions produced different stop reasons");
    }
  }

  std::vector<double> prefill;
  std::vector<double> decode;
  std::vector<double> model_total;
  prefill.reserve(measurements.size());
  decode.reserve(measurements.size());
  model_total.reserve(measurements.size());
  BenchmarkResult result;
  result.mode = mode;
  result.warmup = warmup;
  result.repetitions = measurements.size();
  result.stable_generated_token_ids = reference_tokens;
  result.stable_stop_reason = reference_stop;
  result.measurements = std::move(measurements);
  for (const auto& measurement : result.measurements) {
    prefill.push_back(measurement.prefill_ms);
    decode.push_back(measurement.decode_ms);
    model_total.push_back(measurement.model_total_ms);
    result.total_graph_calls += measurement.counters.graph_calls;
    result.total_drafted_tokens += measurement.counters.drafted_tokens;
    result.total_accepted_draft_tokens +=
        measurement.counters.accepted_draft_tokens;
    result.total_rejected_draft_tokens +=
        measurement.counters.rejected_draft_tokens;
  }
  result.prefill_ms = Summarize(prefill);
  result.decode_ms = Summarize(decode);
  result.model_total_ms = Summarize(model_total);
  if (result.total_drafted_tokens != 0) {
    result.acceptance_rate =
        static_cast<double>(result.total_accepted_draft_tokens) /
        static_cast<double>(result.total_drafted_tokens);
  }
  const double seconds = std::accumulate(
      model_total.begin(), model_total.end(), 0.0) / 1000.0;
  const std::size_t generated =
      reference_tokens.size() * result.measurements.size();
  if (seconds > 0.0) {
    result.generated_tokens_per_second =
        static_cast<double>(generated) / seconds;
  }
  return result;
}

void EmitProgress(
    const ProgressCallback& progress,
    const char* phase,
    GenerationMode mode,
    std::size_t run_index,
    std::size_t run_total,
    const char* stage,
    std::size_t generated_tokens,
    std::size_t max_new_tokens,
    std::size_t prefix_tokens,
    std::size_t graph_calls,
    std::size_t decode_iteration,
    double elapsed_ms) {
  if (!progress) {
    return;
  }
  progress(ProgressEvent{
      phase,
      mode,
      run_index,
      run_total,
      stage,
      generated_tokens,
      max_new_tokens,
      prefix_tokens,
      graph_calls,
      decode_iteration,
      elapsed_ms,
  });
}

}  // namespace

const char* ModeName(GenerationMode mode) noexcept {
  return mode == GenerationMode::kOrdinary ? "ordinary-greedy"
                                            : "dflash-strict-greedy";
}

std::size_t StatefulGraphExecutor::max_speculative_sync_window()
    const noexcept {
  return 1;
}

std::vector<StatefulStep> StatefulGraphExecutor::SpeculativeWindow(
    const std::vector<std::size_t>& logical_proposal_counts) {
  if (logical_proposal_counts.empty() ||
      logical_proposal_counts.size() > max_speculative_sync_window()) {
    throw std::invalid_argument(
        "speculative sync window exceeds executor capability");
  }
  std::vector<StatefulStep> result;
  result.reserve(logical_proposal_counts.size());
  for (const std::size_t logical_proposal_count : logical_proposal_counts) {
    StatefulStep step = SpeculativeStep(logical_proposal_count);
    const bool finished = step.finished;
    result.push_back(std::move(step));
    if (finished) {
      break;
    }
  }
  return result;
}

namespace {

GenerationMeasurement GenerateOnceWithContext(
    GraphExecutor& executor,
    const std::vector<std::int64_t>& prompt_token_ids,
    GenerationMode mode,
    const GenerationOptions& options,
    const ProgressCallback& progress,
    const char* phase,
    std::size_t run_index,
    std::size_t run_total) {
  ValidateInputs(executor, prompt_token_ids, options);
  const std::unordered_set<std::int64_t> eos(
      options.eos_token_ids.begin(), options.eos_token_ids.end());

  std::vector<std::int64_t> prefix = prompt_token_ids;
  prefix.reserve(executor.sequence_length());
  std::vector<std::int64_t> generated;
  generated.reserve(options.max_new_tokens);
  std::vector<std::int64_t> committed;
  committed.reserve(options.max_draft_tokens + 1);
  GenerationMeasurement result;
  result.counters.graph_calls = 1;

  EmitProgress(
      progress,
      phase,
      mode,
      run_index,
      run_total,
      "run-start",
      0,
      options.max_new_tokens,
      prefix.size(),
      0,
      0,
      0.0);
  EmitProgress(
      progress,
      phase,
      mode,
      run_index,
      run_total,
      "prefill-start",
      0,
      options.max_new_tokens,
      prefix.size(),
      0,
      0,
      0.0);
  const auto prefill_start = Clock::now();
  const GraphOutputs& prefill_outputs =
      executor.Execute(prefix, options.pad_token_id);
  ValidateOutputs(executor, prefill_outputs);
  const std::int64_t first = prefill_outputs.target_top1[prefix.size() - 1];
  bool finished = false;
  AppendCommitted(
      std::vector<std::int64_t>{first},
      options.max_new_tokens,
      eos,
      &generated,
      &prefix,
      &finished);
  const auto prefill_end = Clock::now();
  result.prefill_ms = Milliseconds(prefill_start, prefill_end);
  EmitProgress(
      progress,
      phase,
      mode,
      run_index,
      run_total,
      "prefill-done",
      generated.size(),
      options.max_new_tokens,
      prefix.size(),
      result.counters.graph_calls,
      0,
      result.prefill_ms);

  while (!finished && generated.size() < options.max_new_tokens) {
    const std::size_t remaining = options.max_new_tokens - generated.size();
    const std::size_t decode_iteration =
        result.counters.decode_iterations + 1;
    EmitProgress(
        progress,
        phase,
        mode,
        run_index,
        run_total,
        "decode-start",
        generated.size(),
        options.max_new_tokens,
        prefix.size(),
        result.counters.graph_calls,
        decode_iteration,
        0.0);
    const auto decode_start = Clock::now();
    const GraphOutputs& proposal_outputs =
        executor.Execute(prefix, options.pad_token_id);
    ++result.counters.graph_calls;
    ValidateOutputs(executor, proposal_outputs);
    const std::int64_t ordinary_next =
        proposal_outputs.target_top1[prefix.size() - 1];
    if (ordinary_next < 0) {
      throw std::runtime_error("target OM returned a negative token ID");
    }

    committed.clear();
    if (mode == GenerationMode::kOrdinary || remaining == 1) {
      committed.push_back(ordinary_next);
    } else {
      const std::size_t proposal_count_limit = std::min(
          {options.max_draft_tokens,
           executor.draft_width(),
           remaining - 1});
      for (std::size_t index = 0; index < proposal_count_limit; ++index) {
        const std::int64_t proposal = proposal_outputs.draft_top1[index];
        if (proposal < 0) {
          throw std::runtime_error("draft OM returned a negative token ID");
        }
        committed.push_back(proposal);
        if (IsEos(proposal, eos)) {
          break;
        }
      }
      const std::size_t proposal_count = committed.size();
      result.counters.drafted_tokens += proposal_count;
      if (proposal_count == 0) {
        committed.push_back(ordinary_next);
      } else {
        const std::size_t base = prefix.size();
        prefix.insert(prefix.end(), committed.begin(), committed.end());
        const GraphOutputs& verify_outputs =
            executor.Execute(prefix, options.pad_token_id);
        ++result.counters.graph_calls;
        ValidateOutputs(executor, verify_outputs);
        if (verify_outputs.target_top1[base - 1] != ordinary_next) {
          throw std::runtime_error(
              "target OM changed its next token between proposal and verify");
        }
        std::size_t accepted = 0;
        for (; accepted < proposal_count; ++accepted) {
          if (committed[accepted] !=
              verify_outputs.target_top1[base - 1 + accepted]) {
            break;
          }
        }
        result.counters.accepted_draft_tokens += accepted;
        result.counters.rejected_draft_tokens += proposal_count - accepted;
        prefix.resize(base);
        if (accepted < proposal_count) {
          committed.resize(accepted);
          committed.push_back(verify_outputs.target_top1[base - 1 + accepted]);
        } else if (!IsEos(committed.back(), eos)) {
          committed.push_back(
              verify_outputs.target_top1[base + proposal_count - 1]);
        }
      }
    }
    RequireNonNegative(committed, "committed output");
    AppendCommitted(
        committed, remaining, eos, &generated, &prefix, &finished);
    const auto decode_end = Clock::now();
    const double iteration_ms = Milliseconds(decode_start, decode_end);
    result.decode_iteration_ms.push_back(iteration_ms);
    result.decode_ms += iteration_ms;
    ++result.counters.decode_iterations;
    EmitProgress(
        progress,
        phase,
        mode,
        run_index,
        run_total,
        "decode-done",
        generated.size(),
        options.max_new_tokens,
        prefix.size(),
        result.counters.graph_calls,
        decode_iteration,
        iteration_ms);
  }

  result.generated_token_ids = std::move(generated);
  result.stop_reason =
      (!result.generated_token_ids.empty() &&
       IsEos(result.generated_token_ids.back(), eos))
          ? "eos"
          : "length";
  result.model_total_ms = result.prefill_ms + result.decode_ms;
  EmitProgress(
      progress,
      phase,
      mode,
      run_index,
      run_total,
      "run-done",
      result.generated_token_ids.size(),
      options.max_new_tokens,
      prefix.size(),
      result.counters.graph_calls,
      result.counters.decode_iterations,
      result.model_total_ms);
  return result;
}

void ValidateStatefulInputs(
    const StatefulGraphExecutor& executor,
    const std::vector<std::int64_t>& prompt,
    const GenerationOptions& options) {
  if (executor.sequence_length() <= 1) {
    throw std::invalid_argument("stateful OM capacity must exceed one token");
  }
  if (executor.prefill_width() == 0) {
    throw std::invalid_argument("stateful prefill width must be positive");
  }
  if (executor.proposal_width() == 0) {
    throw std::invalid_argument("stateful proposal width must be positive");
  }
  if (prompt.empty()) {
    throw std::invalid_argument("prompt token IDs must not be empty");
  }
  RequireNonNegative(prompt, "prompt");
  RequireNonNegative(options.eos_token_ids, "EOS set");
  if (options.eos_token_ids.size() > executor.eos_table_width()) {
    throw std::invalid_argument("EOS set exceeds the exported fixed table width");
  }
  if (options.pad_token_id < 0) {
    throw std::invalid_argument("pad token ID must be non-negative");
  }
  if (options.max_new_tokens == 0 || options.max_draft_tokens == 0) {
    throw std::invalid_argument("generation limits must be positive");
  }
  if (options.dflash_sync_window == 0 ||
      options.dflash_sync_window >
          executor.max_speculative_sync_window()) {
    throw std::invalid_argument(
        "DFlash sync window exceeds executor capability");
  }
  if (prompt.size() + options.max_new_tokens - 1 >
      executor.sequence_length()) {
    throw std::invalid_argument(
        "prompt plus requested generation exceeds the state cache capacity");
  }
}

void ValidateStatefulStep(
    const StatefulStep& step,
    const std::unordered_set<std::int64_t>& eos,
    bool speculative,
    std::size_t proposal_limit) {
  if (step.model_executions == 0) {
    throw std::runtime_error("stateful executor reported no OM execution");
  }
  if (step.token_ids.empty()) {
    throw std::runtime_error("stateful graph committed no token");
  }
  RequireNonNegative(step.token_ids, "stateful graph output");
  bool host_finished = false;
  for (std::size_t index = 0; index < step.token_ids.size(); ++index) {
    if (IsEos(step.token_ids[index], eos)) {
      if (index + 1 != step.token_ids.size()) {
        throw std::runtime_error("stateful graph returned tokens after EOS");
      }
      host_finished = true;
    }
  }
  if (host_finished != step.finished) {
    throw std::runtime_error("device and host EOS decisions differ");
  }
  if (!speculative) {
    if (step.token_ids.size() != 1 || step.drafted_tokens != 0 ||
        step.accepted_draft_tokens != 0 ||
        step.rejected_draft_tokens != 0) {
      throw std::runtime_error("ordinary Target step returned speculative data");
    }
    return;
  }
  if (step.drafted_tokens == 0 || step.drafted_tokens > proposal_limit) {
    throw std::runtime_error("verify graph returned an invalid drafted count");
  }
  if (step.accepted_draft_tokens + step.rejected_draft_tokens !=
      step.drafted_tokens) {
    throw std::runtime_error("verify acceptance counters do not close");
  }
  if (step.token_ids.size() > step.accepted_draft_tokens + 1) {
    throw std::runtime_error("verify graph committed too many tokens");
  }
}

GenerationMeasurement GenerateStatefulOnceWithContext(
    StatefulGraphExecutor& executor,
    const std::vector<std::int64_t>& prompt_token_ids,
    GenerationMode mode,
    const GenerationOptions& options,
    const ProgressCallback& progress,
    const char* phase,
    std::size_t run_index,
    std::size_t run_total) {
  ValidateStatefulInputs(executor, prompt_token_ids, options);
  const std::unordered_set<std::int64_t> eos(
      options.eos_token_ids.begin(), options.eos_token_ids.end());

  EmitProgress(
      progress, phase, mode, run_index, run_total, "state-reset-start", 0,
      options.max_new_tokens, prompt_token_ids.size(), 0, 0, 0.0);
  executor.Reset(options.pad_token_id, options.eos_token_ids);
  EmitProgress(
      progress, phase, mode, run_index, run_total, "state-reset-staged", 0,
      options.max_new_tokens, prompt_token_ids.size(), 0, 0, 0.0);

  std::vector<std::int64_t> prefix = prompt_token_ids;
  prefix.reserve(executor.sequence_length() + 1);
  std::vector<std::int64_t> generated;
  generated.reserve(options.max_new_tokens);
  GenerationMeasurement result;

  EmitProgress(
      progress, phase, mode, run_index, run_total, "run-start", 0,
      options.max_new_tokens, prefix.size(), 0, 0, 0.0);
  EmitProgress(
      progress, phase, mode, run_index, run_total, "prefill-start", 0,
      options.max_new_tokens, prefix.size(), 0, 0, 0.0);
  const auto prefill_start = Clock::now();

  StatefulStep final_prefill;
  std::size_t prompt_offset = 0;
  std::vector<std::int64_t> chunk;
  chunk.reserve(executor.prefill_width());
  while (prompt_offset < prompt_token_ids.size()) {
    const std::size_t chunk_size = std::min(
        executor.prefill_width(), prompt_token_ids.size() - prompt_offset);
    const bool last_chunk =
        prompt_offset + chunk_size == prompt_token_ids.size();
    const bool prepare_draft =
        mode == GenerationMode::kDFlash && options.max_new_tokens > 1;
    const std::size_t proposal_count = prepare_draft
        ? (last_chunk
               ? std::min(
                     {options.max_draft_tokens,
                      executor.proposal_width(),
                      options.max_new_tokens - 1})
               : std::min(
                     options.max_draft_tokens,
                     executor.proposal_width()))
        : 0;
    chunk.assign(
        prompt_token_ids.begin() + static_cast<std::ptrdiff_t>(prompt_offset),
        prompt_token_ids.begin() +
            static_cast<std::ptrdiff_t>(prompt_offset + chunk_size));
    if (last_chunk) {
      StatefulStep step = executor.PrefillChunk(
          chunk, prepare_draft, proposal_count);
      ValidateStatefulStep(step, eos, false, 0);
      result.counters.graph_calls += step.model_executions;
      final_prefill = std::move(step);
    } else {
      const std::size_t model_executions = executor.PrefillChunkDeferred(
          chunk, prepare_draft, proposal_count);
      if (model_executions == 0) {
        throw std::runtime_error(
            "stateful executor reported no deferred prefill execution");
      }
      result.counters.graph_calls += model_executions;
    }
    prompt_offset += chunk_size;
  }

  bool finished = false;
  AppendCommitted(
      final_prefill.token_ids,
      options.max_new_tokens,
      eos,
      &generated,
      &prefix,
      &finished);
  const auto prefill_end = Clock::now();
  result.prefill_ms = Milliseconds(prefill_start, prefill_end);
  EmitProgress(
      progress, phase, mode, run_index, run_total, "prefill-done",
      generated.size(), options.max_new_tokens, prefix.size(),
      result.counters.graph_calls, 0, result.prefill_ms);

  while (!finished && generated.size() < options.max_new_tokens) {
    const std::size_t remaining = options.max_new_tokens - generated.size();
    const std::size_t decode_iteration =
        result.counters.decode_iterations + 1;
    EmitProgress(
        progress, phase, mode, run_index, run_total, "decode-start",
        generated.size(), options.max_new_tokens, prefix.size(),
        result.counters.graph_calls, decode_iteration, 0.0);
    const auto decode_start = Clock::now();

    std::vector<StatefulStep> steps;
    bool speculative = false;
    std::vector<std::size_t> proposal_counts;
    if (mode == GenerationMode::kOrdinary || remaining == 1) {
      steps.push_back(executor.DecodeOne(prefix.back()));
    } else {
      speculative = true;
      const std::size_t first_proposal_count = std::min(
          {options.max_draft_tokens,
           executor.proposal_width(),
           remaining - 1});
      proposal_counts.push_back(first_proposal_count);
      if (options.dflash_sync_window > 1) {
        const std::size_t remaining_after_worst_first =
            remaining - (first_proposal_count + 1);
        if (remaining_after_worst_first > 1) {
          proposal_counts.push_back(std::min(
              {options.max_draft_tokens,
               executor.proposal_width(),
               remaining_after_worst_first - 1}));
        }
      }
      steps = executor.SpeculativeWindow(proposal_counts);
      if (steps.empty() || steps.size() > proposal_counts.size()) {
        throw std::runtime_error(
            "stateful executor returned an invalid speculative window");
      }
    }
    for (const auto& step : steps) {
      result.counters.graph_calls += step.model_executions;
    }
    if (speculative) {
      result.counters.speculative_transactions += steps.size();
    }
    for (std::size_t step_index = 0;
         step_index < steps.size(); ++step_index) {
      const auto& step = steps[step_index];
      if (finished) {
        break;
      }
      ValidateStatefulStep(
          step,
          eos,
          speculative,
          speculative ? proposal_counts.at(step_index) : 0);
      if (speculative) {
        result.counters.drafted_tokens += step.drafted_tokens;
        result.counters.accepted_draft_tokens +=
            step.accepted_draft_tokens;
        result.counters.rejected_draft_tokens +=
            step.rejected_draft_tokens;
      }
      const std::size_t current_remaining =
          options.max_new_tokens - generated.size();
      AppendCommitted(
          step.token_ids,
          current_remaining,
          eos,
          &generated,
          &prefix,
          &finished);
    }

    const auto decode_end = Clock::now();
    const double iteration_ms = Milliseconds(decode_start, decode_end);
    result.decode_iteration_ms.push_back(iteration_ms);
    result.decode_ms += iteration_ms;
    ++result.counters.decode_iterations;
    EmitProgress(
        progress, phase, mode, run_index, run_total, "decode-done",
        generated.size(), options.max_new_tokens, prefix.size(),
        result.counters.graph_calls, decode_iteration, iteration_ms);
  }

  result.generated_token_ids = std::move(generated);
  result.stop_reason =
      (!result.generated_token_ids.empty() &&
       IsEos(result.generated_token_ids.back(), eos))
          ? "eos"
          : "length";
  result.model_total_ms = result.prefill_ms + result.decode_ms;
  EmitProgress(
      progress, phase, mode, run_index, run_total, "run-done",
      result.generated_token_ids.size(), options.max_new_tokens,
      prefix.size(), result.counters.graph_calls,
      result.counters.decode_iterations, result.model_total_ms);
  return result;
}

}  // namespace

GenerationMeasurement GenerateOnce(
    GraphExecutor& executor,
    const std::vector<std::int64_t>& prompt_token_ids,
    GenerationMode mode,
    const GenerationOptions& options,
    const ProgressCallback& progress) {
  return GenerateOnceWithContext(
      executor,
      prompt_token_ids,
      mode,
      options,
      progress,
      "single",
      1,
      1);
}

BenchmarkResult Benchmark(
    GraphExecutor& executor,
    const std::vector<std::int64_t>& prompt_token_ids,
    GenerationMode mode,
    const GenerationOptions& options,
    std::size_t warmup,
    std::size_t repetitions,
    const ProgressCallback& progress) {
  if (repetitions == 0) {
    throw std::invalid_argument("benchmark repetitions must be positive");
  }
  for (std::size_t index = 0; index < warmup; ++index) {
    static_cast<void>(GenerateOnceWithContext(
        executor,
        prompt_token_ids,
        mode,
        options,
        progress,
        "warmup",
        index + 1,
        warmup));
  }
  std::vector<GenerationMeasurement> measurements;
  measurements.reserve(repetitions);
  for (std::size_t index = 0; index < repetitions; ++index) {
    measurements.push_back(GenerateOnceWithContext(
        executor,
        prompt_token_ids,
        mode,
        options,
        progress,
        "measurement",
        index + 1,
        repetitions));
  }
  return FinalizeBenchmark(mode, warmup, std::move(measurements));
}

PairedBenchmarkResult BenchmarkPair(
    GraphExecutor& executor,
    const std::vector<std::int64_t>& prompt_token_ids,
    const GenerationOptions& options,
    std::size_t warmup,
    std::size_t repetitions,
    const ProgressCallback& progress) {
  if (repetitions == 0) {
    throw std::invalid_argument("benchmark repetitions must be positive");
  }
  for (std::size_t index = 0; index < warmup; ++index) {
    if (index % 2 == 0) {
      static_cast<void>(GenerateOnceWithContext(
          executor, prompt_token_ids, GenerationMode::kOrdinary, options,
          progress, "warmup", index + 1, warmup));
      static_cast<void>(GenerateOnceWithContext(
          executor, prompt_token_ids, GenerationMode::kDFlash, options,
          progress, "warmup", index + 1, warmup));
    } else {
      static_cast<void>(GenerateOnceWithContext(
          executor, prompt_token_ids, GenerationMode::kDFlash, options,
          progress, "warmup", index + 1, warmup));
      static_cast<void>(GenerateOnceWithContext(
          executor, prompt_token_ids, GenerationMode::kOrdinary, options,
          progress, "warmup", index + 1, warmup));
    }
  }

  std::vector<GenerationMeasurement> ordinary;
  std::vector<GenerationMeasurement> dflash;
  ordinary.reserve(repetitions);
  dflash.reserve(repetitions);
  for (std::size_t index = 0; index < repetitions; ++index) {
    if (index % 2 == 0) {
      ordinary.push_back(GenerateOnceWithContext(
          executor, prompt_token_ids, GenerationMode::kOrdinary, options,
          progress, "measurement", index + 1, repetitions));
      dflash.push_back(GenerateOnceWithContext(
          executor, prompt_token_ids, GenerationMode::kDFlash, options,
          progress, "measurement", index + 1, repetitions));
    } else {
      dflash.push_back(GenerateOnceWithContext(
          executor, prompt_token_ids, GenerationMode::kDFlash, options,
          progress, "measurement", index + 1, repetitions));
      ordinary.push_back(GenerateOnceWithContext(
          executor, prompt_token_ids, GenerationMode::kOrdinary, options,
          progress, "measurement", index + 1, repetitions));
    }
  }

  PairedBenchmarkResult result{
      FinalizeBenchmark(
          GenerationMode::kOrdinary, warmup, std::move(ordinary)),
      FinalizeBenchmark(
          GenerationMode::kDFlash, warmup, std::move(dflash)),
      0,
      0,
  };
  const auto& expected = result.ordinary.stable_generated_token_ids;
  const auto& actual = result.dflash.stable_generated_token_ids;
  const std::size_t width = std::max(expected.size(), actual.size());
  for (std::size_t index = 0; index < width; ++index) {
    if (index >= expected.size() || index >= actual.size() ||
        expected[index] != actual[index]) {
      ++result.token_id_mismatches;
    }
  }
  result.eos_mismatches =
      result.ordinary.stable_stop_reason == result.dflash.stable_stop_reason
          ? 0
          : 1;
  if (result.token_id_mismatches != 0 || result.eos_mismatches != 0) {
    throw std::runtime_error(
        "DFlash output differs from the ordinary greedy authority");
  }
  return result;
}

GenerationMeasurement GenerateStatefulOnce(
    StatefulGraphExecutor& executor,
    const std::vector<std::int64_t>& prompt_token_ids,
    GenerationMode mode,
    const GenerationOptions& options,
    const ProgressCallback& progress) {
  return GenerateStatefulOnceWithContext(
      executor,
      prompt_token_ids,
      mode,
      options,
      progress,
      "single",
      1,
      1);
}

BenchmarkResult BenchmarkStateful(
    StatefulGraphExecutor& executor,
    const std::vector<std::int64_t>& prompt_token_ids,
    GenerationMode mode,
    const GenerationOptions& options,
    std::size_t warmup,
    std::size_t repetitions,
    const ProgressCallback& progress) {
  if (repetitions == 0) {
    throw std::invalid_argument("benchmark repetitions must be positive");
  }
  for (std::size_t index = 0; index < warmup; ++index) {
    static_cast<void>(GenerateStatefulOnceWithContext(
        executor,
        prompt_token_ids,
        mode,
        options,
        progress,
        "warmup",
        index + 1,
        warmup));
  }
  std::vector<GenerationMeasurement> measurements;
  measurements.reserve(repetitions);
  for (std::size_t index = 0; index < repetitions; ++index) {
    measurements.push_back(GenerateStatefulOnceWithContext(
        executor,
        prompt_token_ids,
        mode,
        options,
        progress,
        "measurement",
        index + 1,
        repetitions));
  }
  return FinalizeBenchmark(mode, warmup, std::move(measurements));
}

PairedBenchmarkResult BenchmarkPairStateful(
    StatefulGraphExecutor& executor,
    const std::vector<std::int64_t>& prompt_token_ids,
    const GenerationOptions& options,
    std::size_t warmup,
    std::size_t repetitions,
    const ProgressCallback& progress) {
  if (repetitions == 0) {
    throw std::invalid_argument("benchmark repetitions must be positive");
  }
  for (std::size_t index = 0; index < warmup; ++index) {
    if (index % 2 == 0) {
      static_cast<void>(GenerateStatefulOnceWithContext(
          executor, prompt_token_ids, GenerationMode::kOrdinary, options,
          progress, "warmup", index + 1, warmup));
      static_cast<void>(GenerateStatefulOnceWithContext(
          executor, prompt_token_ids, GenerationMode::kDFlash, options,
          progress, "warmup", index + 1, warmup));
    } else {
      static_cast<void>(GenerateStatefulOnceWithContext(
          executor, prompt_token_ids, GenerationMode::kDFlash, options,
          progress, "warmup", index + 1, warmup));
      static_cast<void>(GenerateStatefulOnceWithContext(
          executor, prompt_token_ids, GenerationMode::kOrdinary, options,
          progress, "warmup", index + 1, warmup));
    }
  }

  std::vector<GenerationMeasurement> ordinary;
  std::vector<GenerationMeasurement> dflash;
  ordinary.reserve(repetitions);
  dflash.reserve(repetitions);
  for (std::size_t index = 0; index < repetitions; ++index) {
    if (index % 2 == 0) {
      ordinary.push_back(GenerateStatefulOnceWithContext(
          executor, prompt_token_ids, GenerationMode::kOrdinary, options,
          progress, "measurement", index + 1, repetitions));
      dflash.push_back(GenerateStatefulOnceWithContext(
          executor, prompt_token_ids, GenerationMode::kDFlash, options,
          progress, "measurement", index + 1, repetitions));
    } else {
      dflash.push_back(GenerateStatefulOnceWithContext(
          executor, prompt_token_ids, GenerationMode::kDFlash, options,
          progress, "measurement", index + 1, repetitions));
      ordinary.push_back(GenerateStatefulOnceWithContext(
          executor, prompt_token_ids, GenerationMode::kOrdinary, options,
          progress, "measurement", index + 1, repetitions));
    }
  }

  PairedBenchmarkResult result{
      FinalizeBenchmark(
          GenerationMode::kOrdinary, warmup, std::move(ordinary)),
      FinalizeBenchmark(
          GenerationMode::kDFlash, warmup, std::move(dflash)),
      0,
      0,
  };
  const auto& expected = result.ordinary.stable_generated_token_ids;
  const auto& actual = result.dflash.stable_generated_token_ids;
  const std::size_t width = std::max(expected.size(), actual.size());
  for (std::size_t index = 0; index < width; ++index) {
    if (index >= expected.size() || index >= actual.size() ||
        expected[index] != actual[index]) {
      ++result.token_id_mismatches;
    }
  }
  result.eos_mismatches =
      result.ordinary.stable_stop_reason == result.dflash.stable_stop_reason
          ? 0
          : 1;
  if (result.token_id_mismatches != 0 || result.eos_mismatches != 0) {
    throw std::runtime_error(
        "stateful DFlash output differs from ordinary greedy authority");
  }
  return result;
}

Distribution Summarize(const std::vector<double>& values) {
  if (values.empty()) {
    throw std::invalid_argument("cannot summarize an empty latency set");
  }
  std::vector<double> ordered = values;
  std::sort(ordered.begin(), ordered.end());
  const double sum = std::accumulate(ordered.begin(), ordered.end(), 0.0);
  const double mean = sum / static_cast<double>(ordered.size());
  double variance = 0.0;
  for (const double value : ordered) {
    const double delta = value - mean;
    variance += delta * delta;
  }
  variance /= static_cast<double>(ordered.size());
  auto percentile = [&ordered](double fraction) {
    if (ordered.size() == 1) {
      return ordered.front();
    }
    const double position =
        static_cast<double>(ordered.size() - 1) * fraction;
    const auto low = static_cast<std::size_t>(std::floor(position));
    const auto high = static_cast<std::size_t>(std::ceil(position));
    if (low == high) {
      return ordered[low];
    }
    const double weight = position - static_cast<double>(low);
    return ordered[low] * (1.0 - weight) + ordered[high] * weight;
  };
  return Distribution{
      ordered.size(),
      ordered.front(),
      ordered.back(),
      mean,
      percentile(0.5),
      percentile(0.9),
      std::sqrt(variance),
  };
}

}  // namespace qwen35::dflash
