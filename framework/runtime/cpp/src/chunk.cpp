#include "qwen35_dflash/chunk.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <stdexcept>

#include "qwen35_dflash/sha256.hpp"

namespace qwen35::dflash {
namespace {
void Require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
using Clock = std::chrono::steady_clock;
double Ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

void ValidateVerifyDiscardOutputs(const ChunkGraph& graph, bool mtp) {
  std::vector<TensorSpec> expected;
  for (const auto& spec : graph.inputs) {
    Require(!IsVerifyDiscardState(spec.name),
            "verify discard state must never be an OM input");
    const std::string suffix = "_recurrent";
    if (!mtp && graph.name == "target_verify" && spec.name.size() > suffix.size() + 1 &&
        spec.name[0] == 't' && spec.name[1] >= '0' && spec.name[1] <= '9' &&
        spec.name.compare(spec.name.size() - suffix.size(), suffix.size(), suffix) == 0)
      expected.push_back({"verify_discard_" + spec.name, "float32", spec.shape});
  }
  const auto count = static_cast<std::size_t>(std::count_if(
      graph.outputs.begin(), graph.outputs.end(),
      [](const auto& spec) { return IsVerifyDiscardState(spec.name); }));
  Require(count == expected.size() &&
              (mtp || graph.name != "target_verify" || count > 0),
          "verify discard output count differs from recurrent layer count");
  // Discard buffers follow all committed states, in the same linear-layer order.
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const auto& actual = graph.outputs[graph.outputs.size() - expected.size() + i];
    Require(actual.name == expected[i].name && actual.dtype == expected[i].dtype &&
                actual.shape == expected[i].shape,
            "verify discard output name/order/dtype/shape differs from raw FP32 state");
  }
}
}  // namespace

bool IsVerifyDiscardState(const std::string& name) {
  return name.rfind("verify_discard_", 0) == 0;
}

std::size_t TensorSpec::bytes() const {
  std::size_t size = dtype == "float16" || dtype == "int16" ? 2
                     : dtype == "float32"                   ? 4
                     : dtype == "int64"                     ? 8
                                                            : 0;
  Require(size != 0 && !shape.empty() && shape.size() <= 8,
          "invalid tensor dtype/rank");
  for (auto dim : shape) {
    Require(dim > 0 && static_cast<std::uint64_t>(dim) <= (1ULL << 40) / size,
            "invalid tensor dimension/size");
    size *= static_cast<std::size_t>(dim);
  }
  return size;
}

ChunkPlan ReadChunkPlan(const std::filesystem::path& path,
                        const std::string& mode) {
  std::ifstream input(path);
  std::string word;
  ChunkPlan result;
  Require(static_cast<bool>(input >> result.abi) &&
              (result.abi == "qwen35-dflash-chunk-v3" || result.abi == "qwen35-dflash-mtp-v1"),
          "invalid chunk plan ABI; regenerate AIR/OM and rebuild the C++ runner");
  Require(static_cast<bool>(input >> word >> result.capacity >>
                            result.vocabulary) &&
              word == "capacity",
          "missing chunk capacity");
  Require(result.capacity >= 64 && result.capacity <= 32704 &&
              result.capacity % 64 == 0 && result.vocabulary > 0,
          "invalid chunk capacity/vocabulary");
  const std::set<std::string> roles{"target_prefill", "target_decode",
                                    "target_verify", "draft"};
  while (input >> word && word == "graph") {
    ChunkGraph graph;
    std::string filename;
    Require(static_cast<bool>(input >> graph.name >> std::quoted(filename) >>
                              graph.sha256),
            "truncated graph plan");
    Require(roles.count(graph.name) && !result.graphs.count(graph.name),
            "duplicate/unknown chunk graph");
    graph.model = filename;
    if (!(mode == "dflash" && graph.name == "target_decode") &&
        !(mode == "ordinary" &&
          (graph.name == "target_verify" || graph.name == "draft"))) {
      Require(Sha256File(graph.model) == graph.sha256,
              "chunk OM SHA-256 mismatch: " + graph.name);
    }
    std::set<std::string> inputs, outputs;
    while (input >> word && (word == "I" || word == "O")) {
      TensorSpec tensor;
      std::size_t rank = 0;
      Require(static_cast<bool>(input >> tensor.name >> tensor.dtype >> rank) &&
                  rank > 0 && rank <= 8,
              "invalid tensor descriptor");
      tensor.shape.resize(rank);
      for (auto& dim : tensor.shape)
        Require(static_cast<bool>(input >> dim), "truncated tensor shape");
      static_cast<void>(tensor.bytes());
      Require((word == "I" ? inputs : outputs).insert(tensor.name).second,
              "duplicate tensor name");
      (word == "I" ? graph.inputs : graph.outputs).push_back(std::move(tensor));
    }
    Require(word == "end" && !graph.inputs.empty() && !graph.outputs.empty(),
            "incomplete graph plan");
    ValidateVerifyDiscardOutputs(graph, result.abi == "qwen35-dflash-mtp-v1");
    if (result.abi == "qwen35-dflash-mtp-v1") {
      auto validate_recurrent = [](const auto& specs) {
        for (const auto& spec : specs) {
          if (spec.name.size() >= 10 &&
              spec.name.compare(spec.name.size() - 10, 10, "_recurrent") == 0)
            Require(spec.dtype == "float32", "MTP committed recurrent state must be FP32");
        }
      };
      validate_recurrent(graph.inputs);
      validate_recurrent(graph.outputs);
    }
    result.graphs.emplace(graph.name, std::move(graph));
  }
  Require(word == "done" && result.graphs.count("target_prefill") &&
              result.graphs.count("target_verify") &&
              result.graphs.count("draft"),
          "chunk plan needs prefill, verify and draft");
  Require(mode == "dflash" || result.graphs.count("target_decode"),
          "ordinary/paired execution requires target_decode");
  Require(!(input >> word), "unexpected data after chunk plan");
  return result;
}

const GraphOutputs& ChunkExecutor::Execute(const std::vector<std::int64_t>&,
                                           std::int64_t) {
  throw std::logic_error("incremental executor requires the chunk scheduler");
}

GenerationMeasurement GenerateChunk(ChunkExecutor& executor,
                                    const std::vector<std::int64_t>& prompt,
                                    GenerationMode mode,
                                    const GenerationOptions& options) {
  Require(!prompt.empty() && options.max_new_tokens > 0 &&
              options.max_draft_tokens > 0,
          "invalid generation limits/prompt");
  Require(
      prompt.size() <= executor.sequence_length() &&
          options.max_new_tokens <= executor.sequence_length() - prompt.size(),
      "generation exceeds logical chunk capacity");
  auto valid_token = [&](std::int64_t token) {
    Require(token >= 0 && token < executor.vocabulary_size(),
            "token outside model vocabulary");
  };
  for (auto token : prompt) valid_token(token);
  for (auto token : options.eos_token_ids) valid_token(token);
  valid_token(options.pad_token_id);
  std::set<std::int64_t> eos(options.eos_token_ids.begin(),
                             options.eos_token_ids.end());
  GenerationMeasurement result;
  const auto reset_start = Clock::now();
  executor.Reset(
      options.pad_token_id);  // request reset stays outside measured prefill
  result.request_reset_ms = Ms(reset_start);
  try {
    const auto prefill_start = Clock::now();
    auto anchor = executor.Prefill(prompt, mode == GenerationMode::kDFlash);
    valid_token(anchor);
    result.generated_token_ids.push_back(anchor);
    result.prefill_ms = Ms(prefill_start);
    if (options.trace_rounds) {
      GenerationRound round;
      round.committed_prefix_length = prompt.size();
      round.stage = "target_prefill";
      round.target_token_ids = {anchor};
      round.emitted_token_ids = {anchor};
      round.fallback_token_id = anchor;
      result.rounds.push_back(std::move(round));
    }
    while (!eos.count(anchor) &&
           result.generated_token_ids.size() < options.max_new_tokens) {
      const auto start = Clock::now();
      const auto remaining =
          options.max_new_tokens - result.generated_token_ids.size();
      std::vector<std::int64_t> emitted;
      GenerationRound round;
      if (options.trace_rounds)
        round.committed_prefix_length = prompt.size() + result.generated_token_ids.size();
      if (mode == GenerationMode::kOrdinary) {
        emitted.push_back(executor.Decode(anchor));
        if (options.trace_rounds) {
          round.stage = "target_decode";
          round.target_token_ids = emitted;
          round.fallback_token_id = emitted.front();
        }
      } else {
        std::vector<std::int64_t> proposals;
        const auto proposal_count =
            std::min({remaining, options.max_draft_tokens, executor.draft_width()});
        const auto raw = executor.Propose(anchor, proposal_count);
        Require(raw.size() == executor.draft_width(), "draft width mismatch");
        for (std::size_t i = 0; i < proposal_count; ++i) {
          valid_token(raw[i]);
          proposals.push_back(raw[i]);
          if (eos.count(raw[i])) break;
        }
        std::vector<std::int64_t> block{anchor};
        block.insert(block.end(), proposals.begin(), proposals.end());
        const auto verified = executor.Verify(block);
        Require(verified.size() >= block.size(),
                "verify output is shorter than its valid rows");
        for (std::size_t i = 0; i < block.size(); ++i) valid_token(verified[i]);
        std::size_t accepted = 0;
        while (accepted < proposals.size() &&
               proposals[accepted] == verified[accepted])
          ++accepted;
        executor.Commit(
            accepted +
            1);  // publish/ack the fused OM result; no second OM call
        // A zero-accept round still commits its anchor. Keep drafting from the
        // Target correction on the next round, with the updated context.
        emitted.assign(proposals.begin(), proposals.begin() + accepted);
        if (emitted.empty() ||
            (!eos.count(emitted.back()) && emitted.size() < remaining))
          emitted.push_back(verified[accepted]);
        result.counters.drafted_tokens += proposals.size();
        result.counters.accepted_draft_tokens += accepted;
        result.counters.rejected_draft_tokens += proposals.size() - accepted;
        if (options.trace_rounds) {
          round.stage = "target_verify";
          round.proposed_token_ids = proposals;
          round.target_token_ids.assign(verified.begin(), verified.begin() + block.size());
          round.accepted_draft_token_ids.assign(proposals.begin(), proposals.begin() + accepted);
          if (emitted.size() > accepted) round.fallback_token_id = verified[accepted];
        }
      }
      Require(!emitted.empty() && emitted.size() <= remaining,
              "invalid committed token count");
      for (auto token : emitted) {
        valid_token(token);
        result.generated_token_ids.push_back(token);
      }
      anchor = emitted.back();
      const double elapsed = Ms(start);
      result.decode_iteration_ms.push_back(elapsed);
      result.decode_ms += elapsed;
      ++result.counters.decode_iterations;
      if (options.trace_rounds) {
        round.emitted_token_ids = emitted;
        result.rounds.push_back(std::move(round));
      }
    }
    result.stop_reason = eos.count(anchor) ? "eos" : "max_new_tokens";
    result.model_total_ms = result.prefill_ms + result.decode_ms;
    result.counters.graph_calls = executor.graph_calls();
    result.stage_ms = executor.stage_ms();
  } catch (...) {
    executor.Abort();
    throw;
  }
  return result;
}
}  // namespace qwen35::dflash
