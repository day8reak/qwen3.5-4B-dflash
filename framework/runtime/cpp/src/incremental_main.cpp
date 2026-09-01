#include "qwen35_dflash/generation.hpp"
#include "qwen35_dflash/incremental_acl_executor.hpp"
#include "qwen35_dflash/sha256.hpp"

#include "acl_memory_policy.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef QWEN35_DFLASH_RUNNER_VERSION
#define QWEN35_DFLASH_RUNNER_VERSION "dev"
#endif

namespace {

using qwen35::dflash::BenchmarkResult;
using qwen35::dflash::Distribution;
using qwen35::dflash::GenerationMeasurement;
using qwen35::dflash::IncrementalModelMemory;
using qwen35::dflash::IncrementalDecodeCarrierPolicy;
using qwen35::dflash::IncrementalDraftFeaturePolicy;
using qwen35::dflash::IncrementalStateResetPolicy;
using qwen35::dflash::PairedBenchmarkResult;
using qwen35::dflash::ProgressCallback;
using qwen35::dflash::ProgressEvent;
using qwen35::dflash::ZeroAcceptFallbackPolicy;

enum class MeasurementProtocol {
  kEvidence,
  kProfile,
};

struct ModelArgument {
  const char* role;
  std::filesystem::path path;
  std::string sha256;
};

struct Arguments {
  std::array<ModelArgument, 6> models{{
      {"target-prefill", {}, {}},
      {"target-prefill-head", {}, {}},
      {"target-decode1", {}, {}},
      {"draft-propose", {}, {}},
      {"target-verify-commit", {}, {}},
      {"fused-speculative-step", {}, {}},
  }};
  std::filesystem::path output;
  std::vector<std::int64_t> prompt_token_ids;
  std::vector<std::int64_t> eos_token_ids;
  std::int64_t pad_token_id = 0;
  std::size_t max_new_tokens = 32;
  std::size_t max_draft_tokens = 15;
  std::size_t dflash_sync_window = 1;
  bool coalesce_prefill_with_first_verify = false;
  ZeroAcceptFallbackPolicy zero_accept_fallback_policy =
      ZeroAcceptFallbackPolicy::kDisabled;
  std::size_t warmup = 3;
  std::size_t repetitions = 10;
  int device_id = 0;
  bool progress = true;
  IncrementalStateResetPolicy state_reset_policy =
      IncrementalStateResetPolicy::kAsyncMemset;
  IncrementalDecodeCarrierPolicy decode_carrier_policy =
      IncrementalDecodeCarrierPolicy::kLastTokenDeviceCompact;
  IncrementalDraftFeaturePolicy draft_feature_policy =
      IncrementalDraftFeaturePolicy::kFixedVerifyWidth;
  MeasurementProtocol measurement_protocol = MeasurementProtocol::kEvidence;
};

void Usage(std::ostream& stream) {
  stream
      << "Usage: qwen35_dflash_incremental_acl_runner [options]\n"
      << "  --target-prefill PATH                    hash-locked prefill OM\n"
      << "  --target-prefill-sha256 HEX              expected prefill SHA-256\n"
      << "  --target-prefill-head PATH               final-chunk QLinear head OM\n"
      << "  --target-prefill-head-sha256 HEX         expected head SHA-256\n"
      << "  --target-decode1 PATH                    optional decode-one OM; omit for unified Target step\n"
      << "  --target-decode1-sha256 HEX              required only with target-decode1\n"
      << "  --draft-propose PATH                     hash-locked Draft OM\n"
      << "  --draft-propose-sha256 HEX               expected Draft SHA-256\n"
      << "  --target-verify-commit PATH              hash-locked verify OM\n"
      << "  --target-verify-commit-sha256 HEX        expected verify SHA-256\n"
      << "  --fused-speculative-step PATH            exact Draft+verify supergraph; replaces separate pair\n"
      << "  --fused-speculative-step-sha256 HEX      expected fused OM SHA-256\n"
      << "  --output PATH                            paired JSON report\n"
      << "  --prompt-token-ids CSV                   non-empty prompt\n"
      << "  --eos-token-ids CSV                      optional EOS token IDs\n"
      << "  --pad-token-id ID                        default 0\n"
      << "  --max-new-tokens N                       default 32\n"
      << "  --max-draft-tokens N                     default 15\n"
      << "  --dflash-sync-window N                   exact window in 1..8 (default 1)\n"
      << "  --prefill-completion-policy POLICY       separate (default) or coalesce-first-verify\n"
      << "  --zero-accept-fallback-policy POLICY    disabled (default) or request-target-only\n"
      << "  --warmup N                               evidence=3, profile=1\n"
      << "  --repetitions N                          evidence=10, profile=1\n"
      << "  --device-id N                            default 0\n"
      << "  --measurement-protocol MODE             evidence (default) or profile\n"
      << "  --state-reset-policy POLICY             async-memset (default) or "
         "immutable-zero\n"
      << "  --decode-carrier-policy POLICY          last-token-d2d (default) or "
         "one-token-h2d\n"
      << "  --draft-feature-policy POLICY          fixed-16 (default) or "
         "committed-prefix\n"
      << "  --progress true|false                    live stderr progress\n";
}

std::string Trim(std::string value) {
  const auto first = std::find_if_not(value.begin(), value.end(), [](char item) {
    return std::isspace(static_cast<unsigned char>(item)) != 0;
  });
  const auto last = std::find_if_not(value.rbegin(), value.rend(), [](char item) {
                      return std::isspace(static_cast<unsigned char>(item)) != 0;
                    }).base();
  return first >= last ? std::string{} : std::string(first, last);
}

std::int64_t ParseInt64(const std::string& text, const char* name) {
  std::size_t consumed = 0;
  long long value = 0;
  try {
    value = std::stoll(text, &consumed, 10);
  } catch (const std::exception&) {
    throw std::invalid_argument(std::string(name) + " is not an integer");
  }
  if (consumed != text.size()) {
    throw std::invalid_argument(std::string(name) + " is not an integer");
  }
  return static_cast<std::int64_t>(value);
}

std::size_t ParseSize(const std::string& text, const char* name) {
  const std::int64_t value = ParseInt64(text, name);
  if (value <= 0) {
    throw std::invalid_argument(std::string(name) + " must be positive");
  }
  return static_cast<std::size_t>(value);
}

bool ParseBool(const std::string& text, const char* name) {
  if (text == "true" || text == "1") {
    return true;
  }
  if (text == "false" || text == "0") {
    return false;
  }
  throw std::invalid_argument(std::string(name) + " must be true or false");
}

std::vector<std::int64_t> ParseTokenIds(
    const std::string& text,
    bool allow_empty,
    const char* name) {
  std::vector<std::int64_t> result;
  if (Trim(text).empty()) {
    if (allow_empty) {
      return result;
    }
    throw std::invalid_argument(std::string(name) + " must not be empty");
  }
  std::size_t start = 0;
  while (start <= text.size()) {
    const std::size_t separator = text.find(',', start);
    const std::string item = Trim(text.substr(
        start,
        separator == std::string::npos ? std::string::npos : separator - start));
    if (item.empty()) {
      throw std::invalid_argument(std::string(name) + " contains an empty item");
    }
    const std::int64_t value = ParseInt64(item, name);
    if (value < 0) {
      throw std::invalid_argument(std::string(name) + " contains a negative ID");
    }
    result.push_back(value);
    if (separator == std::string::npos) {
      break;
    }
    start = separator + 1;
  }
  return result;
}

std::map<std::string, std::string> ParseOptions(int argc, char** argv) {
  std::map<std::string, std::string> result;
  for (int index = 1; index < argc; ++index) {
    std::string argument(argv[index]);
    if (argument == "--help" || argument == "-h") {
      Usage(std::cout);
      std::exit(0);
    }
    if (argument.rfind("--", 0) != 0) {
      throw std::invalid_argument("unexpected positional argument: " + argument);
    }
    const std::size_t equals = argument.find('=');
    std::string name;
    std::string value;
    if (equals == std::string::npos) {
      name = argument.substr(2);
      if (index + 1 >= argc) {
        throw std::invalid_argument("missing value for --" + name);
      }
      value = argv[++index];
    } else {
      name = argument.substr(2, equals - 2);
      value = argument.substr(equals + 1);
    }
    if (!result.emplace(name, value).second) {
      throw std::invalid_argument("option repeated: --" + name);
    }
  }
  return result;
}

std::string TakeRequired(
    std::map<std::string, std::string>* values,
    const std::string& name) {
  const auto iterator = values->find(name);
  if (iterator == values->end()) {
    throw std::invalid_argument("missing required option --" + name);
  }
  std::string result = iterator->second;
  values->erase(iterator);
  return result;
}

std::string TakeOptional(
    std::map<std::string, std::string>* values,
    const std::string& name,
    std::string fallback) {
  const auto iterator = values->find(name);
  if (iterator == values->end()) {
    return fallback;
  }
  std::string result = iterator->second;
  values->erase(iterator);
  return result;
}

std::string NormalizeHash(std::string value, const std::string& name) {
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char item) { return static_cast<char>(std::tolower(item)); });
  if (value.size() != 64 ||
      !std::all_of(value.begin(), value.end(), [](unsigned char item) {
        return std::isxdigit(item) != 0;
      })) {
    throw std::invalid_argument(name + " must be 64 hexadecimal characters");
  }
  return value;
}

Arguments ParseArguments(int argc, char** argv) {
  auto values = ParseOptions(argc, argv);
  Arguments result;
  for (std::size_t index = 0; index < result.models.size(); ++index) {
    auto& model = result.models[index];
    if (index >= 2) {
      const std::string path = TakeOptional(&values, model.role, "");
      const std::string hash = TakeOptional(
          &values, std::string(model.role) + "-sha256", "");
      if (path.empty() != hash.empty()) {
        throw std::invalid_argument(
            std::string("--") + model.role + " and --" + model.role +
            "-sha256 must be supplied together");
      }
      if (!path.empty()) {
        model.path = path;
        model.sha256 = NormalizeHash(
            hash, std::string(model.role) + "-sha256");
      }
    } else {
      model.path = TakeRequired(&values, model.role);
      model.sha256 = NormalizeHash(
          TakeRequired(&values, std::string(model.role) + "-sha256"),
          std::string(model.role) + "-sha256");
    }
  }
  const bool fused = !result.models[5].path.empty();
  const bool has_decode = !result.models[2].path.empty();
  const bool has_draft = !result.models[3].path.empty();
  const bool has_verify = !result.models[4].path.empty();
  if (fused) {
    if (!has_decode || has_draft || has_verify) {
      throw std::invalid_argument(
          "fused topology requires target-decode1 and forbids separate "
          "draft-propose/target-verify-commit");
    }
  } else if (!has_draft || !has_verify) {
    throw std::invalid_argument(
        "non-fused topology requires draft-propose and target-verify-commit");
  }
  result.output = TakeRequired(&values, "output");
  result.prompt_token_ids = ParseTokenIds(
      TakeRequired(&values, "prompt-token-ids"), false, "prompt-token-ids");
  result.eos_token_ids = ParseTokenIds(
      TakeOptional(&values, "eos-token-ids", ""), true, "eos-token-ids");
  result.pad_token_id = ParseInt64(
      TakeOptional(&values, "pad-token-id", "0"), "pad-token-id");
  result.max_new_tokens = ParseSize(
      TakeOptional(&values, "max-new-tokens", "32"), "max-new-tokens");
  result.max_draft_tokens = ParseSize(
      TakeOptional(&values, "max-draft-tokens", "15"), "max-draft-tokens");
  result.dflash_sync_window = ParseSize(
      TakeOptional(&values, "dflash-sync-window", "1"),
      "dflash-sync-window");
  if (result.dflash_sync_window == 0 || result.dflash_sync_window > 8) {
    throw std::invalid_argument("dflash-sync-window must be in 1..8");
  }
  const std::string prefill_completion_policy = TakeOptional(
      &values, "prefill-completion-policy", "separate");
  if (prefill_completion_policy == "separate") {
    result.coalesce_prefill_with_first_verify = false;
  } else if (prefill_completion_policy == "coalesce-first-verify") {
    result.coalesce_prefill_with_first_verify = true;
  } else {
    throw std::invalid_argument(
        "prefill-completion-policy must be separate or coalesce-first-verify");
  }
  const std::string zero_accept_fallback_policy = TakeOptional(
      &values, "zero-accept-fallback-policy", "disabled");
  if (zero_accept_fallback_policy == "disabled") {
    result.zero_accept_fallback_policy =
        ZeroAcceptFallbackPolicy::kDisabled;
  } else if (zero_accept_fallback_policy == "request-target-only") {
    result.zero_accept_fallback_policy =
        ZeroAcceptFallbackPolicy::kRequestTargetOnly;
  } else {
    throw std::invalid_argument(
        "zero-accept-fallback-policy must be disabled or request-target-only");
  }
  result.warmup = ParseSize(TakeOptional(&values, "warmup", "3"), "warmup");
  result.repetitions = ParseSize(
      TakeOptional(&values, "repetitions", "10"), "repetitions");
  result.progress = ParseBool(
      TakeOptional(&values, "progress", "true"), "progress");
  const std::string measurement_protocol = TakeOptional(
      &values, "measurement-protocol", "evidence");
  if (measurement_protocol == "evidence") {
    result.measurement_protocol = MeasurementProtocol::kEvidence;
  } else if (measurement_protocol == "profile") {
    result.measurement_protocol = MeasurementProtocol::kProfile;
  } else {
    throw std::invalid_argument(
        "measurement-protocol must be evidence or profile");
  }
  const std::string state_reset_policy = TakeOptional(
      &values, "state-reset-policy", "async-memset");
  if (state_reset_policy == "async-memset") {
    result.state_reset_policy = IncrementalStateResetPolicy::kAsyncMemset;
  } else if (state_reset_policy == "immutable-zero") {
    result.state_reset_policy = IncrementalStateResetPolicy::kImmutableZero;
  } else {
    throw std::invalid_argument(
        "state-reset-policy must be async-memset or immutable-zero");
  }
  const std::string decode_carrier_policy = TakeOptional(
      &values, "decode-carrier-policy", "last-token-d2d");
  if (decode_carrier_policy == "last-token-d2d") {
    result.decode_carrier_policy =
        IncrementalDecodeCarrierPolicy::kLastTokenDeviceCompact;
  } else if (decode_carrier_policy == "one-token-h2d") {
    result.decode_carrier_policy =
        IncrementalDecodeCarrierPolicy::kOneTokenHostFallback;
  } else {
    throw std::invalid_argument(
        "decode-carrier-policy must be last-token-d2d or one-token-h2d");
  }
  const std::string draft_feature_policy = TakeOptional(
      &values, "draft-feature-policy", "fixed-16");
  if (draft_feature_policy == "fixed-16") {
    result.draft_feature_policy =
        IncrementalDraftFeaturePolicy::kFixedVerifyWidth;
  } else if (draft_feature_policy == "committed-prefix") {
    result.draft_feature_policy =
        IncrementalDraftFeaturePolicy::kCommittedPrefix;
  } else {
    throw std::invalid_argument(
        "draft-feature-policy must be fixed-16 or committed-prefix");
  }
  const std::int64_t device_id = ParseInt64(
      TakeOptional(&values, "device-id", "0"), "device-id");
  if (device_id < 0 || result.pad_token_id < 0) {
    throw std::invalid_argument("device and pad token IDs must be non-negative");
  }
  result.device_id = static_cast<int>(device_id);
  if (!values.empty()) {
    throw std::invalid_argument("unknown option --" + values.begin()->first);
  }
  if (result.measurement_protocol == MeasurementProtocol::kEvidence &&
      (result.warmup != 3 || result.repetitions != 10)) {
    throw std::invalid_argument(
        "evidence protocol requires exactly 3 warmups and 10 repetitions");
  }
  if (result.measurement_protocol == MeasurementProtocol::kProfile &&
      (result.warmup != 1 || result.repetitions != 1)) {
    throw std::invalid_argument(
        "profile protocol requires exactly 1 warmup and 1 repetition");
  }
  return result;
}

void PrintProgress(bool enabled, const std::string& message) {
  if (enabled) {
    std::cerr << "[qwen35-dflash-incremental] " << message << std::endl;
  }
}

ProgressCallback MakeProgressCallback(bool enabled) {
  if (!enabled) {
    return {};
  }
  return [](const ProgressEvent& event) {
    std::cerr << "[qwen35-dflash-incremental] phase=" << event.phase
              << " run=" << event.run_index << '/' << event.run_total
              << " mode=" << qwen35::dflash::ModeName(event.mode)
              << " stage=" << event.stage
              << " generated=" << event.generated_tokens << '/'
              << event.max_new_tokens << " prefix=" << event.prefix_tokens
              << " graph_calls=" << event.graph_calls;
    if (event.decode_iteration != 0) {
      std::cerr << " decode_iteration=" << event.decode_iteration;
    }
    if (event.elapsed_ms > 0.0) {
      std::cerr << " elapsed_ms=" << std::fixed << std::setprecision(3)
                << event.elapsed_ms;
    }
    std::cerr << std::endl;
  };
}

std::string JsonEscape(const std::string& value) {
  std::ostringstream output;
  for (const unsigned char item : value) {
    switch (item) {
      case '"': output << "\\\""; break;
      case '\\': output << "\\\\"; break;
      case '\b': output << "\\b"; break;
      case '\f': output << "\\f"; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default:
        if (item < 0x20U) {
          output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<int>(item) << std::dec;
        } else {
          output << static_cast<char>(item);
        }
    }
  }
  return output.str();
}

void WriteTokenIds(std::ostream& output, const std::vector<std::int64_t>& values) {
  output << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) output << ',';
    output << values[index];
  }
  output << ']';
}

void WriteDoubles(std::ostream& output, const std::vector<double>& values) {
  output << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) output << ',';
    output << values[index];
  }
  output << ']';
}

void WriteDistribution(std::ostream& output, const Distribution& value) {
  output << "{\"count\":" << value.count << ",\"min\":" << value.min
         << ",\"max\":" << value.max << ",\"mean\":" << value.mean
         << ",\"median\":" << value.median << ",\"p90\":" << value.p90
         << ",\"population_stdev\":" << value.population_stdev << '}';
}

void WriteMeasurement(
    std::ostream& output,
    const GenerationMeasurement& value,
    std::size_t repetition) {
  output << "{\"repetition\":" << repetition
         << ",\"generated_token_ids\":";
  WriteTokenIds(output, value.generated_token_ids);
  output << ",\"stop_reason\":\"" << JsonEscape(value.stop_reason) << "\""
         << ",\"counters\":{\"graph_calls\":" << value.counters.graph_calls
         << ",\"drafted_tokens\":" << value.counters.drafted_tokens
         << ",\"accepted_draft_tokens\":"
         << value.counters.accepted_draft_tokens
         << ",\"rejected_draft_tokens\":"
         << value.counters.rejected_draft_tokens
         << ",\"speculative_transactions\":"
         << value.counters.speculative_transactions
         << ",\"prefill_speculative_windows\":"
         << value.counters.prefill_speculative_windows
         << ",\"zero_accept_transactions\":"
         << value.counters.zero_accept_transactions
         << ",\"zero_accept_fallback_activations\":"
         << value.counters.zero_accept_fallback_activations
         << ",\"target_only_fallback_iterations\":"
         << value.counters.target_only_fallback_iterations
         << ",\"decode_iterations\":" << value.counters.decode_iterations
         << "},\"latency_ms\":{\"prefill\":" << value.prefill_ms
         << ",\"decode\":" << value.decode_ms
         << ",\"model_total\":" << value.model_total_ms
         << "},\"decode_iteration_ms\":";
  WriteDoubles(output, value.decode_iteration_ms);
  output << '}';
}

void WriteBenchmark(std::ostream& output, const BenchmarkResult& value) {
  output << "{\"status\":\"PASS\",\"generation_mode\":\""
         << qwen35::dflash::ModeName(value.mode) << "\",\"warmup\":"
         << value.warmup << ",\"repetitions\":" << value.repetitions
         << ",\"stable_generated_token_ids\":";
  WriteTokenIds(output, value.stable_generated_token_ids);
  output << ",\"stable_stop_reason\":\""
         << JsonEscape(value.stable_stop_reason) << "\",\"latency_ms\":{";
  output << "\"prefill\":";
  WriteDistribution(output, value.prefill_ms);
  output << ",\"decode\":";
  WriteDistribution(output, value.decode_ms);
  output << ",\"model_total\":";
  WriteDistribution(output, value.model_total_ms);
  output << "},\"totals\":{\"graph_calls\":" << value.total_graph_calls
         << ",\"drafted_tokens\":" << value.total_drafted_tokens
         << ",\"accepted_draft_tokens\":"
         << value.total_accepted_draft_tokens
         << ",\"rejected_draft_tokens\":"
         << value.total_rejected_draft_tokens
         << ",\"zero_accept_transactions\":"
         << value.total_zero_accept_transactions
         << ",\"zero_accept_fallback_activations\":"
         << value.total_zero_accept_fallback_activations
         << ",\"target_only_fallback_iterations\":"
         << value.total_target_only_fallback_iterations
         << "},\"acceptance_rate\":" << value.acceptance_rate
         << ",\"generated_tokens_per_second\":"
         << value.generated_tokens_per_second << ",\"measurements\":[";
  for (std::size_t index = 0; index < value.measurements.size(); ++index) {
    if (index != 0) output << ',';
    WriteMeasurement(output, value.measurements[index], index);
  }
  output << "]}";
}

void WriteReport(
    std::ostream& output,
    const Arguments& arguments,
    const qwen35::dflash::AclIncrementalExecutor& executor,
    double load_ms,
    double benchmark_wall_ms,
    const PairedBenchmarkResult& result) {
  const auto& memory = executor.model_memory();
  std::vector<const ModelArgument*> present_models;
  for (const auto& model : arguments.models) {
    if (!model.path.empty()) {
      present_models.push_back(&model);
    }
  }
  if (memory.size() != present_models.size()) {
    throw std::runtime_error("incremental model memory record count differs");
  }
  const std::size_t sum_work = std::accumulate(
      memory.begin(), memory.end(), std::size_t{0},
      [](std::size_t total, const IncrementalModelMemory& item) {
        return total + item.work_bytes;
      });
  const std::size_t sum_weight = std::accumulate(
      memory.begin(), memory.end(), std::size_t{0},
      [](std::size_t total, const IncrementalModelMemory& item) {
        return total + item.weight_bytes;
      });
  const std::size_t max_work = std::max_element(
      memory.begin(), memory.end(),
      [](const auto& left, const auto& right) {
        return left.work_bytes < right.work_bytes;
      })->work_bytes;
  const auto& execution = executor.execution_stats();
  const bool fused_speculative_step = executor.fused_speculative_step();
  const char* resident_model_count =
      (executor.unified_target_step() || fused_speculative_step)
      ? "four"
      : "five";
  const char* physical_topology = fused_speculative_step
      ? "split-prefill-head-four-resident-fused-speculative-step-v1"
      : (executor.unified_target_step()
             ? "split-prefill-head-four-resident-unified-target-step-v1"
             : "split-prefill-head-five-resident-v1");
  const bool formal_latency_evidence =
      arguments.measurement_protocol == MeasurementProtocol::kEvidence;
  if (result.ordinary.total_zero_accept_transactions != 0 ||
      result.ordinary.total_zero_accept_fallback_activations != 0 ||
      result.ordinary.total_target_only_fallback_iterations != 0 ||
      result.dflash.total_zero_accept_fallback_activations >
          result.dflash.total_zero_accept_transactions ||
      (result.dflash.total_zero_accept_fallback_activations != 0 &&
       result.dflash.total_target_only_fallback_iterations <
           result.dflash.total_zero_accept_fallback_activations) ||
      (arguments.zero_accept_fallback_policy ==
           ZeroAcceptFallbackPolicy::kDisabled &&
       (result.dflash.total_zero_accept_fallback_activations != 0 ||
        result.dflash.total_target_only_fallback_iterations != 0))) {
    throw std::runtime_error(
        "zero-accept Target-only scheduler counters do not close");
  }
  const std::size_t model_executions =
      execution.target_prefill_executions +
      execution.target_prefill_head_executions +
      execution.target_decode1_executions +
      (fused_speculative_step
           ? execution.fused_speculative_step_executions
           : execution.draft_propose_executions +
                 execution.target_verify_commit_executions);
  if (fused_speculative_step
          ? (execution.fused_speculative_step_executions !=
                 execution.draft_propose_executions ||
             execution.fused_speculative_step_executions !=
                 execution.target_verify_commit_executions ||
             execution.draft_to_verify_model_launches_elided !=
                 execution.fused_speculative_step_executions)
          : (execution.fused_speculative_step_executions != 0 ||
             execution.draft_to_verify_model_launches_elided != 0)) {
    throw std::runtime_error(
        "fused Draft-to-verify physical execution counters do not close");
  }
  const auto& model_execution_trace = executor.model_execution_trace();
  if (formal_latency_evidence && !model_execution_trace.empty()) {
    throw std::runtime_error(
        "formal evidence unexpectedly enabled model execution tracing");
  }
  if (!formal_latency_evidence &&
      model_execution_trace.size() != model_executions) {
    throw std::runtime_error(
        "profile model execution trace does not close");
  }
  if (execution.speculative_sync_windows +
              execution.speculative_synchronizations_elided +
              execution.prefill_verify_coalesced_windows !=
          execution.target_verify_commit_executions ||
      execution.speculative_d2h_operations_elided !=
          execution.speculative_synchronizations_elided ||
      execution.speculative_window_staging_bytes !=
          execution.speculative_window_staging_operations *
              execution.compact_verify_result_bytes ||
      execution.speculative_window_staging_operations != 0 ||
      execution.speculative_window_direct_output_bytes !=
          execution.speculative_window_direct_output_bindings *
              execution.compact_verify_result_bytes ||
      execution.speculative_window_direct_output_bindings >
          execution.target_verify_commit_executions ||
      (arguments.dflash_sync_window <= 2 &&
       execution.speculative_window_direct_output_bindings != 0) ||
      execution.speculative_window_staging_device_bytes !=
          executor.max_speculative_sync_window() *
              execution.compact_slot_bytes ||
      execution.speculative_window_staging_pinned_host_bytes !=
          execution.speculative_window_staging_device_bytes ||
      execution.prefill_verify_synchronizations_elided !=
          execution.prefill_verify_coalesced_windows ||
      execution.prefill_verify_d2h_operations_elided !=
          execution.prefill_verify_coalesced_windows ||
      execution.prefill_verify_prefill_slot0_windows +
              execution.prefill_verify_prefill_slot1_windows !=
          execution.prefill_verify_coalesced_windows ||
      execution.compact_slot_bytes <
          std::max(
              execution.compact_ordinary_result_bytes,
              execution.compact_verify_result_bytes) ||
      execution.speculative_d2h_padding_bytes !=
          execution.speculative_d2h_operations_elided *
              (execution.compact_slot_bytes -
               execution.compact_verify_result_bytes) ||
      execution.prefill_verify_d2h_padding_bytes !=
          execution.prefill_verify_prefill_slot0_windows *
                  (execution.compact_slot_bytes -
                   execution.compact_ordinary_result_bytes) +
              execution.prefill_verify_prefill_slot1_windows *
                  (execution.compact_slot_bytes -
                   execution.compact_verify_result_bytes) ||
      execution.stream_synchronizations !=
          execution.prefill_completion_synchronizations +
              execution.target_decode1_executions +
              execution.speculative_sync_windows ||
      execution.stream_synchronizations +
              execution.speculative_synchronizations_elided +
              execution.prefill_verify_synchronizations_elided !=
          execution.prefill_completion_synchronizations +
              execution.target_decode1_executions +
              execution.target_verify_commit_executions ||
      execution.device_to_host_operations +
              execution.speculative_d2h_operations_elided +
              execution.prefill_verify_d2h_operations_elided !=
          execution.prefill_completion_synchronizations +
              execution.target_decode1_executions +
              execution.target_verify_commit_executions) {
    throw std::runtime_error(
        "speculative synchronization window counters do not close");
  }
  if (execution.draft_propose_executions <
      execution.prefill_draft_propose_executions) {
    throw std::runtime_error(
        "verify-source Draft execution count underflowed");
  }
  const std::size_t verify_draft_executions =
      execution.draft_propose_executions -
      execution.prefill_draft_propose_executions;
  if (execution.draft_verify_fixed_width_executions +
              execution.draft_verify_committed_prefix_executions +
              execution.draft_verify_pending_upper_bound_executions !=
          verify_draft_executions ||
      execution.draft_verify_full_width_equivalent_rows !=
          verify_draft_executions * (executor.proposal_width() + 1) ||
      execution.draft_verify_feature_input_rows +
              execution.draft_verify_feature_rows_elided !=
          execution.draft_verify_full_width_equivalent_rows ||
      execution.draft_verify_dynamic_gear_count !=
          executor.proposal_width() + 1 ||
      execution.draft_prefill_dynamic_gear_count !=
          execution.prefill_staging_slots ||
      execution.draft_dynamic_gear_count !=
          execution.draft_verify_dynamic_gear_count +
              execution.draft_prefill_dynamic_gear_count) {
    throw std::runtime_error(
        "Draft committed-feature row counters or dynamic gears do not close");
  }
  if (executor.draft_feature_policy() ==
      IncrementalDraftFeaturePolicy::kFixedVerifyWidth) {
    if (execution.draft_verify_fixed_width_executions !=
            verify_draft_executions ||
        execution.draft_verify_committed_prefix_executions != 0 ||
        execution.draft_verify_pending_upper_bound_executions != 0 ||
        execution.draft_verify_feature_rows_elided != 0) {
      throw std::runtime_error(
          "fixed-16 Draft feature policy counters do not close");
    }
  } else if (execution.draft_verify_fixed_width_executions != 0 ||
             execution.draft_verify_committed_prefix_executions +
                     execution.draft_verify_pending_upper_bound_executions !=
                 verify_draft_executions) {
    throw std::runtime_error(
        "committed-prefix Draft feature policy counters do not close");
  }
  const double speedup = result.dflash.model_total_ms.median > 0.0
      ? result.ordinary.model_total_ms.median /
            result.dflash.model_total_ms.median
      : 0.0;

  output << std::setprecision(17)
         << "{\"schema_version\":12,\"status\":\"PASS\","
         << "\"scope\":\"AscendCL C++ "
         << resident_model_count
         << "-resident-OM paired model loop\","
         << "\"runner_id\":\"qwen35-dflash-ascendcl-cpp-incremental-v3\","
         << "\"runner_version\":\"" << JsonEscape(QWEN35_DFLASH_RUNNER_VERSION)
         << "\",\"candidate_status\":\"APPROVED_IN_IMPLEMENTATION_NOT_ACTIVE\","
         << "\"cpu_fallback\":false,\"device_id\":" << arguments.device_id
         << ",\"models\":[";
  for (std::size_t index = 0; index < present_models.size(); ++index) {
    if (index != 0) output << ',';
    const auto& model = *present_models[index];
    if (memory[index].role != model.role) {
      throw std::runtime_error("incremental model memory role order differs");
    }
    output << "{\"role\":\"" << model.role << "\",\"path\":\""
           << JsonEscape(std::filesystem::absolute(model.path).string())
           << "\",\"sha256\":\"" << model.sha256
           << "\",\"model_id\":" << memory[index].model_id
           << ",\"work_bytes\":" << memory[index].work_bytes
           << ",\"weight_bytes\":" << memory[index].weight_bytes << '}';
  }
  output << "],\"abi\":{"
         << "\"id\":\"qwen35-4b-dflash-ascend310p-incremental-performance-v2\","
         << "\"physical_topology\":\""
         << physical_topology
         << "\","
         << "\"state_policy\":\"explicit device-resident ping-pong\","
         << "\"sequence_capacity\":" << executor.sequence_length()
         << ",\"prefill_width\":" << executor.prefill_width()
         << ",\"proposal_width\":" << executor.proposal_width()
         << ",\"verify_width\":" << executor.proposal_width() + 1
         << ",\"eos_table_width\":" << executor.eos_table_width() << "},"
         << "\"model_memory_query\":{\"source\":\"aclmdlQuerySize\","
         << "\"sum_work_bytes\":" << sum_work
         << ",\"max_work_bytes\":" << max_work
         << ",\"sum_weight_bytes\":" << sum_weight
         << ",\"state_device_bytes\":" << execution.state_device_bytes
         << ",\"working_state_device_bytes\":"
         << execution.working_state_device_bytes
         << ",\"immutable_zero_state_device_bytes\":"
         << execution.immutable_zero_state_device_bytes
         << ",\"state_reset_bytes_per_request\":"
         << execution.state_reset_bytes_per_request
         << ",\"carrier_device_bytes\":" << execution.carrier_device_bytes
         << ",\"compact_ping_pong_device_bytes\":"
         << execution.compact_ping_pong_device_bytes
         << ",\"speculative_window_staging_device_bytes\":"
         << execution.speculative_window_staging_device_bytes
         << ",\"speculative_window_staging_pinned_host_bytes\":"
         << execution.speculative_window_staging_pinned_host_bytes
         << ",\"compact_slot_bytes\":" << execution.compact_slot_bytes
         << ",\"compact_ordinary_result_bytes\":"
         << execution.compact_ordinary_result_bytes
         << ",\"compact_verify_result_bytes\":"
         << execution.compact_verify_result_bytes
         << ",\"prefill_control_bytes_per_slot\":"
         << execution.prefill_control_bytes_per_slot
         << ",\"prefill_base_control_bytes_per_slot\":"
         << execution.prefill_base_control_bytes_per_slot
         << ",\"prefill_count_control_bytes_per_slot\":"
         << execution.prefill_count_control_bytes_per_slot
         << ",\"prefill_proposal_control_bytes_per_slot\":"
         << execution.prefill_proposal_control_bytes_per_slot
         << ",\"prefill_persistent_control_tail_bytes_per_slot\":"
         << execution.prefill_persistent_control_tail_bytes_per_slot
         << ",\"prefill_staging_pinned_host_bytes\":"
         << execution.prefill_staging_pinned_host_bytes
         << ",\"proposal_count_staging_pinned_host_bytes\":"
         << execution.proposal_count_staging_pinned_host_bytes
         << ",\"prefill_feature_slab_bytes\":"
         << execution.prefill_feature_slab_bytes
         << ",\"prefill_feature_arena_bytes\":"
         << execution.prefill_feature_arena_bytes
         << ",\"draft_dynamic_gear_count\":"
         << execution.draft_dynamic_gear_count
         << ",\"draft_verify_dynamic_gear_count\":"
         << execution.draft_verify_dynamic_gear_count
         << ",\"draft_prefill_dynamic_gear_count\":"
         << execution.draft_prefill_dynamic_gear_count
         << ",\"target_step_dynamic_gear_count\":"
         << execution.target_step_dynamic_gear_count
         << ",\"target_step_zero_count_device_bytes\":"
         << execution.target_step_zero_count_device_bytes
         << ",\"explicit_allocated_device_bytes_excluding_runtime\":"
         << max_work + sum_weight + execution.state_device_bytes +
                execution.carrier_device_bytes
         << ",\"load_policy\":\""
         << resident_model_count
         << " aclmdlLoadFromFileWithMem sessions; "
            "one max-sized serial workspace; separate per-artifact weights; "
            "no cross-OM weight sharing assumed\"},"
         << "\"protocol\":{\"warmup\":" << arguments.warmup
         << ",\"repetitions\":" << arguments.repetitions
         << ",\"kind\":\""
         << (formal_latency_evidence ? "evidence" : "profile")
         << "\",\"formal_latency_evidence\":"
         << (formal_latency_evidence ? "true" : "false")
         << ",\"profile_model_execution_trace_enabled\":"
         << (formal_latency_evidence ? "false" : "true")
         << ",\"dflash_sync_window\":"
         << arguments.dflash_sync_window
         << ",\"maximum_supported_dflash_sync_window\":"
         << executor.max_speculative_sync_window()
         << ",\"zero_accept_fallback_policy\":\""
         << (arguments.zero_accept_fallback_policy ==
                     ZeroAcceptFallbackPolicy::kRequestTargetOnly
                 ? "request-target-only"
                 : "disabled")
         << "\",\"zero_accept_fallback_policy_description\":\""
         << (arguments.zero_accept_fallback_policy ==
                     ZeroAcceptFallbackPolicy::kRequestTargetOnly
                 ? "after the first completed zero-accept transaction, consume the full synchronized window and use authoritative one-row Target steps for the rest of that request"
                 : "every eligible DFlash iteration continues to execute Draft and Target verify regardless of observed acceptance")
         << "\""
         << ",\"decode_iteration_scope\":\"one host-visible "
            "synchronization window; a DFlash window may contain one to "
            "eight complete speculative transactions\""
         << ",\"order\":\"alternating ordinary/DFlash in one "
         << resident_model_count
         << "-model process\","
         << "\"model_load_excluded_from_latency\":true,"
         << "\"device_memory_allocation_policy\":\""
         << qwen35::dflash::kDeviceMemoryAllocationPolicyName << "\","
         << "\"prefill_completion_policy\":\""
         << (arguments.coalesce_prefill_with_first_verify
                 ? "coalesce-first-verify"
                 : "separate")
         << "\",\"prefill_completion_policy_description\":\""
         << (arguments.coalesce_prefill_with_first_verify
                 ? "intermediate prompt chunks stay queued; on eligible "
                   "DFlash requests the final prefill and first verify share "
                   "one compact D2H and stream synchronization; first-token "
                   "host visibility is delayed until that verify completes"
                 : "intermediate prompt chunks stay queued; final chunk "
                   "performs the only prefill compact D2H and stream "
                   "synchronization before decode")
         << "\","
         << "\"prefill_control_policy\":\"each chunk uploads one prefix "
            "ending after IDs/effective length, final-Draft total count, a "
            "changed proposal count, or a changed process-resident EOS "
            "table/count; all device subsegments start at 64-byte "
            "boundaries\","
         << "\"prefill_draft_policy\":\""
         << (fused_speculative_step
                 ? "Target feature slabs stay device-resident; no prompt "
                   "chunk launches Draft separately; the first prebound "
                   "dynamic-gear fused transaction consumes the complete "
                   "prompt feature batch"
                 : "Target feature slabs stay device-resident; non-final "
                   "prompt chunks execute no Draft OM; final prompt "
                   "completion executes one prebound dynamic-gear Draft OM")
         << "\","
         << "\"prefill_feature_arena_policy\":\"contiguous 64-row FP16 slabs "
            "with 64-byte-aligned starts and one terminal guard; no D2D "
            "compaction\","
         << "\"prefill_target_lm_head_policy\":\"target-prefill body contains "
            "no LM head; target-prefill-head executes exactly once after the "
            "final physical prompt chunk\","
         << "\"device_suballocation_policy\":\"64-byte segment starts; "
            "ALIGN_UP(payload,32)+32 reserved span\","
         << "\"decode_carrier_policy\":\""
         << qwen35::dflash::IncrementalDecodeCarrierPolicyName(
                executor.decode_carrier_policy())
         << "\",\"decode_input_policy\":\""
         << (executor.decode_carrier_policy() ==
                     IncrementalDecodeCarrierPolicy::kLastTokenDeviceCompact
                 ? "the last committed token from any compact Target result "
                   "stays on device; row zero binds directly and later rows "
                   "use an 8-byte D2D copy into the aligned decode scalar; "
                   "caller overrides use the pinned-host H2D fallback"
                 : "one-token compact Target results bind row zero directly; "
                   "multi-token commits and caller overrides use the "
                   "pinned-host 8-byte H2D fallback")
         << "\",\"draft_feature_policy\":\""
         << qwen35::dflash::IncrementalDraftFeaturePolicyName(
                executor.draft_feature_policy())
         << "\",\"draft_feature_policy_description\":\""
         << (executor.draft_feature_policy() ==
                     IncrementalDraftFeaturePolicy::kCommittedPrefix
                 ? "after a synchronized verify, Draft binds exactly accepted+1 "
                   "leading Target feature rows; each later unsynchronized "
                   "transaction binds its predecessor's causal K+1 upper "
                   "bound; masked "
                   "suffix cache writes are scratch and overwritten before "
                   "becoming visible"
                 : "verify-source Draft binds the original physical N=16; "
                   "this is the rollback and matched-baseline route")
         << "\",\"target_step_zero_count_policy\":\""
         << (executor.unified_target_step()
                 ? "T=1 datasets bind a process-resident aligned INT32 zero; "
                   "positive K stays in the mutable proposal carrier"
                 : "not applicable; target-decode1 is a separate static OM")
         << "\","
         << "\"state_reset_policy\":\""
         << qwen35::dflash::IncrementalStateResetPolicyName(
                executor.state_reset_policy())
         << "\",\"state_reset_description\":\""
         << (executor.state_reset_policy() ==
                     IncrementalStateResetPolicy::kAsyncMemset
                 ? "per-request Target/Draft clear queued inside first prefill"
                 : "read-only Target/Draft zero state initialized at startup")
         << "\",\"state_reset_only_barriers\":0,"
         << "\"state_reset_device_work_included_by_prefill_barrier\":"
         << (executor.state_reset_policy() ==
                     IncrementalStateResetPolicy::kAsyncMemset
                 ? "true"
                 : "false")
         << ",\"state_zero_initialization_included_in_startup\":"
         << (executor.state_reset_policy() ==
                     IncrementalStateResetPolicy::kImmutableZero
                 ? "true"
                 : "false")
         << ','
         << "\"progress_emission_excluded_from_model_timers\":true,"
         << "\"live_progress_enabled\":"
         << (arguments.progress ? "true" : "false") << "},"
         << "\"execution_io_counters\":{"
         << "\"scope\":\"paired warmups and measurements\","
         << "\"proposal_policy\":\""
         << (fused_speculative_step
                 ? "verify IDs remain internal to fused Draft-to-Target; no "
                   "proposal tensor crosses an OM or host boundary"
                 : "Draft-to-verify device carrier; no proposal D2H/H2D")
         << "\","
         << "\"result_policy\":\"one logical compact result per complete "
            "transaction; a two-transaction DFlash window coalesces adjacent "
            "ping-pong slots, while windows of three to eight bind each "
            "452-byte Verify result directly into a 4 KiB staging arena "
            "before one D2H, with no post-Verify staging D2D; an "
            "eligible final prefill "
            "and first verify may independently share one contiguous D2H; "
            "one barrier per completed prompt/decode window; no host-visible "
            "result for intermediate prefill chunks\","
         << "\"model_executions\":" << model_executions
         << ",\"target_prefill_executions\":"
         << execution.target_prefill_executions
         << ",\"target_prefill_head_executions\":"
         << execution.target_prefill_head_executions
         << ",\"target_prefill_head_executions_elided\":"
         << execution.target_prefill_head_executions_elided
         << ",\"target_decode1_executions\":"
         << execution.target_decode1_executions
         << ",\"draft_propose_executions\":"
         << execution.draft_propose_executions
         << ",\"target_verify_commit_executions\":"
         << execution.target_verify_commit_executions
         << ",\"fused_speculative_step_executions\":"
         << execution.fused_speculative_step_executions
         << ",\"draft_to_verify_model_launches_elided\":"
         << execution.draft_to_verify_model_launches_elided
         << ",\"target_step_dynamic_gear_count\":"
         << execution.target_step_dynamic_gear_count
         << ",\"target_step_input_rows\":"
         << execution.target_step_input_rows
         << ",\"target_step_padded_rows_elided\":"
         << execution.target_step_padded_rows_elided
         << ",\"target_step_zero_count_device_bytes\":"
         << execution.target_step_zero_count_device_bytes
         << ",\"target_step_zero_count_bindings\":"
         << execution.target_step_zero_count_bindings
         << ",\"stream_synchronizations\":"
         << execution.stream_synchronizations
         << ",\"speculative_sync_windows\":"
         << execution.speculative_sync_windows
         << ",\"speculative_synchronizations_elided\":"
         << execution.speculative_synchronizations_elided
         << ",\"speculative_d2h_operations_elided\":"
         << execution.speculative_d2h_operations_elided
         << ",\"speculative_d2h_padding_bytes\":"
         << execution.speculative_d2h_padding_bytes
         << ",\"speculative_window_staging_operations\":"
         << execution.speculative_window_staging_operations
         << ",\"speculative_window_staging_bytes\":"
         << execution.speculative_window_staging_bytes
         << ",\"speculative_window_direct_output_bindings\":"
         << execution.speculative_window_direct_output_bindings
         << ",\"speculative_window_direct_output_bytes\":"
         << execution.speculative_window_direct_output_bytes
         << ",\"speculative_window_staging_device_bytes\":"
         << execution.speculative_window_staging_device_bytes
         << ",\"speculative_window_staging_pinned_host_bytes\":"
         << execution.speculative_window_staging_pinned_host_bytes
         << ",\"prefill_verify_coalesced_windows\":"
         << execution.prefill_verify_coalesced_windows
         << ",\"prefill_verify_synchronizations_elided\":"
         << execution.prefill_verify_synchronizations_elided
         << ",\"prefill_verify_d2h_operations_elided\":"
         << execution.prefill_verify_d2h_operations_elided
         << ",\"prefill_verify_d2h_padding_bytes\":"
         << execution.prefill_verify_d2h_padding_bytes
         << ",\"prefill_verify_prefill_slot0_windows\":"
         << execution.prefill_verify_prefill_slot0_windows
         << ",\"prefill_verify_prefill_slot1_windows\":"
         << execution.prefill_verify_prefill_slot1_windows
         << ",\"prefill_completion_synchronizations\":"
         << execution.prefill_completion_synchronizations
         << ",\"deferred_prefill_chunks\":"
         << execution.deferred_prefill_chunks
         << ",\"prefill_synchronizations_elided\":"
         << execution.prefill_synchronizations_elided
         << ",\"prefill_compact_downloads_elided\":"
         << execution.prefill_compact_downloads_elided
         << ",\"prefill_draft_propose_executions\":"
         << execution.prefill_draft_propose_executions
         << ",\"prefill_draft_propose_executions_elided\":"
         << execution.prefill_draft_propose_executions_elided
         << ",\"prefill_feature_rows_batched\":"
         << execution.prefill_feature_rows_batched
         << ",\"draft_verify_feature_input_rows\":"
         << execution.draft_verify_feature_input_rows
         << ",\"draft_verify_full_width_equivalent_rows\":"
         << execution.draft_verify_full_width_equivalent_rows
         << ",\"draft_verify_feature_rows_elided\":"
         << execution.draft_verify_feature_rows_elided
         << ",\"draft_verify_fixed_width_executions\":"
         << execution.draft_verify_fixed_width_executions
         << ",\"draft_verify_committed_prefix_executions\":"
         << execution.draft_verify_committed_prefix_executions
         << ",\"draft_verify_pending_upper_bound_executions\":"
         << execution.draft_verify_pending_upper_bound_executions
         << ",\"prefill_control_upload_operations\":"
         << execution.prefill_control_upload_operations
         << ",\"prefill_control_upload_bytes\":"
         << execution.prefill_control_upload_bytes
         << ",\"prefill_control_full_upload_operations\":"
         << execution.prefill_control_full_upload_operations
         << ",\"prefill_control_base_upload_operations\":"
         << execution.prefill_control_base_upload_operations
         << ",\"prefill_control_count_upload_operations\":"
         << execution.prefill_control_count_upload_operations
         << ",\"prefill_control_proposal_upload_operations\":"
         << execution.prefill_control_proposal_upload_operations
         << ",\"prefill_control_h2d_bytes_elided\":"
         << execution.prefill_control_h2d_bytes_elided
         << ",\"prefill_h2d_operations_elided\":"
         << execution.prefill_h2d_operations_elided
         << ",\"decode_id_upload_operations\":"
         << execution.decode_id_upload_operations
         << ",\"decode_id_upload_bytes\":"
         << execution.decode_id_upload_bytes
         << ",\"decode_id_device_carrier_hits\":"
         << execution.decode_id_device_carrier_hits
         << ",\"decode_id_multi_token_carrier_hits\":"
         << execution.decode_id_multi_token_carrier_hits
         << ",\"decode_id_h2d_operations_elided\":"
         << execution.decode_id_h2d_operations_elided
         << ",\"decode_id_device_compaction_operations\":"
         << execution.decode_id_device_compaction_operations
         << ",\"decode_id_device_compaction_bytes\":"
         << execution.decode_id_device_compaction_bytes
         << ",\"proposal_count_upload_operations\":"
         << execution.proposal_count_upload_operations
         << ",\"proposal_count_upload_bytes\":"
         << execution.proposal_count_upload_bytes
         << ",\"proposal_count_staging_pinned_host_bytes\":"
         << execution.proposal_count_staging_pinned_host_bytes
         << ",\"state_resets\":" << execution.state_resets
         << ",\"state_memset_operations\":"
         << execution.state_memset_operations
         << ",\"state_memset_bytes\":" << execution.state_memset_bytes
         << ",\"state_initialization_memset_operations\":"
         << execution.state_initialization_memset_operations
         << ",\"state_initialization_memset_bytes\":"
         << execution.state_initialization_memset_bytes
         << ",\"state_initialization_stream_synchronizations\":"
         << execution.state_initialization_stream_synchronizations
         << ",\"host_to_device_operations\":"
         << execution.host_to_device_operations
         << ",\"host_to_device_bytes\":" << execution.host_to_device_bytes
         << ",\"device_to_host_operations\":"
         << execution.device_to_host_operations
         << ",\"device_to_host_bytes\":" << execution.device_to_host_bytes
         << ",\"state_device_bytes\":" << execution.state_device_bytes
         << ",\"working_state_device_bytes\":"
         << execution.working_state_device_bytes
         << ",\"immutable_zero_state_device_bytes\":"
         << execution.immutable_zero_state_device_bytes
         << ",\"state_reset_bytes_per_request\":"
         << execution.state_reset_bytes_per_request
         << ",\"carrier_device_bytes\":" << execution.carrier_device_bytes
         << ",\"compact_ping_pong_device_bytes\":"
         << execution.compact_ping_pong_device_bytes
         << ",\"compact_slot_bytes\":" << execution.compact_slot_bytes
         << ",\"compact_ordinary_result_bytes\":"
         << execution.compact_ordinary_result_bytes
         << ",\"compact_verify_result_bytes\":"
         << execution.compact_verify_result_bytes
         << ",\"prefill_staging_slots\":"
         << execution.prefill_staging_slots
         << ",\"prefill_control_bytes_per_slot\":"
         << execution.prefill_control_bytes_per_slot
         << ",\"prefill_base_control_bytes_per_slot\":"
         << execution.prefill_base_control_bytes_per_slot
         << ",\"prefill_count_control_bytes_per_slot\":"
         << execution.prefill_count_control_bytes_per_slot
         << ",\"prefill_proposal_control_bytes_per_slot\":"
         << execution.prefill_proposal_control_bytes_per_slot
         << ",\"prefill_persistent_control_tail_bytes_per_slot\":"
         << execution.prefill_persistent_control_tail_bytes_per_slot
         << ",\"prefill_staging_pinned_host_bytes\":"
         << execution.prefill_staging_pinned_host_bytes
         << ",\"prefill_feature_slab_bytes\":"
         << execution.prefill_feature_slab_bytes
         << ",\"prefill_feature_arena_bytes\":"
         << execution.prefill_feature_arena_bytes
         << ",\"draft_dynamic_gear_count\":"
         << execution.draft_dynamic_gear_count
         << ",\"draft_verify_dynamic_gear_count\":"
         << execution.draft_verify_dynamic_gear_count
         << ",\"draft_prefill_dynamic_gear_count\":"
         << execution.draft_prefill_dynamic_gear_count
         << ",\"target_step_dynamic_gear_count\":"
         << execution.target_step_dynamic_gear_count
         << ",\"target_step_zero_count_device_bytes\":"
         << execution.target_step_zero_count_device_bytes
         << ",\"target_step_zero_count_bindings\":"
         << execution.target_step_zero_count_bindings
         << "},\"profile_model_execution_trace\":[";
  for (std::size_t index = 0; index < model_execution_trace.size(); ++index) {
    if (index != 0) output << ',';
    const auto& event = model_execution_trace[index];
    output << "{\"ordinal\":" << event.ordinal
           << ",\"model_id\":" << event.model_id
           << ",\"physical_rows\":" << event.physical_rows << '}';
  }
  output << "],\"prompt_token_ids\":";
  WriteTokenIds(output, arguments.prompt_token_ids);
  output << ",\"eos_token_ids\":";
  WriteTokenIds(output, arguments.eos_token_ids);
  output << ",\"limits\":{\"max_new_tokens\":"
         << arguments.max_new_tokens << ",\"max_draft_tokens\":"
         << arguments.max_draft_tokens << "},\"startup_ms\":{"
         << "\"acl_and_resident_model_load\":" << load_ms
         << ",\""
         << ((executor.unified_target_step() || fused_speculative_step)
                 ? "acl_and_four_model_load"
                 : "acl_and_five_model_load")
         << "\":" << load_ms
         << ",\"paired_benchmark_wall\":" << benchmark_wall_ms
         << "},\"ordinary\":";
  WriteBenchmark(output, result.ordinary);
  output << ",\"dflash\":";
  WriteBenchmark(output, result.dflash);
  output << ",\"ordinary_parity\":{\"status\":\"PASS\","
         << "\"token_id_mismatches\":" << result.token_id_mismatches
         << ",\"eos_mismatches\":" << result.eos_mismatches
         << "},\"dflash_speedup_over_ordinary_model_total_median\":"
         << speedup
         << ",\"claim_boundary\":\""
         << (formal_latency_evidence
                 ? "Candidate execution report only; promotion requires real "
                   "Ascend310P ordinary parity, memory fit and matched evidence."
                 : "Diagnostic profiling run only; collector-perturbed latency "
                   "is not promotion evidence.")
         << "\"}";
}

void AtomicWrite(const std::filesystem::path& path, const std::string& payload) {
  const std::filesystem::path absolute = std::filesystem::absolute(path);
  if (std::filesystem::exists(absolute)) {
    throw std::runtime_error("refusing to overwrite output: " + absolute.string());
  }
  std::filesystem::create_directories(absolute.parent_path());
  const std::filesystem::path temporary = absolute.string() + ".tmp";
  if (std::filesystem::exists(temporary)) {
    throw std::runtime_error("temporary output already exists: " + temporary.string());
  }
  {
    std::ofstream stream(temporary, std::ios::binary);
    if (!stream) {
      throw std::runtime_error("cannot create output: " + temporary.string());
    }
    stream << payload << '\n';
    stream.flush();
    if (!stream) {
      throw std::runtime_error("failed while writing output: " + temporary.string());
    }
  }
  std::filesystem::rename(temporary, absolute);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Arguments arguments = ParseArguments(argc, argv);
    const bool fused_speculative_step = !arguments.models[5].path.empty();
    const bool unified_target_step =
        !fused_speculative_step && arguments.models[2].path.empty();
    const char* topology_count =
        fused_speculative_step ? "four-fused"
                               : (unified_target_step ? "four" : "five");
    PrintProgress(
        arguments.progress,
        std::string("stage=validate-") + topology_count + "-om-start");
    for (const auto& model : arguments.models) {
      if (model.path.empty()) {
        continue;
      }
      if (!std::filesystem::is_regular_file(model.path)) {
        throw std::runtime_error(
            std::string(model.role) + " OM file does not exist: " +
            model.path.string());
      }
      if (qwen35::dflash::Sha256File(model.path) != model.sha256) {
        throw std::runtime_error(
            std::string(model.role) + " OM SHA-256 differs");
      }
      PrintProgress(
          arguments.progress,
          std::string("stage=validate-om-done role=") + model.role);
    }
    PrintProgress(
        arguments.progress,
        std::string("stage=load-") + topology_count + "-om-start");
    const auto load_start = std::chrono::steady_clock::now();
    qwen35::dflash::IncrementalOmPaths paths{
        arguments.models[0].path,
        arguments.models[1].path,
        arguments.models[2].path,
        arguments.models[3].path,
        arguments.models[4].path,
        arguments.models[5].path,
    };
    qwen35::dflash::AclIncrementalExecutor executor(
        std::move(paths),
        arguments.device_id,
        [&](const char* role, const char* stage, std::size_t work, std::size_t weight) {
          std::ostringstream message;
          message << "stage=model-" << stage << " role=" << role;
          if (work != 0 || weight != 0) {
            message << " work_bytes=" << work << " weight_bytes=" << weight;
          }
          PrintProgress(arguments.progress, message.str());
        },
        arguments.state_reset_policy,
        arguments.decode_carrier_policy,
        arguments.measurement_protocol == MeasurementProtocol::kProfile,
        arguments.draft_feature_policy);
    const auto load_end = std::chrono::steady_clock::now();
    const double load_ms = std::chrono::duration<double, std::milli>(
        load_end - load_start).count();
    {
      std::ostringstream message;
      message << "stage=load-" << topology_count
              << "-om-done sequence_capacity="
              << executor.sequence_length() << " prefill_width="
              << executor.prefill_width() << " proposal_width="
              << executor.proposal_width() << " elapsed_ms=" << std::fixed
              << std::setprecision(3) << load_ms;
      PrintProgress(arguments.progress, message.str());
    }

    qwen35::dflash::GenerationOptions options;
    options.pad_token_id = arguments.pad_token_id;
    options.max_new_tokens = arguments.max_new_tokens;
    options.max_draft_tokens = arguments.max_draft_tokens;
    options.dflash_sync_window = arguments.dflash_sync_window;
    options.coalesce_prefill_with_first_verify =
        arguments.coalesce_prefill_with_first_verify;
    options.zero_accept_fallback_policy =
        arguments.zero_accept_fallback_policy;
    options.eos_token_ids = arguments.eos_token_ids;
    PrintProgress(
        arguments.progress,
        arguments.measurement_protocol == MeasurementProtocol::kEvidence
            ? "stage=benchmark-start protocol=evidence-paired-3-plus-10"
            : "stage=benchmark-start protocol=profile-paired-1-plus-1");
    const auto benchmark_start = std::chrono::steady_clock::now();
    const PairedBenchmarkResult result =
        qwen35::dflash::BenchmarkPairStateful(
            executor,
            arguments.prompt_token_ids,
            options,
            arguments.warmup,
            arguments.repetitions,
            MakeProgressCallback(arguments.progress));
    const auto benchmark_end = std::chrono::steady_clock::now();
    const double benchmark_ms = std::chrono::duration<double, std::milli>(
        benchmark_end - benchmark_start).count();
    {
      std::ostringstream message;
      message << "stage=benchmark-done elapsed_ms=" << std::fixed
              << std::setprecision(3) << benchmark_ms;
      PrintProgress(arguments.progress, message.str());
    }
    PrintProgress(arguments.progress, "stage=write-report-start");
    std::ostringstream report;
    WriteReport(report, arguments, executor, load_ms, benchmark_ms, result);
    AtomicWrite(arguments.output, report.str());
    PrintProgress(arguments.progress, "stage=write-report-done status=PASS");
    std::cout << report.str() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "qwen35_dflash_incremental_acl_runner: " << error.what() << '\n';
    return 1;
  }
}
