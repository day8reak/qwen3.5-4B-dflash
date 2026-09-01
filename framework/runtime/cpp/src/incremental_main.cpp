#include "qwen35_dflash/generation.hpp"
#include "qwen35_dflash/incremental_acl_executor.hpp"
#include "qwen35_dflash/sha256.hpp"

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
using qwen35::dflash::IncrementalStateResetPolicy;
using qwen35::dflash::PairedBenchmarkResult;
using qwen35::dflash::ProgressCallback;
using qwen35::dflash::ProgressEvent;

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
  std::array<ModelArgument, 4> models{{
      {"target-prefill", {}, {}},
      {"target-decode1", {}, {}},
      {"draft-propose", {}, {}},
      {"target-verify-commit", {}, {}},
  }};
  std::filesystem::path output;
  std::vector<std::int64_t> prompt_token_ids;
  std::vector<std::int64_t> eos_token_ids;
  std::int64_t pad_token_id = 0;
  std::size_t max_new_tokens = 32;
  std::size_t max_draft_tokens = 15;
  std::size_t warmup = 3;
  std::size_t repetitions = 10;
  int device_id = 0;
  bool progress = true;
  IncrementalStateResetPolicy state_reset_policy =
      IncrementalStateResetPolicy::kAsyncMemset;
  MeasurementProtocol measurement_protocol = MeasurementProtocol::kEvidence;
};

void Usage(std::ostream& stream) {
  stream
      << "Usage: qwen35_dflash_incremental_acl_runner [options]\n"
      << "  --target-prefill PATH                    hash-locked prefill OM\n"
      << "  --target-prefill-sha256 HEX              expected prefill SHA-256\n"
      << "  --target-decode1 PATH                    hash-locked decode-one OM\n"
      << "  --target-decode1-sha256 HEX              expected decode SHA-256\n"
      << "  --draft-propose PATH                     hash-locked Draft OM\n"
      << "  --draft-propose-sha256 HEX               expected Draft SHA-256\n"
      << "  --target-verify-commit PATH              hash-locked verify OM\n"
      << "  --target-verify-commit-sha256 HEX        expected verify SHA-256\n"
      << "  --output PATH                            paired JSON report\n"
      << "  --prompt-token-ids CSV                   non-empty prompt\n"
      << "  --eos-token-ids CSV                      optional EOS token IDs\n"
      << "  --pad-token-id ID                        default 0\n"
      << "  --max-new-tokens N                       default 32\n"
      << "  --max-draft-tokens N                     default 15\n"
      << "  --warmup N                               evidence=3, profile=1\n"
      << "  --repetitions N                          evidence=10, profile=1\n"
      << "  --device-id N                            default 0\n"
      << "  --measurement-protocol MODE             evidence (default) or profile\n"
      << "  --state-reset-policy POLICY             async-memset (default) or "
         "immutable-zero\n"
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
  for (auto& model : result.models) {
    model.path = TakeRequired(&values, model.role);
    model.sha256 = NormalizeHash(
        TakeRequired(&values, std::string(model.role) + "-sha256"),
        std::string(model.role) + "-sha256");
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
  if (memory.size() != arguments.models.size()) {
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
  const bool formal_latency_evidence =
      arguments.measurement_protocol == MeasurementProtocol::kEvidence;
  const std::size_t model_executions =
      execution.target_prefill_executions +
      execution.target_decode1_executions +
      execution.draft_propose_executions +
      execution.target_verify_commit_executions;
  const double speedup = result.dflash.model_total_ms.median > 0.0
      ? result.ordinary.model_total_ms.median /
            result.dflash.model_total_ms.median
      : 0.0;

  output << std::setprecision(17)
         << "{\"schema_version\":2,\"status\":\"PASS\","
         << "\"scope\":\"AscendCL C++ four-resident-OM paired model loop\","
         << "\"runner_id\":\"qwen35-dflash-ascendcl-cpp-incremental-v2\","
         << "\"runner_version\":\"" << JsonEscape(QWEN35_DFLASH_RUNNER_VERSION)
         << "\",\"candidate_status\":\"APPROVED_IN_IMPLEMENTATION_NOT_ACTIVE\","
         << "\"cpu_fallback\":false,\"device_id\":" << arguments.device_id
         << ",\"models\":[";
  for (std::size_t index = 0; index < arguments.models.size(); ++index) {
    if (index != 0) output << ',';
    const auto& model = arguments.models[index];
    if (memory[index].role != model.role) {
      throw std::runtime_error("incremental model memory role order differs");
    }
    output << "{\"role\":\"" << model.role << "\",\"path\":\""
           << JsonEscape(std::filesystem::absolute(model.path).string())
           << "\",\"sha256\":\"" << model.sha256
           << "\",\"work_bytes\":" << memory[index].work_bytes
           << ",\"weight_bytes\":" << memory[index].weight_bytes << '}';
  }
  output << "],\"abi\":{"
         << "\"id\":\"qwen35-4b-dflash-ascend310p-incremental-performance-v2\","
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
         << ",\"prefill_control_bytes_per_slot\":"
         << execution.prefill_control_bytes_per_slot
         << ",\"prefill_staging_pinned_host_bytes\":"
         << execution.prefill_staging_pinned_host_bytes
         << ",\"prefill_feature_slab_bytes\":"
         << execution.prefill_feature_slab_bytes
         << ",\"prefill_feature_arena_bytes\":"
         << execution.prefill_feature_arena_bytes
         << ",\"draft_dynamic_gear_count\":"
         << execution.draft_dynamic_gear_count
         << ",\"explicit_allocated_device_bytes_excluding_runtime\":"
         << max_work + sum_weight + execution.state_device_bytes +
                execution.carrier_device_bytes
         << ",\"load_policy\":\"four aclmdlLoadFromFileWithMem sessions; "
            "one max-sized serial workspace; separate per-artifact weights; "
            "no cross-OM weight sharing assumed\"},"
         << "\"protocol\":{\"warmup\":" << arguments.warmup
         << ",\"repetitions\":" << arguments.repetitions
         << ",\"kind\":\""
         << (formal_latency_evidence ? "evidence" : "profile")
         << "\",\"formal_latency_evidence\":"
         << (formal_latency_evidence ? "true" : "false")
         << ",\"order\":\"alternating ordinary/DFlash in one four-model process\","
         << "\"model_load_excluded_from_latency\":true,"
         << "\"prefill_completion_policy\":\"intermediate prompt chunks "
            "stay queued; final chunk performs the only compact D2H and "
            "stream synchronization\","
         << "\"prefill_control_policy\":\"IDs, effective length, proposal "
            "count, total prompt count and EOS table share one H2D carrier "
            "with 64-byte-aligned device subsegments per prompt chunk\","
         << "\"prefill_draft_policy\":\"Target feature slabs stay device-resident; "
            "non-final prompt chunks execute no Draft OM; final prompt "
            "completion executes one prebound dynamic-gear Draft OM\","
         << "\"prefill_feature_arena_policy\":\"contiguous 64-row FP16 slabs "
            "with 64-byte-aligned starts and one terminal guard; no D2D "
            "compaction\","
         << "\"prefill_target_lm_head_policy\":\"current target-prefill OM "
            "still executes its LM head for every physical chunk; non-final "
            "elimination remains pending real-profile-driven graph redesign\","
         << "\"device_suballocation_policy\":\"64-byte segment starts; "
            "ALIGN_UP(payload,32)+32 reserved span\","
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
         << "\"proposal_policy\":\"Draft-to-verify device carrier; no proposal D2H/H2D\","
         << "\"result_policy\":\"one compact D2H and one barrier per "
            "complete prompt/decode/speculative transaction; no host-visible "
            "result for intermediate prefill chunks\","
         << "\"model_executions\":" << model_executions
         << ",\"target_prefill_executions\":"
         << execution.target_prefill_executions
         << ",\"target_decode1_executions\":"
         << execution.target_decode1_executions
         << ",\"draft_propose_executions\":"
         << execution.draft_propose_executions
         << ",\"target_verify_commit_executions\":"
         << execution.target_verify_commit_executions
         << ",\"stream_synchronizations\":"
         << execution.stream_synchronizations
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
         << ",\"prefill_control_upload_operations\":"
         << execution.prefill_control_upload_operations
         << ",\"prefill_control_upload_bytes\":"
         << execution.prefill_control_upload_bytes
         << ",\"prefill_h2d_operations_elided\":"
         << execution.prefill_h2d_operations_elided
         << ",\"decode_id_upload_operations\":"
         << execution.decode_id_upload_operations
         << ",\"decode_id_upload_bytes\":"
         << execution.decode_id_upload_bytes
         << ",\"proposal_count_upload_operations\":"
         << execution.proposal_count_upload_operations
         << ",\"proposal_count_upload_bytes\":"
         << execution.proposal_count_upload_bytes
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
         << ",\"prefill_staging_slots\":"
         << execution.prefill_staging_slots
         << ",\"prefill_control_bytes_per_slot\":"
         << execution.prefill_control_bytes_per_slot
         << ",\"prefill_staging_pinned_host_bytes\":"
         << execution.prefill_staging_pinned_host_bytes
         << ",\"prefill_feature_slab_bytes\":"
         << execution.prefill_feature_slab_bytes
         << ",\"prefill_feature_arena_bytes\":"
         << execution.prefill_feature_arena_bytes
         << ",\"draft_dynamic_gear_count\":"
         << execution.draft_dynamic_gear_count
         << "},\"prompt_token_ids\":";
  WriteTokenIds(output, arguments.prompt_token_ids);
  output << ",\"eos_token_ids\":";
  WriteTokenIds(output, arguments.eos_token_ids);
  output << ",\"limits\":{\"max_new_tokens\":"
         << arguments.max_new_tokens << ",\"max_draft_tokens\":"
         << arguments.max_draft_tokens << "},\"startup_ms\":{"
         << "\"acl_and_four_model_load\":" << load_ms
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
    PrintProgress(arguments.progress, "stage=validate-four-om-start");
    for (const auto& model : arguments.models) {
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
    PrintProgress(arguments.progress, "stage=load-four-om-start");
    const auto load_start = std::chrono::steady_clock::now();
    qwen35::dflash::IncrementalOmPaths paths{
        arguments.models[0].path,
        arguments.models[1].path,
        arguments.models[2].path,
        arguments.models[3].path,
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
        arguments.state_reset_policy);
    const auto load_end = std::chrono::steady_clock::now();
    const double load_ms = std::chrono::duration<double, std::milli>(
        load_end - load_start).count();
    {
      std::ostringstream message;
      message << "stage=load-four-om-done sequence_capacity="
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
