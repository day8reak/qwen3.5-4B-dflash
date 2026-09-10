#include "qwen35_dflash/acl_executor.hpp"
#include "qwen35_dflash/chunk.hpp"
#include "qwen35_dflash/generation.hpp"
#include "qwen35_dflash/sha256.hpp"
#include "qwen35_dflash/stage_profile.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
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
using qwen35::dflash::PairedBenchmarkResult;

struct Arguments {
  std::filesystem::path model;
  std::string model_sha256;
  std::filesystem::path output;
  std::vector<std::int64_t> prompt_token_ids;
  std::vector<std::int64_t> eos_token_ids;
  std::int64_t pad_token_id = 0;
  std::size_t max_new_tokens = 32;
  std::size_t max_draft_tokens = 15;
  std::size_t warmup = 3;
  std::size_t repetitions = 10;
  bool trace_rounds = false;
  bool low_memory = false;
  std::size_t debug_draft_replay = 0;
  std::string debug_draft_workspace = "shared";
  std::filesystem::path debug_draft_inputs;
  int device_id = 0;
  std::string model_kind = "recompute";
  std::string mode = "paired";
  qwen35::dflash::ProfileOptions profile;
};

void Usage(std::ostream& stream) {
  stream
      << "Usage: qwen35_dflash_acl_runner [options]\n"
      << "  --model PATH                 hash-locked integrated OM\n"
      << "  --model-sha256 HEX           expected OM SHA-256\n"
      << "  --model-kind TYPE            recompute or chunk (model is a loading plan)\n"
      << "  --mode MODE                  paired, ordinary or dflash; default paired\n"
      << "  --low-memory                 chunk paired: measure ordinary, unload, then DFlash\n"
      << "  --output PATH                paired JSON report\n"
      << "  --prompt-token-ids CSV       non-empty pretokenized prompt\n"
      << "  --eos-token-ids CSV          optional EOS token IDs\n"
      << "  --pad-token-id ID            default 0\n"
      << "  --max-new-tokens N           default 32\n"
      << "  --max-draft-tokens N         default 15\n"
      << "  --warmup N                   target evidence requires 3\n"
      << "  --repetitions N              target evidence requires 10\n"
      << "  --device-id N                default 0\n";
  stream << "  --debug-draft-replay N       frozen-input Draft diagnostic; N calls per phase, no msprof\n"
         << "  --debug-draft-workspace MODE shared (default) or private; debug replay only\n"
         << "  --debug-draft-inputs PATH    reuse a previous replay's inputs directory\n";
  stream << "  --trace-rounds               record chunk proposals, verify and emitted tokens\n";
  stream << "  --profile-stage STAGE       diagnostic prefill/decode (ordinary), prefill/draft/verify (dflash), or all\n"
         << "  --profile-mode MODE         ordinary or dflash; use tools/run_msprof.sh --profile-backend cpp\n"
         << "  --profile-output PATH       new raw msprof directory (controller owned)\n"
         << "  --profile-warmup N          unprofiled fresh-state warmups; default 1\n"
         << "  --profile-audit-draft-inputs true|false  hash Draft inputs outside capture; default false\n"
         << "  --profile-aic-metrics NAME  default PipeUtilization\n";
}

std::string Trim(std::string value) {
  const auto first = std::find_if_not(value.begin(), value.end(), [](char item) {
    return std::isspace(static_cast<unsigned char>(item)) != 0;
  });
  const auto last = std::find_if_not(value.rbegin(), value.rend(), [](char item) {
                      return std::isspace(static_cast<unsigned char>(item)) != 0;
                    }).base();
  if (first >= last) {
    return {};
  }
  return std::string(first, last);
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
    if (argument == "--trace-rounds" || argument == "--low-memory") {
      name = argument.substr(2);
      value = "true";
    } else if (equals != std::string::npos) {
      name = argument.substr(2, equals - 2);
      value = argument.substr(equals + 1);
    } else {
      name = argument.substr(2);
      if (index + 1 >= argc) {
        throw std::invalid_argument("missing value for --" + name);
      }
      value = argv[++index];
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

Arguments ParseArguments(int argc, char** argv) {
  auto values = ParseOptions(argc, argv);
  Arguments result;
  result.model = TakeRequired(&values, "model");
  result.model_sha256 = TakeRequired(&values, "model-sha256");
  result.model_kind = TakeOptional(&values, "model-kind", "recompute");
  result.mode = TakeOptional(&values, "mode", "paired");
  if (result.mode != "paired" && result.mode != "ordinary" && result.mode != "dflash") {
    throw std::invalid_argument("mode must be paired, ordinary or dflash");
  }
  if (result.model_kind != "recompute" && result.model_kind != "chunk") {
    throw std::invalid_argument("model-kind must be recompute or chunk");
  }
  const auto trace_rounds = TakeOptional(&values, "trace-rounds", "false");
  if (trace_rounds != "true" && trace_rounds != "false")
    throw std::invalid_argument("trace-rounds must be true or false");
  result.trace_rounds = trace_rounds == "true";
  const auto low_memory = TakeOptional(&values, "low-memory", "false");
  if (low_memory != "true" && low_memory != "false")
    throw std::invalid_argument("low-memory must be true or false");
  result.low_memory = low_memory == "true";
  if (result.trace_rounds && result.model_kind != "chunk")
    throw std::invalid_argument("trace-rounds requires --model-kind chunk");
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
  result.warmup =
      ParseSize(TakeOptional(&values, "warmup", "3"), "warmup");
  result.repetitions = ParseSize(
      TakeOptional(&values, "repetitions", "10"), "repetitions");
  const std::int64_t device_id = ParseInt64(
      TakeOptional(&values, "device-id", "0"), "device-id");
  if (device_id < 0) {
    throw std::invalid_argument("device-id must be non-negative");
  }
  result.device_id = static_cast<int>(device_id);
  result.profile.stage = TakeOptional(&values, "profile-stage", "");
  result.profile.mode = TakeOptional(&values, "profile-mode", "dflash");
  result.profile.output = TakeOptional(&values, "profile-output", "");
  result.profile.metrics = TakeOptional(&values, "profile-aic-metrics", "PipeUtilization");
  const auto profile_warmup = ParseInt64(TakeOptional(&values, "profile-warmup", "1"), "profile-warmup");
  if (profile_warmup < 0) throw std::invalid_argument("profile-warmup must be non-negative");
  result.profile.warmup = static_cast<std::size_t>(profile_warmup);
  result.profile.device_id = result.device_id;
  const auto audit_draft_inputs = TakeOptional(&values, "profile-audit-draft-inputs", "false");
  if (audit_draft_inputs != "true" && audit_draft_inputs != "false")
    throw std::invalid_argument("profile-audit-draft-inputs must be true or false");
  result.profile.audit_draft_inputs = audit_draft_inputs == "true";
  if (!result.profile.stage.empty()) {
    if (result.model_kind != "chunk") throw std::invalid_argument("stage capture requires --model-kind chunk");
    if (result.mode != "paired" && result.mode != result.profile.mode) throw std::invalid_argument("mode disagrees with profile-mode");
    result.mode = result.profile.mode;
    qwen35::dflash::ValidateProfileOptions(result.profile);
    if (std::filesystem::exists(result.output) || std::filesystem::is_symlink(result.output)) {
      throw std::invalid_argument("profile report must be a new file");
    }
  } else if (!result.profile.output.empty() || result.profile.mode != "dflash" || result.profile.audit_draft_inputs) {
    throw std::invalid_argument("profile-output/profile-mode/profile-audit-draft-inputs requires profile-stage");
  }
  const auto debug_replay = TakeOptional(&values, "debug-draft-replay", "");
  const auto debug_workspace = TakeOptional(&values, "debug-draft-workspace", "");
  result.debug_draft_inputs = TakeOptional(&values, "debug-draft-inputs", "");
  if (!debug_replay.empty()) {
    result.debug_draft_replay = ParseSize(debug_replay, "debug-draft-replay");
    result.debug_draft_workspace = debug_workspace.empty() ? "shared" : debug_workspace;
    if (result.debug_draft_replay > 1000 || result.model_kind != "chunk" ||
        result.mode != "dflash" || !result.profile.stage.empty() || result.low_memory)
      throw std::invalid_argument("debug-draft-replay needs chunk dflash mode, 1..1000 calls and no profiler");
    if (result.debug_draft_workspace != "shared" && result.debug_draft_workspace != "private")
      throw std::invalid_argument("debug-draft-workspace must be shared or private");
    const char* profiling = std::getenv("PROFILING_MODE");
    if (profiling && *profiling && std::string(profiling) != "false")
      throw std::invalid_argument("debug replay requires an environment without PROFILING_MODE");
    if (std::filesystem::exists(result.output) || std::filesystem::is_symlink(result.output))
      throw std::invalid_argument("debug replay report must be new");
  } else if (!debug_workspace.empty() || !result.debug_draft_inputs.empty()) {
    throw std::invalid_argument("debug-draft-workspace/inputs requires debug-draft-replay");
  }
  if (!values.empty()) {
    throw std::invalid_argument("unknown option --" + values.begin()->first);
  }
  if (result.low_memory && (result.model_kind != "chunk" || result.mode != "paired"))
    throw std::invalid_argument("low-memory requires chunk paired mode without profiling");
  if (result.pad_token_id < 0) {
    throw std::invalid_argument("pad-token-id must be non-negative");
  }
  if (result.warmup != 3 || result.repetitions != 10) {
    throw std::invalid_argument(
        "target evidence requires exactly 3 warmups and 10 repetitions");
  }
  std::transform(
      result.model_sha256.begin(),
      result.model_sha256.end(),
      result.model_sha256.begin(),
      [](unsigned char item) { return static_cast<char>(std::tolower(item)); });
  if (result.model_sha256.size() != 64 ||
      !std::all_of(
          result.model_sha256.begin(),
          result.model_sha256.end(),
          [](unsigned char item) { return std::isxdigit(item) != 0; })) {
    throw std::invalid_argument("model-sha256 must be 64 hexadecimal characters");
  }
  return result;
}

std::string JsonEscape(const std::string& value) {
  std::ostringstream output;
  for (const unsigned char item : value) {
    switch (item) {
      case '"':
        output << "\\\"";
        break;
      case '\\':
        output << "\\\\";
        break;
      case '\b':
        output << "\\b";
        break;
      case '\f':
        output << "\\f";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\r':
        output << "\\r";
        break;
      case '\t':
        output << "\\t";
        break;
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

void WriteTokenIds(
    std::ostream& output,
    const std::vector<std::int64_t>& values) {
  output << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      output << ',';
    }
    output << values[index];
  }
  output << ']';
}

void WriteDoubles(std::ostream& output, const std::vector<double>& values) {
  output << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      output << ',';
    }
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
         << ",\"counters\":{\"graph_calls\":"
         << value.counters.graph_calls << ",\"drafted_tokens\":"
         << value.counters.drafted_tokens
         << ",\"accepted_draft_tokens\":"
         << value.counters.accepted_draft_tokens
         << ",\"rejected_draft_tokens\":"
         << value.counters.rejected_draft_tokens
         << ",\"decode_iterations\":"
         << value.counters.decode_iterations
         << ",\"speculation_disable_events\":"
         << value.counters.speculation_disable_events
         << ",\"target_only_fallback_rounds\":"
         << value.counters.target_only_fallback_rounds << "},\"latency_ms\":{"
         << "\"prefill\":" << value.prefill_ms
         << ",\"request_reset\":" << value.request_reset_ms
         << ",\"decode\":" << value.decode_ms
         << ",\"model_total\":" << value.model_total_ms
         << "},\"decode_iteration_ms\":";
  WriteDoubles(output, value.decode_iteration_ms);
  output << ",\"stage_ms\":{";
  bool first_stage = true;
  for (const auto& stage : value.stage_ms) {
    if (!first_stage) output << ',';
    first_stage = false;
    output << '\"' << JsonEscape(stage.first) << "\":";
    WriteDoubles(output, stage.second);
  }
  output << '}';
  if (!value.rounds.empty()) {
    output << ",\"rounds\":[";
    bool first_round = true;
    for (const auto& round : value.rounds) {
      if (!first_round) output << ',';
      first_round = false;
      output << "{\"committed_prefix_length\":" << round.committed_prefix_length
             << ",\"stage\":\"" << JsonEscape(round.stage) << "\""
             << ",\"proposed_token_ids\":";
      WriteTokenIds(output, round.proposed_token_ids);
      output << ",\"target_token_ids\":";
      WriteTokenIds(output, round.target_token_ids);
      output << ",\"accepted_draft_token_ids\":";
      WriteTokenIds(output, round.accepted_draft_token_ids);
      output << ",\"emitted_token_ids\":";
      WriteTokenIds(output, round.emitted_token_ids);
      output << ",\"fallback_token_id\":";
      if (round.fallback_token_id < 0) output << "null";
      else output << round.fallback_token_id;
      output << '}';
    }
    output << ']';
  }
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
    if (index != 0) {
      output << ',';
    }
    WriteMeasurement(output, value.measurements[index], index);
  }
  output << "]}";
}

void WriteReport(
    std::ostream& output,
    const Arguments& arguments,
    const qwen35::dflash::GraphExecutor& executor,
    double load_ms,
    double benchmark_wall_ms,
    const PairedBenchmarkResult& result,
    double unload_ms = 0.0) {
  const double speedup = result.dflash.model_total_ms.median > 0.0
                             ? result.ordinary.model_total_ms.median /
                                   result.dflash.model_total_ms.median
                             : 0.0;
  output << std::setprecision(17)
         << "{\"schema_version\":1,\"status\":\"PASS\","
         << "\"scope\":\"AscendCL C++ paired OM model loop\","
         << "\"runner_id\":\"qwen35-dflash-ascendcl-cpp-v1\","
         << "\"runner_version\":\""
         << JsonEscape(QWEN35_DFLASH_RUNNER_VERSION) << "\","
         << "\"cpu_fallback\":false,\"device_id\":"
         << arguments.device_id << ",\"model\":{\"path\":\""
         << JsonEscape(std::filesystem::absolute(arguments.model).string())
         << "\",\"sha256\":\"" << arguments.model_sha256 << "\"},";
  if (arguments.model_kind == "chunk") {
    output << "\"abi\":{\"id\":\"qwen35-dflash-chunk-v3\",\"graph_count\":4,\"sequence_length\":";
  } else {
    output << "\"abi\":{\"input_names\":[\"input_ids\",\"attention_mask\"],"
         << "\"output_names\":[\"target_top1\",\"draft_top1\"],"
         << "\"dtype\":\"int64\",\"sequence_length\":";
  }
  output << executor.sequence_length() << ",\"draft_width\":"
         << executor.draft_width() << "},\"protocol\":{\"warmup\":"
         << arguments.warmup << ",\"repetitions\":"
         << arguments.repetitions
         << ",\"order\":\"" << (arguments.low_memory
              ? "ordinary then DFlash with model unload between modes"
              : "alternating ordinary/DFlash in one loaded process") << "\","
         << "\"low_memory\":" << (arguments.low_memory ? "true" : "false") << ','
         << "\"max_resident_models\":" << (arguments.model_kind == "chunk"
              ? (arguments.low_memory ? 3 : 4) : 1) << ','
         << "\"synchronization\":\"one aclrtSynchronizeStream after queued H2D, execute, D2H\","
         << "\"model_load_excluded_from_latency\":true,"
         << "\"round_trace_enabled\":" << (arguments.trace_rounds ? "true" : "false") << ","
         << "\"request_reset_excluded_from_latency\":" << (arguments.model_kind == "chunk" ? "true" : "false") << "},"
         << "\"prompt_token_ids\":";
  WriteTokenIds(output, arguments.prompt_token_ids);
  output << ",\"eos_token_ids\":";
  WriteTokenIds(output, arguments.eos_token_ids);
  output << ",\"limits\":{\"max_new_tokens\":"
         << arguments.max_new_tokens << ",\"max_draft_tokens\":"
         << arguments.max_draft_tokens << "},\"startup_ms\":{\"acl_and_model_load\":"
         << load_ms << ",\"paired_benchmark_wall\":" << benchmark_wall_ms
         << ",\"mode_switch_unload\":" << unload_ms
         << "},\"ordinary\":";
  WriteBenchmark(output, result.ordinary);
  output << ",\"dflash\":";
  WriteBenchmark(output, result.dflash);
  output << ",\"ordinary_parity\":{\"status\":\"PASS\","
         << "\"token_id_mismatches\":" << result.token_id_mismatches
         << ",\"eos_mismatches\":" << result.eos_mismatches
         << "},\"dflash_speedup_over_ordinary_model_total_median\":"
         << speedup << '}';
}

void AtomicWrite(
    const std::filesystem::path& path,
    const std::string& payload) {
  const std::filesystem::path absolute = std::filesystem::absolute(path);
  if (std::filesystem::exists(absolute)) {
    throw std::runtime_error("refusing to overwrite output: " + absolute.string());
  }
  std::filesystem::create_directories(absolute.parent_path());
  const std::filesystem::path temporary = absolute.string() + ".tmp";
  if (std::filesystem::exists(temporary)) {
    throw std::runtime_error(
        "temporary output already exists: " + temporary.string());
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
    if (!std::filesystem::is_regular_file(arguments.model)) {
      throw std::runtime_error("OM file does not exist: " + arguments.model.string());
    }
    const std::string actual_hash = qwen35::dflash::Sha256File(arguments.model);
    if (actual_hash != arguments.model_sha256) {
      throw std::runtime_error("OM SHA-256 differs from --model-sha256");
    }

    const auto load_start = std::chrono::steady_clock::now();
    std::unique_ptr<qwen35::dflash::GraphExecutor> executor;
    if (arguments.model_kind == "chunk") {
      executor = std::make_unique<qwen35::dflash::AclChunkExecutor>(
          arguments.model, arguments.device_id, arguments.low_memory ? "ordinary" : arguments.mode,
          arguments.debug_draft_workspace != "private");
    } else {
      executor = std::make_unique<qwen35::dflash::AclExecutor>(arguments.model, arguments.device_id);
    }
    const auto load_end = std::chrono::steady_clock::now();
    qwen35::dflash::GenerationOptions options;
    options.pad_token_id = arguments.pad_token_id;
    options.max_new_tokens = arguments.max_new_tokens;
    options.max_draft_tokens = arguments.max_draft_tokens;
    options.eos_token_ids = arguments.eos_token_ids;
    options.trace_rounds = arguments.trace_rounds;
    if (arguments.debug_draft_replay) {
      auto& chunk = dynamic_cast<qwen35::dflash::AclChunkExecutor&>(*executor);
      const auto output = std::filesystem::absolute(arguments.output);
      std::filesystem::create_directories(output.parent_path());
      auto result = chunk.DebugDraftReplay(
          arguments.prompt_token_ids, arguments.pad_token_id, arguments.max_draft_tokens,
          arguments.debug_draft_replay, output.string() + ".replay", arguments.debug_draft_inputs);
      result.second.pop_back();
      const bool fake = std::string(QWEN35_DFLASH_RUNNER_VERSION).find("fake-acl") != std::string::npos;
      result.second += ",\"runner_version\":\"" + JsonEscape(QWEN35_DFLASH_RUNNER_VERSION) +
          "\",\"model_sha256\":\"" + arguments.model_sha256 +
          "\",\"device_id\":" + std::to_string(arguments.device_id) +
          ",\"fake_acl\":" + (fake ? "true" : "false") + "}";
      chunk.Close();
      AtomicWrite(output, result.second);
      std::cout << result.second << '\n';
      return result.first ? 0 : 1;
    }
    if (!arguments.profile.stage.empty()) {
      auto& chunk = dynamic_cast<qwen35::dflash::AclChunkExecutor&>(*executor);
      auto report = qwen35::dflash::ProfileChunk(chunk, arguments.prompt_token_ids, options, arguments.profile);
      report.pop_back();
      report += ",\"runner_version\":\"" + JsonEscape(QWEN35_DFLASH_RUNNER_VERSION) +
          "\",\"model_sha256\":\"" + arguments.model_sha256 + "\"}";
      chunk.Close();
      AtomicWrite(arguments.output, report);
      std::cout << report << '\n';
      return 0;
    }
    const auto benchmark_start = std::chrono::steady_clock::now();
    if (arguments.mode != "paired") {
      const auto mode = arguments.mode == "dflash" ? qwen35::dflash::GenerationMode::kDFlash : qwen35::dflash::GenerationMode::kOrdinary;
      const auto result = qwen35::dflash::Benchmark(*executor, arguments.prompt_token_ids, mode, options, arguments.warmup, arguments.repetitions);
      std::ostringstream report;
      report << std::setprecision(17) << "{\"schema_version\":1,\"status\":\"PASS\",\"scope\":\"single-mode execution; no ordinary parity claim\",\"runner_version\":\""
             << JsonEscape(QWEN35_DFLASH_RUNNER_VERSION) << "\",\"model_kind\":\"" << arguments.model_kind
             << "\",\"cpu_fallback\":false,\"device_id\":" << arguments.device_id
             << ",\"model_sha256\":\"" << arguments.model_sha256 << "\",\"prompt_token_ids\":";
      WriteTokenIds(report, arguments.prompt_token_ids);
      report << ",\"ordinary_parity\":{\"status\":\"NOT_RUN\"},\"benchmark\":";
      WriteBenchmark(report, result);
      report << '}';
      if (auto* chunk = dynamic_cast<qwen35::dflash::AclChunkExecutor*>(executor.get())) chunk->Close();
      AtomicWrite(arguments.output, report.str());
      std::cout << report.str() << '\n';
      return 0;
    }
    PairedBenchmarkResult result;
    double reload_ms = 0, unload_ms = 0;
    if (arguments.low_memory) {
      auto ordinary = qwen35::dflash::Benchmark(
          *executor, arguments.prompt_token_ids, qwen35::dflash::GenerationMode::kOrdinary,
          options, arguments.warmup, arguments.repetitions);
      const auto unload_start = std::chrono::steady_clock::now();
      auto& chunk = dynamic_cast<qwen35::dflash::AclChunkExecutor&>(*executor);
      chunk.UnloadModels();
      const auto reload_start = std::chrono::steady_clock::now();
      unload_ms = std::chrono::duration<double, std::milli>(reload_start - unload_start).count();
      chunk.LoadMode("dflash");
      reload_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - reload_start).count();
      auto dflash = qwen35::dflash::Benchmark(
          *executor, arguments.prompt_token_ids, qwen35::dflash::GenerationMode::kDFlash,
          options, arguments.warmup, arguments.repetitions);
      result = qwen35::dflash::PairBenchmarks(std::move(ordinary), std::move(dflash));
    } else {
      result = qwen35::dflash::BenchmarkPair(*executor, arguments.prompt_token_ids,
          options, arguments.warmup, arguments.repetitions);
    }
    const auto benchmark_end = std::chrono::steady_clock::now();
    const double load_ms =
        std::chrono::duration<double, std::milli>(load_end - load_start).count() + reload_ms;
    const double benchmark_ms =
        std::chrono::duration<double, std::milli>(
            benchmark_end - benchmark_start)
            .count() - reload_ms - unload_ms;
    std::ostringstream report;
    WriteReport(report, arguments, *executor, load_ms, benchmark_ms, result, unload_ms);
    if (auto* chunk = dynamic_cast<qwen35::dflash::AclChunkExecutor*>(executor.get())) chunk->Close();
    AtomicWrite(arguments.output, report.str());
    std::cout << report.str() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "qwen35_dflash_acl_runner: " << error.what() << '\n';
    return 1;
  }
}
