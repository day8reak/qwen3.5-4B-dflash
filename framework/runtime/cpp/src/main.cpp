#include "qwen35_dflash/acl_executor.hpp"
#include "qwen35_dflash/generation.hpp"
#include "qwen35_dflash/sha256.hpp"

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
using qwen35::dflash::ProgressCallback;
using qwen35::dflash::ProgressEvent;

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
  int device_id = 0;
  bool progress = true;
};

void Usage(std::ostream& stream) {
  stream
      << "Usage: qwen35_dflash_acl_runner [options]\n"
      << "  --model PATH                 hash-locked integrated OM\n"
      << "  --model-sha256 HEX           expected OM SHA-256\n"
      << "  --output PATH                paired JSON report\n"
      << "  --prompt-token-ids CSV       non-empty pretokenized prompt\n"
      << "  --eos-token-ids CSV          optional EOS token IDs\n"
      << "  --pad-token-id ID            default 0\n"
      << "  --max-new-tokens N           default 32\n"
      << "  --max-draft-tokens N         default 15\n"
      << "  --warmup N                   target evidence requires 3\n"
      << "  --repetitions N              target evidence requires 10\n"
      << "  --device-id N                default 0\n"
      << "  --progress true|false        live stderr progress, default true\n";
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
    if (equals != std::string::npos) {
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
  result.progress = ParseBool(
      TakeOptional(&values, "progress", "true"), "progress");
  const std::int64_t device_id = ParseInt64(
      TakeOptional(&values, "device-id", "0"), "device-id");
  if (device_id < 0) {
    throw std::invalid_argument("device-id must be non-negative");
  }
  result.device_id = static_cast<int>(device_id);
  if (!values.empty()) {
    throw std::invalid_argument("unknown option --" + values.begin()->first);
  }
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

void PrintProgress(bool enabled, const std::string& message) {
  if (enabled) {
    std::cerr << "[qwen35-dflash] " << message << std::endl;
  }
}

ProgressCallback MakeProgressCallback(bool enabled) {
  if (!enabled) {
    return {};
  }
  return [](const ProgressEvent& event) {
    std::cerr << "[qwen35-dflash] phase=" << event.phase
              << " run=" << event.run_index << '/' << event.run_total
              << " mode=" << qwen35::dflash::ModeName(event.mode)
              << " stage=" << event.stage
              << " generated=" << event.generated_tokens << '/'
              << event.max_new_tokens
              << " prefix=" << event.prefix_tokens
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
         << value.counters.decode_iterations << "},\"latency_ms\":{"
         << "\"prefill\":" << value.prefill_ms
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
    const qwen35::dflash::AclExecutor& executor,
    double load_ms,
    double benchmark_wall_ms,
    const PairedBenchmarkResult& result) {
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
         << "\",\"sha256\":\"" << arguments.model_sha256 << "\"},"
         << "\"abi\":{\"input_names\":[\"input_ids\",\"attention_mask\"],"
         << "\"output_names\":[\"target_top1\",\"draft_top1\"],"
         << "\"dtype\":\"int64\",\"sequence_length\":"
         << executor.sequence_length() << ",\"draft_width\":"
         << executor.draft_width() << "},\"protocol\":{\"warmup\":"
         << arguments.warmup << ",\"repetitions\":"
         << arguments.repetitions
         << ",\"order\":\"alternating ordinary/DFlash in one loaded process\","
         << "\"synchronization\":\"one aclrtSynchronizeStream after queued H2D, execute, D2H\","
         << "\"model_load_excluded_from_latency\":true,"
         << "\"live_progress_enabled\":"
         << (arguments.progress ? "true" : "false") << ','
         << "\"progress_emission_excluded_from_model_timers\":true},"
         << "\"prompt_token_ids\":";
  WriteTokenIds(output, arguments.prompt_token_ids);
  output << ",\"eos_token_ids\":";
  WriteTokenIds(output, arguments.eos_token_ids);
  output << ",\"limits\":{\"max_new_tokens\":"
         << arguments.max_new_tokens << ",\"max_draft_tokens\":"
         << arguments.max_draft_tokens << "},\"startup_ms\":{\"acl_and_model_load\":"
         << load_ms << ",\"paired_benchmark_wall\":" << benchmark_wall_ms
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
    PrintProgress(arguments.progress, "stage=validate-om-start");
    if (!std::filesystem::is_regular_file(arguments.model)) {
      throw std::runtime_error("OM file does not exist: " + arguments.model.string());
    }
    const std::string actual_hash = qwen35::dflash::Sha256File(arguments.model);
    if (actual_hash != arguments.model_sha256) {
      throw std::runtime_error("OM SHA-256 differs from --model-sha256");
    }
    PrintProgress(arguments.progress, "stage=validate-om-done");

    PrintProgress(arguments.progress, "stage=load-om-start");
    const auto load_start = std::chrono::steady_clock::now();
    qwen35::dflash::AclExecutor executor(arguments.model, arguments.device_id);
    const auto load_end = std::chrono::steady_clock::now();
    const double load_ms =
        std::chrono::duration<double, std::milli>(load_end - load_start).count();
    {
      std::ostringstream message;
      message << "stage=load-om-done sequence_length="
              << executor.sequence_length() << " draft_width="
              << executor.draft_width() << " elapsed_ms=" << std::fixed
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
        "stage=benchmark-start protocol=paired-3-warmup-plus-10-measurements");
    const auto benchmark_start = std::chrono::steady_clock::now();
    const PairedBenchmarkResult result = qwen35::dflash::BenchmarkPair(
        executor,
        arguments.prompt_token_ids,
        options,
        arguments.warmup,
        arguments.repetitions,
        MakeProgressCallback(arguments.progress));
    const auto benchmark_end = std::chrono::steady_clock::now();
    const double benchmark_ms =
        std::chrono::duration<double, std::milli>(
            benchmark_end - benchmark_start)
            .count();
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
    std::cerr << "qwen35_dflash_acl_runner: " << error.what() << '\n';
    return 1;
  }
}
