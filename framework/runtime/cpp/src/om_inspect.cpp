#include "qwen35_dflash/sha256.hpp"

#include <acl/acl.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct ModelArgument {
  std::string role;
  std::filesystem::path path;
};

struct ModelMemory {
  ModelArgument model;
  std::string sha256;
  std::size_t work_bytes = 0;
  std::size_t weight_bytes = 0;
};

struct Arguments {
  std::vector<ModelArgument> models;
  std::filesystem::path output;
  std::size_t state_bytes = 0;
  std::size_t io_and_runtime_margin_bytes = 0;
  std::size_t device_budget_bytes = 0;
  bool has_device_budget = false;
};

void Check(aclError code, const char* operation) {
  if (code != ACL_SUCCESS) {
    std::ostringstream message;
    message << operation << " failed with ACL error " << code;
    throw std::runtime_error(message.str());
  }
}

std::size_t AddChecked(
    std::size_t left, std::size_t right, const char* description) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    throw std::overflow_error(std::string(description) + " overflows size_t");
  }
  return left + right;
}

std::size_t ParseBytes(const std::string& value, const char* name) {
  if (value.empty() || value.front() == '-') {
    throw std::invalid_argument(std::string(name) + " must be a non-negative integer");
  }
  std::size_t consumed = 0;
  unsigned long long parsed = 0;
  try {
    parsed = std::stoull(value, &consumed, 10);
  } catch (const std::exception&) {
    throw std::invalid_argument(std::string(name) + " must be a non-negative integer");
  }
  if (consumed != value.size() ||
      parsed > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument(std::string(name) + " is outside size_t");
  }
  return static_cast<std::size_t>(parsed);
}

ModelArgument ParseModel(const std::string& value) {
  const std::size_t equals = value.find('=');
  if (equals == std::string::npos || equals == 0 || equals + 1 == value.size()) {
    throw std::invalid_argument("--model must use ROLE=PATH");
  }
  ModelArgument result{value.substr(0, equals), value.substr(equals + 1)};
  if (!std::all_of(result.role.begin(), result.role.end(), [](unsigned char item) {
        return std::isalnum(item) != 0 || item == '-' || item == '_';
      })) {
    throw std::invalid_argument("model role contains an invalid character");
  }
  return result;
}

void Usage(std::ostream& output) {
  output
      << "Usage: qwen35_dflash_om_inspect [options]\n"
      << "  --model ROLE=PATH            repeat for every candidate OM\n"
      << "  --output PATH                write a new JSON memory report\n"
      << "  --state-bytes N              persistent/peak state budget, default 0\n"
      << "  --io-runtime-margin-bytes N  I/O/runtime safety margin, default 0\n"
      << "  --device-budget-bytes N      optional fit calculation\n";
}

Arguments ParseArguments(int argc, char** argv) {
  Arguments result;
  for (int index = 1; index < argc; ++index) {
    const std::string option(argv[index]);
    if (option == "--help" || option == "-h") {
      Usage(std::cout);
      std::exit(0);
    }
    if (index + 1 >= argc) {
      throw std::invalid_argument("missing value for " + option);
    }
    const std::string value(argv[++index]);
    if (option == "--model") {
      result.models.push_back(ParseModel(value));
    } else if (option == "--output") {
      if (!result.output.empty()) {
        throw std::invalid_argument("--output was repeated");
      }
      result.output = value;
    } else if (option == "--state-bytes") {
      result.state_bytes = ParseBytes(value, "state-bytes");
    } else if (option == "--io-runtime-margin-bytes") {
      result.io_and_runtime_margin_bytes =
          ParseBytes(value, "io-runtime-margin-bytes");
    } else if (option == "--device-budget-bytes") {
      result.device_budget_bytes = ParseBytes(value, "device-budget-bytes");
      result.has_device_budget = true;
    } else {
      throw std::invalid_argument("unknown option " + option);
    }
  }
  if (result.models.empty()) {
    throw std::invalid_argument("at least one --model ROLE=PATH is required");
  }
  if (result.output.empty()) {
    throw std::invalid_argument("--output is required");
  }
  std::map<std::string, bool> roles;
  for (const auto& model : result.models) {
    if (!roles.emplace(model.role, true).second) {
      throw std::invalid_argument("model role was repeated: " + model.role);
    }
    if (!std::filesystem::is_regular_file(model.path)) {
      throw std::invalid_argument("OM path is not a regular file: " + model.path.string());
    }
  }
  return result;
}

std::string JsonEscape(const std::string& value) {
  std::ostringstream output;
  for (const unsigned char item : value) {
    if (item == '"' || item == '\\') {
      output << '\\' << static_cast<char>(item);
    } else if (item == '\n') {
      output << "\\n";
    } else if (item < 0x20U) {
      output << '?';
    } else {
      output << static_cast<char>(item);
    }
  }
  return output.str();
}

void AtomicWrite(const std::filesystem::path& path, const std::string& payload) {
  const auto absolute = std::filesystem::absolute(path);
  if (std::filesystem::exists(absolute)) {
    throw std::runtime_error("refusing to overwrite output: " + absolute.string());
  }
  std::filesystem::create_directories(absolute.parent_path());
  const auto temporary = std::filesystem::path(absolute.string() + ".tmp");
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
  bool initialized = false;
  try {
    const Arguments arguments = ParseArguments(argc, argv);
    Check(aclInit(nullptr), "aclInit");
    initialized = true;

    std::vector<ModelMemory> models;
    models.reserve(arguments.models.size());
    std::size_t weight_sum = 0;
    std::size_t maximum_workspace = 0;
    for (const auto& model : arguments.models) {
      ModelMemory memory;
      memory.model = model;
      memory.sha256 = qwen35::dflash::Sha256File(model.path);
      Check(
          aclmdlQuerySize(
              model.path.c_str(), &memory.work_bytes, &memory.weight_bytes),
          "aclmdlQuerySize");
      weight_sum = AddChecked(weight_sum, memory.weight_bytes, "weight sum");
      maximum_workspace = std::max(maximum_workspace, memory.work_bytes);
      models.push_back(std::move(memory));
    }
    std::size_t minimum_resident = AddChecked(
        weight_sum, maximum_workspace, "minimum resident bytes");
    minimum_resident = AddChecked(
        minimum_resident, arguments.state_bytes, "minimum resident bytes");
    minimum_resident = AddChecked(
        minimum_resident,
        arguments.io_and_runtime_margin_bytes,
        "minimum resident bytes");

    std::ostringstream report;
    report << "{\"schema_version\":1,\"status\":\"QUERIED\",\"models\":[";
    for (std::size_t index = 0; index < models.size(); ++index) {
      if (index != 0) {
        report << ',';
      }
      const auto& memory = models[index];
      report << "{\"role\":\"" << JsonEscape(memory.model.role)
             << "\",\"path\":\""
             << JsonEscape(std::filesystem::absolute(memory.model.path).string())
             << "\",\"sha256\":\"" << memory.sha256
             << "\",\"work_bytes\":" << memory.work_bytes
             << ",\"weight_bytes\":" << memory.weight_bytes << '}';
    }
    report << "],\"budget\":{\"weight_bytes_sum\":" << weight_sum
           << ",\"serial_shared_workspace_bytes\":" << maximum_workspace
           << ",\"state_bytes\":" << arguments.state_bytes
           << ",\"io_and_runtime_margin_bytes\":"
           << arguments.io_and_runtime_margin_bytes
           << ",\"minimum_resident_bytes_excluding_unlisted_overhead\":"
           << minimum_resident << ",\"device_budget_bytes\":";
    if (arguments.has_device_budget) {
      report << arguments.device_budget_bytes << ",\"fits_declared_budget\":"
             << (minimum_resident <= arguments.device_budget_bytes ? "true" : "false");
    } else {
      report << "null,\"fits_declared_budget\":null";
    }
    report << "},\"assumptions\":["
           << "\"all models execute serially so one maximum workspace may be shared\","
           << "\"weight bytes are summed because different OM files are not assumed to share weights\""
           << "],\"claim_boundary\":\"aclmdlQuerySize planning evidence only; this does not prove that the complete model set loads, fits at runtime, is correct, or is fast\"}";

    AtomicWrite(arguments.output, report.str());
    Check(aclFinalize(), "aclFinalize");
    initialized = false;
    std::cerr << "[qwen35-dflash] stage=inspect-om-done models="
              << models.size() << " weight_bytes_sum=" << weight_sum
              << " shared_workspace_bytes=" << maximum_workspace
              << " minimum_resident_bytes=" << minimum_resident << '\n';
    return 0;
  } catch (const std::exception& error) {
    if (initialized) {
      static_cast<void>(aclFinalize());
    }
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}
