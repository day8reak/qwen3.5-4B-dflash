#include <acl/acl.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

#include "qwen35_dflash/chunk.hpp"
#include "qwen35_dflash/sha256.hpp"

namespace qwen35::dflash {
namespace {
void Check(aclError code, const char* op) {
  if (code != ACL_SUCCESS)
    throw std::runtime_error(std::string(op) +
                             " failed: " + std::to_string(code));
}
void Require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
struct ModelMemory {
  aclError status = ACL_SUCCESS;
  std::size_t work = 0, weight = 0;

  std::string Describe() const {
    std::ostringstream out;
    out << "query_status=" << status;
    if (status == ACL_SUCCESS)
      out << " weight_bytes=" << weight << " work_bytes=" << work;
    else
      out << " weight_bytes=unavailable work_bytes=unavailable";
    return out.str();
  }
};

void LogDeviceMemory(const char* phase, const std::string& graph = "all") {
  // These attributes may describe the same physical pool on some devices.
  // Report them separately; never add them or use an unsupported query as zero
  // available memory. Diagnostics must not reject an otherwise runnable model.
  for (auto attr : {ACL_HBM_MEM, ACL_DDR_MEM}) {
    std::size_t free = 0, total = 0;
    const auto status = aclrtGetMemInfo(attr, &free, &total);
    std::cerr << "[chunk-runtime] device-memory phase=" << phase
              << " graph=" << graph
              << " pool=" << (attr == ACL_HBM_MEM ? "HBM" : "DDR")
              << " query_status=" << status;
    if (status == ACL_SUCCESS && total > 0)
      std::cerr << " free_bytes=" << free << " total_bytes=" << total;
    else
      std::cerr << " free_bytes=unavailable total_bytes=unavailable";
    std::cerr << '\n';
  }
}

void LogProcessIdentity() {
  std::cerr << "[chunk-runtime] process pid=" << getpid() << " ppid=" << getppid();
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    if (line.rfind("NSpid:", 0) == 0) {
      std::cerr << " nspid=" << std::quoted(line.substr(6));
      break;
    }
  }
  std::error_code error;
  const auto ns = std::filesystem::read_symlink("/proc/self/ns/pid", error);
  std::cerr << " pid_namespace=" << std::quoted(error ? "unavailable" : ns.string()) << '\n';
}

struct CleanupAudit {
  std::size_t errors = 0, released_models = 0;
  std::size_t allocated_device_bytes = 0, freed_device_bytes = 0;

  void Check(aclError status, const char* operation) noexcept {
    if (status != ACL_SUCCESS) {
      ++errors;
      std::cerr << "[chunk-runtime] cleanup-error operation=" << operation
                << " status=" << status << '\n';
    }
  }
};

bool State(const std::string& name) {
  return name.size() > 2 && (name[0] == 't' || name[0] == 'd') &&
         name[1] >= '0' && name[1] <= '9';
}
aclDataType Dtype(const std::string& name) {
  if (name == "int64") return ACL_INT64;
  if (name == "int16") return ACL_INT16;
  if (name == "float32") return ACL_FLOAT;
  if (name == "float16") return ACL_FLOAT16;
  throw std::runtime_error("unsupported chunk tensor dtype");
}
std::string DtypeName(aclDataType dtype) {
  const char* name = "unknown";
  switch (dtype) {
    case ACL_FLOAT: name = "float32"; break;
    case ACL_FLOAT16: name = "float16"; break;
    case ACL_INT16: name = "int16"; break;
    case ACL_INT32: name = "int32"; break;
    case ACL_INT64: name = "int64"; break;
    default: break;
  }
  return std::string(name) + "(" + std::to_string(static_cast<int>(dtype)) + ")";
}
struct OmTensor {
  aclmdlIODims dims{};
  aclError dims_status = ACL_SUCCESS;
  aclDataType dtype = ACL_DT_UNDEFINED;
  std::size_t bytes = 0;

  bool Matches(const TensorSpec& spec) const {
    const auto limit = sizeof(dims.dims) / sizeof(dims.dims[0]);
    return dims_status == ACL_SUCCESS && dtype == Dtype(spec.dtype) &&
           bytes == spec.bytes() && dims.dimCount == spec.shape.size() &&
           dims.dimCount <= limit &&
           std::equal(spec.shape.begin(), spec.shape.end(), dims.dims);
  }
  std::string Describe() const {
    std::ostringstream out;
    const std::string name(dims.name,
        std::find(dims.name, dims.name + sizeof(dims.name), '\0'));
    out << "name=" << std::quoted(name) << " dtype=" << DtypeName(dtype)
        << " bytes=" << bytes << " rank=" << dims.dimCount << " shape=[";
    const auto limit = sizeof(dims.dims) / sizeof(dims.dims[0]);
    for (std::size_t i = 0; i < std::min(dims.dimCount, limit); ++i) {
      if (i) out << ',';
      out << dims.dims[i];
    }
    if (dims.dimCount > limit) out << ",...invalid-rank";
    out << "] dims_status=" << dims_status;
    return out.str();
  }
};
OmTensor ReadOmTensor(aclmdlDesc* desc, bool output, std::size_t index) {
  OmTensor tensor;
  tensor.dims_status = output ? aclmdlGetOutputDims(desc, index, &tensor.dims)
                             : aclmdlGetInputDims(desc, index, &tensor.dims);
  tensor.dtype = output ? aclmdlGetOutputDataType(desc, index)
                        : aclmdlGetInputDataType(desc, index);
  tensor.bytes = output ? aclmdlGetOutputSizeByIndex(desc, index)
                        : aclmdlGetInputSizeByIndex(desc, index);
  return tensor;
}
std::string DescribePlanTensor(const TensorSpec& spec) {
  std::ostringstream out;
  out << "name=" << std::quoted(spec.name) << " dtype=" << DtypeName(Dtype(spec.dtype))
      << " bytes=" << spec.bytes() << " rank=" << spec.shape.size() << " shape=[";
  for (std::size_t i = 0; i < spec.shape.size(); ++i) {
    if (i) out << ',';
    out << spec.shape[i];
  }
  out << ']';
  return out.str();
}
std::string DescribeModelIo(aclmdlDesc* desc, const ChunkGraph& graph) {
  std::ostringstream out;
  out << "\n[chunk-runtime] OM I/O descriptors graph=" << graph.name
      << " model=" << std::quoted(graph.model.string());
  for (bool output : {false, true}) {
    const auto count = output ? aclmdlGetNumOutputs(desc) : aclmdlGetNumInputs(desc);
    const auto& specs = output ? graph.outputs : graph.inputs;
    out << '\n' << (output ? "outputs" : "inputs") << ": plan=" << specs.size()
        << " om=" << count;
    for (std::size_t i = 0; i < std::max(count, specs.size()); ++i) {
      out << '\n' << (output ? "output[" : "input[") << i << "] expected={"
          << (i < specs.size() ? DescribePlanTensor(specs[i]) : "absent") << "} actual={"
          << (i < count ? ReadOmTensor(desc, output, i).Describe() : "absent") << '}';
    }
  }
  return out.str();
}
void ValidateModelIo(aclmdlDesc* desc, const ChunkGraph& graph) {
  const auto inputs = aclmdlGetNumInputs(desc), outputs = aclmdlGetNumOutputs(desc);
  if (inputs != graph.inputs.size() || outputs != graph.outputs.size()) {
    throw std::runtime_error(
        "OM tensor count differs from chunk plan: graph=" + graph.name +
        " inputs(plan=" + std::to_string(graph.inputs.size()) + ",om=" + std::to_string(inputs) +
        ") outputs(plan=" + std::to_string(graph.outputs.size()) + ",om=" + std::to_string(outputs) + ")" +
        DescribeModelIo(desc, graph));
  }
  for (bool output : {false, true}) {
    const auto& specs = output ? graph.outputs : graph.inputs;
    for (std::size_t i = 0; i < specs.size(); ++i) {
      const auto actual = ReadOmTensor(desc, output, i);
      if (!actual.Matches(specs[i])) {
        throw std::runtime_error(
            "OM tensor ABI differs from chunk plan: graph=" + graph.name +
            (output ? " output[" : " input[") + std::to_string(i) + "] expected={" +
            DescribePlanTensor(specs[i]) + "} actual={" + actual.Describe() + "}" +
            DescribeModelIo(desc, graph));
      }
    }
  }
}

std::string DebugQuote(const std::string& value) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 0x20) out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                           << static_cast<unsigned>(c) << std::dec;
    else out << c;
  }
  out << '"';
  return out.str();
}
std::string DebugMap(const std::map<std::string, std::string>& values) {
  std::ostringstream out;
  out << '{';
  bool first = true;
  for (const auto& item : values) {
    if (!first) out << ',';
    first = false;
    out << DebugQuote(item.first) << ':' << DebugQuote(item.second);
  }
  out << '}';
  return out.str();
}
std::string DebugTokens(const std::vector<std::int64_t>& tokens) {
  std::ostringstream out;
  out << '[';
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    if (i) out << ',';
    out << tokens[i];
  }
  out << ']';
  return out.str();
}
struct Memory {
  explicit Memory(CleanupAudit& audit) : audit(audit) {}
  CleanupAudit& audit;
  void* device = nullptr;
  void* host = nullptr;
  std::size_t bytes = 0;
  TensorSpec spec;
  ~Memory() {
    if (device) {
      const auto status = aclrtFree(device);
      audit.Check(status, "aclrtFree");
      if (status == ACL_SUCCESS) audit.freed_device_bytes += bytes;
      else std::cerr << "[chunk-runtime] unreleased-buffer name=" << spec.name
                     << " bytes=" << bytes << '\n';
    }
    if (host) audit.Check(aclrtFreeHost(host), "aclrtFreeHost");
  }
};
struct Loaded {
  explicit Loaded(CleanupAudit& audit) : audit(audit) {}
  CleanupAudit& audit;
  std::string name;
  std::uint32_t id = 0;
  bool loaded = false;
  aclmdlDesc* desc = nullptr;
  aclmdlDataset* inputs = nullptr;
  aclmdlDataset* outputs = nullptr;
  std::vector<aclDataBuffer*> input_buffers, output_buffers;
  ~Loaded() {
    for (auto* data : input_buffers)
      audit.Check(aclDestroyDataBuffer(data), "aclDestroyDataBuffer(input)");
    for (auto* data : output_buffers)
      audit.Check(aclDestroyDataBuffer(data), "aclDestroyDataBuffer(output)");
    if (inputs) audit.Check(aclmdlDestroyDataset(inputs), "aclmdlDestroyDataset(input)");
    if (outputs) audit.Check(aclmdlDestroyDataset(outputs), "aclmdlDestroyDataset(output)");
    if (desc) audit.Check(aclmdlDestroyDesc(desc), "aclmdlDestroyDesc");
    if (loaded) {
      const auto status = aclmdlUnload(id);
      audit.Check(status, "aclmdlUnload");
      if (status == ACL_SUCCESS) ++audit.released_models;
      std::cerr << "[chunk-runtime] unload graph=" << name
                << " model_id=" << id << " status=" << status << '\n';
    }
  }
};
}  // namespace

class AclChunkExecutor::Impl {
 public:
  Impl(const std::filesystem::path& path, int device, const std::string& mode,
       bool share_workspace)
      : plan(ReadChunkPlan(path, mode)), plan_path(path), plan_sha256(Sha256File(path)),
        device_id(device), share_workspace(share_workspace) {
    Require(device >= 0, "negative device ID");
    try {
      LogProcessIdentity();
      Check(aclInit(nullptr), "aclInit");
      initialized = true;
      Check(aclrtSetDevice(device), "aclrtSetDevice");
      device_set = true;
      Check(aclrtCreateContext(&context, device), "aclrtCreateContext");
      Check(aclrtSetCurrentContext(context), "aclrtSetCurrentContext");
      Check(aclrtCreateStream(&stream), "aclrtCreateStream");
      LoadMode(mode);
    } catch (...) {
      Cleanup();
      throw;
    }
  }
  ~Impl() { Cleanup(); }

  void LoadMode(const std::string& mode) {
    Require(!cleaned && !cleanup.errors && models.empty() && memory.empty() && !workspace,
            "unload models before changing mode");
    std::vector<const ChunkGraph*> selected;
    for (const auto& item : plan.graphs) {
      if (mode == "dflash" && item.first == "target_decode") continue;
      if (mode == "ordinary" &&
          (item.first == "target_verify" || item.first == "draft"))
        continue;
      selected.push_back(&item.second);
    }
    std::cerr << "[chunk-runtime] model-memory mode=" << mode
              << " selected_models=" << selected.size()
              << " weights=independent_per_om\n";
    LogDeviceMemory("before_models");
    // Query every selected OM before any is loaded, so an OOM on an early
    // model does not hide the requirements of the remaining graphs.
    std::map<std::string, ModelMemory> requirements;
    for (const auto* graph : selected) {
      auto& requirement = requirements[graph->name];
      requirement.status = aclmdlQuerySize(
          graph->model.c_str(), &requirement.work, &requirement.weight);
      std::cerr << "[chunk-runtime] om-memory graph=" << graph->name << ' '
                << requirement.Describe() << '\n';
    }
    std::size_t total_work = 0, maximum_work = 0;
    const bool queried_all = std::all_of(
        requirements.begin(), requirements.end(), [](const auto& item) {
          return item.second.status == ACL_SUCCESS;
        });
    if (queried_all && share_workspace) {
      for (const auto& item : requirements) {
        total_work += item.second.work;
        maximum_work = std::max(maximum_work, item.second.work);
      }
      if (maximum_work > 0) {
        workspace = std::make_unique<Memory>(cleanup);
        workspace->spec.name = "shared_workspace";
        workspace->bytes = maximum_work;
        Check(aclrtMalloc(&workspace->device, maximum_work,
                          ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc(shared_workspace)");
        cleanup.allocated_device_bytes += maximum_work;
      }
      std::cerr << "[chunk-runtime] workspace policy=shared_serial"
                << " shared_bytes=" << maximum_work
                << " separate_sum_bytes=" << total_work
                << " saved_work_bytes=" << total_work - maximum_work << '\n';
    } else {
      std::cerr << "[chunk-runtime] workspace policy=per_model"
                << " reason=" << (share_workspace ? "memory_query_unavailable" : "debug_private")
                << '\n';
    }
    for (const auto* graph : selected) {
      Load(*graph, requirements.at(graph->name));
    }
    // Cross-graph state/feature shapes are checked by the shared pool.
    std::size_t bytes = 0, discard_bytes = 0;
    for (const auto& item : memory) {
      bytes += item.second->bytes;
      if (IsVerifyDiscardState(item.second->spec.name))
        discard_bytes += item.second->bytes;
    }
    std::cerr << "[chunk-runtime] loaded_models=" << models.size()
              << " persistent_buffer_bytes=" << bytes
              << " verify_discard_buffer_bytes=" << discard_bytes << '\n';
  }

  void UnloadModels() {
    Require(!cleaned && !pending, "mode change requires a completed request");
    Check(aclrtSetCurrentContext(context), "aclrtSetCurrentContext(mode change)");
    Check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream(mode change)");
    invalid = true;
    models.clear();
    workspace.reset();
    memory.clear();
    LogDeviceMemory("after_mode_unload");
    if (cleanup.errors)
      throw std::runtime_error("model unload failed; refusing to load the next mode");
  }

  void Cleanup() noexcept {
    if (cleaned) return;
    cleaned = true;
    std::cerr << "[chunk-runtime] cleanup_begin loaded_models=" << models.size()
              << " device_buffers=" << memory.size() << '\n';
    if (context) cleanup.Check(aclrtSetCurrentContext(context), "aclrtSetCurrentContext");
    if (stream) cleanup.Check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");
    models.clear();
    // Every model must be unloaded before its borrowed work memory is freed.
    workspace.reset();
    memory.clear();
    if (device_set) LogDeviceMemory("after_release");
    if (stream) {
      cleanup.Check(aclrtDestroyStream(stream), "aclrtDestroyStream");
      stream = nullptr;
    }
    if (context) {
      cleanup.Check(aclrtDestroyContext(context), "aclrtDestroyContext");
      context = nullptr;
    }
    if (device_set) {
      cleanup.Check(aclrtResetDevice(device_id), "aclrtResetDevice");
      device_set = false;
    }
    if (initialized) {
      cleanup.Check(aclFinalize(), "aclFinalize");
      initialized = false;
    }
    std::cerr << "[chunk-runtime] cleanup_end released_models=" << cleanup.released_models
              << " allocated_device_bytes=" << cleanup.allocated_device_bytes
              << " freed_device_bytes=" << cleanup.freed_device_bytes
              << " errors=" << cleanup.errors << '\n';
  }

  std::string Key(const TensorSpec& spec, const std::string& graph,
                  bool output) const {
    if (State(spec.name)) return spec.name + (output ? ".next" : ".current");
    if (spec.name == "features" || spec.name == "start_position" ||
        spec.name == "valid_rows" || spec.name == "anchor" ||
        spec.name == "proposal_count")
      return spec.name;
    return graph + "." + spec.name;
  }
  Memory& Get(const TensorSpec& spec, const std::string& graph, bool output) {
    const auto key = Key(spec, graph, output);
    auto found = memory.find(key);
    if (found != memory.end()) {
      Require(found->second->spec.dtype == spec.dtype &&
                  found->second->spec.shape == spec.shape,
              "cross-graph tensor ABI mismatch");
      return *found->second;
    }
    auto buffer = std::make_unique<Memory>(cleanup);
    buffer->spec = spec;
    buffer->bytes = spec.bytes();
    Check(
        aclrtMalloc(&buffer->device, buffer->bytes, ACL_MEM_MALLOC_NORMAL_ONLY),
        "aclrtMalloc");
    cleanup.allocated_device_bytes += buffer->bytes;
    // Discard outputs have private graph-scoped device allocations. No host
    // allocation means Call queues neither H2D nor D2H for them.
    if (!State(spec.name) && spec.name != "features" &&
        !IsVerifyDiscardState(spec.name))
      Check(aclrtMallocHost(&buffer->host, buffer->bytes), "aclrtMallocHost");
    auto* pointer = buffer.get();
    memory.emplace(key, std::move(buffer));
    return *pointer;
  }

  void Load(const ChunkGraph& graph, const ModelMemory& requirement) {
    auto owner = std::make_unique<Loaded>(cleanup);
    auto& model = *owner;
    model.name = graph.name;
    std::cerr << "[chunk-runtime] load graph=" << graph.name
              << " model=" << std::quoted(graph.model.string()) << '\n';
    LogDeviceMemory("before_load", graph.name);
    // Call() uses one stream and synchronizes every execute. Temporary work
    // memory can therefore be shared; weights keep independent GE ownership.
    const auto* operation = workspace ? "aclmdlLoadFromFileWithMem" : "aclmdlLoadFromFile";
    const auto status = workspace
        ? aclmdlLoadFromFileWithMem(graph.model.c_str(), &model.id,
                                   workspace->device, workspace->bytes, nullptr, 0)
        : aclmdlLoadFromFile(graph.model.c_str(), &model.id);
    if (status != ACL_SUCCESS) {
      LogDeviceMemory("load_failed", graph.name);
      throw std::runtime_error(
          std::string(operation) + " failed: " + std::to_string(status) +
          " graph=" + graph.name + " model=" + graph.model.string() +
          " loaded_models=" + std::to_string(models.size()) + " " +
          requirement.Describe() +
          "; this model's I/O buffers have not been allocated and it has not executed");
    }
    model.loaded = true;
    model.desc = aclmdlCreateDesc();
    Require(model.desc != nullptr, "aclmdlCreateDesc returned null");
    Check(aclmdlGetDesc(model.desc, model.id), "aclmdlGetDesc");
    ValidateModelIo(model.desc, graph);
    model.inputs = aclmdlCreateDataset();
    model.outputs = aclmdlCreateDataset();
    Require(model.inputs && model.outputs, "aclmdlCreateDataset returned null");
    for (bool output : {false, true}) {
      const auto& specs = output ? graph.outputs : graph.inputs;
      auto& buffers = output ? model.output_buffers : model.input_buffers;
      for (std::size_t index = 0; index < specs.size(); ++index) {
        const auto& spec = specs[index];
        auto& mem = Get(spec, graph.name, output);
        auto* data = aclCreateDataBuffer(mem.device, mem.bytes);
        Require(data != nullptr, "aclCreateDataBuffer returned null");
        buffers.push_back(data);
        Check(
            aclmdlAddDatasetBuffer(output ? model.outputs : model.inputs, data),
            "aclmdlAddDatasetBuffer");
      }
    }
    models.emplace(graph.name, std::move(owner));
    LogDeviceMemory("after_load_and_io", graph.name);
  }

  void Reset(std::int64_t padding) {
    Check(aclrtSetCurrentContext(context), "aclrtSetCurrentContext");
    Check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream(reset)");
    for (const auto& item : memory) {
      if (State(item.second->spec.name))
        Check(aclrtMemsetAsync(item.second->device, item.second->bytes, 0,
                               item.second->bytes, stream),
              "aclrtMemsetAsync");
    }
    Check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream(reset)");
    pad = padding;
    cursor = 0;
    draft_cursor = 0;
    pending = 0;
    feature_rows = 0;
    invalid = false;
    calls = 0;
    timings.clear();
  }

  void Healthy() const {
    Require(!invalid, "request invalidated; reset before reuse");
  }
  void Swap(char kind) {
    for (const auto& spec :
         plan.graphs.at(kind == 't' ? "target_prefill" : "draft").inputs) {
      if (State(spec.name) && spec.name[0] == kind)
        memory.at(spec.name + ".current").swap(memory.at(spec.name + ".next"));
    }
  }
  void Scalar(const std::string& name, std::int64_t value) {
    auto& mem = *memory.at(name);
    Require(mem.host != nullptr, "scalar is not host accessible");
    if (mem.spec.dtype == "int16") {
      Require(mem.bytes == 2 && value >= 0 && value <= 32767,
              "invalid INT16 scalar");
      *static_cast<std::int16_t*>(mem.host) = static_cast<std::int16_t>(value);
    } else {
      Require(mem.spec.dtype == "int64" && mem.bytes == 8,
              "invalid INT64 scalar");
      *static_cast<std::int64_t*>(mem.host) = value;
    }
  }
  void Inputs(const std::string& graph, const std::vector<std::int64_t>& ids) {
    Require(!ids.empty() && cursor + ids.size() <= plan.capacity,
            "target rows exceed logical capacity");
    auto& mem = *memory.at(graph + ".input_ids");
    Require(mem.spec.dtype == "int64" && mem.bytes / 8 >= ids.size(),
            "target input gear is too short");
    auto* data = static_cast<std::int64_t*>(mem.host);
    std::fill_n(data, mem.bytes / 8, pad);
    std::copy(ids.begin(), ids.end(), data);
    Scalar("start_position", static_cast<std::int64_t>(cursor));
    Scalar("valid_rows", static_cast<std::int64_t>(ids.size()));
  }
  void Call(const std::string& name) {
    Healthy();
    const auto start = std::chrono::steady_clock::now();
    auto& model = *models.at(name);
    const auto& graph = plan.graphs.at(name);
    try {
      for (bool output : {false, true}) {
        const auto& specs = output ? graph.outputs : graph.inputs;
        const auto& buffers =
            output ? model.output_buffers : model.input_buffers;
        for (std::size_t index = 0; index < specs.size(); ++index) {
          auto& mem = Get(specs[index], name, output);
          Check(aclUpdateDataBuffer(buffers[index], mem.device, mem.bytes),
                "aclUpdateDataBuffer");
          if (!output && mem.host)
            Check(aclrtMemcpyAsync(mem.device, mem.bytes, mem.host, mem.bytes,
                                   ACL_MEMCPY_HOST_TO_DEVICE, stream),
                  "aclrtMemcpyAsync(H2D)");
        }
      }
      Check(aclmdlExecuteAsync(model.id, model.inputs, model.outputs, stream),
            "aclmdlExecuteAsync");
      for (const auto& spec : graph.outputs) {
        auto& mem = Get(spec, name, true);
        if (mem.host)
          Check(aclrtMemcpyAsync(mem.host, mem.bytes, mem.device, mem.bytes,
                                 ACL_MEMCPY_DEVICE_TO_HOST, stream),
                "aclrtMemcpyAsync(D2H)");
      }
      Check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");
    } catch (...) {
      invalid = true;
      throw;
    }
    ++calls;
    timings[name].push_back(std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - start)
                                .count());
  }
  std::vector<std::int64_t> Tokens(const std::string& graph,
                                   const std::string& name = "target_top1") {
    auto& mem = *memory.at(graph + "." + name);
    Require(mem.spec.dtype == "int64" && mem.host, "token output ABI mismatch");
    auto* values = static_cast<std::int64_t*>(mem.host);
    return {values, values + mem.bytes / 8};
  }

  std::int64_t Prefill(const std::vector<std::int64_t>& ids, bool draft) {
    Healthy();
    Require(cursor == 0 && pending == 0, "prefill needs a fresh request");
    std::int64_t token = 0;
    for (std::size_t offset = 0; offset < ids.size(); offset += 64) {
      const auto rows = std::min<std::size_t>(64, ids.size() - offset);
      Inputs("target_prefill",
             {ids.begin() + offset, ids.begin() + offset + rows});
      Call("target_prefill");
      token = Tokens("target_prefill").at(0);
      Swap('t');
      feature_start = cursor;
      feature_rows = rows;
      cursor += rows;
      // Only the final chunk's proposal is needed. Earlier calls initialize
      // Draft KV; the extra block computation is an explicit four-OM tradeoff.
      if (draft && offset + rows < ids.size())
        static_cast<void>(Propose(token, 15));
    }
    return token;
  }
  void PrepareDraft(std::int64_t anchor, std::size_t proposal_count) {
    Healthy();
    Require(proposal_count > 0 && proposal_count <= 15, "Draft proposal_count must be 1..15");
    Require(!pending && feature_rows > 0 && draft_cursor == feature_start &&
                feature_start + feature_rows == cursor,
            "Draft context cursor is inconsistent");
    Scalar("start_position", static_cast<std::int64_t>(feature_start));
    Scalar("valid_rows", static_cast<std::int64_t>(feature_rows));
    Scalar("anchor", anchor);
    Scalar("proposal_count", static_cast<std::int64_t>(proposal_count));
  }
  std::map<std::string, std::string> DraftInputHashes(std::int64_t anchor,
                                                    std::size_t proposal_count) {
    PrepareDraft(anchor, proposal_count);
    Check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream(input audit)");
    const auto& inputs = plan.graphs.at("draft").inputs;
    Memory scratch(cleanup);
    for (const auto& spec : inputs) scratch.bytes = std::max(scratch.bytes, spec.bytes());
    Check(aclrtMallocHost(&scratch.host, scratch.bytes), "aclrtMallocHost(input audit)");
    std::map<std::string, std::string> hashes;
    for (const auto& spec : inputs) {
      auto& mem = Get(spec, "draft", false);
      const void* data = mem.host;
      if (!data) {
        Check(aclrtMemcpyAsync(scratch.host, scratch.bytes, mem.device, mem.bytes,
                               ACL_MEMCPY_DEVICE_TO_HOST, stream), "aclrtMemcpyAsync(input audit)");
        Check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream(input audit)");
        data = scratch.host;
      }
      // Control scalars are hashed from the exact host bytes Call() will upload;
      // features/current KV are read from device. No output buffer is sampled.
      hashes.emplace(spec.name, Sha256(std::string_view(static_cast<const char*>(data), mem.bytes)));
    }
    return hashes;
  }

  std::pair<bool, std::string> DebugDraftReplay(
      const std::vector<std::int64_t>& prompt, std::int64_t padding,
      std::size_t proposal_count, std::size_t repetitions,
      const std::filesystem::path& output_directory,
      const std::filesystem::path& input_directory) {
    Require(!prompt.empty() && prompt.size() + 16 <= plan.capacity,
            "debug replay prompt needs room for a full Draft block");
    Require(padding >= 0 && padding < plan.vocabulary &&
                std::all_of(prompt.begin(), prompt.end(), [&](auto id) {
                  return id >= 0 && id < plan.vocabulary;
                }), "debug replay token outside vocabulary");
    Require(repetitions > 0 && repetitions <= 1000 && proposal_count > 0 &&
                proposal_count <= 15, "debug replay requires 1..1000 repetitions and 1..15 proposals");
    Require(std::filesystem::create_directory(output_directory),
            "debug replay directory must be new");
    const auto snapshot_dir = output_directory / "inputs";
    std::filesystem::create_directory(snapshot_dir);
    std::ofstream trace(output_directory / "iterations.jsonl");
    trace.exceptions(std::ios::badbit | std::ios::failbit);

    const auto& graph = plan.graphs.at("draft");
    Memory scratch(cleanup);
    for (const auto& spec : graph.inputs) scratch.bytes = std::max(scratch.bytes, spec.bytes());
    for (const auto& spec : graph.outputs) scratch.bytes = std::max(scratch.bytes, spec.bytes());
    Check(aclrtMallocHost(&scratch.host, scratch.bytes), "aclrtMallocHost(debug replay)");
    auto read_device = [&](Memory& mem) {
      Check(aclrtMemcpyAsync(scratch.host, scratch.bytes, mem.device, mem.bytes,
                             ACL_MEMCPY_DEVICE_TO_HOST, stream), "aclrtMemcpyAsync(debug read)");
      Check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream(debug read)");
      return std::string(static_cast<const char*>(scratch.host), mem.bytes);
    };
    auto write_file = [](const std::filesystem::path& path, const std::string& bytes) {
      std::ofstream out(path, std::ios::binary);
      out.exceptions(std::ios::badbit | std::ios::failbit);
      out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    };
    auto read_file = [](const std::filesystem::path& path, std::size_t bytes) {
      Require(std::filesystem::is_regular_file(path) && std::filesystem::file_size(path) == bytes,
              "debug snapshot file size mismatch");
      std::string result(bytes, '\0');
      std::ifstream in(path, std::ios::binary);
      in.exceptions(std::ios::badbit | std::ios::failbit);
      in.read(result.data(), static_cast<std::streamsize>(bytes));
      return result;
    };

    // Bind imported bytes to the exact Draft OM, shape ABI, prompt and controls.
    // ATC precision changes need a new snapshot contract, not a silent A/B.
    std::ostringstream contract;
    contract << "qwen35-draft-replay-inputs-v1\n" << graph.sha256 << '\n'
             << plan.capacity << ' ' << padding << ' ' << proposal_count << '\n';
    for (auto id : prompt) contract << id << ' ';
    contract << '\n';
    for (const auto& spec : graph.inputs) {
      Require(spec.name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
                  == std::string::npos && !spec.name.empty(), "unsafe snapshot tensor name");
      contract << spec.name << ' ' << spec.dtype << ' ' << spec.bytes();
      for (auto dim : spec.shape) contract << ' ' << dim;
      contract << '\n';
    }
    if (!input_directory.empty())
      Require(std::filesystem::is_regular_file(input_directory / "contract.txt") &&
                  std::filesystem::file_size(input_directory / "contract.txt") == contract.str().size() &&
                  read_file(input_directory / "contract.txt", contract.str().size()) == contract.str(),
              "debug snapshot contract differs from Draft OM/prompt/ABI");
    write_file(snapshot_dir / "contract.txt", contract.str());

    Reset(padding);
    const auto anchor = Prefill(prompt, true);
    PrepareDraft(anchor, proposal_count);
    std::ifstream imported_hashes;
    if (!input_directory.empty()) {
      imported_hashes.open(input_directory / "sha256.txt");
      imported_hashes.exceptions(std::ios::badbit | std::ios::failbit);
    }
    std::map<std::string, std::string> frozen, frozen_hashes;
    std::ostringstream snapshot_hashes;
    for (const auto& spec : graph.inputs) {
      auto& mem = Get(spec, "draft", false);
      std::string bytes;
      if (input_directory.empty()) {
        bytes = mem.host ? std::string(static_cast<const char*>(mem.host), mem.bytes) : read_device(mem);
      } else {
        bytes = read_file(input_directory / (spec.name + ".bin"), mem.bytes);
        std::string expected_name, expected_hash;
        imported_hashes >> expected_name >> expected_hash;
        Require(expected_name == spec.name && Sha256(bytes) == expected_hash,
                "debug snapshot input hash mismatch");
      }
      frozen_hashes[spec.name] = Sha256(bytes);
      snapshot_hashes << spec.name << ' ' << frozen_hashes.at(spec.name) << '\n';
      write_file(snapshot_dir / (spec.name + ".bin"), bytes);
      frozen.emplace(spec.name, std::move(bytes));
    }
    write_file(snapshot_dir / "sha256.txt", snapshot_hashes.str());
    auto hashes = [&](bool output) {
      std::map<std::string, std::string> result;
      for (const auto& spec : output ? graph.outputs : graph.inputs)
        result[spec.name] = Sha256(read_device(Get(spec, "draft", output)));
      return result;
    };
    auto addresses = [&](bool output) {
      std::map<std::string, std::string> result;
      for (const auto& spec : output ? graph.outputs : graph.inputs) {
        std::ostringstream address;
        address << Get(spec, "draft", output).device;
        result[spec.name] = address.str();
      }
      return result;
    };

    bool stable = true;
    std::vector<std::int64_t> reference;
    std::map<std::string, std::string> reference_outputs;
    std::ostringstream phases;
    for (const std::string phase : {"isolated", "interleaved_prefill"}) {
      std::size_t token_mismatches = 0, restore_mismatches = 0, input_mutations = 0, output_hash_mismatches = 0;
      if (phase != "isolated") phases << ',';
      for (std::size_t iteration = 0; iteration < repetitions; ++iteration) {
        if (phase == "interleaved_prefill") {
          Reset(padding);
          static_cast<void>(Prefill(prompt, true));
        }
        // Restore all 17 actual inputs, including every physical KV row and
        // controls. Never publish the replay's Draft outputs with Swap('d').
        for (const auto& spec : graph.inputs) {
          auto& mem = Get(spec, "draft", false);
          const auto& bytes = frozen.at(spec.name);
          if (mem.host) std::memcpy(mem.host, bytes.data(), bytes.size());
          std::memcpy(scratch.host, bytes.data(), bytes.size());
          Check(aclrtMemcpyAsync(mem.device, mem.bytes, scratch.host, mem.bytes,
                                 ACL_MEMCPY_HOST_TO_DEVICE, stream), "aclrtMemcpyAsync(debug restore)");
          // The single pinned staging buffer must stay alive until H2D completes.
          Check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream(debug restore)");
        }
        const auto before = hashes(false);
        trace << "{\"event\":\"prepared\",\"phase\":" << DebugQuote(phase)
              << ",\"iteration\":" << iteration << ",\"input_sha256\":" << DebugMap(before) << "}\n";
        trace.flush();
        Call("draft");
        auto tokens = Tokens("draft", "draft_top1");
        Require(tokens.size() >= proposal_count, "debug Draft token output too short");
        tokens.resize(proposal_count);
        const auto after = hashes(false), outputs = hashes(true);
        if (phase == "isolated" && iteration == 0) {
          reference = tokens;
          reference_outputs = outputs;
        }
        const bool input_match = before == frozen_hashes;
        const bool readonly = before == after;
        const bool token_match = reference == tokens;
        const bool tokens_valid = std::all_of(tokens.begin(), tokens.end(), [&](auto id) {
          return id >= 0 && id < plan.vocabulary;
        });
        if (!input_match) ++restore_mismatches;
        if (!readonly) ++input_mutations;
        if (!token_match || !tokens_valid) ++token_mismatches;
        if (outputs != reference_outputs) ++output_hash_mismatches;
        const bool ok = input_match && readonly && token_match && tokens_valid;
        stable = stable && ok;
        trace << "{\"event\":\"completed\",\"phase\":" << DebugQuote(phase)
              << ",\"iteration\":" << iteration << ",\"profiled\":false"
              << ",\"status\":" << DebugQuote(ok ? "PASS" : "FAIL")
              << ",\"input_sha256\":" << DebugMap(before)
              << ",\"input_after_sha256\":" << DebugMap(after)
              << ",\"output_sha256\":" << DebugMap(outputs)
              << ",\"input_device_addresses\":" << DebugMap(addresses(false))
              << ",\"output_device_addresses\":" << DebugMap(addresses(true))
              << ",\"input_matches_snapshot\":" << (input_match ? "true" : "false")
              << ",\"inputs_unchanged\":" << (readonly ? "true" : "false")
              << ",\"valid_tokens_match\":" << (token_match ? "true" : "false")
              << ",\"valid_tokens_in_range\":" << (tokens_valid ? "true" : "false")
              << ",\"all_output_bytes_match\":" << (outputs == reference_outputs ? "true" : "false")
              << ",\"output_token_ids\":" << DebugTokens(tokens)
              << ",\"first_token_difference\":";
        if (token_match) trace << "null";
        else {
          const auto index = static_cast<std::size_t>(std::mismatch(reference.begin(), reference.end(), tokens.begin()).first - reference.begin());
          trace << "{\"index\":" << index << ",\"reference\":" << reference[index]
                << ",\"actual\":" << tokens[index] << '}';
        }
        trace << "}\n";
        trace.flush();
        std::cerr << "[draft-replay] phase=" << phase << " iteration=" << iteration
                  << " input_match=" << input_match << " readonly=" << readonly
                  << " tokens_match=" << token_match << '\n';
      }
      phases << "{\"phase\":" << DebugQuote(phase) << ",\"iterations\":" << repetitions
             << ",\"token_mismatch_iterations\":" << token_mismatches
             << ",\"input_restore_mismatch_iterations\":" << restore_mismatches
             << ",\"input_mutation_iterations\":" << input_mutations
             << ",\"full_output_hash_mismatch_iterations\":" << output_hash_mismatches << '}';
    }
    // This diagnostic leaves no cache branch eligible for continued generation.
    invalid = true;
    std::ostringstream report;
    report << "{\"schema_version\":1,\"status\":" << DebugQuote(stable ? "PASS_REPLAY_CHECKS" : "FAIL_REPLAY_CHECKS")
           << ",\"scope\":\"frozen Draft OM input replay; valid tokens and readonly inputs\""
           << ",\"formal_latency_evidence\":false,\"ordinary_parity\":\"NOT_RUN\""
           << ",\"profiled\":false,\"draft_outputs_committed\":false"
           << ",\"requested_workspace\":" << DebugQuote(share_workspace ? "shared" : "private")
           << ",\"actual_workspace_policy\":" << DebugQuote(workspace ? "shared_serial" : "per_model")
           << ",\"draft_om_sha256\":" << DebugQuote(graph.sha256)
           << ",\"input_directory\":" << DebugQuote(snapshot_dir.string())
           << ",\"trace\":" << DebugQuote((output_directory / "iterations.jsonl").string())
           << ",\"snapshot_sha256\":" << DebugMap(frozen_hashes)
           << ",\"reference_token_ids\":" << DebugTokens(reference)
           << ",\"reference_output_sha256\":" << DebugMap(reference_outputs)
           << ",\"phases\":[" << phases.str() << "]"
           << ",\"note\":\"All output hashes include physical padding; inspect them separately. "
              "Restores, readbacks and synchronization perturb execution. A passing replay does not close the original instability.\"}";
    return {stable, report.str()};
  }
  std::vector<std::int64_t> Propose(std::int64_t anchor, std::size_t proposal_count) {
    PrepareDraft(anchor, proposal_count);
    Call("draft");
    Swap('d');
    draft_cursor = cursor;
    feature_rows = 0;
    return Tokens("draft", "draft_top1");
  }
  std::vector<std::int64_t> Verify(const std::vector<std::int64_t>& ids) {
    Healthy();
    Require(!pending && cursor > 0 && ids.size() <= 16,
            "verify requires committed state and 1..16 rows");
    Inputs("target_verify", ids);
    Call("target_verify");
    const auto accepted = Tokens("target_verify", "accepted_count").at(0);
    Require(accepted >= 0 && static_cast<std::size_t>(accepted) < ids.size(),
            "OM returned invalid accepted_count");
    pending = ids.size();
    pending_commit = static_cast<std::size_t>(accepted) + 1;
    return Tokens("target_verify");
  }
  void Commit(std::size_t rows) {
    Healthy();
    Require(pending && rows == pending_commit && rows <= pending,
            "host acceptance disagrees with fused OM");
    // Publish only named Target cache outputs, containing second-pass GDR
    // states. First-pass verify_discard_* buffers never enter Swap('t').
    Swap('t');
    feature_start = cursor;
    feature_rows = rows;
    cursor += rows;
    pending = 0;
  }
  std::int64_t Decode(std::int64_t anchor) {
    Healthy();
    Require(!pending && cursor > 0, "decode requires committed state");
    Inputs("target_decode", {anchor});
    Call("target_decode");
    Swap('t');
    ++cursor;
    feature_rows = 0;
    return Tokens("target_decode").at(0);
  }

  ChunkPlan plan;
  std::filesystem::path plan_path;
  std::string plan_sha256;
  int device_id;
  bool share_workspace;
  bool initialized = false, device_set = false, invalid = true, cleaned = false;
  aclrtContext context = nullptr;
  aclrtStream stream = nullptr;
  CleanupAudit cleanup;
  std::unique_ptr<Memory> workspace;
  std::map<std::string, std::unique_ptr<Loaded>> models;
  std::map<std::string, std::unique_ptr<Memory>> memory;
  std::map<std::string, std::vector<double>> timings;
  std::size_t cursor = 0, draft_cursor = 0, pending = 0, pending_commit = 0;
  std::size_t feature_start = 0, feature_rows = 0, calls = 0;
  std::int64_t pad = 0;
};

AclChunkExecutor::AclChunkExecutor(const std::filesystem::path& plan,
                                   int device, const std::string& mode,
                                   bool share_workspace)
    : impl_(std::make_unique<Impl>(plan, device, mode, share_workspace)) {}
AclChunkExecutor::~AclChunkExecutor() = default;
void AclChunkExecutor::Close() {
  impl_->Cleanup();
  if (impl_->cleanup.errors)
    throw std::runtime_error("chunk runner cleanup failed; see cleanup-error log");
}
void AclChunkExecutor::UnloadModels() { impl_->UnloadModels(); }
void AclChunkExecutor::LoadMode(const std::string& mode) {
  Require(!impl_->cleaned && impl_->models.empty() && impl_->memory.empty(),
          "unload models before changing mode");
  Require(Sha256File(impl_->plan_path) == impl_->plan_sha256,
          "chunk plan changed between modes");
  // Recheck hashes for the new mode's OMs before allocating device memory.
  impl_->plan = ReadChunkPlan(impl_->plan_path, mode);
  impl_->LoadMode(mode);
}
void AclChunkExecutor::Synchronize() {
  Check(aclrtSetCurrentContext(impl_->context),
        "aclrtSetCurrentContext(profile)");
  Check(aclrtSynchronizeStream(impl_->stream),
        "aclrtSynchronizeStream(profile)");
}
std::size_t AclChunkExecutor::sequence_length() const noexcept {
  return impl_->plan.capacity;
}
std::int64_t AclChunkExecutor::vocabulary_size() const noexcept {
  return impl_->plan.vocabulary;
}
void AclChunkExecutor::Reset(std::int64_t pad) { impl_->Reset(pad); }
void AclChunkExecutor::Abort() noexcept {
  impl_->invalid = true;
  impl_->pending = 0;
}
std::int64_t AclChunkExecutor::Prefill(const std::vector<std::int64_t>& ids,
                                       bool draft) {
  return impl_->Prefill(ids, draft);
}
std::vector<std::int64_t> AclChunkExecutor::Propose(std::int64_t anchor, std::size_t count) {
  return impl_->Propose(anchor, count);
}
std::map<std::string, std::string> AclChunkExecutor::DraftInputHashes(
    std::int64_t anchor, std::size_t count) {
  return impl_->DraftInputHashes(anchor, count);
}
std::vector<std::int64_t> AclChunkExecutor::Verify(
    const std::vector<std::int64_t>& ids) {
  return impl_->Verify(ids);
}
std::pair<bool, std::string> AclChunkExecutor::DebugDraftReplay(
    const std::vector<std::int64_t>& prompt, std::int64_t pad,
    std::size_t proposal_count, std::size_t repetitions,
    const std::filesystem::path& output_directory,
    const std::filesystem::path& input_directory) {
  return impl_->DebugDraftReplay(prompt, pad, proposal_count, repetitions,
                                output_directory, input_directory);
}
void AclChunkExecutor::Commit(std::size_t rows) { impl_->Commit(rows); }
std::int64_t AclChunkExecutor::Decode(std::int64_t anchor) {
  return impl_->Decode(anchor);
}
bool AclChunkExecutor::HasOrdinaryDecode() const noexcept {
  return impl_->models.count("target_decode") != 0;
}
std::size_t AclChunkExecutor::graph_calls() const noexcept {
  return impl_->calls;
}
const std::map<std::string, std::vector<double>>& AclChunkExecutor::stage_ms()
    const {
  return impl_->timings;
}
}  // namespace qwen35::dflash
