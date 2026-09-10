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
  Impl(const std::filesystem::path& path, int device, const std::string& mode)
      : plan(ReadChunkPlan(path, mode)), plan_path(path), plan_sha256(Sha256File(path)), device_id(device) {
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
    if (queried_all) {
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
                << " reason=memory_query_unavailable\n";
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
                                   int device, const std::string& mode)
    : impl_(std::make_unique<Impl>(plan, device, mode)) {}
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
