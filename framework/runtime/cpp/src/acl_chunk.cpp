#include <acl/acl.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include "qwen35_dflash/chunk.hpp"

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
struct Memory {
  void* device = nullptr;
  void* host = nullptr;
  std::size_t bytes = 0;
  TensorSpec spec;
  ~Memory() {
    if (device) static_cast<void>(aclrtFree(device));
    if (host) static_cast<void>(aclrtFreeHost(host));
  }
};
struct Loaded {
  std::uint32_t id = 0;
  bool loaded = false;
  aclmdlDesc* desc = nullptr;
  aclmdlDataset* inputs = nullptr;
  aclmdlDataset* outputs = nullptr;
  std::vector<aclDataBuffer*> input_buffers, output_buffers;
  ~Loaded() {
    for (auto* data : input_buffers)
      static_cast<void>(aclDestroyDataBuffer(data));
    for (auto* data : output_buffers)
      static_cast<void>(aclDestroyDataBuffer(data));
    if (inputs) static_cast<void>(aclmdlDestroyDataset(inputs));
    if (outputs) static_cast<void>(aclmdlDestroyDataset(outputs));
    if (desc) static_cast<void>(aclmdlDestroyDesc(desc));
    if (loaded) static_cast<void>(aclmdlUnload(id));
  }
};
}  // namespace

class AclChunkExecutor::Impl {
 public:
  Impl(const std::filesystem::path& path, int device, const std::string& mode)
      : plan(ReadChunkPlan(path, mode)), device_id(device) {
    Require(device >= 0, "negative device ID");
    try {
      Check(aclInit(nullptr), "aclInit");
      initialized = true;
      Check(aclrtSetDevice(device), "aclrtSetDevice");
      device_set = true;
      Check(aclrtCreateContext(&context, device), "aclrtCreateContext");
      Check(aclrtSetCurrentContext(context), "aclrtSetCurrentContext");
      Check(aclrtCreateStream(&stream), "aclrtCreateStream");
      for (const auto& item : plan.graphs) {
        if (mode == "dflash" && item.first == "target_decode") continue;
        if (mode == "ordinary" &&
            (item.first == "target_verify" || item.first == "draft"))
          continue;
        Load(item.second);
      }
      // Cross-graph state/feature shapes are checked by the shared pool.
      std::size_t bytes = 0;
      for (const auto& item : memory) bytes += item.second->bytes;
      std::cerr << "[chunk-runtime] loaded_models=" << models.size()
                << " persistent_buffer_bytes=" << bytes << '\n';
    } catch (...) {
      Cleanup();
      throw;
    }
  }
  ~Impl() { Cleanup(); }

  void Cleanup() noexcept {
    if (context) static_cast<void>(aclrtSetCurrentContext(context));
    if (stream) static_cast<void>(aclrtSynchronizeStream(stream));
    models.clear();
    memory.clear();
    if (stream) {
      static_cast<void>(aclrtDestroyStream(stream));
      stream = nullptr;
    }
    if (context) {
      static_cast<void>(aclrtDestroyContext(context));
      context = nullptr;
    }
    if (device_set) {
      static_cast<void>(aclrtResetDevice(device_id));
      device_set = false;
    }
    if (initialized) {
      static_cast<void>(aclFinalize());
      initialized = false;
    }
  }

  std::string Key(const TensorSpec& spec, const std::string& graph,
                  bool output) const {
    if (State(spec.name)) return spec.name + (output ? ".next" : ".current");
    if (spec.name == "features" || spec.name == "start_position" ||
        spec.name == "valid_rows" || spec.name == "anchor")
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
    auto buffer = std::make_unique<Memory>();
    buffer->spec = spec;
    buffer->bytes = spec.bytes();
    Check(
        aclrtMalloc(&buffer->device, buffer->bytes, ACL_MEM_MALLOC_NORMAL_ONLY),
        "aclrtMalloc");
    if (!State(spec.name) && spec.name != "features")
      Check(aclrtMallocHost(&buffer->host, buffer->bytes), "aclrtMallocHost");
    auto* pointer = buffer.get();
    memory.emplace(key, std::move(buffer));
    return *pointer;
  }

  void Load(const ChunkGraph& graph) {
    auto owner = std::make_unique<Loaded>();
    auto& model = *owner;
    Check(aclmdlLoadFromFile(graph.model.c_str(), &model.id),
          "aclmdlLoadFromFile");
    model.loaded = true;
    model.desc = aclmdlCreateDesc();
    Require(model.desc != nullptr, "aclmdlCreateDesc returned null");
    Check(aclmdlGetDesc(model.desc, model.id), "aclmdlGetDesc");
    Require(aclmdlGetNumInputs(model.desc) == graph.inputs.size() &&
                aclmdlGetNumOutputs(model.desc) == graph.outputs.size(),
            "OM tensor count differs from chunk plan");
    model.inputs = aclmdlCreateDataset();
    model.outputs = aclmdlCreateDataset();
    Require(model.inputs && model.outputs, "aclmdlCreateDataset returned null");
    for (bool output : {false, true}) {
      const auto& specs = output ? graph.outputs : graph.inputs;
      auto& buffers = output ? model.output_buffers : model.input_buffers;
      for (std::size_t index = 0; index < specs.size(); ++index) {
        const auto& spec = specs[index];
        aclmdlIODims dims{};
        Check(output ? aclmdlGetOutputDims(model.desc, index, &dims)
                     : aclmdlGetInputDims(model.desc, index, &dims),
              "aclmdlGetIODims");
        const auto dtype = output ? aclmdlGetOutputDataType(model.desc, index)
                                  : aclmdlGetInputDataType(model.desc, index);
        const auto bytes = output
                               ? aclmdlGetOutputSizeByIndex(model.desc, index)
                               : aclmdlGetInputSizeByIndex(model.desc, index);
        Require(dtype == Dtype(spec.dtype) && bytes == spec.bytes() &&
                    dims.dimCount == spec.shape.size(),
                "OM dtype/bytes/rank differs from chunk plan");
        for (std::size_t d = 0; d < dims.dimCount; ++d)
          Require(dims.dims[d] == spec.shape[d],
                  "OM shape differs from chunk plan");
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
        static_cast<void>(Propose(token));
    }
    return token;
  }
  std::vector<std::int64_t> Propose(std::int64_t anchor) {
    Healthy();
    Require(!pending && feature_rows > 0 && draft_cursor == feature_start &&
                feature_start + feature_rows == cursor,
            "Draft context cursor is inconsistent");
    Scalar("start_position", static_cast<std::int64_t>(feature_start));
    Scalar("valid_rows", static_cast<std::int64_t>(feature_rows));
    Scalar("anchor", anchor);
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
    // All layers finished both GDR passes before this atomic host publication.
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
  int device_id;
  bool initialized = false, device_set = false, invalid = true;
  aclrtContext context = nullptr;
  aclrtStream stream = nullptr;
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
std::vector<std::int64_t> AclChunkExecutor::Propose(std::int64_t anchor) {
  return impl_->Propose(anchor);
}
std::vector<std::int64_t> AclChunkExecutor::Verify(
    const std::vector<std::int64_t>& ids) {
  return impl_->Verify(ids);
}
void AclChunkExecutor::Commit(std::size_t rows) { impl_->Commit(rows); }
std::int64_t AclChunkExecutor::Decode(std::int64_t anchor) {
  return impl_->Decode(anchor);
}
std::size_t AclChunkExecutor::graph_calls() const noexcept {
  return impl_->calls;
}
const std::map<std::string, std::vector<double>>& AclChunkExecutor::stage_ms()
    const {
  return impl_->timings;
}
}  // namespace qwen35::dflash
