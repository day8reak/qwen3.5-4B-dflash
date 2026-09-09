#include <acl/acl.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <new>
#include <fstream>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <vector>

struct aclDataBuffer {
  void* data;
  std::size_t size;
};

struct aclmdlDataset {
  std::vector<aclDataBuffer*> buffers;
};

struct aclmdlDesc { std::uint32_t id = 0; };

namespace {

constexpr std::size_t kSequenceLength = 32;
constexpr std::size_t kDraftWidth = 15;

struct FixtureTensor {
  std::string name;
  aclDataType dtype = ACL_INT64;
  std::vector<std::int64_t> shape;
  std::size_t bytes = 0;
};
struct FixtureModel {
  std::string role;
  std::vector<FixtureTensor> inputs, outputs;
};
std::map<std::uint32_t, FixtureModel> fixtures;
std::uint32_t next_id = 1;
std::map<void*, std::size_t> device_allocations;
std::set<void*> discard_allocations;

bool TouchesDiscard(const void* pointer, std::size_t bytes) {
  const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
  for (auto* discard : discard_allocations) {
    const auto base = reinterpret_cast<std::uintptr_t>(discard);
    if (begin < base + device_allocations.at(discard) && base < begin + bytes)
      return true;
  }
  return false;
}

aclError ExecuteChunk(const FixtureModel& model, const aclmdlDataset* input, aclmdlDataset* output) {
  if (input->buffers.size() != model.inputs.size() || output->buffers.size() != model.outputs.size()) return 21;
  const char* failure = std::getenv("QWEN35_FAKE_FAIL_GRAPH");
  if (failure && model.role == failure) return 22;
  std::map<std::string, aclDataBuffer*> in, out;
  for (std::size_t i = 0; i < model.inputs.size(); ++i) in[model.inputs[i].name] = input->buffers[i];
  for (std::size_t i = 0; i < model.outputs.size(); ++i) out[model.outputs[i].name] = output->buffers[i];
  for (const auto& item : in)
    if (TouchesDiscard(item.second->data, item.second->size)) return 29;
  for (const auto& item : out) {
    if (item.first.rfind("verify_discard_", 0) != 0) continue;
    auto* buffer = item.second;
    if (model.role != "target_verify" || !buffer->data ||
        !device_allocations.count(buffer->data) ||
        device_allocations.at(buffer->data) != buffer->size) return 30;
    for (const auto& other : in)
      if (other.second->data == buffer->data) return 31;
    for (const auto& other : out)
      if (other.first != item.first && other.second->data == buffer->data) return 32;
    discard_allocations.insert(buffer->data);
    // A poison value distinct from every committed cursor detects accidental
    // publication on the next verify, decode, or reset/prefill.
    std::fill_n(static_cast<float*>(buffer->data), buffer->size / sizeof(float), -1024.5f);
    if (const auto* path = std::getenv("QWEN35_FAKE_MEMORY_LOG")) {
      std::ofstream log(path, std::ios::app);
      log << "[\"discard\",\"" << item.first << "\"," << buffer->size << ','
          << reinterpret_cast<std::uintptr_t>(buffer->data) << "]\n";
    }
  }
  const auto start = *static_cast<std::int64_t*>(in.at("start_position")->data);
  const auto valid = *static_cast<std::int16_t*>(in.at("valid_rows")->data);
  if (const auto* path = std::getenv("QWEN35_FAKE_EVENT_LOG")) {
    const auto* active = std::getenv("TEST_ACTIVE");
    std::ofstream log(path, std::ios::app);
    log << "[\"" << model.role << "\"," << (active && std::filesystem::exists(active) ? "true" : "false")
        << ',' << start << ',' << valid << "]\n";
  }
  if (start < 0 || valid <= 0 || valid > 64) return 23;
  std::size_t committed = static_cast<std::size_t>(valid);
  if (model.role == "draft") {
    if (*static_cast<std::uint16_t*>(in.at("features")->data) != start) return 24;
    const auto anchor = *static_cast<std::int64_t*>(in.at("anchor")->data);
    const auto proposal_count = *static_cast<std::int16_t*>(in.at("proposal_count")->data);
    if (proposal_count < 1 || proposal_count > 15) return 28;
    if (const auto* path = std::getenv("QWEN35_FAKE_PROPOSAL_LOG")) {
      std::ofstream log(path, std::ios::app);
      log << start << ' ' << valid << ' ' << proposal_count << '\n';
    }
    auto* proposals = static_cast<std::int64_t*>(out.at("draft_top1")->data);
    const char* requested = std::getenv("QWEN35_FAKE_ACCEPT");
    const int accepted = requested ? std::atoi(requested) : 15;
    for (int i = 0; i < 15; ++i)
      proposals[i] = i < proposal_count ? (anchor + i + 1 + (i == accepted ? 7 : 0)) % 64 : 0;
  } else {
    auto* ids = static_cast<std::int64_t*>(in.at("input_ids")->data);
    auto* predictions = static_cast<std::int64_t*>(out.at("target_top1")->data);
    if (model.role == "target_verify") {
      if (valid > 16) return 25;
      for (int i = 0; i < 16; ++i) predictions[i] = (ids[i] + 1) % 64;
      std::size_t accepted = 0;
      while (accepted + 1 < static_cast<std::size_t>(valid) && ids[accepted + 1] == predictions[accepted]) ++accepted;
      committed = accepted + 1;
      *static_cast<std::int64_t*>(out.at("accepted_count")->data) = std::getenv("QWEN35_FAKE_BAD_ACCEPT") ? 0 : static_cast<std::int64_t>(accepted);
    } else {
      predictions[0] = (ids[valid - 1] + 1) % 64;
    }
    if (out.count("features")) {
      std::memset(out.at("features")->data, 0, out.at("features")->size);
      *static_cast<std::uint16_t*>(out.at("features")->data) = static_cast<std::uint16_t>(start);
    }
  }
  for (const auto& item : in) {
    if (item.first.size() > 2 && (item.first[0] == 't' || item.first[0] == 'd') && item.first[1] >= '0' && item.first[1] <= '9') {
      auto* result = out.at(item.first);
      if (result->data == item.second->data || result->size != item.second->size) return 26;
      if (*static_cast<std::uint16_t*>(item.second->data) != start) return 27;
      std::memcpy(result->data, item.second->data, result->size);
      *static_cast<std::uint16_t*>(result->data) = static_cast<std::uint16_t>(start + committed);
    }
  }
  return ACL_SUCCESS;
}

aclError FixtureDims(const FixtureTensor& tensor, aclmdlIODims* dimensions) {
  if (!dimensions) return 1;
  std::memset(dimensions, 0, sizeof(*dimensions));
  std::strncpy(dimensions->name, tensor.name.c_str(), sizeof(dimensions->name) - 1);
  dimensions->dimCount = tensor.shape.size();
  std::copy(tensor.shape.begin(), tensor.shape.end(), dimensions->dims);
  return ACL_SUCCESS;
}

aclError SetDims(aclmdlIODims* dimensions, std::int64_t width) {
  if (dimensions == nullptr) {
    return 1;
  }
  std::memset(dimensions, 0, sizeof(*dimensions));
  dimensions->dimCount = 2;
  dimensions->dims[0] = 1;
  dimensions->dims[1] = width;
  return ACL_SUCCESS;
}

}  // namespace

extern "C" {

aclError aclInit(const char*) { return ACL_SUCCESS; }
aclError aclFinalize() { return ACL_SUCCESS; }
aclError aclrtSetDevice(int) { return ACL_SUCCESS; }
aclError aclrtResetDevice(int) { return ACL_SUCCESS; }

aclError aclrtCreateContext(aclrtContext* context, int) {
  if (context == nullptr) {
    return 1;
  }
  *context = new (std::nothrow) int(1);
  return *context == nullptr ? 1 : ACL_SUCCESS;
}

aclError aclrtDestroyContext(aclrtContext context) {
  delete static_cast<int*>(context);
  return ACL_SUCCESS;
}

aclError aclrtSetCurrentContext(aclrtContext) { return ACL_SUCCESS; }

aclError aclrtCreateStream(aclrtStream* stream) {
  if (stream == nullptr) {
    return 1;
  }
  *stream = new (std::nothrow) int(2);
  return *stream == nullptr ? 1 : ACL_SUCCESS;
}

aclError aclrtDestroyStream(aclrtStream stream) {
  delete static_cast<int*>(stream);
  return ACL_SUCCESS;
}

aclError aclrtSynchronizeStream(aclrtStream) { return ACL_SUCCESS; }
aclError aclrtMemsetAsync(void* ptr, std::size_t maximum, std::int32_t value, std::size_t count, aclrtStream) {
  if (!ptr || count > maximum) return 1;
  if (TouchesDiscard(ptr, count)) return 33;
  std::memset(ptr, value, count); return ACL_SUCCESS;
}
aclError aclUpdateDataBuffer(aclDataBuffer* buffer, void* ptr, std::size_t size) {
  if (!buffer || !ptr || !size) return 1;
  buffer->data = ptr; buffer->size = size; return ACL_SUCCESS;
}

aclError aclrtMallocHost(void** host_ptr, std::size_t size) {
  if (host_ptr == nullptr || size == 0) {
    return 1;
  }
  *host_ptr = std::malloc(size);
  if (const auto* path = std::getenv("QWEN35_FAKE_MEMORY_LOG")) {
    std::ofstream log(path, std::ios::app);
    log << "[\"host_alloc\"," << size << "]\n";
  }
  return *host_ptr == nullptr ? 1 : ACL_SUCCESS;
}

aclError aclrtFreeHost(void* host_ptr) {
  std::free(host_ptr);
  return ACL_SUCCESS;
}

aclError aclrtMalloc(void** device_ptr, std::size_t size, aclrtMemMallocPolicy) {
  if (device_ptr == nullptr || size == 0) {
    return 1;
  }
  *device_ptr = std::malloc(size);
  if (*device_ptr) device_allocations[*device_ptr] = size;
  return *device_ptr == nullptr ? 1 : ACL_SUCCESS;
}

aclError aclrtFree(void* device_ptr) {
  discard_allocations.erase(device_ptr);
  device_allocations.erase(device_ptr);
  std::free(device_ptr);
  return ACL_SUCCESS;
}

aclError aclrtMemcpyAsync(
    void* destination,
    std::size_t destination_max,
    const void* source,
    std::size_t count,
    aclrtMemcpyKind,
    aclrtStream) {
  if (destination == nullptr || source == nullptr || count > destination_max) {
    return 1;
  }
  if (TouchesDiscard(source, count) || TouchesDiscard(destination, count)) return 34;
  std::memcpy(destination, source, count);
  return ACL_SUCCESS;
}

aclError aclmdlLoadFromFile(const char* path, std::uint32_t* model_id) {
  if (model_id == nullptr) {
    return 1;
  }
  *model_id = next_id++;
  FixtureModel model;
  std::ifstream file(path);
  std::string word;
  if (file >> word && word == "FAKE_CHUNK") {
    file >> model.role;
    while (file >> word && (word == "I" || word == "O")) {
      FixtureTensor tensor;
      std::string dtype;
      std::size_t rank;
      file >> tensor.name >> dtype >> rank;
      if (rank > 8) return 2;
      tensor.dtype = dtype == "float16" ? ACL_FLOAT16 : dtype == "int16" ? ACL_INT16 : dtype == "float32" ? ACL_FLOAT : ACL_INT64;
      tensor.bytes = tensor.dtype == ACL_INT64 ? 8 : tensor.dtype == ACL_FLOAT ? 4 : 2;
      tensor.shape.resize(rank);
      for (auto& dim : tensor.shape) { file >> dim; tensor.bytes *= static_cast<std::size_t>(dim); }
      (word == "I" ? model.inputs : model.outputs).push_back(tensor);
    }
  }
  if (model.role == "draft") {
    const char* fault = std::getenv("QWEN35_FAKE_CHUNK_IO_FAULT");
    const std::string kind = fault ? fault : "";
    auto& tensors = kind.find("output-") == 0 ? model.outputs : model.inputs;
    if (kind == "input-order") {
      std::swap(tensors.at(1), tensors.at(2));
    } else if (!kind.empty()) {
      auto& tensor = tensors.at(0);
      if (kind.find("-dtype") != std::string::npos) tensor.dtype = ACL_INT32;
      if (kind.find("-bytes") != std::string::npos) tensor.bytes += 32;
      if (kind.find("-rank") != std::string::npos) tensor.shape.insert(tensor.shape.begin(), 1);
      if (kind.find("-shape") != std::string::npos) tensor.shape.back() += 1;
      if (kind.find("-count") != std::string::npos) tensors.pop_back();
    }
  }
  fixtures[*model_id] = std::move(model);
  return ACL_SUCCESS;
}

aclError aclmdlUnload(std::uint32_t id) { fixtures.erase(id); return ACL_SUCCESS; }

aclmdlDesc* aclmdlCreateDesc() { return new (std::nothrow) aclmdlDesc(); }

aclError aclmdlDestroyDesc(aclmdlDesc* description) {
  delete description;
  return ACL_SUCCESS;
}

aclError aclmdlGetDesc(aclmdlDesc* desc, std::uint32_t id) { desc->id = id; return ACL_SUCCESS; }
std::size_t aclmdlGetNumInputs(const aclmdlDesc* desc) { return fixtures.at(desc->id).role.empty() ? 2 : fixtures.at(desc->id).inputs.size(); }
std::size_t aclmdlGetNumOutputs(const aclmdlDesc* desc) { return fixtures.at(desc->id).role.empty() ? 2 : fixtures.at(desc->id).outputs.size(); }

aclError aclmdlGetInputDims(const aclmdlDesc* desc, std::size_t index, aclmdlIODims* dims) {
  if (!fixtures.at(desc->id).role.empty()) return FixtureDims(fixtures.at(desc->id).inputs.at(index), dims);
  return index < 2 ? SetDims(dims, kSequenceLength) : 1;
}

aclError aclmdlGetOutputDims(
    const aclmdlDesc* desc, std::size_t index, aclmdlIODims* dims) {
  if (!fixtures.at(desc->id).role.empty()) return FixtureDims(fixtures.at(desc->id).outputs.at(index), dims);
  if (index == 0) {
    return SetDims(dims, kSequenceLength);
  }
  return index == 1 ? SetDims(dims, kDraftWidth) : 1;
}

aclDataType aclmdlGetInputDataType(const aclmdlDesc* desc, std::size_t index) {
  if (!fixtures.at(desc->id).role.empty()) return fixtures.at(desc->id).inputs.at(index).dtype;
  return index < 2 ? ACL_INT64 : ACL_DT_UNDEFINED;
}

aclDataType aclmdlGetOutputDataType(const aclmdlDesc* desc, std::size_t index) {
  if (!fixtures.at(desc->id).role.empty()) return fixtures.at(desc->id).outputs.at(index).dtype;
  return index < 2 ? ACL_INT64 : ACL_DT_UNDEFINED;
}

std::size_t aclmdlGetInputSizeByIndex(aclmdlDesc* desc, std::size_t index) {
  if (!fixtures.at(desc->id).role.empty()) return fixtures.at(desc->id).inputs.at(index).bytes;
  return index < 2 ? kSequenceLength * sizeof(std::int64_t) : 0;
}

std::size_t aclmdlGetOutputSizeByIndex(aclmdlDesc* desc, std::size_t index) {
  if (!fixtures.at(desc->id).role.empty()) return fixtures.at(desc->id).outputs.at(index).bytes;
  if (index == 0) {
    return kSequenceLength * sizeof(std::int64_t);
  }
  return index == 1 ? kDraftWidth * sizeof(std::int64_t) : 0;
}

aclmdlDataset* aclmdlCreateDataset() {
  return new (std::nothrow) aclmdlDataset();
}

aclError aclmdlDestroyDataset(aclmdlDataset* dataset) {
  delete dataset;
  return ACL_SUCCESS;
}

aclDataBuffer* aclCreateDataBuffer(void* data, std::size_t size) {
  if (data == nullptr || size == 0) {
    return nullptr;
  }
  return new (std::nothrow) aclDataBuffer{data, size};
}

aclError aclDestroyDataBuffer(aclDataBuffer* buffer) {
  delete buffer;
  return ACL_SUCCESS;
}

aclError aclmdlAddDatasetBuffer(
    aclmdlDataset* dataset, aclDataBuffer* data_buffer) {
  if (dataset == nullptr || data_buffer == nullptr) {
    return 1;
  }
  dataset->buffers.push_back(data_buffer);
  return ACL_SUCCESS;
}

aclError aclmdlExecuteAsync(
    std::uint32_t id,
    const aclmdlDataset* input,
    aclmdlDataset* output,
    aclrtStream) {
  if (!fixtures.at(id).role.empty()) return ExecuteChunk(fixtures.at(id), input, output);
  if (input == nullptr || output == nullptr || input->buffers.size() != 2 ||
      output->buffers.size() != 2) {
    return 1;
  }
  const auto* ids = static_cast<const std::int64_t*>(input->buffers[0]->data);
  const auto* mask = static_cast<const std::int64_t*>(input->buffers[1]->data);
  auto* target = static_cast<std::int64_t*>(output->buffers[0]->data);
  auto* draft = static_cast<std::int64_t*>(output->buffers[1]->data);
  std::fill_n(target, kSequenceLength, 0);
  std::size_t prefix = 0;
  while (prefix < kSequenceLength && mask[prefix] == 1) {
    target[prefix] = ids[prefix] + 1;
    ++prefix;
  }
  if (prefix == 0) {
    return 1;
  }
  for (std::size_t index = 0; index < kDraftWidth; ++index) {
    draft[index] = ids[prefix - 1] + static_cast<std::int64_t>(index) + 1;
  }
  return ACL_SUCCESS;
}

}  // extern "C"
