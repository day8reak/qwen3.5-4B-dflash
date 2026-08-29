#include <acl/acl.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

struct aclDataBuffer {
  void* data;
  std::size_t size;
};

struct aclmdlDataset {
  std::vector<aclDataBuffer*> buffers;
};

struct aclmdlDesc {};

namespace {

constexpr std::size_t kSequenceLength = 32;
constexpr std::size_t kDraftWidth = 15;

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

aclError aclrtMallocHost(void** host_ptr, std::size_t size) {
  if (host_ptr == nullptr || size == 0) {
    return 1;
  }
  *host_ptr = std::malloc(size);
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
  return *device_ptr == nullptr ? 1 : ACL_SUCCESS;
}

aclError aclrtFree(void* device_ptr) {
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
  std::memcpy(destination, source, count);
  return ACL_SUCCESS;
}

aclError aclmdlLoadFromFile(const char*, std::uint32_t* model_id) {
  if (model_id == nullptr) {
    return 1;
  }
  *model_id = 1;
  return ACL_SUCCESS;
}

aclError aclmdlUnload(std::uint32_t) { return ACL_SUCCESS; }

aclmdlDesc* aclmdlCreateDesc() { return new (std::nothrow) aclmdlDesc(); }

aclError aclmdlDestroyDesc(aclmdlDesc* description) {
  delete description;
  return ACL_SUCCESS;
}

aclError aclmdlGetDesc(aclmdlDesc*, std::uint32_t) { return ACL_SUCCESS; }
std::size_t aclmdlGetNumInputs(const aclmdlDesc*) { return 2; }
std::size_t aclmdlGetNumOutputs(const aclmdlDesc*) { return 2; }

aclError aclmdlGetInputDims(const aclmdlDesc*, std::size_t index, aclmdlIODims* dims) {
  return index < 2 ? SetDims(dims, kSequenceLength) : 1;
}

aclError aclmdlGetOutputDims(
    const aclmdlDesc*, std::size_t index, aclmdlIODims* dims) {
  if (index == 0) {
    return SetDims(dims, kSequenceLength);
  }
  return index == 1 ? SetDims(dims, kDraftWidth) : 1;
}

aclDataType aclmdlGetInputDataType(const aclmdlDesc*, std::size_t index) {
  return index < 2 ? ACL_INT64 : ACL_DT_UNDEFINED;
}

aclDataType aclmdlGetOutputDataType(const aclmdlDesc*, std::size_t index) {
  return index < 2 ? ACL_INT64 : ACL_DT_UNDEFINED;
}

std::size_t aclmdlGetInputSizeByIndex(const aclmdlDesc*, std::size_t index) {
  return index < 2 ? kSequenceLength * sizeof(std::int64_t) : 0;
}

std::size_t aclmdlGetOutputSizeByIndex(const aclmdlDesc*, std::size_t index) {
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
    std::uint32_t,
    const aclmdlDataset* input,
    aclmdlDataset* output,
    aclrtStream) {
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
