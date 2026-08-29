#include "qwen35_dflash/acl_executor.hpp"

#include <acl/acl.h>

#include <algorithm>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qwen35::dflash {
namespace {

void Check(aclError code, const char* operation) {
  if (code != ACL_SUCCESS) {
    std::ostringstream message;
    message << operation << " failed with ACL error " << code;
    throw std::runtime_error(message.str());
  }
}

struct Buffer {
  void* host = nullptr;
  void* device = nullptr;
  std::size_t bytes = 0;
  aclDataBuffer* data = nullptr;
};

std::vector<std::int64_t> Shape(
    const aclmdlDesc* description,
    std::size_t index,
    bool input) {
  aclmdlIODims dimensions{};
  Check(
      input ? aclmdlGetInputDims(description, index, &dimensions)
            : aclmdlGetOutputDims(description, index, &dimensions),
      input ? "aclmdlGetInputDims" : "aclmdlGetOutputDims");
  std::vector<std::int64_t> result;
  result.reserve(dimensions.dimCount);
  for (std::size_t dimension = 0; dimension < dimensions.dimCount; ++dimension) {
    const std::int64_t value = dimensions.dims[dimension];
    if (value <= 0) {
      throw std::runtime_error("C++ runner requires a fully static OM ABI");
    }
    result.push_back(value);
  }
  return result;
}

void RequireInt64(
    const aclmdlDesc* description,
    std::size_t index,
    bool input) {
  const aclDataType type =
      input ? aclmdlGetInputDataType(description, index)
            : aclmdlGetOutputDataType(description, index);
  if (type != ACL_INT64) {
    throw std::runtime_error(
        std::string(input ? "input" : "output") +
        " tensor is not ACL_INT64");
  }
}

void RequireDenseInt64Size(
    const std::vector<std::int64_t>& shape,
    std::size_t bytes,
    const char* description) {
  std::size_t elements = 1;
  for (const std::int64_t dimension : shape) {
    const std::size_t value = static_cast<std::size_t>(dimension);
    if (elements > static_cast<std::size_t>(-1) / value) {
      throw std::overflow_error(std::string(description) + " shape overflows size_t");
    }
    elements *= value;
  }
  if (elements > static_cast<std::size_t>(-1) / sizeof(std::int64_t) ||
      elements * sizeof(std::int64_t) != bytes) {
    throw std::runtime_error(
        std::string(description) + " is not a dense INT64 tensor");
  }
}

Buffer AllocateBuffer(std::size_t bytes) {
  Buffer buffer;
  buffer.bytes = bytes;
  Check(aclrtMallocHost(&buffer.host, bytes), "aclrtMallocHost");
  try {
    Check(
        aclrtMalloc(&buffer.device, bytes, ACL_MEM_MALLOC_NORMAL_ONLY),
        "aclrtMalloc");
    buffer.data = aclCreateDataBuffer(buffer.device, bytes);
    if (buffer.data == nullptr) {
      throw std::runtime_error("aclCreateDataBuffer returned null");
    }
  } catch (...) {
    if (buffer.data != nullptr) {
      static_cast<void>(aclDestroyDataBuffer(buffer.data));
    }
    if (buffer.device != nullptr) {
      static_cast<void>(aclrtFree(buffer.device));
    }
    static_cast<void>(aclrtFreeHost(buffer.host));
    throw;
  }
  return buffer;
}

void ReleaseBuffer(Buffer* buffer) noexcept {
  if (buffer->data != nullptr) {
    static_cast<void>(aclDestroyDataBuffer(buffer->data));
    buffer->data = nullptr;
  }
  if (buffer->device != nullptr) {
    static_cast<void>(aclrtFree(buffer->device));
    buffer->device = nullptr;
  }
  if (buffer->host != nullptr) {
    static_cast<void>(aclrtFreeHost(buffer->host));
    buffer->host = nullptr;
  }
}

}  // namespace

class AclExecutor::Impl {
 public:
  Impl(const std::filesystem::path& model_path, int device_id)
      : device_id_(device_id) {
    if (device_id < 0) {
      throw std::invalid_argument("device ID must be non-negative");
    }
    if (!std::filesystem::is_regular_file(model_path)) {
      throw std::invalid_argument("OM path is not a regular file");
    }
    try {
      Check(aclInit(nullptr), "aclInit");
      initialized_ = true;
      Check(aclrtSetDevice(device_id_), "aclrtSetDevice");
      device_set_ = true;
      Check(aclrtCreateContext(&context_, device_id_), "aclrtCreateContext");
      Check(aclrtSetCurrentContext(context_), "aclrtSetCurrentContext");
      Check(aclrtCreateStream(&stream_), "aclrtCreateStream");
      Check(
          aclmdlLoadFromFile(model_path.c_str(), &model_id_),
          "aclmdlLoadFromFile");
      model_loaded_ = true;
      description_ = aclmdlCreateDesc();
      if (description_ == nullptr) {
        throw std::runtime_error("aclmdlCreateDesc returned null");
      }
      Check(aclmdlGetDesc(description_, model_id_), "aclmdlGetDesc");
      ValidateAndAllocate();
    } catch (...) {
      Cleanup();
      throw;
    }
  }

  ~Impl() { Cleanup(); }

  std::size_t sequence_length() const noexcept { return sequence_length_; }
  std::size_t draft_width() const noexcept { return draft_width_; }

  const GraphOutputs& Execute(
      const std::vector<std::int64_t>& prefix,
      std::int64_t pad_token_id) {
    if (prefix.empty() || prefix.size() > sequence_length_) {
      throw std::invalid_argument("committed prefix is outside the OM gear");
    }
    if (std::any_of(prefix.begin(), prefix.end(), [](std::int64_t token) {
          return token < 0;
        })) {
      throw std::invalid_argument("committed prefix contains a negative token ID");
    }
    auto* input_ids = static_cast<std::int64_t*>(inputs_[0].host);
    auto* attention_mask = static_cast<std::int64_t*>(inputs_[1].host);
    std::fill_n(input_ids, sequence_length_, pad_token_id);
    std::fill_n(attention_mask, sequence_length_, 0);
    std::copy(prefix.begin(), prefix.end(), input_ids);
    std::fill_n(attention_mask, prefix.size(), 1);

    for (const Buffer& input : inputs_) {
      Check(
          aclrtMemcpyAsync(
              input.device,
              input.bytes,
              input.host,
              input.bytes,
              ACL_MEMCPY_HOST_TO_DEVICE,
              stream_),
          "aclrtMemcpyAsync(host_to_device)");
    }
    Check(
        aclmdlExecuteAsync(model_id_, input_dataset_, output_dataset_, stream_),
        "aclmdlExecuteAsync");
    for (const Buffer& output : outputs_) {
      Check(
          aclrtMemcpyAsync(
              output.host,
              output.bytes,
              output.device,
              output.bytes,
              ACL_MEMCPY_DEVICE_TO_HOST,
              stream_),
          "aclrtMemcpyAsync(device_to_host)");
    }
    Check(aclrtSynchronizeStream(stream_), "aclrtSynchronizeStream");
    std::memcpy(
        graph_outputs_.target_top1.data(),
        outputs_[0].host,
        outputs_[0].bytes);
    std::memcpy(
        graph_outputs_.draft_top1.data(),
        outputs_[1].host,
        outputs_[1].bytes);
    return graph_outputs_;
  }

 private:
  void ValidateAndAllocate() {
    if (aclmdlGetNumInputs(description_) != 2 ||
        aclmdlGetNumOutputs(description_) != 2) {
      throw std::runtime_error("integrated OM must have exactly two inputs and outputs");
    }
    const auto ids_shape = Shape(description_, 0, true);
    const auto mask_shape = Shape(description_, 1, true);
    const auto target_shape = Shape(description_, 0, false);
    const auto draft_shape = Shape(description_, 1, false);
    if (ids_shape.size() != 2 || ids_shape[0] != 1 ||
        ids_shape != mask_shape || target_shape != ids_shape) {
      throw std::runtime_error(
          "integrated OM input/target shapes must be static [1,S]");
    }
    if (draft_shape.size() != 2 || draft_shape[0] != 1 ||
        draft_shape[1] <= 0) {
      throw std::runtime_error("integrated OM draft output must be [1,K]");
    }
    sequence_length_ = static_cast<std::size_t>(ids_shape[1]);
    draft_width_ = static_cast<std::size_t>(draft_shape[1]);
    for (std::size_t index = 0; index < 2; ++index) {
      RequireInt64(description_, index, true);
      RequireInt64(description_, index, false);
    }

    input_dataset_ = aclmdlCreateDataset();
    output_dataset_ = aclmdlCreateDataset();
    if (input_dataset_ == nullptr || output_dataset_ == nullptr) {
      throw std::runtime_error("aclmdlCreateDataset returned null");
    }
    for (std::size_t index = 0; index < 2; ++index) {
      const std::size_t bytes = aclmdlGetInputSizeByIndex(description_, index);
      RequireDenseInt64Size(
          index == 0 ? ids_shape : mask_shape, bytes, "OM input");
      inputs_.push_back(AllocateBuffer(bytes));
      Check(
          aclmdlAddDatasetBuffer(input_dataset_, inputs_.back().data),
          "aclmdlAddDatasetBuffer(input)");
    }
    for (std::size_t index = 0; index < 2; ++index) {
      const std::size_t bytes = aclmdlGetOutputSizeByIndex(description_, index);
      RequireDenseInt64Size(
          index == 0 ? target_shape : draft_shape, bytes, "OM output");
      outputs_.push_back(AllocateBuffer(bytes));
      Check(
          aclmdlAddDatasetBuffer(output_dataset_, outputs_.back().data),
          "aclmdlAddDatasetBuffer(output)");
    }
    graph_outputs_.target_top1.resize(sequence_length_);
    graph_outputs_.draft_top1.resize(draft_width_);
  }

  void Cleanup() noexcept {
    if (stream_ != nullptr) {
      static_cast<void>(aclrtSynchronizeStream(stream_));
    }
    for (auto iterator = outputs_.rbegin(); iterator != outputs_.rend(); ++iterator) {
      ReleaseBuffer(&*iterator);
    }
    for (auto iterator = inputs_.rbegin(); iterator != inputs_.rend(); ++iterator) {
      ReleaseBuffer(&*iterator);
    }
    outputs_.clear();
    inputs_.clear();
    if (output_dataset_ != nullptr) {
      static_cast<void>(aclmdlDestroyDataset(output_dataset_));
      output_dataset_ = nullptr;
    }
    if (input_dataset_ != nullptr) {
      static_cast<void>(aclmdlDestroyDataset(input_dataset_));
      input_dataset_ = nullptr;
    }
    if (description_ != nullptr) {
      static_cast<void>(aclmdlDestroyDesc(description_));
      description_ = nullptr;
    }
    if (model_loaded_) {
      static_cast<void>(aclmdlUnload(model_id_));
      model_loaded_ = false;
    }
    if (stream_ != nullptr) {
      static_cast<void>(aclrtDestroyStream(stream_));
      stream_ = nullptr;
    }
    if (context_ != nullptr) {
      static_cast<void>(aclrtDestroyContext(context_));
      context_ = nullptr;
    }
    if (device_set_) {
      static_cast<void>(aclrtResetDevice(device_id_));
      device_set_ = false;
    }
    if (initialized_) {
      static_cast<void>(aclFinalize());
      initialized_ = false;
    }
  }

  int device_id_ = 0;
  bool initialized_ = false;
  bool device_set_ = false;
  aclrtContext context_ = nullptr;
  aclrtStream stream_ = nullptr;
  std::uint32_t model_id_ = 0;
  bool model_loaded_ = false;
  aclmdlDesc* description_ = nullptr;
  aclmdlDataset* input_dataset_ = nullptr;
  aclmdlDataset* output_dataset_ = nullptr;
  std::vector<Buffer> inputs_;
  std::vector<Buffer> outputs_;
  std::size_t sequence_length_ = 0;
  std::size_t draft_width_ = 0;
  GraphOutputs graph_outputs_;
};

AclExecutor::AclExecutor(
    const std::filesystem::path& model_path,
    int device_id)
    : impl_(std::make_unique<Impl>(model_path, device_id)) {}

AclExecutor::~AclExecutor() = default;
AclExecutor::AclExecutor(AclExecutor&&) noexcept = default;
AclExecutor& AclExecutor::operator=(AclExecutor&&) noexcept = default;

std::size_t AclExecutor::sequence_length() const noexcept {
  return impl_->sequence_length();
}

std::size_t AclExecutor::draft_width() const noexcept {
  return impl_->draft_width();
}

const GraphOutputs& AclExecutor::Execute(
    const std::vector<std::int64_t>& committed_prefix,
    std::int64_t pad_token_id) {
  return impl_->Execute(committed_prefix, pad_token_id);
}

}  // namespace qwen35::dflash
