#include "qwen35_dflash/acl_executor.hpp"

#include <acl/acl.h>

#include <algorithm>
#include <cstddef>
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

struct ElementRange {
  std::size_t begin = 0;
  std::size_t end = 0;

  bool empty() const noexcept { return begin == end; }
  std::size_t size() const noexcept { return end - begin; }
};

ElementRange ChangedTokenRange(
    std::int64_t* values,
    std::size_t count,
    const std::vector<std::int64_t>& prefix,
    std::int64_t pad_token_id) {
  ElementRange changed{count, count};
  for (std::size_t index = 0; index < count; ++index) {
    const std::int64_t desired =
        index < prefix.size() ? prefix[index] : pad_token_id;
    if (values[index] != desired) {
      values[index] = desired;
      changed.begin = std::min(changed.begin, index);
      changed.end = index + 1;
    }
  }
  return changed;
}

ElementRange ChangedMaskRange(
    std::int64_t* values,
    std::size_t count,
    std::size_t prefix_length) {
  ElementRange changed{count, count};
  for (std::size_t index = 0; index < count; ++index) {
    const std::int64_t desired = index < prefix_length ? 1 : 0;
    if (values[index] != desired) {
      values[index] = desired;
      changed.begin = std::min(changed.begin, index);
      changed.end = index + 1;
    }
  }
  return changed;
}

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
          aclmdlQuerySize(
              model_path.c_str(),
              &model_work_bytes_,
              &model_weight_bytes_),
          "aclmdlQuerySize");
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
  std::size_t model_work_bytes() const noexcept { return model_work_bytes_; }
  std::size_t model_weight_bytes() const noexcept { return model_weight_bytes_; }
  const AclExecutionStats& execution_stats() const noexcept { return stats_; }

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
    ElementRange ids_changed;
    ElementRange mask_changed;
    if (!inputs_initialized_) {
      std::fill_n(input_ids, sequence_length_, pad_token_id);
      std::fill_n(attention_mask, sequence_length_, 0);
      std::copy(prefix.begin(), prefix.end(), input_ids);
      std::fill_n(attention_mask, prefix.size(), 1);
      ids_changed = ElementRange{0, sequence_length_};
      mask_changed = ElementRange{0, sequence_length_};
      inputs_initialized_ = true;
    } else {
      ids_changed = ChangedTokenRange(
          input_ids,
          sequence_length_,
          prefix,
          pad_token_id);
      mask_changed = ChangedMaskRange(
          attention_mask,
          sequence_length_,
          prefix.size());
    }
    stats_.full_host_to_device_bytes += inputs_[0].bytes + inputs_[1].bytes;
    QueueHostToDevice(inputs_[0], ids_changed);
    QueueHostToDevice(inputs_[1], mask_changed);

    Check(
        aclmdlExecuteAsync(model_id_, input_dataset_, output_dataset_, stream_),
        "aclmdlExecuteAsync");
    ++stats_.model_executions;

    // Every current scheduler read is in this suffix: prefill/proposal reads
    // only prefix[-1], while verify reads anchor plus at most draft_width_
    // proposal rows.  Keep the full logical vector ABI, but transfer only the
    // suffix that can be observed by the caller.
    const std::size_t maximum_target_rows = draft_width_ + 1;
    const std::size_t target_begin =
        prefix.size() > maximum_target_rows
            ? prefix.size() - maximum_target_rows
            : 0;
    const ElementRange target_range{target_begin, prefix.size()};
    stats_.full_device_to_host_bytes += outputs_[0].bytes + outputs_[1].bytes;
    QueueDeviceToHost(outputs_[0], target_range);
    QueueDeviceToHost(
        outputs_[1], ElementRange{0, draft_width_});
    stats_.target_elements_downloaded += target_range.size();
    stats_.maximum_target_elements_per_call = std::max(
        stats_.maximum_target_elements_per_call,
        target_range.size());

    Check(aclrtSynchronizeStream(stream_), "aclrtSynchronizeStream");
    ++stats_.stream_synchronizations;
    const std::size_t target_bytes =
        target_range.size() * sizeof(std::int64_t);
    std::memcpy(
        graph_outputs_.target_top1.data() + target_range.begin,
        static_cast<const std::int64_t*>(outputs_[0].host) + target_range.begin,
        target_bytes);
    std::memcpy(
        graph_outputs_.draft_top1.data(),
        outputs_[1].host,
        outputs_[1].bytes);
    return graph_outputs_;
  }

 private:
  void QueueHostToDevice(const Buffer& buffer, const ElementRange& range) {
    if (range.begin > range.end ||
        range.end > buffer.bytes / sizeof(std::int64_t)) {
      throw std::runtime_error("host-to-device range exceeds its INT64 buffer");
    }
    if (range.empty()) {
      ++stats_.host_to_device_copies_skipped;
      return;
    }
    const std::size_t offset = range.begin * sizeof(std::int64_t);
    const std::size_t bytes = range.size() * sizeof(std::int64_t);
    auto* destination = static_cast<std::byte*>(buffer.device) + offset;
    const auto* source = static_cast<const std::byte*>(buffer.host) + offset;
    Check(
        aclrtMemcpyAsync(
            destination,
            buffer.bytes - offset,
            source,
            bytes,
            ACL_MEMCPY_HOST_TO_DEVICE,
            stream_),
        "aclrtMemcpyAsync(host_to_device_range)");
    ++stats_.host_to_device_operations;
    stats_.host_to_device_bytes += bytes;
  }

  void QueueDeviceToHost(const Buffer& buffer, const ElementRange& range) {
    if (range.begin > range.end ||
        range.end > buffer.bytes / sizeof(std::int64_t)) {
      throw std::runtime_error("device-to-host range exceeds its INT64 buffer");
    }
    if (range.empty()) {
      throw std::runtime_error("device-to-host output range must not be empty");
    }
    const std::size_t offset = range.begin * sizeof(std::int64_t);
    const std::size_t bytes = range.size() * sizeof(std::int64_t);
    auto* destination = static_cast<std::byte*>(buffer.host) + offset;
    const auto* source = static_cast<const std::byte*>(buffer.device) + offset;
    Check(
        aclrtMemcpyAsync(
            destination,
            buffer.bytes - offset,
            source,
            bytes,
            ACL_MEMCPY_DEVICE_TO_HOST,
            stream_),
        "aclrtMemcpyAsync(device_to_host_range)");
    ++stats_.device_to_host_operations;
    stats_.device_to_host_bytes += bytes;
  }

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
  std::size_t model_work_bytes_ = 0;
  std::size_t model_weight_bytes_ = 0;
  bool inputs_initialized_ = false;
  AclExecutionStats stats_;
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

std::size_t AclExecutor::model_work_bytes() const noexcept {
  return impl_->model_work_bytes();
}

std::size_t AclExecutor::model_weight_bytes() const noexcept {
  return impl_->model_weight_bytes();
}

const AclExecutionStats& AclExecutor::execution_stats() const noexcept {
  return impl_->execution_stats();
}

const GraphOutputs& AclExecutor::Execute(
    const std::vector<std::int64_t>& committed_prefix,
    std::int64_t pad_token_id) {
  return impl_->Execute(committed_prefix, pad_token_id);
}

}  // namespace qwen35::dflash
