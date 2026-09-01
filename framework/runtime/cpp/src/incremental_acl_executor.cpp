#include "qwen35_dflash/incremental_acl_executor.hpp"

#include <acl/acl.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qwen35::dflash {
namespace {

constexpr const char* kDynamicTensorName = "ascend_mbatch_shape_data";
constexpr std::size_t kBufferAlignment = 64;

void Check(aclError code, const std::string& operation) {
  if (code != ACL_SUCCESS) {
    std::ostringstream message;
    message << operation << " failed with ACL error " << code;
    throw std::runtime_error(message.str());
  }
}

std::size_t Align(std::size_t value, std::size_t alignment) {
  if (alignment == 0 || value > std::numeric_limits<std::size_t>::max() -
                                    (alignment - 1)) {
    throw std::overflow_error("device arena alignment overflow");
  }
  return (value + alignment - 1) / alignment * alignment;
}

struct TensorSpec {
  std::string name;
  aclDataType dtype = ACL_DT_UNDEFINED;
  std::vector<std::int64_t> shape;
  std::size_t bytes = 0;
};

TensorSpec ReadTensorSpec(
    const aclmdlDesc* description,
    std::size_t index,
    bool input) {
  aclmdlIODims dimensions{};
  Check(
      input ? aclmdlGetInputDims(description, index, &dimensions)
            : aclmdlGetOutputDims(description, index, &dimensions),
      input ? "aclmdlGetInputDims" : "aclmdlGetOutputDims");
  if (dimensions.dimCount == 0 || dimensions.dimCount > 128) {
    throw std::runtime_error("OM tensor has an invalid dimension count");
  }
  TensorSpec result;
  result.name = dimensions.name;
  result.shape.reserve(dimensions.dimCount);
  for (std::size_t dimension = 0; dimension < dimensions.dimCount; ++dimension) {
    if (dimensions.dims[dimension] == 0 || dimensions.dims[dimension] < -1) {
      throw std::runtime_error("OM tensor has an invalid dimension");
    }
    result.shape.push_back(dimensions.dims[dimension]);
  }
  result.dtype = input
      ? aclmdlGetInputDataType(description, index)
      : aclmdlGetOutputDataType(description, index);
  result.bytes = input
      ? aclmdlGetInputSizeByIndex(description, index)
      : aclmdlGetOutputSizeByIndex(description, index);
  if (result.dtype == ACL_DT_UNDEFINED || result.bytes == 0) {
    throw std::runtime_error("OM tensor has an invalid dtype or byte size");
  }
  return result;
}

bool SameTensor(const TensorSpec& first, const TensorSpec& second) {
  return first.dtype == second.dtype && first.shape == second.shape &&
         first.bytes == second.bytes;
}

void RequireSameTensor(
    const TensorSpec& first,
    const TensorSpec& second,
    const char* description) {
  if (!SameTensor(first, second)) {
    throw std::runtime_error(std::string(description) + " tensor ABI differs");
  }
}

void RequireTensor(
    const TensorSpec& spec,
    aclDataType dtype,
    const std::vector<std::int64_t>& shape,
    const char* description) {
  if (spec.dtype != dtype || spec.shape != shape) {
    throw std::runtime_error(std::string(description) + " tensor ABI differs");
  }
}

struct ModelSession {
  std::string role;
  std::filesystem::path path;
  std::uint32_t id = 0;
  bool loaded = false;
  aclmdlDesc* description = nullptr;
  std::vector<TensorSpec> inputs;
  std::vector<TensorSpec> outputs;
  std::size_t work_bytes = 0;
  std::size_t weight_bytes = 0;
  std::size_t dynamic_input_index = std::numeric_limits<std::size_t>::max();
  std::vector<std::size_t> public_input_indices;
  std::vector<aclmdlIODims> dynamic_gears;

  void Query(
      const std::filesystem::path& model_path,
      std::string model_role,
      const IncrementalModelProgress& progress) {
    role = std::move(model_role);
    path = model_path;
    if (!std::filesystem::is_regular_file(path)) {
      throw std::invalid_argument(role + " OM path is not a regular file");
    }
    if (progress) {
      progress(role.c_str(), "query-start", 0, 0);
    }
    Check(
        aclmdlQuerySize(path.c_str(), &work_bytes, &weight_bytes),
        role + ": aclmdlQuerySize");
    if (progress) {
      progress(role.c_str(), "query-done", work_bytes, weight_bytes);
    }
  }

  void LoadWithMemory(
      void* shared_work,
      std::size_t shared_work_bytes,
      void* weight,
      std::size_t allocated_weight_bytes,
      bool require_dynamic_gears,
      const IncrementalModelProgress& progress) {
    if ((work_bytes != 0 && shared_work == nullptr) ||
        shared_work_bytes < work_bytes ||
        (weight_bytes != 0 && weight == nullptr) ||
        allocated_weight_bytes < weight_bytes) {
      throw std::runtime_error(role + ": explicit model memory is too small");
    }
    if (progress) {
      progress(role.c_str(), "load-start", work_bytes, weight_bytes);
    }
    Check(
        aclmdlLoadFromFileWithMem(
            path.c_str(),
            &id,
            shared_work,
            shared_work_bytes,
            weight,
            allocated_weight_bytes),
        role + ": aclmdlLoadFromFileWithMem");
    loaded = true;
    description = aclmdlCreateDesc();
    if (description == nullptr) {
      throw std::runtime_error(role + ": aclmdlCreateDesc returned null");
    }
    Check(aclmdlGetDesc(description, id), role + ": aclmdlGetDesc");
    const std::size_t input_count = aclmdlGetNumInputs(description);
    const std::size_t output_count = aclmdlGetNumOutputs(description);
    inputs.reserve(input_count);
    outputs.reserve(output_count);
    for (std::size_t index = 0; index < input_count; ++index) {
      inputs.push_back(ReadTensorSpec(description, index, true));
    }
    for (std::size_t index = 0; index < output_count; ++index) {
      outputs.push_back(ReadTensorSpec(description, index, false));
    }

    if (require_dynamic_gears) {
      std::size_t index = 0;
      Check(
          aclmdlGetInputIndexByName(
              description, kDynamicTensorName, &index),
          role + ": aclmdlGetInputIndexByName(dynamic dims)");
      if (index >= inputs.size()) {
        throw std::runtime_error(role + ": dynamic input index is invalid");
      }
      dynamic_input_index = index;
      std::size_t gear_count = 0;
      Check(
          aclmdlGetInputDynamicGearCount(
              description,
              std::numeric_limits<std::size_t>::max(),
              &gear_count),
          role + ": aclmdlGetInputDynamicGearCount");
      if (gear_count == 0) {
        throw std::runtime_error(
            role + ": OM has no dynamic-dimension profiles");
      }
      dynamic_gears.resize(gear_count);
      Check(
          aclmdlGetInputDynamicDims(
              description,
              std::numeric_limits<std::size_t>::max(),
              dynamic_gears.data(),
              dynamic_gears.size()),
          role + ": aclmdlGetInputDynamicDims");
    }
    for (std::size_t index = 0; index < inputs.size(); ++index) {
      if (index != dynamic_input_index) {
        public_input_indices.push_back(index);
      }
    }
    if (progress) {
      progress(role.c_str(), "load-done", work_bytes, weight_bytes);
    }
  }

  const TensorSpec& PublicInput(std::size_t index) const {
    if (index >= public_input_indices.size()) {
      throw std::out_of_range(role + ": public input index is invalid");
    }
    return inputs[public_input_indices[index]];
  }

  void Release() noexcept {
    if (description != nullptr) {
      static_cast<void>(aclmdlDestroyDesc(description));
      description = nullptr;
    }
    if (loaded) {
      static_cast<void>(aclmdlUnload(id));
      loaded = false;
    }
  }
};

struct BufferView {
  void* data = nullptr;
  std::size_t bytes = 0;
};

struct DeviceAllocation {
  void* data = nullptr;
  std::size_t bytes = 0;

  void Allocate(std::size_t requested) {
    if (requested == 0 || data != nullptr) {
      throw std::logic_error("invalid device allocation request");
    }
    bytes = requested;
    Check(
        aclrtMalloc(&data, bytes, ACL_MEM_MALLOC_NORMAL_ONLY),
        "aclrtMalloc");
  }

  BufferView View(std::size_t requested = 0) const {
    const std::size_t size = requested == 0 ? bytes : requested;
    if (data == nullptr || size > bytes) {
      throw std::out_of_range("device buffer view exceeds its allocation");
    }
    return BufferView{data, size};
  }

  void Release() noexcept {
    if (data != nullptr) {
      static_cast<void>(aclrtFree(data));
      data = nullptr;
    }
    bytes = 0;
  }
};

struct MirrorBuffer {
  void* host = nullptr;
  DeviceAllocation device;
  std::size_t bytes = 0;

  void Allocate(std::size_t requested) {
    if (requested == 0 || host != nullptr) {
      throw std::logic_error("invalid mirrored allocation request");
    }
    bytes = requested;
    Check(aclrtMallocHost(&host, bytes), "aclrtMallocHost");
    try {
      device.Allocate(bytes);
    } catch (...) {
      static_cast<void>(aclrtFreeHost(host));
      host = nullptr;
      bytes = 0;
      throw;
    }
    std::memset(host, 0, bytes);
  }

  BufferView View(std::size_t requested = 0) const {
    return device.View(requested);
  }

  void Release() noexcept {
    device.Release();
    if (host != nullptr) {
      static_cast<void>(aclrtFreeHost(host));
      host = nullptr;
    }
    bytes = 0;
  }
};

struct StateArena {
  DeviceAllocation allocation;
  std::vector<BufferView> tensors;

  void Allocate(const std::vector<TensorSpec>& specs) {
    std::vector<std::size_t> offsets;
    offsets.reserve(specs.size());
    std::size_t total = 0;
    for (const auto& spec : specs) {
      total = Align(total, kBufferAlignment);
      offsets.push_back(total);
      if (spec.bytes > std::numeric_limits<std::size_t>::max() - total) {
        throw std::overflow_error("state arena size overflow");
      }
      total += spec.bytes;
    }
    allocation.Allocate(Align(total, kBufferAlignment));
    tensors.reserve(specs.size());
    for (std::size_t index = 0; index < specs.size(); ++index) {
      auto* pointer = static_cast<std::byte*>(allocation.data) + offsets[index];
      tensors.push_back(BufferView{pointer, specs[index].bytes});
    }
  }

  void Release() noexcept {
    tensors.clear();
    allocation.Release();
  }
};

struct DatasetPlan {
  aclmdlDataset* input = nullptr;
  aclmdlDataset* output = nullptr;
  std::vector<aclDataBuffer*> input_buffers;
  std::vector<aclDataBuffer*> output_buffers;

  static void Add(
      aclmdlDataset* dataset,
      const BufferView& view,
      std::vector<aclDataBuffer*>* owned,
      const char* description) {
    if (view.data == nullptr || view.bytes == 0) {
      throw std::runtime_error(std::string(description) + " binding is empty");
    }
    aclDataBuffer* buffer = aclCreateDataBuffer(view.data, view.bytes);
    if (buffer == nullptr) {
      throw std::runtime_error(
          std::string(description) + ": aclCreateDataBuffer returned null");
    }
    owned->push_back(buffer);
    Check(
        aclmdlAddDatasetBuffer(dataset, buffer),
        std::string(description) + ": aclmdlAddDatasetBuffer");
  }

  void Build(
      const ModelSession& session,
      const std::vector<BufferView>& public_inputs,
      const std::vector<BufferView>& outputs,
      const BufferView& dynamic_control = {}) {
    if (public_inputs.size() != session.public_input_indices.size() ||
        outputs.size() != session.outputs.size()) {
      throw std::runtime_error(session.role + ": dataset binding count differs");
    }
    input = aclmdlCreateDataset();
    output = aclmdlCreateDataset();
    if (input == nullptr || output == nullptr) {
      throw std::runtime_error(session.role + ": aclmdlCreateDataset failed");
    }
    std::size_t public_index = 0;
    for (std::size_t index = 0; index < session.inputs.size(); ++index) {
      if (index == session.dynamic_input_index) {
        Add(input, dynamic_control, &input_buffers, "dynamic input");
      } else {
        Add(
            input,
            public_inputs[public_index++],
            &input_buffers,
            "model input");
      }
    }
    for (const auto& view : outputs) {
      Add(output, view, &output_buffers, "model output");
    }
  }

  void Release() noexcept {
    for (auto iterator = output_buffers.rbegin();
         iterator != output_buffers.rend();
         ++iterator) {
      static_cast<void>(aclDestroyDataBuffer(*iterator));
    }
    for (auto iterator = input_buffers.rbegin();
         iterator != input_buffers.rend();
         ++iterator) {
      static_cast<void>(aclDestroyDataBuffer(*iterator));
    }
    output_buffers.clear();
    input_buffers.clear();
    if (output != nullptr) {
      static_cast<void>(aclmdlDestroyDataset(output));
      output = nullptr;
    }
    if (input != nullptr) {
      static_cast<void>(aclmdlDestroyDataset(input));
      input = nullptr;
    }
  }
};

template <typename Value>
Value ReadAt(const MirrorBuffer& buffer, std::size_t offset) {
  if (offset > buffer.bytes || sizeof(Value) > buffer.bytes - offset) {
    throw std::out_of_range("compact result read exceeds its host buffer");
  }
  Value value{};
  std::memcpy(&value, static_cast<const std::byte*>(buffer.host) + offset,
              sizeof(value));
  return value;
}

}  // namespace

class AclIncrementalExecutor::Impl {
 public:
  Impl(
      const IncrementalOmPaths& paths,
      int device_id,
      const IncrementalModelProgress& progress)
      : device_id_(device_id) {
    if (device_id < 0) {
      throw std::invalid_argument("device ID must be non-negative");
    }
    try {
      Check(aclInit(nullptr), "aclInit");
      initialized_ = true;
      Check(aclrtSetDevice(device_id_), "aclrtSetDevice");
      device_set_ = true;
      Check(aclrtCreateContext(&context_, device_id_), "aclrtCreateContext");
      Check(aclrtSetCurrentContext(context_), "aclrtSetCurrentContext");
      Check(aclrtCreateStream(&stream_), "aclrtCreateStream");

      prefill_.Query(paths.target_prefill, "target-prefill", progress);
      decode_.Query(paths.target_decode1, "target-decode1", progress);
      draft_.Query(paths.draft_propose, "draft-propose", progress);
      verify_.Query(
          paths.target_verify_commit, "target-verify-commit", progress);
      memory_ = {
          {prefill_.role, prefill_.work_bytes, prefill_.weight_bytes},
          {decode_.role, decode_.work_bytes, decode_.weight_bytes},
          {draft_.role, draft_.work_bytes, draft_.weight_bytes},
          {verify_.role, verify_.work_bytes, verify_.weight_bytes},
      };
      const std::size_t shared_work_bytes = std::max(
          {prefill_.work_bytes,
           decode_.work_bytes,
           draft_.work_bytes,
           verify_.work_bytes});
      if (shared_work_bytes != 0) {
        shared_model_work_.Allocate(shared_work_bytes);
      }
      const std::array<std::size_t, 4> weight_bytes{
          prefill_.weight_bytes,
          decode_.weight_bytes,
          draft_.weight_bytes,
          verify_.weight_bytes,
      };
      for (std::size_t index = 0; index < model_weights_.size(); ++index) {
        if (weight_bytes[index] != 0) {
          model_weights_[index].Allocate(weight_bytes[index]);
        }
      }
      const auto load = [this, &progress](
                            ModelSession& session,
                            DeviceAllocation& weight,
                            bool dynamic) {
        session.LoadWithMemory(
            shared_model_work_.data,
            shared_model_work_.bytes,
            weight.data,
            weight.bytes,
            dynamic,
            progress);
      };
      load(prefill_, model_weights_[0], false);
      load(decode_, model_weights_[1], false);
      load(draft_, model_weights_[2], true);
      load(verify_, model_weights_[3], false);
      ValidateAbi();
      AllocateBuffers();
      BuildPlans();
    } catch (...) {
      Cleanup();
      throw;
    }
  }

  ~Impl() { Cleanup(); }

  std::size_t sequence_length() const noexcept { return sequence_length_; }
  std::size_t prefill_width() const noexcept { return prefill_width_; }
  std::size_t proposal_width() const noexcept { return proposal_width_; }
  std::size_t eos_table_width() const noexcept { return eos_table_width_; }
  const std::vector<IncrementalModelMemory>& model_memory() const noexcept {
    return memory_;
  }
  const IncrementalAclExecutionStats& execution_stats() const noexcept {
    return stats_;
  }

  void Reset(
      std::int64_t pad_token_id,
      const std::vector<std::int64_t>& eos_token_ids) {
    if (pad_token_id < 0) {
      throw std::invalid_argument("pad token ID must be non-negative");
    }
    if (eos_token_ids.size() > eos_table_width_) {
      throw std::invalid_argument("EOS set exceeds the fixed OM table width");
    }
    if (std::any_of(
            eos_token_ids.begin(), eos_token_ids.end(),
            [](std::int64_t value) { return value < 0; })) {
      throw std::invalid_argument("EOS set contains a negative token ID");
    }
    pad_token_id_ = pad_token_id;
    target_state_index_ = 0;
    draft_state_index_ = 0;
    proposal_ready_ = false;
    prepared_proposal_count_ = 0;
    feature_source_ = FeatureSource::kNone;

    if (!eos_uploaded_ || eos_token_ids != uploaded_eos_ids_) {
      auto* eos_values = static_cast<std::int64_t*>(eos_ids_.host);
      std::fill_n(eos_values, eos_table_width_, 0);
      std::copy(eos_token_ids.begin(), eos_token_ids.end(), eos_values);
      *static_cast<std::int32_t*>(eos_count_.host) =
          static_cast<std::int32_t>(eos_token_ids.size());
      pending_eos_ids_ = eos_token_ids;
      eos_upload_pending_ = true;
    } else {
      eos_upload_pending_ = false;
    }
    ++stats_.state_resets;
    reset_pending_ = true;
    reset_ = true;
  }

  StatefulStep PrefillChunk(
      const std::vector<std::int64_t>& token_ids,
      bool prepare_draft,
      std::size_t logical_proposal_count) {
    RequireReset();
    if (token_ids.empty() || token_ids.size() > prefill_width_) {
      throw std::invalid_argument("prefill chunk is outside the fixed 64-row gear");
    }
    if (std::any_of(token_ids.begin(), token_ids.end(), [](std::int64_t value) {
          return value < 0;
        })) {
      throw std::invalid_argument("prefill chunk contains a negative token ID");
    }
    if (prepare_draft) {
      RequireProposalCount(logical_proposal_count);
    } else if (logical_proposal_count != 0) {
      throw std::invalid_argument("proposal count supplied without Draft execution");
    }
    ApplyPendingReset();
    if (prepare_draft) {
      SetProposalCount(logical_proposal_count);
    }

    auto* values = static_cast<std::int64_t*>(prefill_ids_.host);
    std::fill_n(values, prefill_width_, pad_token_id_);
    std::copy(token_ids.begin(), token_ids.end(), values);
    *static_cast<std::int16_t*>(effective_length_.host) =
        static_cast<std::int16_t>(token_ids.size());
    Upload(prefill_ids_, prefill_ids_.bytes);
    Upload(effective_length_, effective_length_.bytes);

    Execute(prefill_, prefill_plans_[target_state_index_]);
    ++stats_.target_prefill_executions;
    target_state_index_ = 1 - target_state_index_;
    feature_source_ = FeatureSource::kPrefill;
    std::size_t executions = 1;
    if (prepare_draft) {
      ExecuteDraft(FeatureSource::kPrefill, prefill_width_);
      proposal_ready_ = true;
      prepared_proposal_count_ = logical_proposal_count;
      ++executions;
    } else {
      proposal_ready_ = false;
    }
    DownloadCompact(false);
    Synchronize();
    return ReadCompact(false, executions);
  }

  StatefulStep DecodeOne(std::int64_t input_token_id) {
    RequireReset();
    RequirePrefilled();
    if (input_token_id < 0) {
      throw std::invalid_argument("decode input token ID must be non-negative");
    }
    *static_cast<std::int64_t*>(decode_id_.host) = input_token_id;
    Upload(decode_id_, decode_id_.bytes);
    Execute(decode_, decode_plans_[target_state_index_]);
    ++stats_.target_decode1_executions;
    target_state_index_ = 1 - target_state_index_;
    proposal_ready_ = false;
    feature_source_ = FeatureSource::kNone;
    DownloadCompact(false);
    Synchronize();
    return ReadCompact(false, 1);
  }

  StatefulStep SpeculativeStep(std::size_t logical_proposal_count) {
    RequireReset();
    RequirePrefilled();
    RequireProposalCount(logical_proposal_count);
    std::size_t executions = 1;
    if (proposal_ready_) {
      if (prepared_proposal_count_ != logical_proposal_count) {
        throw std::runtime_error(
            "logical proposal count changed after Draft preparation");
      }
    } else {
      if (feature_source_ != FeatureSource::kVerify) {
        throw std::runtime_error(
            "no committed Target feature carrier is available for Draft");
      }
      SetProposalCount(logical_proposal_count);
      ExecuteDraft(FeatureSource::kVerify, verify_width_);
      ++executions;
    }
    Execute(verify_, verify_plans_[target_state_index_]);
    ++stats_.target_verify_commit_executions;
    target_state_index_ = 1 - target_state_index_;
    proposal_ready_ = false;
    feature_source_ = FeatureSource::kVerify;
    DownloadCompact(true);
    Synchronize();
    return ReadCompact(true, executions);
  }

 private:
  enum class FeatureSource : std::size_t {
    kPrefill = 0,
    kVerify = 1,
    kNone = 2,
  };

  static std::vector<TensorSpec> SelectSpecs(
      const std::vector<TensorSpec>& values,
      const std::vector<std::size_t>& indices) {
    std::vector<TensorSpec> result;
    result.reserve(indices.size());
    for (const std::size_t index : indices) {
      result.push_back(values.at(index));
    }
    return result;
  }

  static void RequireStateSet(
      const std::vector<TensorSpec>& expected,
      const std::vector<TensorSpec>& actual,
      const char* description) {
    if (expected.size() != actual.size()) {
      throw std::runtime_error(std::string(description) + " state count differs");
    }
    for (std::size_t index = 0; index < expected.size(); ++index) {
      RequireSameTensor(expected[index], actual[index], description);
    }
  }

  void ValidateAbi() {
    if (prefill_.public_input_indices.size() != 9 ||
        prefill_.outputs.size() != 10 ||
        decode_.public_input_indices.size() != 8 ||
        decode_.outputs.size() != 8 ||
        draft_.public_input_indices.size() != 8 ||
        draft_.outputs.size() != 4 ||
        verify_.public_input_indices.size() != 9 ||
        verify_.outputs.size() != 13) {
      throw std::runtime_error("incremental OM binding counts differ from v2 ABI");
    }
    const auto& prefill_ids = prefill_.PublicInput(0);
    if (prefill_ids.dtype != ACL_INT64 || prefill_ids.shape.size() != 2 ||
        prefill_ids.shape[0] != 1 || prefill_ids.shape[1] != 64) {
      throw std::runtime_error("target-prefill input_ids must be INT64[1,64]");
    }
    prefill_width_ = 64;
    RequireTensor(
        prefill_.PublicInput(1), ACL_INT16, {1}, "prefill effective_length");
    if (prefill_.PublicInput(2).dtype != ACL_INT64 ||
        prefill_.PublicInput(2).shape.size() != 1 ||
        prefill_.PublicInput(2).shape[0] <= 0) {
      throw std::runtime_error("prefill EOS table ABI differs");
    }
    eos_table_width_ =
        static_cast<std::size_t>(prefill_.PublicInput(2).shape[0]);
    RequireTensor(
        prefill_.PublicInput(3), ACL_INT32, {1}, "prefill eos_token_count");

    RequireTensor(decode_.PublicInput(0), ACL_INT64, {1, 1}, "decode input_ids");
    RequireSameTensor(
        prefill_.PublicInput(2), decode_.PublicInput(1), "decode EOS table");
    RequireSameTensor(
        prefill_.PublicInput(3), decode_.PublicInput(2), "decode EOS count");

    const auto& verify_ids = verify_.PublicInput(0);
    if (verify_ids.dtype != ACL_INT64 || verify_ids.shape.size() != 2 ||
        verify_ids.shape[0] != 1 || verify_ids.shape[1] != 16) {
      throw std::runtime_error("verify input IDs must be INT64[1,16]");
    }
    verify_width_ = 16;
    proposal_width_ = verify_width_ - 1;
    RequireTensor(
        verify_.PublicInput(1), ACL_INT32, {1}, "verify proposal count");
    RequireSameTensor(
        prefill_.PublicInput(2), verify_.PublicInput(2), "verify EOS table");
    RequireSameTensor(
        prefill_.PublicInput(3), verify_.PublicInput(3), "verify EOS count");

    target_state_specs_ = SelectSpecs(
        prefill_.inputs,
        {prefill_.public_input_indices[4], prefill_.public_input_indices[5],
         prefill_.public_input_indices[6], prefill_.public_input_indices[7],
         prefill_.public_input_indices[8]});
    RequireStateSet(
        target_state_specs_,
        SelectSpecs(
            decode_.inputs,
            {decode_.public_input_indices[3], decode_.public_input_indices[4],
             decode_.public_input_indices[5], decode_.public_input_indices[6],
             decode_.public_input_indices[7]}),
        "decode Target");
    RequireStateSet(
        target_state_specs_,
        SelectSpecs(
            verify_.inputs,
            {verify_.public_input_indices[4], verify_.public_input_indices[5],
             verify_.public_input_indices[6], verify_.public_input_indices[7],
             verify_.public_input_indices[8]}),
        "verify Target");
    RequireStateSet(
        target_state_specs_,
        SelectSpecs(prefill_.outputs, {5, 6, 7, 8, 9}),
        "prefill Target outputs");
    RequireStateSet(
        target_state_specs_,
        SelectSpecs(decode_.outputs, {3, 4, 5, 6, 7}),
        "decode Target outputs");
    RequireStateSet(
        target_state_specs_,
        SelectSpecs(verify_.outputs, {5, 6, 11, 12, 9}),
        "verify Target outputs");

    draft_state_specs_ = SelectSpecs(
        draft_.inputs,
        {draft_.public_input_indices[5], draft_.public_input_indices[6],
         draft_.public_input_indices[7]});
    RequireStateSet(
        draft_state_specs_,
        SelectSpecs(draft_.outputs, {1, 2, 3}),
        "Draft outputs");
    const auto& draft_key = draft_state_specs_[0];
    if (draft_key.dtype != ACL_FLOAT16 || draft_key.shape.size() != 5 ||
        draft_key.shape[0] != 6 || draft_key.shape[1] != 1 ||
        draft_key.shape[3] <= 0) {
      throw std::runtime_error("Draft key cache ABI differs");
    }
    sequence_length_ = static_cast<std::size_t>(draft_key.shape[3]);
    RequireSameTensor(draft_state_specs_[0], draft_state_specs_[1], "Draft K/V");
    RequireTensor(draft_state_specs_[2], ACL_INT64, {1}, "Draft cursor");

    const auto& target_key = target_state_specs_[2];
    if (target_key.dtype != ACL_FLOAT16 || target_key.shape.size() != 5 ||
        target_key.shape[0] != 8 || target_key.shape[1] <= 0 ||
        static_cast<std::size_t>(target_key.shape[1]) * 64 != sequence_length_) {
      throw std::runtime_error("Target paged KV capacity differs from Draft KV");
    }
    RequireSameTensor(target_state_specs_[2], target_state_specs_[3], "Target K/V");
    RequireTensor(target_state_specs_[4], ACL_INT64, {1}, "Target cursor");

    RequireTensor(prefill_.outputs[0], ACL_INT64, {1, 16}, "prefill committed IDs");
    RequireTensor(prefill_.outputs[1], ACL_INT32, {1}, "prefill commit count");
    RequireTensor(prefill_.outputs[2], ACL_BOOL, {1}, "prefill finished");
    if (prefill_.outputs[3].dtype != ACL_FLOAT16 ||
        prefill_.outputs[3].shape.size() != 3 ||
        prefill_.outputs[3].shape[0] != 1 ||
        prefill_.outputs[3].shape[1] != 64 ||
        prefill_.outputs[3].shape[2] <= 0) {
      throw std::runtime_error("prefill Target feature carrier ABI differs");
    }
    feature_width_ = static_cast<std::size_t>(prefill_.outputs[3].shape[2]);
    RequireTensor(prefill_.outputs[4], ACL_INT32, {1}, "prefill feature count");
    RequireSameTensor(prefill_.outputs[0], decode_.outputs[0], "decode committed IDs");
    RequireSameTensor(prefill_.outputs[1], decode_.outputs[1], "decode commit count");
    RequireSameTensor(prefill_.outputs[2], decode_.outputs[2], "decode finished");

    RequireSameTensor(prefill_.outputs[0], verify_.outputs[0], "verify committed IDs");
    for (std::size_t index = 1; index <= 4; ++index) {
      RequireTensor(
          verify_.outputs[index], ACL_INT32, {1}, "verify compact counter");
    }
    RequireTensor(verify_.outputs[10], ACL_BOOL, {1}, "verify finished");
    if (verify_.outputs[7].dtype != ACL_FLOAT16 ||
        verify_.outputs[7].shape !=
            std::vector<std::int64_t>{1, 16, static_cast<std::int64_t>(feature_width_)}) {
      throw std::runtime_error("verify Target feature carrier ABI differs");
    }
    RequireSameTensor(prefill_.outputs[4], verify_.outputs[8], "feature count");

    const auto& draft_feature = draft_.PublicInput(0);
    if (draft_feature.dtype != ACL_FLOAT16 || draft_feature.shape.size() != 3 ||
        draft_feature.shape[0] != 1 || draft_feature.shape[1] != -1 ||
        draft_feature.shape[2] != static_cast<std::int64_t>(feature_width_)) {
      throw std::runtime_error(
          "Draft feature input must be FP16[1,-1,feature_width]");
    }
    RequireSameTensor(prefill_.outputs[4], draft_.PublicInput(1), "Draft feature count");
    RequireSameTensor(prefill_.outputs[0], draft_.PublicInput(2), "Draft previous IDs");
    RequireSameTensor(prefill_.outputs[1], draft_.PublicInput(3), "Draft previous count");
    RequireSameTensor(verify_.PublicInput(1), draft_.PublicInput(4), "Draft proposal count");
    RequireSameTensor(verify_.PublicInput(0), draft_.outputs[0], "Draft verify IDs");
    ResolveDraftGears();
  }

  std::vector<std::int64_t> FlattenDraftShape(std::size_t feature_rows) const {
    std::vector<std::int64_t> result;
    for (std::size_t public_index = 0;
         public_index < draft_.public_input_indices.size();
         ++public_index) {
      const auto& spec = draft_.PublicInput(public_index);
      for (std::size_t dimension = 0; dimension < spec.shape.size(); ++dimension) {
        std::int64_t value = spec.shape[dimension];
        if (value == -1) {
          if (public_index != 0 || dimension != 1) {
            throw std::runtime_error(
                "Draft OM has an unexpected dynamic dimension");
          }
          value = static_cast<std::int64_t>(feature_rows);
        }
        result.push_back(value);
      }
    }
    return result;
  }

  void ResolveDraftGears() {
    const auto find = [this](std::size_t rows) -> aclmdlIODims {
      const auto expected = FlattenDraftShape(rows);
      for (const auto& gear : draft_.dynamic_gears) {
        if (gear.dimCount != expected.size()) {
          continue;
        }
        bool matches = true;
        for (std::size_t index = 0; index < expected.size(); ++index) {
          matches = matches && gear.dims[index] == expected[index];
        }
        if (matches) {
          return gear;
        }
      }
      throw std::runtime_error(
          "Draft OM is missing required dynamic feature gear N=" +
          std::to_string(rows));
    };
    draft_gear_verify_ = find(verify_width_);
    draft_gear_prefill_ = find(prefill_width_);
  }

  void AllocateBuffers() {
    target_states_[0].Allocate(target_state_specs_);
    target_states_[1].Allocate(target_state_specs_);
    draft_states_[0].Allocate(draft_state_specs_);
    draft_states_[1].Allocate(draft_state_specs_);
    stats_.state_device_bytes =
        target_states_[0].allocation.bytes +
        target_states_[1].allocation.bytes +
        draft_states_[0].allocation.bytes +
        draft_states_[1].allocation.bytes;

    prefill_ids_.Allocate(prefill_.PublicInput(0).bytes);
    effective_length_.Allocate(prefill_.PublicInput(1).bytes);
    eos_ids_.Allocate(prefill_.PublicInput(2).bytes);
    eos_count_.Allocate(prefill_.PublicInput(3).bytes);
    decode_id_.Allocate(decode_.PublicInput(0).bytes);
    proposal_count_.Allocate(verify_.PublicInput(1).bytes);

    compact_token_offset_ = 0;
    compact_commit_offset_ = Align(prefill_.outputs[0].bytes, 4);
    compact_finished_offset_ =
        Align(compact_commit_offset_ + prefill_.outputs[1].bytes, 4);
    compact_drafted_offset_ =
        Align(compact_finished_offset_ + prefill_.outputs[2].bytes, 4);
    compact_accepted_offset_ =
        Align(compact_drafted_offset_ + verify_.outputs[2].bytes, 4);
    compact_rejected_offset_ =
        Align(compact_accepted_offset_ + verify_.outputs[3].bytes, 4);
    compact_ordinary_bytes_ =
        compact_finished_offset_ + prefill_.outputs[2].bytes;
    compact_verify_bytes_ =
        compact_rejected_offset_ + verify_.outputs[4].bytes;
    compact_.Allocate(Align(compact_verify_bytes_, kBufferAlignment));

    const std::size_t feature_allocation = std::max(
        {prefill_.outputs[3].bytes,
         verify_.outputs[7].bytes,
         draft_.PublicInput(0).bytes});
    prefill_features_.Allocate(feature_allocation);
    verify_features_.Allocate(feature_allocation);
    committed_input_count_.Allocate(prefill_.outputs[4].bytes);
    verify_ids_.Allocate(verify_.PublicInput(0).bytes);
    for (auto& control : dynamic_controls_) {
      control.Allocate(draft_.inputs.at(draft_.dynamic_input_index).bytes);
    }

    stats_.carrier_device_bytes =
        prefill_ids_.device.bytes + effective_length_.device.bytes +
        eos_ids_.device.bytes + eos_count_.device.bytes +
        decode_id_.device.bytes + proposal_count_.device.bytes +
        compact_.device.bytes + prefill_features_.bytes +
        verify_features_.bytes + committed_input_count_.bytes +
        verify_ids_.bytes + dynamic_controls_[0].bytes +
        dynamic_controls_[1].bytes;
  }

  BufferView CompactView(std::size_t offset, std::size_t bytes) const {
    if (offset > compact_.device.bytes ||
        bytes > compact_.device.bytes - offset) {
      throw std::out_of_range("compact device binding exceeds its arena");
    }
    return BufferView{
        static_cast<std::byte*>(compact_.device.data) + offset,
        bytes,
    };
  }

  std::vector<BufferView> TargetInputs(
      const std::vector<BufferView>& prefix,
      std::size_t state_index) const {
    std::vector<BufferView> result = prefix;
    result.insert(
        result.end(),
        target_states_[state_index].tensors.begin(),
        target_states_[state_index].tensors.end());
    return result;
  }

  void BuildPlans() {
    for (std::size_t current = 0; current < 2; ++current) {
      const std::size_t next = 1 - current;
      prefill_plans_[current].Build(
          prefill_,
          TargetInputs(
              {prefill_ids_.View(), effective_length_.View(), eos_ids_.View(),
               eos_count_.View()},
              current),
          {CompactView(compact_token_offset_, prefill_.outputs[0].bytes),
           CompactView(compact_commit_offset_, prefill_.outputs[1].bytes),
           CompactView(compact_finished_offset_, prefill_.outputs[2].bytes),
           prefill_features_.View(prefill_.outputs[3].bytes),
           committed_input_count_.View(),
           target_states_[next].tensors[0],
           target_states_[next].tensors[1],
           target_states_[next].tensors[2],
           target_states_[next].tensors[3],
           target_states_[next].tensors[4]});
      decode_plans_[current].Build(
          decode_,
          TargetInputs(
              {decode_id_.View(), eos_ids_.View(), eos_count_.View()},
              current),
          {CompactView(compact_token_offset_, decode_.outputs[0].bytes),
           CompactView(compact_commit_offset_, decode_.outputs[1].bytes),
           CompactView(compact_finished_offset_, decode_.outputs[2].bytes),
           target_states_[next].tensors[0],
           target_states_[next].tensors[1],
           target_states_[next].tensors[2],
           target_states_[next].tensors[3],
           target_states_[next].tensors[4]});
      verify_plans_[current].Build(
          verify_,
          TargetInputs(
              {verify_ids_.View(), proposal_count_.View(), eos_ids_.View(),
               eos_count_.View()},
              current),
          {CompactView(compact_token_offset_, verify_.outputs[0].bytes),
           CompactView(compact_commit_offset_, verify_.outputs[1].bytes),
           CompactView(compact_drafted_offset_, verify_.outputs[2].bytes),
           CompactView(compact_accepted_offset_, verify_.outputs[3].bytes),
           CompactView(compact_rejected_offset_, verify_.outputs[4].bytes),
           target_states_[next].tensors[0],
           target_states_[next].tensors[1],
           verify_features_.View(verify_.outputs[7].bytes),
           committed_input_count_.View(),
           target_states_[next].tensors[4],
           CompactView(compact_finished_offset_, verify_.outputs[10].bytes),
           target_states_[next].tensors[2],
           target_states_[next].tensors[3]});

      for (std::size_t source = 0; source < 2; ++source) {
        const BufferView feature = source == 0
            ? prefill_features_.View(draft_.PublicInput(0).bytes)
            : verify_features_.View(draft_.PublicInput(0).bytes);
        draft_plans_[source][current].Build(
            draft_,
            {feature,
             committed_input_count_.View(),
             CompactView(compact_token_offset_, draft_.PublicInput(2).bytes),
             CompactView(compact_commit_offset_, draft_.PublicInput(3).bytes),
             proposal_count_.View(),
             draft_states_[current].tensors[0],
             draft_states_[current].tensors[1],
             draft_states_[current].tensors[2]},
            {verify_ids_.View(),
             draft_states_[next].tensors[0],
             draft_states_[next].tensors[1],
             draft_states_[next].tensors[2]},
            dynamic_controls_[source].View());
        const aclmdlIODims& gear = source == 0
            ? draft_gear_prefill_
            : draft_gear_verify_;
        Check(
            aclmdlSetInputDynamicDims(
                draft_.id,
                draft_plans_[source][current].input,
                draft_.dynamic_input_index,
                &gear),
            "draft-propose: aclmdlSetInputDynamicDims(prebind)");
      }
    }
  }

  void RequireReset() const {
    if (!reset_) {
      throw std::logic_error("incremental executor must be reset before use");
    }
  }

  void RequirePrefilled() const {
    if (reset_pending_) {
      throw std::logic_error(
          "incremental executor requires prefill after reset");
    }
  }

  void ApplyPendingReset() {
    if (!reset_pending_) {
      return;
    }
    Check(
        aclrtMemsetAsync(
            target_states_[0].allocation.data,
            target_states_[0].allocation.bytes,
            0,
            target_states_[0].allocation.bytes,
            stream_),
        "aclrtMemsetAsync(target state)");
    ++stats_.state_memset_operations;
    stats_.state_memset_bytes += target_states_[0].allocation.bytes;
    Check(
        aclrtMemsetAsync(
            draft_states_[0].allocation.data,
            draft_states_[0].allocation.bytes,
            0,
            draft_states_[0].allocation.bytes,
            stream_),
        "aclrtMemsetAsync(draft state)");
    ++stats_.state_memset_operations;
    stats_.state_memset_bytes += draft_states_[0].allocation.bytes;
    if (eos_upload_pending_) {
      Upload(eos_ids_, eos_ids_.bytes);
      Upload(eos_count_, eos_count_.bytes);
      uploaded_eos_ids_ = pending_eos_ids_;
      eos_uploaded_ = true;
      eos_upload_pending_ = false;
    }
    reset_pending_ = false;
  }

  void RequireProposalCount(std::size_t value) const {
    if (value == 0 || value > proposal_width_) {
      throw std::invalid_argument("logical proposal count is outside 1..15");
    }
  }

  void SetProposalCount(std::size_t value) {
    RequireProposalCount(value);
    if (proposal_value_valid_ && proposal_value_ == value) {
      return;
    }
    *static_cast<std::int32_t*>(proposal_count_.host) =
        static_cast<std::int32_t>(value);
    Upload(proposal_count_, proposal_count_.bytes);
    proposal_value_ = value;
    proposal_value_valid_ = true;
  }

  void ExecuteDraft(FeatureSource source, std::size_t feature_rows) {
    if (source == FeatureSource::kNone ||
        (feature_rows != prefill_width_ && feature_rows != verify_width_)) {
      throw std::logic_error("invalid Draft feature source/gear");
    }
    DatasetPlan& plan =
        draft_plans_[static_cast<std::size_t>(source)][draft_state_index_];
    Execute(draft_, plan);
    ++stats_.draft_propose_executions;
    draft_state_index_ = 1 - draft_state_index_;
  }

  void Execute(const ModelSession& session, DatasetPlan& plan) {
    Check(
        aclmdlExecuteAsync(session.id, plan.input, plan.output, stream_),
        session.role + ": aclmdlExecuteAsync");
  }

  void Upload(const MirrorBuffer& buffer, std::size_t bytes) {
    if (bytes == 0 || bytes > buffer.bytes) {
      throw std::out_of_range("host-to-device copy exceeds mirrored buffer");
    }
    Check(
        aclrtMemcpyAsync(
            buffer.device.data,
            buffer.device.bytes,
            buffer.host,
            bytes,
            ACL_MEMCPY_HOST_TO_DEVICE,
            stream_),
        "aclrtMemcpyAsync(host-to-device)");
    ++stats_.host_to_device_operations;
    stats_.host_to_device_bytes += bytes;
  }

  void DownloadCompact(bool verify) {
    const std::size_t bytes =
        verify ? compact_verify_bytes_ : compact_ordinary_bytes_;
    Check(
        aclrtMemcpyAsync(
            compact_.host,
            compact_.bytes,
            compact_.device.data,
            bytes,
            ACL_MEMCPY_DEVICE_TO_HOST,
            stream_),
        "aclrtMemcpyAsync(compact device-to-host)");
    ++stats_.device_to_host_operations;
    stats_.device_to_host_bytes += bytes;
  }

  void Synchronize() {
    Check(aclrtSynchronizeStream(stream_), "aclrtSynchronizeStream");
    ++stats_.stream_synchronizations;
  }

  StatefulStep ReadCompact(bool verify, std::size_t model_executions) const {
    const std::int32_t commit_count =
        ReadAt<std::int32_t>(compact_, compact_commit_offset_);
    if (commit_count <= 0 ||
        static_cast<std::size_t>(commit_count) > verify_width_) {
      throw std::runtime_error("Target graph returned an invalid commit count");
    }
    std::vector<std::int64_t> tokens(
        static_cast<std::size_t>(commit_count));
    std::memcpy(
        tokens.data(),
        static_cast<const std::byte*>(compact_.host) + compact_token_offset_,
        tokens.size() * sizeof(std::int64_t));
    const bool finished =
        ReadAt<std::uint8_t>(compact_, compact_finished_offset_) != 0;
    StatefulStep result;
    result.token_ids = std::move(tokens);
    result.model_executions = model_executions;
    result.finished = finished;
    if (verify) {
      const std::int32_t drafted =
          ReadAt<std::int32_t>(compact_, compact_drafted_offset_);
      const std::int32_t accepted =
          ReadAt<std::int32_t>(compact_, compact_accepted_offset_);
      const std::int32_t rejected =
          ReadAt<std::int32_t>(compact_, compact_rejected_offset_);
      if (drafted <= 0 || accepted < 0 || rejected < 0) {
        throw std::runtime_error("verify graph returned negative/zero counters");
      }
      result.drafted_tokens = static_cast<std::size_t>(drafted);
      result.accepted_draft_tokens = static_cast<std::size_t>(accepted);
      result.rejected_draft_tokens = static_cast<std::size_t>(rejected);
    }
    return result;
  }

  void Cleanup() noexcept {
    for (auto& by_source : draft_plans_) {
      for (auto& plan : by_source) {
        plan.Release();
      }
    }
    for (auto& plan : verify_plans_) {
      plan.Release();
    }
    for (auto& plan : decode_plans_) {
      plan.Release();
    }
    for (auto& plan : prefill_plans_) {
      plan.Release();
    }

    for (auto& control : dynamic_controls_) {
      control.Release();
    }
    verify_ids_.Release();
    committed_input_count_.Release();
    verify_features_.Release();
    prefill_features_.Release();
    compact_.Release();
    proposal_count_.Release();
    decode_id_.Release();
    eos_count_.Release();
    eos_ids_.Release();
    effective_length_.Release();
    prefill_ids_.Release();
    for (auto& state : draft_states_) {
      state.Release();
    }
    for (auto& state : target_states_) {
      state.Release();
    }

    verify_.Release();
    draft_.Release();
    decode_.Release();
    prefill_.Release();
    for (auto& weight : model_weights_) {
      weight.Release();
    }
    shared_model_work_.Release();
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
  bool reset_ = false;
  aclrtContext context_ = nullptr;
  aclrtStream stream_ = nullptr;

  ModelSession prefill_;
  ModelSession decode_;
  ModelSession draft_;
  ModelSession verify_;
  std::vector<IncrementalModelMemory> memory_;
  DeviceAllocation shared_model_work_;
  std::array<DeviceAllocation, 4> model_weights_;

  std::size_t sequence_length_ = 0;
  std::size_t prefill_width_ = 0;
  std::size_t verify_width_ = 0;
  std::size_t proposal_width_ = 0;
  std::size_t eos_table_width_ = 0;
  std::size_t feature_width_ = 0;
  std::vector<TensorSpec> target_state_specs_;
  std::vector<TensorSpec> draft_state_specs_;
  aclmdlIODims draft_gear_prefill_{};
  aclmdlIODims draft_gear_verify_{};

  std::array<StateArena, 2> target_states_;
  std::array<StateArena, 2> draft_states_;
  MirrorBuffer prefill_ids_;
  MirrorBuffer effective_length_;
  MirrorBuffer eos_ids_;
  MirrorBuffer eos_count_;
  MirrorBuffer decode_id_;
  MirrorBuffer proposal_count_;
  MirrorBuffer compact_;
  DeviceAllocation prefill_features_;
  DeviceAllocation verify_features_;
  DeviceAllocation committed_input_count_;
  DeviceAllocation verify_ids_;
  std::array<DeviceAllocation, 2> dynamic_controls_;

  std::size_t compact_token_offset_ = 0;
  std::size_t compact_commit_offset_ = 0;
  std::size_t compact_finished_offset_ = 0;
  std::size_t compact_drafted_offset_ = 0;
  std::size_t compact_accepted_offset_ = 0;
  std::size_t compact_rejected_offset_ = 0;
  std::size_t compact_ordinary_bytes_ = 0;
  std::size_t compact_verify_bytes_ = 0;

  std::array<DatasetPlan, 2> prefill_plans_;
  std::array<DatasetPlan, 2> decode_plans_;
  std::array<DatasetPlan, 2> verify_plans_;
  std::array<std::array<DatasetPlan, 2>, 2> draft_plans_;

  std::size_t target_state_index_ = 0;
  std::size_t draft_state_index_ = 0;
  bool proposal_ready_ = false;
  std::size_t prepared_proposal_count_ = 0;
  FeatureSource feature_source_ = FeatureSource::kNone;
  bool proposal_value_valid_ = false;
  std::size_t proposal_value_ = 0;
  std::int64_t pad_token_id_ = 0;
  bool eos_uploaded_ = false;
  bool eos_upload_pending_ = false;
  bool reset_pending_ = false;
  std::vector<std::int64_t> uploaded_eos_ids_;
  std::vector<std::int64_t> pending_eos_ids_;
  IncrementalAclExecutionStats stats_;
};

AclIncrementalExecutor::AclIncrementalExecutor(
    IncrementalOmPaths model_paths,
    int device_id,
    IncrementalModelProgress progress)
    : impl_(std::make_unique<Impl>(model_paths, device_id, progress)) {}

AclIncrementalExecutor::~AclIncrementalExecutor() = default;
AclIncrementalExecutor::AclIncrementalExecutor(
    AclIncrementalExecutor&&) noexcept = default;
AclIncrementalExecutor& AclIncrementalExecutor::operator=(
    AclIncrementalExecutor&&) noexcept = default;

std::size_t AclIncrementalExecutor::sequence_length() const noexcept {
  return impl_->sequence_length();
}

std::size_t AclIncrementalExecutor::prefill_width() const noexcept {
  return impl_->prefill_width();
}

std::size_t AclIncrementalExecutor::proposal_width() const noexcept {
  return impl_->proposal_width();
}

std::size_t AclIncrementalExecutor::eos_table_width() const noexcept {
  return impl_->eos_table_width();
}

void AclIncrementalExecutor::Reset(
    std::int64_t pad_token_id,
    const std::vector<std::int64_t>& eos_token_ids) {
  impl_->Reset(pad_token_id, eos_token_ids);
}

StatefulStep AclIncrementalExecutor::PrefillChunk(
    const std::vector<std::int64_t>& token_ids,
    bool prepare_draft,
    std::size_t logical_proposal_count) {
  return impl_->PrefillChunk(
      token_ids, prepare_draft, logical_proposal_count);
}

StatefulStep AclIncrementalExecutor::DecodeOne(
    std::int64_t input_token_id) {
  return impl_->DecodeOne(input_token_id);
}

StatefulStep AclIncrementalExecutor::SpeculativeStep(
    std::size_t logical_proposal_count) {
  return impl_->SpeculativeStep(logical_proposal_count);
}

const std::vector<IncrementalModelMemory>&
AclIncrementalExecutor::model_memory() const noexcept {
  return impl_->model_memory();
}

const IncrementalAclExecutionStats&
AclIncrementalExecutor::execution_stats() const noexcept {
  return impl_->execution_stats();
}

}  // namespace qwen35::dflash
