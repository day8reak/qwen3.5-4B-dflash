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
constexpr std::size_t kCompactSlotCount = 2;

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

std::size_t ReserveDeviceSegment(
    std::size_t* cursor,
    std::size_t payload_bytes) {
  if (cursor == nullptr || payload_bytes == 0) {
    throw std::logic_error("invalid device segment reservation");
  }
  const std::size_t offset = Align(*cursor, kBufferAlignment);
  const std::size_t padded_bytes = Align(payload_bytes, 32);
  if (padded_bytes > std::numeric_limits<std::size_t>::max() - 32) {
    throw std::overflow_error("device segment size overflow");
  }
  const std::size_t segment_bytes = padded_bytes + 32;
  if (segment_bytes > std::numeric_limits<std::size_t>::max() - offset) {
    throw std::overflow_error("device arena size overflow");
  }
  *cursor = offset + segment_bytes;
  return offset;
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

struct HostAllocation {
  void* data = nullptr;
  std::size_t bytes = 0;

  void Allocate(std::size_t requested) {
    if (requested == 0 || data != nullptr) {
      throw std::logic_error("invalid pinned host allocation request");
    }
    bytes = requested;
    Check(aclrtMallocHost(&data, bytes), "aclrtMallocHost(staging)");
    std::memset(data, 0, bytes);
  }

  void Release() noexcept {
    if (data != nullptr) {
      static_cast<void>(aclrtFreeHost(data));
      data = nullptr;
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
      offsets.push_back(ReserveDeviceSegment(&total, spec.bytes));
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

const char* IncrementalStateResetPolicyName(
    IncrementalStateResetPolicy policy) noexcept {
  switch (policy) {
    case IncrementalStateResetPolicy::kAsyncMemset:
      return "async-memset";
    case IncrementalStateResetPolicy::kImmutableZero:
      return "immutable-zero";
  }
  return "unknown";
}

const char* IncrementalDecodeCarrierPolicyName(
    IncrementalDecodeCarrierPolicy policy) noexcept {
  switch (policy) {
    case IncrementalDecodeCarrierPolicy::kOneTokenHostFallback:
      return "one-token-h2d";
    case IncrementalDecodeCarrierPolicy::kLastTokenDeviceCompact:
      return "last-token-d2d";
  }
  return "unknown";
}

const char* IncrementalDraftFeaturePolicyName(
    IncrementalDraftFeaturePolicy policy) noexcept {
  switch (policy) {
    case IncrementalDraftFeaturePolicy::kFixedVerifyWidth:
      return "fixed-16";
    case IncrementalDraftFeaturePolicy::kCommittedPrefix:
      return "committed-prefix";
  }
  return "unknown";
}

class AclIncrementalExecutor::Impl {
 public:
  Impl(
      const IncrementalOmPaths& paths,
      int device_id,
      const IncrementalModelProgress& progress,
      IncrementalStateResetPolicy state_reset_policy,
      IncrementalDecodeCarrierPolicy decode_carrier_policy,
      bool profile_model_executions,
      IncrementalDraftFeaturePolicy draft_feature_policy)
      : device_id_(device_id),
        unified_target_step_(paths.target_decode1.empty()),
        state_reset_policy_(state_reset_policy),
        decode_carrier_policy_(decode_carrier_policy),
        draft_feature_policy_(draft_feature_policy),
        profile_model_executions_(profile_model_executions) {
    if (device_id < 0) {
      throw std::invalid_argument("device ID must be non-negative");
    }
    if (state_reset_policy_ != IncrementalStateResetPolicy::kAsyncMemset &&
        state_reset_policy_ != IncrementalStateResetPolicy::kImmutableZero) {
      throw std::invalid_argument("unknown incremental state reset policy");
    }
    if (decode_carrier_policy_ !=
            IncrementalDecodeCarrierPolicy::kOneTokenHostFallback &&
        decode_carrier_policy_ !=
            IncrementalDecodeCarrierPolicy::kLastTokenDeviceCompact) {
      throw std::invalid_argument("unknown incremental decode carrier policy");
    }
    if (draft_feature_policy_ !=
            IncrementalDraftFeaturePolicy::kFixedVerifyWidth &&
        draft_feature_policy_ !=
            IncrementalDraftFeaturePolicy::kCommittedPrefix) {
      throw std::invalid_argument("unknown incremental Draft feature policy");
    }
    if (profile_model_executions_) {
      model_execution_trace_.reserve(4096);
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
      prefill_head_.Query(
          paths.target_prefill_head, "target-prefill-head", progress);
      if (!unified_target_step_) {
        decode_.Query(paths.target_decode1, "target-decode1", progress);
      }
      draft_.Query(paths.draft_propose, "draft-propose", progress);
      verify_.Query(
          paths.target_verify_commit, "target-verify-commit", progress);
      if (prefill_head_.weight_bytes >= prefill_.weight_bytes) {
        throw std::runtime_error(
            "target-prefill-head weightSize must be smaller than the "
            "head-free target-prefill body; refusing a duplicated Target");
      }
      const std::size_t shared_work_bytes = std::max(
          {prefill_.work_bytes,
           prefill_head_.work_bytes,
           unified_target_step_ ? 0 : decode_.work_bytes,
           draft_.work_bytes,
           verify_.work_bytes});
      if (shared_work_bytes != 0) {
        shared_model_work_.Allocate(shared_work_bytes);
      }
      const std::array<std::size_t, 5> weight_bytes{
          prefill_.weight_bytes,
          prefill_head_.weight_bytes,
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
      load(prefill_head_, model_weights_[1], false);
      if (!unified_target_step_) {
        load(decode_, model_weights_[2], false);
      }
      load(draft_, model_weights_[3], true);
      load(verify_, model_weights_[4], unified_target_step_);
      memory_.push_back(
          {prefill_.role,
           prefill_.id,
           prefill_.work_bytes,
           prefill_.weight_bytes});
      memory_.push_back(
          {prefill_head_.role,
           prefill_head_.id,
           prefill_head_.work_bytes,
           prefill_head_.weight_bytes});
      if (!unified_target_step_) {
        memory_.push_back(
            {decode_.role,
             decode_.id,
             decode_.work_bytes,
             decode_.weight_bytes});
      }
      memory_.push_back(
          {draft_.role, draft_.id, draft_.work_bytes, draft_.weight_bytes});
      memory_.push_back(
          {verify_.role,
           verify_.id,
           verify_.work_bytes,
           verify_.weight_bytes});
      ValidateAbi();
      AllocateBuffers();
      InitializeImmutableZeroState();
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
  const std::vector<IncrementalModelExecutionTrace>&
  model_execution_trace() const noexcept {
    return model_execution_trace_;
  }
  const IncrementalAclExecutionStats& execution_stats() const noexcept {
    return stats_;
  }
  IncrementalStateResetPolicy state_reset_policy() const noexcept {
    return state_reset_policy_;
  }
  IncrementalDecodeCarrierPolicy decode_carrier_policy() const noexcept {
    return decode_carrier_policy_;
  }
  IncrementalDraftFeaturePolicy draft_feature_policy() const noexcept {
    return draft_feature_policy_;
  }
  bool unified_target_step() const noexcept { return unified_target_step_; }

  void Reset(
      std::int64_t pad_token_id,
      const std::vector<std::int64_t>& eos_token_ids) {
    if (deferred_prefill_pending_ || stream_work_pending_) {
      throw std::logic_error(
          "cannot reset while deferred prefill work is pending");
    }
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
    proposal_count_upload_staging_index_ = 0;
    committed_feature_rows_ = 0;
    committed_feature_rows_valid_ = false;
    decode_carrier_valid_ = false;
    decode_carrier_row_ = 0;
    feature_source_ = FeatureSource::kNone;
    prefill_staging_index_ = 0;
    prefill_total_token_count_ = 0;
    draft_reset_pending_ = true;

    configured_eos_token_ids_ = eos_token_ids;
    eos_control_upload_pending_ =
        !device_eos_token_ids_valid_ ||
        device_eos_token_ids_ != configured_eos_token_ids_;
    ++stats_.state_resets;
    reset_pending_ = true;
    reset_ = true;
  }

  StatefulStep PrefillChunk(
      const std::vector<std::int64_t>& token_ids,
      bool prepare_draft,
      std::size_t logical_proposal_count) {
    return PrefillChunkImpl(
        token_ids, prepare_draft, logical_proposal_count, true, false);
  }

  std::size_t PrefillChunkDeferred(
      const std::vector<std::int64_t>& token_ids,
      bool prepare_draft,
      std::size_t logical_proposal_count) {
    return PrefillChunkImpl(
        token_ids, prepare_draft, logical_proposal_count, false, false)
        .model_executions;
  }

  StatefulStep PrefillChunkImpl(
      const std::vector<std::int64_t>& token_ids,
      bool prepare_draft,
      std::size_t logical_proposal_count,
      bool complete,
      bool defer_completion) {
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
    if (defer_completion && (!complete || !prepare_draft)) {
      throw std::invalid_argument(
          "deferred completion requires a final prefill with Draft");
    }
    if (prefill_staging_index_ >= stats_.prefill_staging_slots) {
      throw std::length_error("prefill staging ring capacity was exceeded");
    }
    if (token_ids.size() > sequence_length_ - prefill_total_token_count_) {
      throw std::length_error("prefill token count exceeds sequence capacity");
    }
    const std::size_t staging_index = prefill_staging_index_++;
    prefill_total_token_count_ += token_ids.size();
    const bool use_immutable_zero = ApplyPendingReset();
    const std::size_t control_upload_bytes = PreparePrefillControl(
        staging_index,
        token_ids,
        prepare_draft,
        logical_proposal_count,
        complete);
    UploadPrefillControl(staging_index, control_upload_bytes);

    if (use_immutable_zero) {
      Execute(prefill_, initial_prefill_plan_, prefill_width_);
      target_state_index_ = 0;
    } else {
      Execute(
          prefill_,
          prefill_plans_[staging_index][target_state_index_],
          prefill_width_);
      target_state_index_ = 1 - target_state_index_;
    }
    ++stats_.target_prefill_executions;
    feature_source_ = FeatureSource::kPrefill;
    std::size_t executions = 1;
    if (complete) {
      Execute(
          prefill_head_, prefill_head_plans_[target_state_index_], 1);
      ++stats_.target_prefill_head_executions;
      ++executions;
    } else {
      ++stats_.target_prefill_head_executions_elided;
    }
    if (prepare_draft && complete) {
      const std::size_t feature_rows =
          (staging_index + 1) * prefill_width_;
      ExecuteDraft(
          FeatureSource::kPrefill,
          feature_rows,
          DraftFeatureRoute::kPrefill);
      proposal_ready_ = true;
      prepared_proposal_count_ = logical_proposal_count;
      ++executions;
    } else {
      proposal_ready_ = false;
      if (prepare_draft) {
        ++stats_.prefill_draft_propose_executions_elided;
      }
    }
    if (!complete) {
      deferred_prefill_pending_ = true;
      ++stats_.deferred_prefill_chunks;
      ++stats_.prefill_synchronizations_elided;
      ++stats_.prefill_compact_downloads_elided;
      StatefulStep deferred;
      deferred.model_executions = executions;
      return deferred;
    }
    deferred_prefill_pending_ = false;
    prefill_staging_index_ = 0;
    prefill_total_token_count_ = 0;
    if (defer_completion) {
      StatefulStep deferred;
      deferred.model_executions = executions;
      return deferred;
    }
    DownloadCompact(false, target_state_index_);
    Synchronize();
    ++stats_.prefill_completion_synchronizations;
    return ReadCompactAndTrackCarrier(false, executions, target_state_index_);
  }

  bool supports_prefill_verify_coalescing() const noexcept { return true; }

  std::vector<StatefulStep> PrefillChunkAndSpeculative(
      const std::vector<std::int64_t>& token_ids,
      std::size_t logical_proposal_count) {
    StatefulStep prefill = PrefillChunkImpl(
        token_ids, true, logical_proposal_count, true, true);
    const std::size_t prefill_state_index = target_state_index_;
    const auto verify = EnqueueSpeculative(
        logical_proposal_count, true);
    if (prefill_state_index == verify.compact_state_index) {
      throw std::logic_error(
          "prefill/verify window did not alternate compact output arenas");
    }
    DownloadPrefillVerifyPair(
        prefill_state_index, verify.compact_state_index);
    Synchronize();
    ++stats_.prefill_completion_synchronizations;
    ++stats_.prefill_verify_coalesced_windows;
    ++stats_.prefill_verify_synchronizations_elided;
    std::vector<StatefulStep> result;
    result.reserve(2);
    result.push_back(ReadCompactAndTrackCarrier(
        false, prefill.model_executions, prefill_state_index));
    result.push_back(ReadCompactAndTrackCarrier(
        true, verify.model_executions, verify.compact_state_index));
    return result;
  }

  StatefulStep DecodeOne(std::int64_t input_token_id) {
    RequireReset();
    RequirePrefilled();
    RequireCompletedPrefill();
    if (input_token_id < 0) {
      throw std::invalid_argument("decode input token ID must be non-negative");
    }
    if (unified_target_step_ && !target_step_zero_count_valid_) {
      throw std::logic_error(
          "unified Target-step zero count was not initialized by prefill");
    }
    DatasetPlan* plan = nullptr;
    if (decode_carrier_valid_ && decode_carrier_token_id_ == input_token_id) {
      if (decode_carrier_row_ == 0) {
        plan = &decode_carrier_plans_[target_state_index_];
      } else {
        CompactDecodeIdToAlignedInput(
            target_state_index_, decode_carrier_row_);
        plan = &decode_upload_plans_[target_state_index_];
      }
      ++stats_.decode_id_device_carrier_hits;
      if (decode_carrier_row_ != 0) {
        ++stats_.decode_id_multi_token_carrier_hits;
      }
      ++stats_.decode_id_h2d_operations_elided;
    } else {
      *static_cast<std::int64_t*>(decode_id_.host) = input_token_id;
      Upload(decode_id_, decode_id_.bytes);
      ++stats_.decode_id_upload_operations;
      stats_.decode_id_upload_bytes += decode_id_.bytes;
      plan = &decode_upload_plans_[target_state_index_];
    }
    Execute(unified_target_step_ ? verify_ : decode_, *plan, 1);
    ++stats_.target_decode1_executions;
    if (unified_target_step_) {
      ++stats_.target_step_zero_count_bindings;
      ++stats_.target_step_input_rows;
      stats_.target_step_padded_rows_elided += verify_width_ - 1;
    }
    target_state_index_ = 1 - target_state_index_;
    proposal_ready_ = false;
    feature_source_ = unified_target_step_
        ? FeatureSource::kVerify
        : FeatureSource::kNone;
    DownloadCompact(false, target_state_index_);
    Synchronize();
    committed_feature_rows_ = unified_target_step_ ? 1 : 0;
    committed_feature_rows_valid_ = unified_target_step_;
    return ReadCompactAndTrackCarrier(false, 1, target_state_index_);
  }

  StatefulStep SpeculativeStep(std::size_t logical_proposal_count) {
    const auto pending = EnqueueSpeculative(
        logical_proposal_count, false);
    DownloadCompact(true, pending.compact_state_index);
    Synchronize();
    ++stats_.speculative_sync_windows;
    return ReadCompactAndTrackCarrier(
        true, pending.model_executions, pending.compact_state_index);
  }

  std::size_t max_speculative_sync_window() const noexcept { return 2; }

  std::vector<StatefulStep> SpeculativeWindow(
      const std::vector<std::size_t>& logical_proposal_counts) {
    if (logical_proposal_counts.size() == 1) {
      return {SpeculativeStep(logical_proposal_counts.front())};
    }
    if (logical_proposal_counts.size() != 2) {
      throw std::invalid_argument(
          "ACL speculative sync window must be one or two");
    }
    for (const std::size_t count : logical_proposal_counts) {
      RequireProposalCount(count);
    }
    const auto first = EnqueueSpeculative(
        logical_proposal_counts[0], false);
    const auto second = EnqueueSpeculative(
        logical_proposal_counts[1],
        true,
        logical_proposal_counts[0] + 1);
    if (first.compact_state_index == second.compact_state_index) {
      throw std::logic_error(
          "speculative pair did not alternate compact output arenas");
    }
    DownloadCompactPair(
        first.compact_state_index, second.compact_state_index);
    Synchronize();
    ++stats_.speculative_sync_windows;
    ++stats_.speculative_synchronizations_elided;
    std::vector<StatefulStep> result;
    result.reserve(2);
    result.push_back(ReadCompactAndTrackCarrier(
        true, first.model_executions, first.compact_state_index));
    result.push_back(ReadCompactAndTrackCarrier(
        true, second.model_executions, second.compact_state_index));
    return result;
  }

 private:
  struct PendingSpeculative {
    std::size_t model_executions = 0;
    std::size_t compact_state_index = 0;
  };

  PendingSpeculative EnqueueSpeculative(
      std::size_t logical_proposal_count,
      bool allow_pending_stream_work,
      std::size_t pending_feature_upper_bound = 0) {
    RequireReset();
    RequirePrefilled();
    RequireCompletedPrefill(allow_pending_stream_work);
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
      std::size_t feature_rows = verify_width_;
      DraftFeatureRoute feature_route = DraftFeatureRoute::kFixedVerifyWidth;
      if (draft_feature_policy_ ==
          IncrementalDraftFeaturePolicy::kCommittedPrefix) {
        if (allow_pending_stream_work) {
          if (pending_feature_upper_bound == 0 ||
              pending_feature_upper_bound > verify_width_) {
            throw std::logic_error(
                "pending Draft feature upper bound is invalid");
          }
          feature_rows = pending_feature_upper_bound;
          feature_route = DraftFeatureRoute::kPendingUpperBound;
        } else {
          if (!committed_feature_rows_valid_ ||
              committed_feature_rows_ == 0 ||
              committed_feature_rows_ > verify_width_) {
            throw std::logic_error(
                "host-visible committed Draft feature prefix is unavailable");
          }
          feature_rows = committed_feature_rows_;
          feature_route = DraftFeatureRoute::kCommittedPrefix;
        }
      }
      ExecuteDraft(FeatureSource::kVerify, feature_rows, feature_route);
      ++executions;
    }
    const std::size_t physical_rows = logical_proposal_count + 1;
    DatasetPlan& target_plan = unified_target_step_
        ? target_step_plans_.at(physical_rows - 1)[target_state_index_]
        : verify_plans_[target_state_index_];
    Execute(
        verify_,
        target_plan,
        unified_target_step_ ? physical_rows : verify_width_);
    ++stats_.target_verify_commit_executions;
    stats_.target_step_input_rows +=
        unified_target_step_ ? physical_rows : verify_width_;
    if (unified_target_step_) {
      stats_.target_step_padded_rows_elided += verify_width_ - physical_rows;
    }
    target_state_index_ = 1 - target_state_index_;
    proposal_ready_ = false;
    feature_source_ = FeatureSource::kVerify;
    return PendingSpeculative{executions, target_state_index_};
  }

  enum class FeatureSource : std::size_t {
    kPrefill = 0,
    kVerify = 1,
    kNone = 2,
  };

  enum class DraftFeatureRoute : std::size_t {
    kPrefill = 0,
    kFixedVerifyWidth = 1,
    kCommittedPrefix = 2,
    kPendingUpperBound = 3,
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
    if (prefill_.public_input_indices.size() != 7 ||
        prefill_.outputs.size() != 8 ||
        prefill_head_.public_input_indices.size() != 3 ||
        prefill_head_.outputs.size() != 3 ||
        (!unified_target_step_ &&
         (decode_.public_input_indices.size() != 8 ||
          decode_.outputs.size() != 8)) ||
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
    RequireSameTensor(
        prefill_.outputs[0], prefill_head_.PublicInput(0),
        "prefill last hidden");
    if (prefill_.outputs[0].dtype != ACL_FLOAT16 ||
        prefill_.outputs[0].shape.size() != 3 ||
        prefill_.outputs[0].shape[0] != 1 ||
        prefill_.outputs[0].shape[1] != 1 ||
        prefill_.outputs[0].shape[2] <= 0) {
      throw std::runtime_error(
          "prefill last hidden must be FP16[1,1,H]");
    }
    if (prefill_head_.PublicInput(1).dtype != ACL_INT64 ||
        prefill_head_.PublicInput(1).shape.size() != 1 ||
        prefill_head_.PublicInput(1).shape[0] <= 0) {
      throw std::runtime_error("prefill-head EOS table ABI differs");
    }
    eos_table_width_ =
        static_cast<std::size_t>(prefill_head_.PublicInput(1).shape[0]);
    RequireTensor(
        prefill_head_.PublicInput(2), ACL_INT32, {1},
        "prefill-head eos_token_count");

    if (!unified_target_step_) {
      RequireTensor(
          decode_.PublicInput(0), ACL_INT64, {1, 1}, "decode input_ids");
      RequireSameTensor(
          prefill_head_.PublicInput(1), decode_.PublicInput(1),
          "decode EOS table");
      RequireSameTensor(
          prefill_head_.PublicInput(2), decode_.PublicInput(2),
          "decode EOS count");
    }

    const auto& verify_ids = verify_.PublicInput(0);
    const std::int64_t expected_verify_rows = unified_target_step_ ? -1 : 16;
    if (verify_ids.dtype != ACL_INT64 || verify_ids.shape.size() != 2 ||
        verify_ids.shape[0] != 1 ||
        verify_ids.shape[1] != expected_verify_rows) {
      throw std::runtime_error(
          unified_target_step_
              ? "unified Target input IDs must be dynamic INT64[1,-1]"
              : "verify input IDs must be INT64[1,16]");
    }
    verify_width_ = 16;
    proposal_width_ = verify_width_ - 1;
    RequireTensor(
        verify_.PublicInput(1), ACL_INT32, {1}, "verify proposal count");
    RequireSameTensor(
        prefill_head_.PublicInput(1), verify_.PublicInput(2),
        "verify EOS table");
    RequireSameTensor(
        prefill_head_.PublicInput(2), verify_.PublicInput(3),
        "verify EOS count");

    target_state_specs_ = SelectSpecs(
        prefill_.inputs,
        {prefill_.public_input_indices[2], prefill_.public_input_indices[3],
         prefill_.public_input_indices[4], prefill_.public_input_indices[5],
         prefill_.public_input_indices[6]});
    if (!unified_target_step_) {
      RequireStateSet(
          target_state_specs_,
          SelectSpecs(
              decode_.inputs,
              {decode_.public_input_indices[3], decode_.public_input_indices[4],
               decode_.public_input_indices[5], decode_.public_input_indices[6],
               decode_.public_input_indices[7]}),
          "decode Target");
    }
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
        SelectSpecs(prefill_.outputs, {3, 4, 5, 6, 7}),
        "prefill Target outputs");
    if (!unified_target_step_) {
      RequireStateSet(
          target_state_specs_,
          SelectSpecs(decode_.outputs, {3, 4, 5, 6, 7}),
          "decode Target outputs");
    }
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

    RequireTensor(
        prefill_head_.outputs[0], ACL_INT64, {1, 16},
        "prefill-head committed IDs");
    RequireTensor(
        prefill_head_.outputs[1], ACL_INT32, {1},
        "prefill-head commit count");
    RequireTensor(
        prefill_head_.outputs[2], ACL_BOOL, {1},
        "prefill-head finished");
    if (prefill_.outputs[1].dtype != ACL_FLOAT16 ||
        prefill_.outputs[1].shape.size() != 3 ||
        prefill_.outputs[1].shape[0] != 1 ||
        prefill_.outputs[1].shape[1] != 64 ||
        prefill_.outputs[1].shape[2] <= 0) {
      throw std::runtime_error("prefill Target feature carrier ABI differs");
    }
    feature_width_ = static_cast<std::size_t>(prefill_.outputs[1].shape[2]);
    RequireTensor(prefill_.outputs[2], ACL_INT32, {1}, "prefill feature count");
    if (!unified_target_step_) {
      RequireSameTensor(
          prefill_head_.outputs[0], decode_.outputs[0],
          "decode committed IDs");
      RequireSameTensor(
          prefill_head_.outputs[1], decode_.outputs[1],
          "decode commit count");
      RequireSameTensor(
          prefill_head_.outputs[2], decode_.outputs[2], "decode finished");
    }

    RequireSameTensor(
        prefill_head_.outputs[0], verify_.outputs[0],
        "verify committed IDs");
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
    RequireSameTensor(prefill_.outputs[2], verify_.outputs[8], "feature count");

    const auto& draft_feature = draft_.PublicInput(0);
    if (draft_feature.dtype != ACL_FLOAT16 || draft_feature.shape.size() != 3 ||
        draft_feature.shape[0] != 1 || draft_feature.shape[1] != -1 ||
        draft_feature.shape[2] != static_cast<std::int64_t>(feature_width_)) {
      throw std::runtime_error(
          "Draft feature input must be FP16[1,-1,feature_width]");
    }
    RequireSameTensor(prefill_.outputs[2], draft_.PublicInput(1), "Draft feature count");
    RequireSameTensor(
        prefill_head_.outputs[0], draft_.PublicInput(2),
        "Draft previous IDs");
    RequireSameTensor(
        prefill_head_.outputs[1], draft_.PublicInput(3),
        "Draft previous count");
    RequireSameTensor(verify_.PublicInput(1), draft_.PublicInput(4), "Draft proposal count");
    if (unified_target_step_) {
      if (verify_.PublicInput(0).dtype != draft_.outputs[0].dtype ||
          verify_.PublicInput(0).bytes != draft_.outputs[0].bytes) {
        throw std::runtime_error(
            "dynamic Target-step maximum input differs from Draft verify IDs");
      }
    } else {
      RequireSameTensor(
          verify_.PublicInput(0), draft_.outputs[0], "Draft verify IDs");
    }
    ResolveDraftGears();
    if (unified_target_step_) {
      ResolveTargetStepGears();
    }
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
    draft_gear_verify_.reserve(verify_width_);
    for (std::size_t rows = 1; rows <= verify_width_; ++rows) {
      draft_gear_verify_.push_back(find(rows));
    }
    const std::size_t prefill_gears =
        (sequence_length_ - 1) / prefill_width_ + 1;
    if (draft_.dynamic_gears.size() != prefill_gears + verify_width_) {
      throw std::runtime_error(
          "Draft OM dynamic gear count differs from N=1..16 plus every "
          "64-row prompt batch");
    }
    draft_gear_prefill_.reserve(prefill_gears);
    for (std::size_t index = 0; index < prefill_gears; ++index) {
      draft_gear_prefill_.push_back(find((index + 1) * prefill_width_));
    }
    stats_.draft_dynamic_gear_count = draft_.dynamic_gears.size();
    stats_.draft_verify_dynamic_gear_count = draft_gear_verify_.size();
    stats_.draft_prefill_dynamic_gear_count = draft_gear_prefill_.size();
  }

  std::vector<std::int64_t> FlattenTargetStepShape(
      std::size_t physical_rows) const {
    std::vector<std::int64_t> result;
    for (std::size_t public_index = 0;
         public_index < verify_.public_input_indices.size();
         ++public_index) {
      const auto& spec = verify_.PublicInput(public_index);
      for (std::size_t dimension = 0; dimension < spec.shape.size();
           ++dimension) {
        std::int64_t value = spec.shape[dimension];
        if (value == -1) {
          if (public_index != 0 || dimension != 1) {
            throw std::runtime_error(
                "unified Target step has an unexpected dynamic dimension");
          }
          value = static_cast<std::int64_t>(physical_rows);
        }
        result.push_back(value);
      }
    }
    return result;
  }

  void ResolveTargetStepGears() {
    if (verify_.dynamic_gears.size() != verify_width_) {
      throw std::runtime_error(
          "unified Target step must expose exactly T=1..16 gears");
    }
    target_step_gears_.reserve(verify_width_);
    for (std::size_t rows = 1; rows <= verify_width_; ++rows) {
      const auto expected = FlattenTargetStepShape(rows);
      const auto match = std::find_if(
          verify_.dynamic_gears.begin(),
          verify_.dynamic_gears.end(),
          [&expected](const aclmdlIODims& gear) {
            if (gear.dimCount != expected.size()) {
              return false;
            }
            for (std::size_t index = 0; index < expected.size(); ++index) {
              if (gear.dims[index] != expected[index]) {
                return false;
              }
            }
            return true;
          });
      if (match == verify_.dynamic_gears.end()) {
        throw std::runtime_error(
            "unified Target step is missing dynamic gear T=" +
            std::to_string(rows));
      }
      target_step_gears_.push_back(*match);
    }
    stats_.target_step_dynamic_gear_count = target_step_gears_.size();
  }

  void AllocateBuffers() {
    target_states_[0].Allocate(target_state_specs_);
    target_states_[1].Allocate(target_state_specs_);
    draft_states_[0].Allocate(draft_state_specs_);
    draft_states_[1].Allocate(draft_state_specs_);
    stats_.working_state_device_bytes =
        target_states_[0].allocation.bytes +
        target_states_[1].allocation.bytes +
        draft_states_[0].allocation.bytes +
        draft_states_[1].allocation.bytes;
    stats_.state_reset_bytes_per_request =
        target_states_[0].allocation.bytes +
        draft_states_[0].allocation.bytes;
    if (state_reset_policy_ ==
        IncrementalStateResetPolicy::kImmutableZero) {
      target_zero_state_.Allocate(target_state_specs_);
      draft_zero_state_.Allocate(draft_state_specs_);
      stats_.immutable_zero_state_device_bytes =
          target_zero_state_.allocation.bytes +
          draft_zero_state_.allocation.bytes;
      if (stats_.immutable_zero_state_device_bytes !=
          stats_.state_reset_bytes_per_request) {
        throw std::logic_error("immutable zero state size differs from reset set");
      }
    }
    stats_.state_device_bytes =
        stats_.working_state_device_bytes +
        stats_.immutable_zero_state_device_bytes;

    std::size_t control_cursor = 0;
    prefill_ids_offset_ = ReserveDeviceSegment(
        &control_cursor, prefill_.PublicInput(0).bytes);
    effective_length_offset_ = ReserveDeviceSegment(
        &control_cursor, prefill_.PublicInput(1).bytes);
    prefill_base_control_bytes_ =
        effective_length_offset_ + prefill_.PublicInput(1).bytes;
    prefill_total_count_offset_ = ReserveDeviceSegment(
        &control_cursor, prefill_.outputs[2].bytes);
    prefill_count_control_bytes_ =
        prefill_total_count_offset_ + prefill_.outputs[2].bytes;
    proposal_count_offset_ = ReserveDeviceSegment(
        &control_cursor, verify_.PublicInput(1).bytes);
    prefill_proposal_control_bytes_ =
        proposal_count_offset_ + verify_.PublicInput(1).bytes;
    eos_ids_offset_ = ReserveDeviceSegment(
        &control_cursor, prefill_head_.PublicInput(1).bytes);
    eos_count_offset_ = ReserveDeviceSegment(
        &control_cursor, prefill_head_.PublicInput(2).bytes);
    if (unified_target_step_) {
      target_step_zero_count_offset_ = ReserveDeviceSegment(
          &control_cursor, verify_.PublicInput(1).bytes);
      stats_.target_step_zero_count_device_bytes =
          verify_.PublicInput(1).bytes;
    }
    prefill_control_.Allocate(Align(control_cursor, kBufferAlignment));
    if (prefill_base_control_bytes_ == 0 ||
        prefill_base_control_bytes_ >= prefill_count_control_bytes_ ||
        prefill_count_control_bytes_ >= prefill_proposal_control_bytes_ ||
        prefill_proposal_control_bytes_ >= prefill_control_.bytes) {
      throw std::logic_error("prefill control prefix layout is invalid");
    }
    if (unified_target_step_ &&
        (target_step_zero_count_offset_ % kBufferAlignment != 0 ||
         target_step_zero_count_offset_ > prefill_control_.bytes ||
         verify_.PublicInput(1).bytes >
             prefill_control_.bytes - target_step_zero_count_offset_)) {
      throw std::logic_error(
          "unified Target-step zero count layout is invalid");
    }
    decode_id_.Allocate(
        unified_target_step_ ? sizeof(std::int64_t)
                             : decode_.PublicInput(0).bytes);
    stats_.prefill_staging_slots =
        (sequence_length_ - 1) / prefill_width_ + 1;
    if (stats_.prefill_staging_slots == 0) {
      throw std::logic_error("prefill staging ring has no slots");
    }
    stats_.prefill_control_bytes_per_slot = prefill_control_.bytes;
    stats_.prefill_base_control_bytes_per_slot =
        prefill_base_control_bytes_;
    stats_.prefill_count_control_bytes_per_slot =
        prefill_count_control_bytes_;
    stats_.prefill_proposal_control_bytes_per_slot =
        prefill_proposal_control_bytes_;
    stats_.prefill_persistent_control_tail_bytes_per_slot =
        prefill_control_.bytes - prefill_proposal_control_bytes_;
    const std::size_t staging_bytes_per_slot = prefill_control_.bytes;
    if (stats_.prefill_staging_slots >
        std::numeric_limits<std::size_t>::max() / staging_bytes_per_slot) {
      throw std::overflow_error("prefill pinned host staging size overflow");
    }
    stats_.prefill_staging_pinned_host_bytes =
        stats_.prefill_staging_slots * staging_bytes_per_slot;
    const std::size_t extra_slots = stats_.prefill_staging_slots - 1;
    extra_prefill_control_host_.resize(extra_slots);
    for (std::size_t index = 0; index < extra_slots; ++index) {
      extra_prefill_control_host_[index].Allocate(prefill_control_.bytes);
    }
    for (auto& host : proposal_count_upload_host_) {
      host.Allocate(verify_.PublicInput(1).bytes);
      stats_.proposal_count_staging_pinned_host_bytes += host.bytes;
    }

    std::size_t compact_cursor = 0;
    compact_token_offset_ = ReserveDeviceSegment(
        &compact_cursor, prefill_head_.outputs[0].bytes);
    compact_commit_offset_ = ReserveDeviceSegment(
        &compact_cursor, prefill_head_.outputs[1].bytes);
    compact_finished_offset_ = ReserveDeviceSegment(
        &compact_cursor, prefill_head_.outputs[2].bytes);
    compact_drafted_offset_ = ReserveDeviceSegment(
        &compact_cursor, verify_.outputs[2].bytes);
    compact_accepted_offset_ = ReserveDeviceSegment(
        &compact_cursor, verify_.outputs[3].bytes);
    compact_rejected_offset_ = ReserveDeviceSegment(
        &compact_cursor, verify_.outputs[4].bytes);
    compact_ordinary_bytes_ =
        compact_finished_offset_ + prefill_head_.outputs[2].bytes;
    compact_verify_bytes_ =
        compact_rejected_offset_ + verify_.outputs[4].bytes;
    stats_.compact_ordinary_result_bytes = compact_ordinary_bytes_;
    stats_.compact_verify_result_bytes = compact_verify_bytes_;
    compact_slot_bytes_ = Align(compact_cursor, kBufferAlignment);
    if (compact_slot_bytes_ >
        std::numeric_limits<std::size_t>::max() / kCompactSlotCount) {
      throw std::overflow_error("compact ping-pong arena size overflow");
    }
    compact_.Allocate(compact_slot_bytes_ * kCompactSlotCount);
    stats_.compact_slot_bytes = compact_slot_bytes_;
    stats_.compact_ping_pong_device_bytes = compact_.device.bytes;

    const std::size_t prefill_feature_payload = prefill_.outputs[1].bytes;
    if (prefill_feature_payload % kBufferAlignment != 0) {
      throw std::runtime_error(
          "prefill feature slab does not preserve 64-byte alignment");
    }
    if (stats_.prefill_staging_slots >
        std::numeric_limits<std::size_t>::max() / prefill_feature_payload) {
      throw std::overflow_error("prefill feature arena size overflow");
    }
    const std::size_t packed_feature_bytes =
        stats_.prefill_staging_slots * prefill_feature_payload;
    if (draft_.PublicInput(0).bytes < packed_feature_bytes) {
      throw std::runtime_error(
          "Draft feature input buffer is smaller than the maximum prompt batch");
    }
    const std::size_t feature_payload =
        std::max(draft_.PublicInput(0).bytes, packed_feature_bytes);
    if (feature_payload > std::numeric_limits<std::size_t>::max() - 32) {
      throw std::overflow_error("prefill feature terminal guard overflow");
    }
    prefill_features_.Allocate(
        Align(feature_payload + 32, kBufferAlignment));
    stats_.prefill_feature_slab_bytes = prefill_feature_payload;
    stats_.prefill_feature_arena_bytes = prefill_features_.bytes;
    prefill_last_hidden_.Allocate(prefill_.outputs[0].bytes);
    committed_input_count_.Allocate(prefill_.outputs[2].bytes);
    verify_ids_.Allocate(verify_.PublicInput(0).bytes);
    prefill_dynamic_controls_.resize(stats_.prefill_staging_slots);
    for (auto& control : prefill_dynamic_controls_) {
      control.Allocate(draft_.inputs.at(draft_.dynamic_input_index).bytes);
    }
    verify_dynamic_control_.Allocate(
        draft_.inputs.at(draft_.dynamic_input_index).bytes);
    if (unified_target_step_) {
      target_step_dynamic_control_.Allocate(
          verify_.inputs.at(verify_.dynamic_input_index).bytes);
      target_step_plans_.resize(verify_width_);
    }

    prefill_plans_.resize(stats_.prefill_staging_slots);
    prefill_draft_plans_.resize(stats_.prefill_staging_slots);
    verify_draft_plans_.resize(verify_width_);
    if (state_reset_policy_ == IncrementalStateResetPolicy::kImmutableZero) {
      initial_draft_plans_.resize(stats_.prefill_staging_slots);
    }

    stats_.carrier_device_bytes =
        prefill_control_.device.bytes + decode_id_.device.bytes +
        stats_.compact_ping_pong_device_bytes + prefill_features_.bytes +
        prefill_last_hidden_.bytes + committed_input_count_.bytes +
        verify_ids_.bytes +
        verify_dynamic_control_.bytes;
    if (unified_target_step_) {
      stats_.carrier_device_bytes += target_step_dynamic_control_.bytes;
    }
    for (const auto& control : prefill_dynamic_controls_) {
      stats_.carrier_device_bytes += control.bytes;
    }
  }

  void* PrefillControlHost(std::size_t index) {
    if (index == 0) {
      return prefill_control_.host;
    }
    if (index - 1 >= extra_prefill_control_host_.size() ||
        extra_prefill_control_host_[index - 1].data == nullptr) {
      throw std::out_of_range("prefill control staging index is invalid");
    }
    return extra_prefill_control_host_[index - 1].data;
  }

  void* PrefillIdsHost(std::size_t index) {
    return static_cast<std::byte*>(PrefillControlHost(index)) +
        prefill_ids_offset_;
  }

  void* EffectiveLengthHost(std::size_t index) {
    return static_cast<std::byte*>(PrefillControlHost(index)) +
        effective_length_offset_;
  }

  void* PrefillTotalCountHost(std::size_t index) {
    return static_cast<std::byte*>(PrefillControlHost(index)) +
        prefill_total_count_offset_;
  }

  void* ProposalCountHost(std::size_t index) {
    return static_cast<std::byte*>(PrefillControlHost(index)) +
        proposal_count_offset_;
  }

  void* EosIdsHost(std::size_t index) {
    return static_cast<std::byte*>(PrefillControlHost(index)) +
        eos_ids_offset_;
  }

  void* EosCountHost(std::size_t index) {
    return static_cast<std::byte*>(PrefillControlHost(index)) +
        eos_count_offset_;
  }

  void* TargetStepZeroCountHost(std::size_t index) {
    if (!unified_target_step_ ||
        target_step_zero_count_offset_ ==
            std::numeric_limits<std::size_t>::max()) {
      throw std::logic_error(
          "Target-step zero count is unavailable in the five-OM topology");
    }
    return static_cast<std::byte*>(PrefillControlHost(index)) +
        target_step_zero_count_offset_;
  }

  BufferView PrefillControlView(
      std::size_t offset,
      std::size_t bytes) const {
    if (offset > prefill_control_.device.bytes ||
        bytes > prefill_control_.device.bytes - offset) {
      throw std::out_of_range("prefill control device view is invalid");
    }
    return BufferView{
        static_cast<std::byte*>(prefill_control_.device.data) + offset,
        bytes,
    };
  }

  BufferView PrefillIdsView() const {
    return PrefillControlView(
        prefill_ids_offset_, prefill_.PublicInput(0).bytes);
  }

  BufferView EffectiveLengthView() const {
    return PrefillControlView(
        effective_length_offset_, prefill_.PublicInput(1).bytes);
  }

  BufferView EosIdsView() const {
    return PrefillControlView(
        eos_ids_offset_, prefill_head_.PublicInput(1).bytes);
  }

  BufferView EosCountView() const {
    return PrefillControlView(
        eos_count_offset_, prefill_head_.PublicInput(2).bytes);
  }

  BufferView ProposalCountView() const {
    return PrefillControlView(
        proposal_count_offset_, verify_.PublicInput(1).bytes);
  }

  BufferView TargetStepZeroCountView() const {
    if (!unified_target_step_ ||
        target_step_zero_count_offset_ ==
            std::numeric_limits<std::size_t>::max()) {
      throw std::logic_error(
          "Target-step zero count is unavailable in the five-OM topology");
    }
    return PrefillControlView(
        target_step_zero_count_offset_, verify_.PublicInput(1).bytes);
  }

  BufferView PrefillTotalCountView() const {
    return PrefillControlView(
        prefill_total_count_offset_, prefill_.outputs[2].bytes);
  }

  BufferView PrefillFeatureSlabView(std::size_t index) const {
    if (index >= stats_.prefill_staging_slots) {
      throw std::out_of_range("prefill feature slab index is invalid");
    }
    const std::size_t offset = index * stats_.prefill_feature_slab_bytes;
    if (offset > prefill_features_.bytes ||
        stats_.prefill_feature_slab_bytes > prefill_features_.bytes - offset) {
      throw std::out_of_range("prefill feature slab exceeds its arena");
    }
    return BufferView{
        static_cast<std::byte*>(prefill_features_.data) + offset,
        stats_.prefill_feature_slab_bytes,
    };
  }

  BufferView PrefillFeatureBatchView() const {
    return prefill_features_.View(draft_.PublicInput(0).bytes);
  }

  std::size_t PreparePrefillControl(
      std::size_t staging_index,
      const std::vector<std::int64_t>& token_ids,
      bool prepare_draft,
      std::size_t logical_proposal_count,
      bool complete) {
    auto* input_ids = static_cast<std::int64_t*>(
        PrefillIdsHost(staging_index));
    std::fill_n(input_ids, prefill_width_, pad_token_id_);
    std::copy(token_ids.begin(), token_ids.end(), input_ids);
    *static_cast<std::int16_t*>(EffectiveLengthHost(staging_index)) =
        static_cast<std::int16_t>(token_ids.size());
    std::size_t upload_bytes = prefill_base_control_bytes_;
    const bool final_draft = prepare_draft && complete;
    if (final_draft) {
      *static_cast<std::int32_t*>(PrefillTotalCountHost(staging_index)) =
          static_cast<std::int32_t>(prefill_total_token_count_);
      upload_bytes = prefill_count_control_bytes_;
    }

    const std::size_t desired_proposal_count = final_draft
        ? logical_proposal_count
        : (proposal_value_valid_ ? proposal_value_ : 1);
    const bool proposal_upload_required =
        final_draft &&
        (!proposal_value_valid_ ||
         proposal_value_ != desired_proposal_count);
    const bool eos_upload_required = complete && eos_control_upload_pending_;
    if (proposal_upload_required || eos_upload_required) {
      *static_cast<std::int32_t*>(ProposalCountHost(staging_index)) =
          static_cast<std::int32_t>(desired_proposal_count);
      upload_bytes = prefill_proposal_control_bytes_;
    }

    if (eos_upload_required) {
      auto* eos_ids = static_cast<std::int64_t*>(
          EosIdsHost(staging_index));
      std::fill_n(eos_ids, eos_table_width_, 0);
      std::copy(
          configured_eos_token_ids_.begin(),
          configured_eos_token_ids_.end(),
          eos_ids);
      *static_cast<std::int32_t*>(EosCountHost(staging_index)) =
          static_cast<std::int32_t>(configured_eos_token_ids_.size());
      if (unified_target_step_) {
        *static_cast<std::int32_t*>(
            TargetStepZeroCountHost(staging_index)) = 0;
      }
      upload_bytes = prefill_control_.bytes;
    }
    return upload_bytes;
  }

  void UploadPrefillControl(
      std::size_t staging_index,
      std::size_t bytes) {
    if (bytes != prefill_base_control_bytes_ &&
        bytes != prefill_count_control_bytes_ &&
        bytes != prefill_proposal_control_bytes_ &&
        bytes != prefill_control_.bytes) {
      throw std::logic_error("prefill control upload prefix is invalid");
    }
    UploadFromHost(
        prefill_control_,
        PrefillControlHost(staging_index),
        bytes);
    ++stats_.prefill_control_upload_operations;
    stats_.prefill_control_upload_bytes += bytes;
    if (bytes == prefill_control_.bytes) {
      ++stats_.prefill_control_full_upload_operations;
      device_eos_token_ids_ = configured_eos_token_ids_;
      device_eos_token_ids_valid_ = true;
      eos_control_upload_pending_ = false;
      if (unified_target_step_) {
        const auto zero_count = *static_cast<const std::int32_t*>(
            TargetStepZeroCountHost(staging_index));
        if (zero_count != 0) {
          throw std::logic_error(
              "unified Target-step zero count is not zero");
        }
        target_step_zero_count_valid_ = true;
      }
    } else if (bytes == prefill_proposal_control_bytes_) {
      ++stats_.prefill_control_proposal_upload_operations;
    } else if (bytes == prefill_count_control_bytes_) {
      ++stats_.prefill_control_count_upload_operations;
    } else {
      ++stats_.prefill_control_base_upload_operations;
    }
    if (bytes >= prefill_proposal_control_bytes_) {
      const auto proposal = *static_cast<const std::int32_t*>(
          ProposalCountHost(staging_index));
      if (proposal <= 0 ||
          static_cast<std::size_t>(proposal) > proposal_width_) {
        throw std::logic_error("uploaded prefill proposal count is invalid");
      }
      proposal_value_ = static_cast<std::size_t>(proposal);
      proposal_value_valid_ = true;
    }
    stats_.prefill_control_h2d_bytes_elided +=
        prefill_control_.bytes - bytes;
    ++stats_.prefill_h2d_operations_elided;
  }

  void InitializeImmutableZeroState() {
    if (state_reset_policy_ !=
        IncrementalStateResetPolicy::kImmutableZero) {
      return;
    }
    Check(
        aclrtMemsetAsync(
            target_zero_state_.allocation.data,
            target_zero_state_.allocation.bytes,
            0,
            target_zero_state_.allocation.bytes,
            stream_),
        "aclrtMemsetAsync(immutable Target zero state)");
    stream_work_pending_ = true;
    ++stats_.state_initialization_memset_operations;
    stats_.state_initialization_memset_bytes +=
        target_zero_state_.allocation.bytes;
    Check(
        aclrtMemsetAsync(
            draft_zero_state_.allocation.data,
            draft_zero_state_.allocation.bytes,
            0,
            draft_zero_state_.allocation.bytes,
            stream_),
        "aclrtMemsetAsync(immutable Draft zero state)");
    stream_work_pending_ = true;
    ++stats_.state_initialization_memset_operations;
    stats_.state_initialization_memset_bytes +=
        draft_zero_state_.allocation.bytes;
    Check(
        aclrtSynchronizeStream(stream_),
        "aclrtSynchronizeStream(immutable zero state initialization)");
    stream_work_pending_ = false;
    ++stats_.state_initialization_stream_synchronizations;
  }

  BufferView CompactView(
      std::size_t state_index,
      std::size_t offset,
      std::size_t bytes) const {
    if (state_index >= kCompactSlotCount) {
      throw std::out_of_range("compact state index is invalid");
    }
    if (offset > compact_slot_bytes_ ||
        bytes > compact_slot_bytes_ - offset) {
      throw std::out_of_range("compact device binding exceeds its arena");
    }
    const std::size_t slot_offset = state_index * compact_slot_bytes_;
    return BufferView{
        static_cast<std::byte*>(compact_.device.data) + slot_offset + offset,
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
    const auto build_draft = [this](
                                 DatasetPlan& plan,
                                 std::size_t target_state_index,
                                 const BufferView& features,
                                 const BufferView& committed_count,
                                 const std::vector<BufferView>& input_state,
                                 const std::vector<BufferView>& output_state,
                                 const BufferView& dynamic_control,
                                 const aclmdlIODims& gear,
                                 const char* gear_description) {
      plan.Build(
          draft_,
          {features,
           committed_count,
           CompactView(
               target_state_index,
               compact_token_offset_,
               draft_.PublicInput(2).bytes),
           CompactView(
               target_state_index,
               compact_commit_offset_,
               draft_.PublicInput(3).bytes),
           ProposalCountView(),
           input_state.at(0),
           input_state.at(1),
           input_state.at(2)},
          {verify_ids_.View(),
           output_state.at(0),
           output_state.at(1),
           output_state.at(2)},
          dynamic_control);
      Check(
          aclmdlSetInputDynamicDims(
              draft_.id,
              plan.input,
              draft_.dynamic_input_index,
              &gear),
          std::string("draft-propose: aclmdlSetInputDynamicDims(") +
              gear_description + ")");
    };

    for (std::size_t current = 0; current < 2; ++current) {
      const std::size_t next = 1 - current;
      for (std::size_t slot = 0; slot < prefill_plans_.size(); ++slot) {
        prefill_plans_[slot][current].Build(
            prefill_,
            TargetInputs(
                {PrefillIdsView(), EffectiveLengthView()},
                current),
            {prefill_last_hidden_.View(),
             PrefillFeatureSlabView(slot),
             committed_input_count_.View(),
             target_states_[next].tensors[0],
             target_states_[next].tensors[1],
             target_states_[next].tensors[2],
             target_states_[next].tensors[3],
             target_states_[next].tensors[4]});
      }
      const auto verify_outputs = [this, next]() {
        return std::vector<BufferView>{
            CompactView(
                next, compact_token_offset_, verify_.outputs[0].bytes),
            CompactView(
                next, compact_commit_offset_, verify_.outputs[1].bytes),
            CompactView(
                next, compact_drafted_offset_, verify_.outputs[2].bytes),
            CompactView(
                next, compact_accepted_offset_, verify_.outputs[3].bytes),
            CompactView(
                next, compact_rejected_offset_, verify_.outputs[4].bytes),
            target_states_[next].tensors[0],
            target_states_[next].tensors[1],
            prefill_features_.View(verify_.outputs[7].bytes),
            committed_input_count_.View(),
            target_states_[next].tensors[4],
            CompactView(
                next, compact_finished_offset_, verify_.outputs[10].bytes),
            target_states_[next].tensors[2],
            target_states_[next].tensors[3]};
      };
      if (unified_target_step_) {
        const auto build_target_step = [this, current, &verify_outputs](
                                           DatasetPlan& plan,
                                           const BufferView& input_ids,
                                           std::size_t rows,
                                           const BufferView& proposal_count) {
          plan.Build(
              verify_,
              TargetInputs(
                  {input_ids, proposal_count, EosIdsView(), EosCountView()},
                  current),
              verify_outputs(),
              target_step_dynamic_control_.View());
          Check(
              aclmdlSetInputDynamicDims(
                  verify_.id,
                  plan.input,
                  verify_.dynamic_input_index,
                  &target_step_gears_.at(rows - 1)),
              "target-verify-commit: aclmdlSetInputDynamicDims(T=" +
                  std::to_string(rows) + ")");
        };
        build_target_step(
            decode_upload_plans_[current],
            decode_id_.View(),
            1,
            TargetStepZeroCountView());
        build_target_step(
            decode_carrier_plans_[current],
            CompactView(
                current, compact_token_offset_, sizeof(std::int64_t)),
            1,
            TargetStepZeroCountView());
        for (std::size_t rows = 2; rows <= verify_width_; ++rows) {
          build_target_step(
              target_step_plans_[rows - 1][current],
              verify_ids_.View(rows * sizeof(std::int64_t)),
              rows,
              ProposalCountView());
        }
      } else {
        const auto decode_outputs = [this, next]() {
          return std::vector<BufferView>{
              CompactView(
                  next, compact_token_offset_, decode_.outputs[0].bytes),
              CompactView(
                  next, compact_commit_offset_, decode_.outputs[1].bytes),
              CompactView(
                  next, compact_finished_offset_, decode_.outputs[2].bytes),
              target_states_[next].tensors[0],
              target_states_[next].tensors[1],
              target_states_[next].tensors[2],
              target_states_[next].tensors[3],
              target_states_[next].tensors[4]};
        };
        decode_upload_plans_[current].Build(
            decode_,
            TargetInputs(
                {decode_id_.View(), EosIdsView(), EosCountView()},
                current),
            decode_outputs());
        decode_carrier_plans_[current].Build(
            decode_,
            TargetInputs(
                {CompactView(
                     current,
                     compact_token_offset_,
                     decode_.PublicInput(0).bytes),
                 EosIdsView(), EosCountView()},
                current),
            decode_outputs());
        verify_plans_[current].Build(
            verify_,
            TargetInputs(
                {verify_ids_.View(), ProposalCountView(), EosIdsView(),
                 EosCountView()},
                current),
            verify_outputs());
      }
      prefill_head_plans_[current].Build(
          prefill_head_,
          {prefill_last_hidden_.View(), EosIdsView(), EosCountView()},
          {CompactView(
               current,
               compact_token_offset_,
               prefill_head_.outputs[0].bytes),
           CompactView(
               current,
               compact_commit_offset_,
               prefill_head_.outputs[1].bytes),
           CompactView(
               current,
               compact_finished_offset_,
               prefill_head_.outputs[2].bytes)});
    }

    for (std::size_t target = 0; target < 2; ++target) {
      for (std::size_t current = 0; current < 2; ++current) {
        const std::size_t next = 1 - current;
        for (std::size_t rows = 1; rows <= verify_width_; ++rows) {
          build_draft(
              verify_draft_plans_.at(rows - 1)[target][current],
              target,
              PrefillFeatureBatchView(),
              committed_input_count_.View(),
              draft_states_[current].tensors,
              draft_states_[next].tensors,
              verify_dynamic_control_.View(),
              draft_gear_verify_.at(rows - 1),
              "verify committed-prefix prebind");
        }
        for (std::size_t slot = 0; slot < prefill_draft_plans_.size(); ++slot) {
          build_draft(
              prefill_draft_plans_[slot][target][current],
              target,
              PrefillFeatureBatchView(),
              PrefillTotalCountView(),
              draft_states_[current].tensors,
              draft_states_[next].tensors,
              prefill_dynamic_controls_[slot].View(),
              draft_gear_prefill_[slot],
              "prefill batch prebind");
        }
      }
    }
    if (state_reset_policy_ ==
        IncrementalStateResetPolicy::kImmutableZero) {
      initial_prefill_plan_.Build(
          prefill_,
          [&]() {
            std::vector<BufferView> inputs{
                PrefillIdsView(), EffectiveLengthView()};
            inputs.insert(
                inputs.end(),
                target_zero_state_.tensors.begin(),
                target_zero_state_.tensors.end());
            return inputs;
          }(),
          {prefill_last_hidden_.View(),
           PrefillFeatureSlabView(0),
           committed_input_count_.View(),
           target_states_[0].tensors[0],
           target_states_[0].tensors[1],
           target_states_[0].tensors[2],
           target_states_[0].tensors[3],
           target_states_[0].tensors[4]});
      for (std::size_t slot = 0; slot < initial_draft_plans_.size(); ++slot) {
        for (std::size_t target = 0; target < 2; ++target) {
          build_draft(
              initial_draft_plans_[slot][target],
              target,
              PrefillFeatureBatchView(),
              PrefillTotalCountView(),
              draft_zero_state_.tensors,
              draft_states_[0].tensors,
              prefill_dynamic_controls_[slot].View(),
              draft_gear_prefill_[slot],
              "immutable zero prefill batch prebind");
        }
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

  void RequireCompletedPrefill(
      bool allow_pending_stream_work = false) const {
    if (deferred_prefill_pending_ ||
        (stream_work_pending_ && !allow_pending_stream_work)) {
      throw std::logic_error(
          "incremental executor requires a completing prefill chunk");
    }
  }

  bool ApplyPendingReset() {
    if (!reset_pending_) {
      return false;
    }
    const bool use_immutable_zero = state_reset_policy_ ==
        IncrementalStateResetPolicy::kImmutableZero;
    if (!use_immutable_zero) {
      Check(
          aclrtMemsetAsync(
              target_states_[0].allocation.data,
              target_states_[0].allocation.bytes,
              0,
              target_states_[0].allocation.bytes,
              stream_),
          "aclrtMemsetAsync(target state)");
      stream_work_pending_ = true;
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
      stream_work_pending_ = true;
      ++stats_.state_memset_operations;
      stats_.state_memset_bytes += draft_states_[0].allocation.bytes;
      draft_reset_pending_ = false;
    }
    reset_pending_ = false;
    return use_immutable_zero;
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
    HostAllocation& staging =
        proposal_count_upload_host_[proposal_count_upload_staging_index_];
    proposal_count_upload_staging_index_ =
        1 - proposal_count_upload_staging_index_;
    void* host = staging.data;
    *static_cast<std::int32_t*>(host) =
        static_cast<std::int32_t>(value);
    const std::size_t bytes = verify_.PublicInput(1).bytes;
    UploadFromHost(ProposalCountView(), host, bytes);
    ++stats_.proposal_count_upload_operations;
    stats_.proposal_count_upload_bytes += bytes;
    proposal_value_ = value;
    proposal_value_valid_ = true;
  }

  void ExecuteDraft(
      FeatureSource source,
      std::size_t feature_rows,
      DraftFeatureRoute feature_route) {
    if (source == FeatureSource::kNone) {
      throw std::logic_error("invalid Draft feature source/gear");
    }
    const bool prefill_source = source == FeatureSource::kPrefill;
    if ((!prefill_source &&
         (feature_rows == 0 || feature_rows > verify_width_)) ||
        (prefill_source &&
         (feature_rows == 0 || feature_rows % prefill_width_ != 0 ||
          feature_rows > sequence_length_))) {
      throw std::logic_error("invalid Draft feature source/gear");
    }
    if ((prefill_source && feature_route != DraftFeatureRoute::kPrefill) ||
        (!prefill_source && feature_route == DraftFeatureRoute::kPrefill)) {
      throw std::logic_error("Draft feature route does not match its source");
    }
    const bool use_immutable_zero =
        draft_reset_pending_ &&
        state_reset_policy_ == IncrementalStateResetPolicy::kImmutableZero;
    if (draft_reset_pending_ && !use_immutable_zero) {
      throw std::logic_error("Draft reset was not consumed by prefill");
    }
    if (use_immutable_zero && !prefill_source) {
      throw std::logic_error(
          "immutable Draft zero state is valid only for first prefill");
    }
    DatasetPlan* plan = nullptr;
    if (prefill_source) {
      const std::size_t gear_index = feature_rows / prefill_width_ - 1;
      plan = use_immutable_zero
          ? &initial_draft_plans_.at(gear_index)[target_state_index_]
          : &prefill_draft_plans_.at(gear_index)
                 [target_state_index_][draft_state_index_];
    } else {
      plan = &verify_draft_plans_.at(feature_rows - 1)
                  [target_state_index_][draft_state_index_];
    }
    Execute(draft_, *plan, feature_rows);
    ++stats_.draft_propose_executions;
    if (prefill_source) {
      ++stats_.prefill_draft_propose_executions;
      stats_.prefill_feature_rows_batched += feature_rows;
    } else {
      stats_.draft_verify_feature_input_rows += feature_rows;
      stats_.draft_verify_full_width_equivalent_rows += verify_width_;
      stats_.draft_verify_feature_rows_elided +=
          verify_width_ - feature_rows;
      switch (feature_route) {
        case DraftFeatureRoute::kFixedVerifyWidth:
          ++stats_.draft_verify_fixed_width_executions;
          break;
        case DraftFeatureRoute::kCommittedPrefix:
          ++stats_.draft_verify_committed_prefix_executions;
          break;
        case DraftFeatureRoute::kPendingUpperBound:
          ++stats_.draft_verify_pending_upper_bound_executions;
          break;
        case DraftFeatureRoute::kPrefill:
          throw std::logic_error("prefill Draft route reached verify stats");
      }
    }
    draft_state_index_ = use_immutable_zero ? 0 : 1 - draft_state_index_;
    draft_reset_pending_ = false;
  }

  void Execute(
      const ModelSession& session,
      DatasetPlan& plan,
      std::size_t physical_rows) {
    if (physical_rows == 0) {
      throw std::invalid_argument("model execution rows must be positive");
    }
    if (profile_model_executions_) {
      model_execution_trace_.push_back(
          {model_execution_trace_.size(), session.id, physical_rows});
    }
    Check(
        aclmdlExecuteAsync(session.id, plan.input, plan.output, stream_),
        session.role + ": aclmdlExecuteAsync");
    stream_work_pending_ = true;
  }

  void CompactDecodeIdToAlignedInput(
      std::size_t state_index,
      std::size_t row) {
    if (row == 0 || row >= verify_width_) {
      throw std::out_of_range("decode carrier compaction row is invalid");
    }
    const BufferView source = CompactView(
        state_index,
        compact_token_offset_ + row * sizeof(std::int64_t),
        decode_id_.bytes);
    Check(
        aclrtMemcpyAsync(
            decode_id_.device.data,
            decode_id_.device.bytes,
            source.data,
            source.bytes,
            ACL_MEMCPY_DEVICE_TO_DEVICE,
            stream_),
        "aclrtMemcpyAsync(decode carrier device compaction)");
    stream_work_pending_ = true;
    ++stats_.decode_id_device_compaction_operations;
    stats_.decode_id_device_compaction_bytes += source.bytes;
  }

  void Upload(const MirrorBuffer& buffer, std::size_t bytes) {
    UploadFromHost(buffer.device.View(), buffer.host, bytes);
  }

  void UploadFromHost(
      const MirrorBuffer& buffer,
      const void* host,
      std::size_t bytes) {
    UploadFromHost(buffer.device.View(), host, bytes);
  }

  void UploadFromHost(
      const BufferView& destination,
      const void* host,
      std::size_t bytes) {
    if (bytes == 0 || bytes > destination.bytes) {
      throw std::out_of_range("host-to-device copy exceeds device buffer");
    }
    if (host == nullptr) {
      throw std::invalid_argument("host-to-device source is null");
    }
    Check(
        aclrtMemcpyAsync(
            destination.data,
            destination.bytes,
            host,
            bytes,
            ACL_MEMCPY_HOST_TO_DEVICE,
            stream_),
        "aclrtMemcpyAsync(host-to-device)");
    stream_work_pending_ = true;
    ++stats_.host_to_device_operations;
    stats_.host_to_device_bytes += bytes;
  }

  void DownloadCompact(bool verify, std::size_t state_index) {
    if (state_index >= kCompactSlotCount) {
      throw std::out_of_range("compact download state index is invalid");
    }
    const std::size_t bytes =
        verify ? compact_verify_bytes_ : compact_ordinary_bytes_;
    const std::size_t slot_offset = state_index * compact_slot_bytes_;
    Check(
        aclrtMemcpyAsync(
            static_cast<std::byte*>(compact_.host) + slot_offset,
            compact_.bytes - slot_offset,
            static_cast<std::byte*>(compact_.device.data) + slot_offset,
            bytes,
            ACL_MEMCPY_DEVICE_TO_HOST,
            stream_),
        "aclrtMemcpyAsync(compact device-to-host)");
    stream_work_pending_ = true;
    ++stats_.device_to_host_operations;
    stats_.device_to_host_bytes += bytes;
  }

  void DownloadCompactPair(
      std::size_t first_state_index,
      std::size_t second_state_index) {
    if (first_state_index >= kCompactSlotCount ||
        second_state_index >= kCompactSlotCount ||
        first_state_index == second_state_index) {
      throw std::out_of_range("compact pair state indices are invalid");
    }
    const std::size_t bytes = compact_slot_bytes_ + compact_verify_bytes_;
    Check(
        aclrtMemcpyAsync(
            compact_.host,
            compact_.bytes,
            compact_.device.data,
            bytes,
            ACL_MEMCPY_DEVICE_TO_HOST,
            stream_),
        "aclrtMemcpyAsync(compact pair device-to-host)");
    stream_work_pending_ = true;
    ++stats_.device_to_host_operations;
    stats_.device_to_host_bytes += bytes;
    ++stats_.speculative_d2h_operations_elided;
    stats_.speculative_d2h_padding_bytes +=
        compact_slot_bytes_ - compact_verify_bytes_;
  }

  void DownloadPrefillVerifyPair(
      std::size_t prefill_state_index,
      std::size_t verify_state_index) {
    if (prefill_state_index >= kCompactSlotCount ||
        verify_state_index >= kCompactSlotCount ||
        prefill_state_index == verify_state_index) {
      throw std::out_of_range(
          "prefill/verify compact pair state indices are invalid");
    }
    const std::size_t prefill_end =
        prefill_state_index * compact_slot_bytes_ + compact_ordinary_bytes_;
    const std::size_t verify_end =
        verify_state_index * compact_slot_bytes_ + compact_verify_bytes_;
    const std::size_t bytes = std::max(prefill_end, verify_end);
    if (bytes > compact_.bytes ||
        bytes < compact_ordinary_bytes_ + compact_verify_bytes_) {
      throw std::logic_error(
          "prefill/verify compact pair span is invalid");
    }
    Check(
        aclrtMemcpyAsync(
            compact_.host,
            compact_.bytes,
            compact_.device.data,
            bytes,
            ACL_MEMCPY_DEVICE_TO_HOST,
            stream_),
        "aclrtMemcpyAsync(prefill/verify compact pair device-to-host)");
    stream_work_pending_ = true;
    ++stats_.device_to_host_operations;
    stats_.device_to_host_bytes += bytes;
    ++stats_.prefill_verify_d2h_operations_elided;
    stats_.prefill_verify_d2h_padding_bytes +=
        bytes - compact_ordinary_bytes_ - compact_verify_bytes_;
    if (prefill_state_index == 0) {
      ++stats_.prefill_verify_prefill_slot0_windows;
    } else {
      ++stats_.prefill_verify_prefill_slot1_windows;
    }
  }

  void Synchronize() {
    Check(aclrtSynchronizeStream(stream_), "aclrtSynchronizeStream");
    stream_work_pending_ = false;
    ++stats_.stream_synchronizations;
  }

  StatefulStep ReadCompact(
      bool verify,
      std::size_t model_executions,
      std::size_t state_index) const {
    if (state_index >= kCompactSlotCount) {
      throw std::out_of_range("compact read state index is invalid");
    }
    const std::size_t slot_offset = state_index * compact_slot_bytes_;
    const std::int32_t commit_count =
        ReadAt<std::int32_t>(
            compact_, slot_offset + compact_commit_offset_);
    if (commit_count <= 0 ||
        static_cast<std::size_t>(commit_count) > verify_width_) {
      throw std::runtime_error("Target graph returned an invalid commit count");
    }
    std::vector<std::int64_t> tokens(
        static_cast<std::size_t>(commit_count));
    std::memcpy(
        tokens.data(),
        static_cast<const std::byte*>(compact_.host) + slot_offset +
            compact_token_offset_,
        tokens.size() * sizeof(std::int64_t));
    const bool finished =
        ReadAt<std::uint8_t>(
            compact_, slot_offset + compact_finished_offset_) != 0;
    StatefulStep result;
    result.token_ids = std::move(tokens);
    result.model_executions = model_executions;
    result.finished = finished;
    if (verify) {
      const std::int32_t drafted =
          ReadAt<std::int32_t>(
              compact_, slot_offset + compact_drafted_offset_);
      const std::int32_t accepted =
          ReadAt<std::int32_t>(
              compact_, slot_offset + compact_accepted_offset_);
      const std::int32_t rejected =
          ReadAt<std::int32_t>(
              compact_, slot_offset + compact_rejected_offset_);
      if (drafted <= 0 || accepted < 0 || rejected < 0) {
        throw std::runtime_error("verify graph returned negative/zero counters");
      }
      result.drafted_tokens = static_cast<std::size_t>(drafted);
      result.accepted_draft_tokens = static_cast<std::size_t>(accepted);
      result.rejected_draft_tokens = static_cast<std::size_t>(rejected);
    }
    return result;
  }

  StatefulStep ReadCompactAndTrackCarrier(
      bool verify,
      std::size_t model_executions,
      std::size_t state_index) {
    StatefulStep result = ReadCompact(verify, model_executions, state_index);
    if (verify) {
      if (result.accepted_draft_tokens >= verify_width_) {
        throw std::runtime_error(
            "verify accepted count exceeds the feature carrier");
      }
      committed_feature_rows_ = result.accepted_draft_tokens + 1;
      committed_feature_rows_valid_ = true;
    }
    decode_carrier_valid_ =
        result.token_ids.size() == 1 ||
        decode_carrier_policy_ ==
            IncrementalDecodeCarrierPolicy::kLastTokenDeviceCompact;
    if (decode_carrier_valid_) {
      decode_carrier_row_ = result.token_ids.size() - 1;
      decode_carrier_token_id_ = result.token_ids.back();
    } else {
      decode_carrier_row_ = 0;
    }
    return result;
  }

  void Cleanup() noexcept {
    if (stream_ != nullptr && stream_work_pending_) {
      static_cast<void>(aclrtSynchronizeStream(stream_));
      stream_work_pending_ = false;
    }
    for (auto& by_target : initial_draft_plans_) {
      for (auto& plan : by_target) {
        plan.Release();
      }
    }
    initial_draft_plans_.clear();
    for (auto& plan : prefill_head_plans_) {
      plan.Release();
    }
    initial_prefill_plan_.Release();
    for (auto& by_gear : prefill_draft_plans_) {
      for (auto& by_target : by_gear) {
        for (auto& plan : by_target) {
          plan.Release();
        }
      }
    }
    prefill_draft_plans_.clear();
    for (auto& by_rows : verify_draft_plans_) {
      for (auto& by_target : by_rows) {
        for (auto& plan : by_target) {
          plan.Release();
        }
      }
    }
    verify_draft_plans_.clear();
    for (auto& plan : verify_plans_) {
      plan.Release();
    }
    for (auto& by_state : target_step_plans_) {
      for (auto& plan : by_state) {
        plan.Release();
      }
    }
    target_step_plans_.clear();
    for (auto& plan : decode_carrier_plans_) {
      plan.Release();
    }
    for (auto& plan : decode_upload_plans_) {
      plan.Release();
    }
    for (auto& by_slot : prefill_plans_) {
      for (auto& plan : by_slot) {
        plan.Release();
      }
    }
    prefill_plans_.clear();

    target_step_dynamic_control_.Release();
    verify_dynamic_control_.Release();
    for (auto& control : prefill_dynamic_controls_) {
      control.Release();
    }
    prefill_dynamic_controls_.clear();
    verify_ids_.Release();
    committed_input_count_.Release();
    prefill_last_hidden_.Release();
    prefill_features_.Release();
    compact_.Release();
    for (auto& host : extra_prefill_control_host_) {
      host.Release();
    }
    extra_prefill_control_host_.clear();
    for (auto& host : proposal_count_upload_host_) {
      host.Release();
    }
    decode_id_.Release();
    prefill_control_.Release();
    for (auto& state : draft_states_) {
      state.Release();
    }
    for (auto& state : target_states_) {
      state.Release();
    }
    draft_zero_state_.Release();
    target_zero_state_.Release();

    verify_.Release();
    draft_.Release();
    decode_.Release();
    prefill_head_.Release();
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
  bool unified_target_step_ = false;
  IncrementalStateResetPolicy state_reset_policy_ =
      IncrementalStateResetPolicy::kAsyncMemset;
  IncrementalDecodeCarrierPolicy decode_carrier_policy_ =
      IncrementalDecodeCarrierPolicy::kLastTokenDeviceCompact;
  IncrementalDraftFeaturePolicy draft_feature_policy_ =
      IncrementalDraftFeaturePolicy::kFixedVerifyWidth;
  bool profile_model_executions_ = false;
  bool initialized_ = false;
  bool device_set_ = false;
  bool reset_ = false;
  aclrtContext context_ = nullptr;
  aclrtStream stream_ = nullptr;

  ModelSession prefill_;
  ModelSession prefill_head_;
  ModelSession decode_;
  ModelSession draft_;
  ModelSession verify_;
  std::vector<IncrementalModelMemory> memory_;
  std::vector<IncrementalModelExecutionTrace> model_execution_trace_;
  DeviceAllocation shared_model_work_;
  std::array<DeviceAllocation, 5> model_weights_;

  std::size_t sequence_length_ = 0;
  std::size_t prefill_width_ = 0;
  std::size_t verify_width_ = 0;
  std::size_t proposal_width_ = 0;
  std::size_t eos_table_width_ = 0;
  std::size_t feature_width_ = 0;
  std::vector<TensorSpec> target_state_specs_;
  std::vector<TensorSpec> draft_state_specs_;
  std::vector<aclmdlIODims> draft_gear_prefill_;
  std::vector<aclmdlIODims> draft_gear_verify_;
  std::vector<aclmdlIODims> target_step_gears_;

  std::array<StateArena, 2> target_states_;
  std::array<StateArena, 2> draft_states_;
  StateArena target_zero_state_;
  StateArena draft_zero_state_;
  MirrorBuffer prefill_control_;
  MirrorBuffer decode_id_;
  std::vector<HostAllocation> extra_prefill_control_host_;
  std::array<HostAllocation, 2> proposal_count_upload_host_;
  MirrorBuffer compact_;
  DeviceAllocation prefill_features_;
  DeviceAllocation prefill_last_hidden_;
  DeviceAllocation committed_input_count_;
  DeviceAllocation verify_ids_;
  std::vector<DeviceAllocation> prefill_dynamic_controls_;
  DeviceAllocation verify_dynamic_control_;
  DeviceAllocation target_step_dynamic_control_;

  std::size_t compact_token_offset_ = 0;
  std::size_t compact_commit_offset_ = 0;
  std::size_t compact_finished_offset_ = 0;
  std::size_t compact_drafted_offset_ = 0;
  std::size_t compact_accepted_offset_ = 0;
  std::size_t compact_rejected_offset_ = 0;
  std::size_t compact_ordinary_bytes_ = 0;
  std::size_t compact_verify_bytes_ = 0;
  std::size_t compact_slot_bytes_ = 0;
  std::size_t prefill_ids_offset_ = 0;
  std::size_t effective_length_offset_ = 0;
  std::size_t eos_ids_offset_ = 0;
  std::size_t eos_count_offset_ = 0;
  std::size_t proposal_count_offset_ = 0;
  std::size_t prefill_total_count_offset_ = 0;
  std::size_t prefill_base_control_bytes_ = 0;
  std::size_t prefill_count_control_bytes_ = 0;
  std::size_t prefill_proposal_control_bytes_ = 0;
  std::size_t target_step_zero_count_offset_ =
      std::numeric_limits<std::size_t>::max();

  std::vector<std::array<DatasetPlan, 2>> prefill_plans_;
  std::array<DatasetPlan, 2> decode_upload_plans_;
  std::array<DatasetPlan, 2> decode_carrier_plans_;
  std::array<DatasetPlan, 2> verify_plans_;
  std::vector<std::array<DatasetPlan, 2>> target_step_plans_;
  std::vector<std::array<std::array<DatasetPlan, 2>, 2>>
      prefill_draft_plans_;
  std::vector<std::array<std::array<DatasetPlan, 2>, 2>>
      verify_draft_plans_;
  DatasetPlan initial_prefill_plan_;
  std::array<DatasetPlan, 2> prefill_head_plans_;
  std::vector<std::array<DatasetPlan, 2>> initial_draft_plans_;

  std::size_t target_state_index_ = 0;
  std::size_t draft_state_index_ = 0;
  std::size_t prefill_staging_index_ = 0;
  std::size_t prefill_total_token_count_ = 0;
  bool deferred_prefill_pending_ = false;
  bool stream_work_pending_ = false;
  bool proposal_ready_ = false;
  std::size_t prepared_proposal_count_ = 0;
  bool decode_carrier_valid_ = false;
  std::size_t decode_carrier_row_ = 0;
  std::int64_t decode_carrier_token_id_ = 0;
  FeatureSource feature_source_ = FeatureSource::kNone;
  bool proposal_value_valid_ = false;
  std::size_t proposal_value_ = 0;
  std::size_t proposal_count_upload_staging_index_ = 0;
  std::size_t committed_feature_rows_ = 0;
  bool committed_feature_rows_valid_ = false;
  std::int64_t pad_token_id_ = 0;
  bool reset_pending_ = false;
  bool draft_reset_pending_ = false;
  std::vector<std::int64_t> configured_eos_token_ids_;
  std::vector<std::int64_t> device_eos_token_ids_;
  bool device_eos_token_ids_valid_ = false;
  bool eos_control_upload_pending_ = true;
  bool target_step_zero_count_valid_ = false;
  IncrementalAclExecutionStats stats_;
};

AclIncrementalExecutor::AclIncrementalExecutor(
    IncrementalOmPaths model_paths,
    int device_id,
    IncrementalModelProgress progress,
    IncrementalStateResetPolicy state_reset_policy,
    IncrementalDecodeCarrierPolicy decode_carrier_policy,
    bool profile_model_executions,
    IncrementalDraftFeaturePolicy draft_feature_policy)
    : impl_(std::make_unique<Impl>(
          model_paths,
          device_id,
          progress,
          state_reset_policy,
          decode_carrier_policy,
          profile_model_executions,
          draft_feature_policy)) {}

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

std::size_t AclIncrementalExecutor::PrefillChunkDeferred(
    const std::vector<std::int64_t>& token_ids,
    bool prepare_draft,
    std::size_t logical_proposal_count) {
  return impl_->PrefillChunkDeferred(
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

bool AclIncrementalExecutor::supports_prefill_verify_coalescing()
    const noexcept {
  return impl_->supports_prefill_verify_coalescing();
}

std::vector<StatefulStep>
AclIncrementalExecutor::PrefillChunkAndSpeculative(
    const std::vector<std::int64_t>& token_ids,
    std::size_t logical_proposal_count) {
  return impl_->PrefillChunkAndSpeculative(
      token_ids, logical_proposal_count);
}

std::size_t AclIncrementalExecutor::max_speculative_sync_window()
    const noexcept {
  return impl_->max_speculative_sync_window();
}

std::vector<StatefulStep> AclIncrementalExecutor::SpeculativeWindow(
    const std::vector<std::size_t>& logical_proposal_counts) {
  return impl_->SpeculativeWindow(logical_proposal_counts);
}

const std::vector<IncrementalModelMemory>&
AclIncrementalExecutor::model_memory() const noexcept {
  return impl_->model_memory();
}

const std::vector<IncrementalModelExecutionTrace>&
AclIncrementalExecutor::model_execution_trace() const noexcept {
  return impl_->model_execution_trace();
}

const IncrementalAclExecutionStats&
AclIncrementalExecutor::execution_stats() const noexcept {
  return impl_->execution_stats();
}

IncrementalStateResetPolicy
AclIncrementalExecutor::state_reset_policy() const noexcept {
  return impl_->state_reset_policy();
}

IncrementalDecodeCarrierPolicy
AclIncrementalExecutor::decode_carrier_policy() const noexcept {
  return impl_->decode_carrier_policy();
}

IncrementalDraftFeaturePolicy
AclIncrementalExecutor::draft_feature_policy() const noexcept {
  return impl_->draft_feature_policy();
}

bool AclIncrementalExecutor::unified_target_step() const noexcept {
  return impl_->unified_target_step();
}

}  // namespace qwen35::dflash
