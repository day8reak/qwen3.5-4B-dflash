#include <acl/acl.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <new>
#include <string>
#include <vector>

struct aclDataBuffer {
  void* data;
  std::size_t size;
};

struct aclmdlDataset {
  std::vector<aclDataBuffer*> buffers;
  aclmdlIODims dynamic_dims{};
};

namespace {

enum class Role {
  kIntegrated,
  kTargetPrefill,
  kTargetPrefillHead,
  kTargetDecode,
  kDraftPropose,
  kTargetVerify,
  kTargetStep,
};

struct Spec {
  aclDataType dtype;
  std::vector<std::int64_t> shape;
  const char* name;
};

constexpr std::size_t kSequenceLength = 32;
constexpr std::size_t kIncrementalSequenceLength = 128;
constexpr std::size_t kIntegratedDraftWidth = 15;
constexpr std::size_t kPrefillRows = 64;
constexpr std::size_t kVerifyRows = 16;
constexpr std::size_t kModelWorkBytes = 64;
constexpr std::size_t kModelWeightBytes = 256;
constexpr std::size_t kPrefillHeadWeightBytes = 64;

const Spec kTargetConv{ACL_FLOAT16, {2, 1, 8, 4}, "target_conv_state"};
const Spec kTargetRecurrent{
    ACL_FLOAT, {2, 1, 2, 4, 4}, "target_recurrent_state"};
const Spec kTargetKey{
    ACL_FLOAT16, {8, 2, 1, 64, 16}, "target_key_cache"};
const Spec kTargetValue{
    ACL_FLOAT16, {8, 2, 1, 64, 16}, "target_value_cache"};
const Spec kTargetCursor{ACL_INT64, {1}, "logical_target_cursor"};
const Spec kDraftKey{
    ACL_FLOAT16,
    {6, 1, 2, static_cast<std::int64_t>(kIncrementalSequenceLength), 4},
    "draft_key_cache"};
const Spec kDraftValue{
    ACL_FLOAT16,
    {6, 1, 2, static_cast<std::int64_t>(kIncrementalSequenceLength), 4},
    "draft_value_cache"};
const Spec kDraftCursor{ACL_INT64, {1}, "logical_draft_cursor"};

std::map<std::uint32_t, Role> g_models;
std::uint32_t g_next_model_id = 1;
void* g_incremental_shared_work = nullptr;
std::vector<void*> g_incremental_weights;
std::vector<const void*> g_pending_h2d_sources;

Role RoleFromPath(const char* path) {
  const std::string value = path == nullptr ? "" : path;
  if (value.find("target-prefill-head") != std::string::npos) {
    return Role::kTargetPrefillHead;
  }
  if (value.find("target-prefill") != std::string::npos) {
    return Role::kTargetPrefill;
  }
  if (value.find("target-decode1") != std::string::npos) {
    return Role::kTargetDecode;
  }
  if (value.find("draft-propose") != std::string::npos) {
    return Role::kDraftPropose;
  }
  if (value.find("target-verify-commit-dynamic") != std::string::npos) {
    return Role::kTargetStep;
  }
  if (value.find("target-verify-commit") != std::string::npos) {
    return Role::kTargetVerify;
  }
  return Role::kIntegrated;
}

std::size_t TypeBytes(aclDataType dtype) {
  switch (dtype) {
    case ACL_FLOAT:
    case ACL_INT32:
    case ACL_UINT32:
      return 4;
    case ACL_FLOAT16:
    case ACL_INT16:
    case ACL_UINT16:
      return 2;
    case ACL_INT64:
    case ACL_UINT64:
    case ACL_DOUBLE:
      return 8;
    case ACL_INT8:
    case ACL_UINT8:
    case ACL_BOOL:
      return 1;
    default:
      return 0;
  }
}

std::size_t Bytes(const Spec& spec) {
  std::size_t result = TypeBytes(spec.dtype);
  for (const std::int64_t raw : spec.shape) {
    const std::size_t value = raw == -1
        ? (std::string(spec.name) == "verify_input_ids"
               ? kVerifyRows
               : kIncrementalSequenceLength)
        : static_cast<std::size_t>(raw);
    result *= value;
  }
  return result;
}

const std::vector<Spec>& Inputs(Role role) {
  static const std::vector<Spec> integrated{
      {ACL_INT64, {1, 32}, "input_ids"},
      {ACL_INT64, {1, 32}, "attention_mask"},
  };
  static const std::vector<Spec> prefill{
      {ACL_INT64, {1, 64}, "input_ids"},
      {ACL_INT16, {1}, "effective_length"},
      kTargetConv,
      kTargetRecurrent,
      kTargetKey,
      kTargetValue,
      kTargetCursor,
  };
  static const std::vector<Spec> prefill_head{
      {ACL_FLOAT16, {1, 1, 4}, "last_hidden"},
      {ACL_INT64, {4}, "eos_token_ids"},
      {ACL_INT32, {1}, "eos_token_count"},
  };
  static const std::vector<Spec> decode{
      {ACL_INT64, {1, 1}, "input_ids"},
      {ACL_INT64, {4}, "eos_token_ids"},
      {ACL_INT32, {1}, "eos_token_count"},
      kTargetConv,
      kTargetRecurrent,
      kTargetKey,
      kTargetValue,
      kTargetCursor,
  };
  static const std::vector<Spec> draft{
      {ACL_FLOAT16, {1, -1, 8}, "target_feature_tail"},
      {ACL_INT32, {1}, "committed_input_count"},
      {ACL_INT64, {1, 16}, "previous_committed_token_ids"},
      {ACL_INT32, {1}, "previous_commit_count"},
      {ACL_INT32, {1}, "logical_proposal_count"},
      kDraftKey,
      kDraftValue,
      kDraftCursor,
      {ACL_UINT64, {64}, "ascend_mbatch_shape_data"},
  };
  static const std::vector<Spec> verify{
      {ACL_INT64, {1, 16}, "verify_input_ids"},
      {ACL_INT32, {1}, "logical_proposal_count"},
      {ACL_INT64, {4}, "eos_token_ids"},
      {ACL_INT32, {1}, "eos_token_count"},
      kTargetConv,
      kTargetRecurrent,
      kTargetKey,
      kTargetValue,
      kTargetCursor,
  };
  static const std::vector<Spec> target_step{
      {ACL_INT64, {1, -1}, "verify_input_ids"},
      {ACL_INT32, {1}, "logical_proposal_count"},
      {ACL_INT64, {4}, "eos_token_ids"},
      {ACL_INT32, {1}, "eos_token_count"},
      kTargetConv,
      kTargetRecurrent,
      kTargetKey,
      kTargetValue,
      kTargetCursor,
      {ACL_UINT64, {64}, "ascend_mbatch_shape_data"},
  };
  switch (role) {
    case Role::kTargetPrefill:
      return prefill;
    case Role::kTargetPrefillHead:
      return prefill_head;
    case Role::kTargetDecode:
      return decode;
    case Role::kDraftPropose:
      return draft;
    case Role::kTargetVerify:
      return verify;
    case Role::kTargetStep:
      return target_step;
    case Role::kIntegrated:
      return integrated;
  }
  return integrated;
}

const std::vector<Spec>& Outputs(Role role) {
  static const std::vector<Spec> integrated{
      {ACL_INT64, {1, 32}, "target_top1"},
      {ACL_INT64, {1, 15}, "draft_top1"},
  };
  static const std::vector<Spec> prefill{
      {ACL_FLOAT16, {1, 1, 4}, "last_hidden"},
      {ACL_FLOAT16, {1, 64, 8}, "target_feature_tail"},
      {ACL_INT32, {1}, "committed_input_count"},
      kTargetConv,
      kTargetRecurrent,
      kTargetKey,
      kTargetValue,
      kTargetCursor,
  };
  static const std::vector<Spec> prefill_head{
      {ACL_INT64, {1, 16}, "committed_token_ids"},
      {ACL_INT32, {1}, "commit_count"},
      {ACL_BOOL, {1}, "finished"},
  };
  static const std::vector<Spec> decode{
      {ACL_INT64, {1, 16}, "committed_token_ids"},
      {ACL_INT32, {1}, "commit_count"},
      {ACL_BOOL, {1}, "finished"},
      kTargetConv,
      kTargetRecurrent,
      kTargetKey,
      kTargetValue,
      kTargetCursor,
  };
  static const std::vector<Spec> draft{
      {ACL_INT64, {1, 16}, "verify_input_ids"},
      kDraftKey,
      kDraftValue,
      kDraftCursor,
  };
  static const std::vector<Spec> verify{
      {ACL_INT64, {1, 16}, "committed_token_ids"},
      {ACL_INT32, {1}, "commit_count"},
      {ACL_INT32, {1}, "drafted_count"},
      {ACL_INT32, {1}, "accepted_count"},
      {ACL_INT32, {1}, "rejected_count"},
      kTargetConv,
      kTargetRecurrent,
      {ACL_FLOAT16, {1, 16, 8}, "target_feature_tail"},
      {ACL_INT32, {1}, "committed_input_count"},
      kTargetCursor,
      {ACL_BOOL, {1}, "finished"},
      kTargetKey,
      kTargetValue,
  };
  switch (role) {
    case Role::kTargetPrefill:
      return prefill;
    case Role::kTargetPrefillHead:
      return prefill_head;
    case Role::kTargetDecode:
      return decode;
    case Role::kDraftPropose:
      return draft;
    case Role::kTargetVerify:
    case Role::kTargetStep:
      return verify;
    case Role::kIntegrated:
      return integrated;
  }
  return integrated;
}

aclError SetDims(aclmdlIODims* dimensions, const Spec& spec) {
  if (dimensions == nullptr || spec.shape.size() > 128) {
    return 1;
  }
  std::memset(dimensions, 0, sizeof(*dimensions));
  std::strncpy(dimensions->name, spec.name, sizeof(dimensions->name) - 1);
  dimensions->dimCount = spec.shape.size();
  std::copy(spec.shape.begin(), spec.shape.end(), dimensions->dims);
  return ACL_SUCCESS;
}

template <typename Value>
Value Scalar(const aclDataBuffer* buffer) {
  Value value{};
  std::memcpy(&value, buffer->data, sizeof(value));
  return value;
}

template <typename Value>
void SetScalar(aclDataBuffer* buffer, Value value) {
  std::memcpy(buffer->data, &value, sizeof(value));
}

void Copy(aclDataBuffer* output, const aclDataBuffer* input) {
  std::memcpy(output->data, input->data, std::min(output->size, input->size));
}

bool IsEos(
    std::int64_t token,
    const aclDataBuffer* eos_ids,
    const aclDataBuffer* eos_count) {
  const auto count = Scalar<std::int32_t>(eos_count);
  const auto* values = static_cast<const std::int64_t*>(eos_ids->data);
  for (std::int32_t index = 0; index < count; ++index) {
    if (values[index] == token) {
      return true;
    }
  }
  return false;
}

void FillCommitted(aclDataBuffer* output, const std::vector<std::int64_t>& values) {
  auto* tokens = static_cast<std::int64_t*>(output->data);
  std::fill_n(tokens, kVerifyRows, 0);
  std::copy(values.begin(), values.end(), tokens);
}

aclError ExecuteIntegrated(const aclmdlDataset* input, aclmdlDataset* output) {
  if (input->buffers.size() != 2 || output->buffers.size() != 2) {
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
  for (std::size_t index = 0; index < kIntegratedDraftWidth; ++index) {
    draft[index] = ids[prefix - 1] + static_cast<std::int64_t>(index) + 1;
  }
  return ACL_SUCCESS;
}

aclError ExecutePrefill(const aclmdlDataset* input, aclmdlDataset* output) {
  const auto length = Scalar<std::int16_t>(input->buffers[1]);
  if (length <= 0 || length > static_cast<std::int16_t>(kPrefillRows)) {
    return 1;
  }
  const auto* ids = static_cast<const std::int64_t*>(input->buffers[0]->data);
  SetScalar<std::uint16_t>(
      output->buffers[0], static_cast<std::uint16_t>(ids[length - 1] + 1));
  std::memset(output->buffers[1]->data, 0, output->buffers[1]->size);
  SetScalar<std::int32_t>(output->buffers[2], length);
  for (std::size_t index = 0; index < 4; ++index) {
    Copy(output->buffers[3 + index], input->buffers[2 + index]);
  }
  SetScalar<std::int64_t>(
      output->buffers[7],
      Scalar<std::int64_t>(input->buffers[6]) + length);
  return ACL_SUCCESS;
}

aclError ExecutePrefillHead(
    const aclmdlDataset* input,
    aclmdlDataset* output) {
  const std::int64_t token = Scalar<std::uint16_t>(input->buffers[0]);
  FillCommitted(output->buffers[0], {token});
  SetScalar<std::int32_t>(output->buffers[1], 1);
  SetScalar<std::uint8_t>(
      output->buffers[2], IsEos(token, input->buffers[1], input->buffers[2]));
  return ACL_SUCCESS;
}

aclError ExecuteDecode(const aclmdlDataset* input, aclmdlDataset* output) {
  const std::int64_t token = Scalar<std::int64_t>(input->buffers[0]) + 1;
  FillCommitted(output->buffers[0], {token});
  SetScalar<std::int32_t>(output->buffers[1], 1);
  SetScalar<std::uint8_t>(
      output->buffers[2], IsEos(token, input->buffers[1], input->buffers[2]));
  for (std::size_t index = 0; index < 4; ++index) {
    Copy(output->buffers[3 + index], input->buffers[3 + index]);
  }
  SetScalar<std::int64_t>(
      output->buffers[7], Scalar<std::int64_t>(input->buffers[7]) + 1);
  return ACL_SUCCESS;
}

aclError ExecuteDraft(const aclmdlDataset* input, aclmdlDataset* output) {
  if (input->dynamic_dims.dimCount < 3) {
    return 1;
  }
  const auto feature_rows = input->dynamic_dims.dims[1];
  const auto committed_input_count = Scalar<std::int32_t>(input->buffers[1]);
  if (!((feature_rows >= 1 && feature_rows <= 16) ||
        feature_rows == 64 || feature_rows == 128) ||
      committed_input_count <= 0 || committed_input_count > feature_rows ||
      input->buffers[0]->size <
          static_cast<std::size_t>(feature_rows) * 8 * sizeof(std::uint16_t)) {
    return 1;
  }
  const auto commit_count = Scalar<std::int32_t>(input->buffers[3]);
  const auto proposal_count = Scalar<std::int32_t>(input->buffers[4]);
  if (commit_count <= 0 || commit_count > 16 || proposal_count <= 0 ||
      proposal_count > 15) {
    return 1;
  }
  const auto* previous =
      static_cast<const std::int64_t*>(input->buffers[2]->data);
  const std::int64_t anchor = previous[commit_count - 1];
  auto* verify = static_cast<std::int64_t*>(output->buffers[0]->data);
  verify[0] = anchor;
  for (std::size_t index = 1; index < kVerifyRows; ++index) {
    verify[index] = anchor + static_cast<std::int64_t>(index);
  }
  Copy(output->buffers[1], input->buffers[5]);
  Copy(output->buffers[2], input->buffers[6]);
  SetScalar<std::int64_t>(
      output->buffers[3],
      Scalar<std::int64_t>(input->buffers[7]) +
          Scalar<std::int32_t>(input->buffers[1]));
  return ACL_SUCCESS;
}

aclError ExecuteVerify(const aclmdlDataset* input, aclmdlDataset* output) {
  const auto proposal_count = Scalar<std::int32_t>(input->buffers[1]);
  if (proposal_count < 0 || proposal_count > 15) {
    return 1;
  }
  if (input->dynamic_dims.dimCount != 0) {
    if (input->dynamic_dims.dimCount < 2 ||
        input->dynamic_dims.dims[1] != proposal_count + 1 ||
        input->buffers[0]->size <
            static_cast<std::size_t>(proposal_count + 1) *
                sizeof(std::int64_t)) {
      return 1;
    }
  } else if (proposal_count == 0) {
    return 1;
  }
  const auto* verify = static_cast<const std::int64_t*>(input->buffers[0]->data);
  std::vector<std::int64_t> proposals;
  for (std::int32_t index = 0; index < proposal_count; ++index) {
    const std::int64_t token = verify[index + 1];
    proposals.push_back(token);
    if (IsEos(token, input->buffers[2], input->buffers[3])) {
      break;
    }
  }
  std::size_t accepted = 0;
  while (accepted < proposals.size() &&
         proposals[accepted] == verify[accepted] + 1) {
    ++accepted;
  }
  std::vector<std::int64_t> committed(
      proposals.begin(),
      proposals.begin() + static_cast<std::ptrdiff_t>(accepted));
  const bool accepted_eos =
      !committed.empty() &&
      IsEos(committed.back(), input->buffers[2], input->buffers[3]);
  if (!accepted_eos) {
    committed.push_back(verify[accepted] + 1);
  }
  FillCommitted(output->buffers[0], committed);
  SetScalar<std::int32_t>(output->buffers[1], committed.size());
  SetScalar<std::int32_t>(output->buffers[2], proposals.size());
  SetScalar<std::int32_t>(output->buffers[3], accepted);
  SetScalar<std::int32_t>(
      output->buffers[4], proposals.size() - accepted);
  Copy(output->buffers[5], input->buffers[4]);
  Copy(output->buffers[6], input->buffers[5]);
  std::memset(output->buffers[7]->data, 0, output->buffers[7]->size);
  SetScalar<std::int32_t>(output->buffers[8], accepted + 1);
  SetScalar<std::int64_t>(
      output->buffers[9],
      Scalar<std::int64_t>(input->buffers[8]) + accepted + 1);
  SetScalar<std::uint8_t>(
      output->buffers[10],
      IsEos(committed.back(), input->buffers[2], input->buffers[3]));
  Copy(output->buffers[11], input->buffers[6]);
  Copy(output->buffers[12], input->buffers[7]);
  return ACL_SUCCESS;
}

}  // namespace

struct aclmdlDesc {
  Role role = Role::kIntegrated;
};

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

aclError aclrtSynchronizeStream(aclrtStream) {
  g_pending_h2d_sources.clear();
  return ACL_SUCCESS;
}

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
  const std::size_t aligned_size = (size + 63) / 64 * 64;
  *device_ptr = std::aligned_alloc(64, aligned_size);
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
    aclrtMemcpyKind kind,
    aclrtStream) {
  if (destination == nullptr || source == nullptr || count > destination_max) {
    return 1;
  }
  if (kind == ACL_MEMCPY_HOST_TO_DEVICE) {
    if (std::find(
            g_pending_h2d_sources.begin(),
            g_pending_h2d_sources.end(),
            source) != g_pending_h2d_sources.end()) {
      return 1;
    }
    g_pending_h2d_sources.push_back(source);
  }
  std::memcpy(destination, source, count);
  return ACL_SUCCESS;
}

aclError aclrtMemsetAsync(
    void* device_ptr,
    std::size_t max_count,
    std::int32_t value,
    std::size_t count,
    aclrtStream) {
  if (device_ptr == nullptr || count > max_count) {
    return 1;
  }
  std::memset(device_ptr, value, count);
  return ACL_SUCCESS;
}

aclError aclmdlLoadFromFile(const char* path, std::uint32_t* model_id) {
  if (model_id == nullptr || RoleFromPath(path) != Role::kIntegrated) {
    return 1;
  }
  *model_id = g_next_model_id++;
  g_models[*model_id] = RoleFromPath(path);
  return ACL_SUCCESS;
}

aclError aclmdlLoadFromFileWithMem(
    const char* path,
    std::uint32_t* model_id,
    void* work_ptr,
    std::size_t work_size,
    void* weight_ptr,
    std::size_t weight_size) {
  const std::size_t required_weight =
      RoleFromPath(path) == Role::kTargetPrefillHead
      ? kPrefillHeadWeightBytes
      : kModelWeightBytes;
  if (work_ptr == nullptr || work_size < kModelWorkBytes ||
      weight_ptr == nullptr || weight_size < required_weight) {
    return 1;
  }
  if (g_incremental_shared_work == nullptr) {
    g_incremental_shared_work = work_ptr;
  } else if (g_incremental_shared_work != work_ptr) {
    return 1;
  }
  if (std::find(
          g_incremental_weights.begin(),
          g_incremental_weights.end(),
          weight_ptr) != g_incremental_weights.end()) {
    return 1;
  }
  g_incremental_weights.push_back(weight_ptr);
  if (model_id == nullptr) {
    return 1;
  }
  *model_id = g_next_model_id++;
  g_models[*model_id] = RoleFromPath(path);
  return ACL_SUCCESS;
}

aclError aclmdlQuerySize(
    const char* path, std::size_t* work_size, std::size_t* weight_size) {
  if (work_size == nullptr || weight_size == nullptr) {
    return 1;
  }
  *work_size = kModelWorkBytes;
  *weight_size = RoleFromPath(path) == Role::kTargetPrefillHead
      ? kPrefillHeadWeightBytes
      : kModelWeightBytes;
  return ACL_SUCCESS;
}

aclError aclmdlUnload(std::uint32_t model_id) {
  g_models.erase(model_id);
  if (g_models.empty()) {
    g_incremental_shared_work = nullptr;
    g_incremental_weights.clear();
  }
  return ACL_SUCCESS;
}

aclmdlDesc* aclmdlCreateDesc() { return new (std::nothrow) aclmdlDesc(); }

aclError aclmdlDestroyDesc(aclmdlDesc* description) {
  delete description;
  return ACL_SUCCESS;
}

aclError aclmdlGetDesc(aclmdlDesc* description, std::uint32_t model_id) {
  const auto iterator = g_models.find(model_id);
  if (description == nullptr || iterator == g_models.end()) {
    return 1;
  }
  description->role = iterator->second;
  return ACL_SUCCESS;
}

std::size_t aclmdlGetNumInputs(const aclmdlDesc* description) {
  return description == nullptr ? 0 : Inputs(description->role).size();
}

std::size_t aclmdlGetNumOutputs(const aclmdlDesc* description) {
  return description == nullptr ? 0 : Outputs(description->role).size();
}

aclError aclmdlGetInputDims(
    const aclmdlDesc* description,
    std::size_t index,
    aclmdlIODims* dimensions) {
  if (description == nullptr || index >= Inputs(description->role).size()) {
    return 1;
  }
  return SetDims(dimensions, Inputs(description->role)[index]);
}

aclError aclmdlGetOutputDims(
    const aclmdlDesc* description,
    std::size_t index,
    aclmdlIODims* dimensions) {
  if (description == nullptr || index >= Outputs(description->role).size()) {
    return 1;
  }
  return SetDims(dimensions, Outputs(description->role)[index]);
}

aclDataType aclmdlGetInputDataType(
    const aclmdlDesc* description, std::size_t index) {
  return description != nullptr && index < Inputs(description->role).size()
      ? Inputs(description->role)[index].dtype
      : ACL_DT_UNDEFINED;
}

aclDataType aclmdlGetOutputDataType(
    const aclmdlDesc* description, std::size_t index) {
  return description != nullptr && index < Outputs(description->role).size()
      ? Outputs(description->role)[index].dtype
      : ACL_DT_UNDEFINED;
}

std::size_t aclmdlGetInputSizeByIndex(
    const aclmdlDesc* description, std::size_t index) {
  return description != nullptr && index < Inputs(description->role).size()
      ? Bytes(Inputs(description->role)[index])
      : 0;
}

std::size_t aclmdlGetOutputSizeByIndex(
    const aclmdlDesc* description, std::size_t index) {
  return description != nullptr && index < Outputs(description->role).size()
      ? Bytes(Outputs(description->role)[index])
      : 0;
}

aclError aclmdlGetInputIndexByName(
    const aclmdlDesc* description,
    const char* name,
    std::size_t* index) {
  if (description == nullptr || name == nullptr || index == nullptr) {
    return 1;
  }
  const auto& inputs = Inputs(description->role);
  for (std::size_t candidate = 0; candidate < inputs.size(); ++candidate) {
    if (inputs[candidate].name == std::string(name)) {
      *index = candidate;
      return ACL_SUCCESS;
    }
  }
  return 1;
}

aclError aclmdlGetInputDynamicGearCount(
    const aclmdlDesc* description,
    std::size_t,
    std::size_t* gear_count) {
  if (description == nullptr || gear_count == nullptr ||
      (description->role != Role::kDraftPropose &&
       description->role != Role::kTargetStep)) {
    return 1;
  }
  *gear_count = description->role == Role::kDraftPropose ? 18 : 16;
  return ACL_SUCCESS;
}

aclError aclmdlGetInputDynamicDims(
    const aclmdlDesc* description,
    std::size_t,
    aclmdlIODims* dimensions,
    std::size_t gear_count) {
  if (description == nullptr || dimensions == nullptr ||
      (description->role != Role::kDraftPropose &&
       description->role != Role::kTargetStep)) {
    return 1;
  }
  const std::size_t expected_gears =
      description->role == Role::kDraftPropose ? 18 : 16;
  if (gear_count != expected_gears) {
    return 1;
  }
  for (std::size_t gear = 0; gear < expected_gears; ++gear) {
    const std::int64_t rows = description->role == Role::kDraftPropose
        ? (gear < 16
               ? static_cast<std::int64_t>(gear + 1)
               : static_cast<std::int64_t>((gear - 15) * 64))
        : static_cast<std::int64_t>(gear + 1);
    std::memset(&dimensions[gear], 0, sizeof(aclmdlIODims));
    std::vector<std::int64_t> flat;
    const auto& inputs = Inputs(description->role);
    for (std::size_t input = 0; input + 1 < inputs.size(); ++input) {
      for (const std::int64_t value : inputs[input].shape) {
        flat.push_back(value == -1 ? rows : value);
      }
    }
    dimensions[gear].dimCount = flat.size();
    std::copy(flat.begin(), flat.end(), dimensions[gear].dims);
  }
  return ACL_SUCCESS;
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

aclError aclmdlSetInputDynamicDims(
    std::uint32_t model_id,
    aclmdlDataset* dataset,
    std::size_t index,
    const aclmdlIODims* dimensions) {
  const auto iterator = g_models.find(model_id);
  if (iterator == g_models.end() || dataset == nullptr || dimensions == nullptr) {
    return 1;
  }
  const std::size_t expected_index =
      iterator->second == Role::kDraftPropose
      ? 8
      : (iterator->second == Role::kTargetStep ? 9 : 999);
  if (index != expected_index) {
    return 1;
  }
  dataset->dynamic_dims = *dimensions;
  return ACL_SUCCESS;
}

aclError aclmdlExecuteAsync(
    std::uint32_t model_id,
    const aclmdlDataset* input,
    aclmdlDataset* output,
    aclrtStream) {
  const auto iterator = g_models.find(model_id);
  if (iterator == g_models.end() || input == nullptr || output == nullptr) {
    return 1;
  }
  const auto aligned = [](const aclDataBuffer* buffer) {
    return buffer != nullptr && buffer->data != nullptr &&
        reinterpret_cast<std::uintptr_t>(buffer->data) % 64 == 0;
  };
  if (!std::all_of(input->buffers.begin(), input->buffers.end(), aligned) ||
      !std::all_of(output->buffers.begin(), output->buffers.end(), aligned)) {
    return 1;
  }
  switch (iterator->second) {
    case Role::kIntegrated:
      return ExecuteIntegrated(input, output);
    case Role::kTargetPrefill:
      return ExecutePrefill(input, output);
    case Role::kTargetPrefillHead:
      return ExecutePrefillHead(input, output);
    case Role::kTargetDecode:
      return ExecuteDecode(input, output);
    case Role::kDraftPropose:
      return ExecuteDraft(input, output);
    case Role::kTargetVerify:
    case Role::kTargetStep:
      return ExecuteVerify(input, output);
  }
  return 1;
}

}  // extern "C"
