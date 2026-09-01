#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef int aclError;
static const aclError ACL_SUCCESS = 0;

typedef enum aclDataType {
  ACL_DT_UNDEFINED = -1,
  ACL_FLOAT = 0,
  ACL_FLOAT16 = 1,
  ACL_INT8 = 2,
  ACL_INT32 = 3,
  ACL_UINT8 = 4,
  ACL_INT16 = 6,
  ACL_UINT16 = 7,
  ACL_UINT32 = 8,
  ACL_INT64 = 9,
  ACL_UINT64 = 10,
  ACL_DOUBLE = 11,
  ACL_BOOL = 12,
} aclDataType;

typedef enum aclrtMemMallocPolicy {
  ACL_MEM_MALLOC_HUGE_FIRST = 0,
  ACL_MEM_MALLOC_HUGE_ONLY = 1,
  ACL_MEM_MALLOC_NORMAL_ONLY = 2,
} aclrtMemMallocPolicy;

typedef enum aclrtMemcpyKind {
  ACL_MEMCPY_HOST_TO_HOST = 0,
  ACL_MEMCPY_HOST_TO_DEVICE = 1,
  ACL_MEMCPY_DEVICE_TO_HOST = 2,
  ACL_MEMCPY_DEVICE_TO_DEVICE = 3,
  ACL_MEMCPY_DEFAULT = 4,
} aclrtMemcpyKind;

typedef void* aclrtContext;
typedef void* aclrtStream;

typedef struct aclmdlIODims {
  char name[128];
  std::size_t dimCount;
  std::int64_t dims[128];
} aclmdlIODims;

typedef struct aclDataBuffer aclDataBuffer;
typedef struct aclmdlDataset aclmdlDataset;
typedef struct aclmdlDesc aclmdlDesc;

aclError aclInit(const char* config_path);
aclError aclFinalize();
aclError aclrtSetDevice(int device_id);
aclError aclrtResetDevice(int device_id);
aclError aclrtCreateContext(aclrtContext* context, int device_id);
aclError aclrtDestroyContext(aclrtContext context);
aclError aclrtSetCurrentContext(aclrtContext context);
aclError aclrtCreateStream(aclrtStream* stream);
aclError aclrtDestroyStream(aclrtStream stream);
aclError aclrtSynchronizeStream(aclrtStream stream);
aclError aclrtMallocHost(void** host_ptr, std::size_t size);
aclError aclrtFreeHost(void* host_ptr);
aclError aclrtMalloc(
    void** device_ptr,
    std::size_t size,
    aclrtMemMallocPolicy policy);
aclError aclrtFree(void* device_ptr);
aclError aclrtMemcpyAsync(
    void* destination,
    std::size_t destination_max,
    const void* source,
    std::size_t count,
    aclrtMemcpyKind kind,
    aclrtStream stream);
aclError aclrtMemsetAsync(
    void* device_ptr,
    std::size_t max_count,
    std::int32_t value,
    std::size_t count,
    aclrtStream stream);

aclError aclmdlLoadFromFile(const char* model_path, std::uint32_t* model_id);
aclError aclmdlLoadFromFileWithMem(
    const char* model_path,
    std::uint32_t* model_id,
    void* work_ptr,
    std::size_t work_size,
    void* weight_ptr,
    std::size_t weight_size);
aclError aclmdlQuerySize(
    const char* model_path,
    std::size_t* work_size,
    std::size_t* weight_size);
aclError aclmdlUnload(std::uint32_t model_id);
aclmdlDesc* aclmdlCreateDesc();
aclError aclmdlDestroyDesc(aclmdlDesc* description);
aclError aclmdlGetDesc(aclmdlDesc* description, std::uint32_t model_id);
std::size_t aclmdlGetNumInputs(const aclmdlDesc* description);
std::size_t aclmdlGetNumOutputs(const aclmdlDesc* description);
aclError aclmdlGetInputDims(
    const aclmdlDesc* description,
    std::size_t index,
    aclmdlIODims* dimensions);
aclError aclmdlGetOutputDims(
    const aclmdlDesc* description,
    std::size_t index,
    aclmdlIODims* dimensions);
aclDataType aclmdlGetInputDataType(
    const aclmdlDesc* description,
    std::size_t index);
aclDataType aclmdlGetOutputDataType(
    const aclmdlDesc* description,
    std::size_t index);
std::size_t aclmdlGetInputSizeByIndex(
    const aclmdlDesc* description,
    std::size_t index);
std::size_t aclmdlGetOutputSizeByIndex(
    const aclmdlDesc* description,
    std::size_t index);
aclError aclmdlGetInputIndexByName(
    const aclmdlDesc* description,
    const char* name,
    std::size_t* index);
aclError aclmdlGetInputDynamicGearCount(
    const aclmdlDesc* description,
    std::size_t index,
    std::size_t* gear_count);
aclError aclmdlGetInputDynamicDims(
    const aclmdlDesc* description,
    std::size_t index,
    aclmdlIODims* dimensions,
    std::size_t gear_count);
aclmdlDataset* aclmdlCreateDataset();
aclError aclmdlDestroyDataset(aclmdlDataset* dataset);
aclDataBuffer* aclCreateDataBuffer(void* data, std::size_t size);
aclError aclDestroyDataBuffer(aclDataBuffer* buffer);
aclError aclmdlAddDatasetBuffer(
    aclmdlDataset* dataset,
    aclDataBuffer* data_buffer);
aclError aclmdlExecuteAsync(
    std::uint32_t model_id,
    const aclmdlDataset* input,
    aclmdlDataset* output,
    aclrtStream stream);
aclError aclmdlSetInputDynamicDims(
    std::uint32_t model_id,
    aclmdlDataset* dataset,
    std::size_t index,
    const aclmdlIODims* dimensions);

#ifdef __cplusplus
}
#endif
