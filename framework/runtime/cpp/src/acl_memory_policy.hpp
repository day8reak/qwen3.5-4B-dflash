#pragma once

#include <acl/acl.h>

#ifndef QWEN35_DFLASH_DEVICE_MEMORY_POLICY_HUGE_FIRST
#define QWEN35_DFLASH_DEVICE_MEMORY_POLICY_HUGE_FIRST 0
#endif

namespace qwen35::dflash {

#if QWEN35_DFLASH_DEVICE_MEMORY_POLICY_HUGE_FIRST
inline constexpr aclrtMemMallocPolicy kDeviceMemoryAllocationPolicy =
    ACL_MEM_MALLOC_HUGE_FIRST;
inline constexpr const char* kDeviceMemoryAllocationPolicyName =
    "huge-first";
#else
inline constexpr aclrtMemMallocPolicy kDeviceMemoryAllocationPolicy =
    ACL_MEM_MALLOC_NORMAL_ONLY;
inline constexpr const char* kDeviceMemoryAllocationPolicyName =
    "normal-only";
#endif

}  // namespace qwen35::dflash
