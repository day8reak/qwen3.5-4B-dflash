#pragma once

#include "qwen35_dflash/chunk.hpp"

namespace qwen35::dflash {
struct ProfileOptions {
  std::string stage, mode = "dflash", metrics = "PipeUtilization";
  std::filesystem::path output;
  std::size_t warmup = 1;
  bool audit_draft_inputs = false;
  int device_id = 0;
};
// Standard POSIX socket barriers only; msprof CLI is owned by the parent.
std::string ProfileChunk(AclChunkExecutor&, const std::vector<std::int64_t>&,
                         const GenerationOptions&, const ProfileOptions&);
void ValidateProfileOptions(const ProfileOptions&);
}  // namespace qwen35::dflash
