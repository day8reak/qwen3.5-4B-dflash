#pragma once

#include <filesystem>
#include <string>
#include <string_view>

namespace qwen35::dflash {

std::string Sha256(std::string_view value);
std::string Sha256File(const std::filesystem::path& path);

}  // namespace qwen35::dflash
