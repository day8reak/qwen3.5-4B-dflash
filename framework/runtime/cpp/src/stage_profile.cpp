#include "qwen35_dflash/stage_profile.hpp"

#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>

namespace qwen35::dflash {
namespace {
using Clock = std::chrono::steady_clock;
void Require(bool ok, const char* why) {
  if (!ok) throw std::runtime_error(why);
}
std::string Quote(const std::string& text) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : text) {
    if (c == '"' || c == '\\')
      out << '\\' << c;
    else if (c < 32)
      out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c)
          << std::dec;
    else
      out << c;
  }
  out << '"';
  return out.str();
}
std::vector<std::string> Stages(const std::string& mode) {
  return mode == "ordinary"
             ? std::vector<std::string>{"prefill", "decode"}
             : std::vector<std::string>{"prefill", "draft", "verify"};
}
class Channel {
 public:
  Channel() {
    const char* value = std::getenv("DFLASH_MSPROF_CONTROL_FD");
    Require(value != nullptr,
            "stage profiling requires tools/run_msprof.sh controller");
    std::size_t used = 0;
    fd_ = std::stoi(value, &used);
    Require(used == std::string(value).size() && fd_ >= 0,
            "invalid profile control FD");
    const char* raw_timeout = std::getenv("DFLASH_MSPROF_CONTROL_TIMEOUT");
    const double timeout = raw_timeout ? std::stod(raw_timeout) : 2400;
    Require(std::isfinite(timeout) && timeout > 0 && timeout <= 1e8,
            "invalid profile control timeout");
    timeval tv{};
    tv.tv_sec = static_cast<long>(timeout);
    tv.tv_usec = static_cast<long>((timeout - tv.tv_sec) * 1e6);
    if (tv.tv_sec == 0 && tv.tv_usec == 0) tv.tv_usec = 1;
    Require(
        setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) == 0 &&
            setsockopt(fd_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) == 0 &&
            fcntl(fd_, F_SETFD, FD_CLOEXEC) == 0,
        "cannot configure profile socket");
    unsetenv("DFLASH_MSPROF_CONTROL_FD");
  }
  ~Channel() {
    if (fd_ >= 0) close(fd_);
  }
  void Exchange(const std::string& message, const char* expected) {
    const std::string line = message + '\n';
    std::size_t offset = 0;
    while (offset < line.size()) {
      auto n =
          send(fd_, line.data() + offset, line.size() - offset, MSG_NOSIGNAL);
      if (n < 0 && errno == EINTR) continue;
      Require(n > 0, "profile controller send failed or timed out");
      offset += static_cast<std::size_t>(n);
    }
    std::string response;
    for (;;) {
      char c = 0;
      const auto n = recv(fd_, &c, 1, 0);
      if (n < 0 && errno == EINTR) continue;
      Require(n == 1, "profile controller disconnected or timed out");
      if (c == '\n') break;
      Require(response.size() < 4096, "profile response exceeds limit");
      if (c != ' ' && c != '\t' && c != '\r') response += c;
    }
    Require(response == std::string("{\"event\":\"") + expected + "\"}",
            "unexpected profile controller acknowledgement");
  }

 private:
  int fd_ = -1;
};
std::string Scope(const std::string& stage) {
  if (stage == "prefill")
    return "one complete Target prefill, all 64-row prompt chunks and fused "
           "Top1; reset outside; no Draft execution";
  if (stage == "decode")
    return "one target_decode.om call with one real token, fused Top1 and "
           "state update; prefill/cache setup outside";
  if (stage == "draft")
    return "one draft.om call: final prompt feature projection, Draft KV "
           "append and one proposal block with Draft Top1; earlier prompt "
           "chunks outside";
  return "one target_verify.om call: first GDR, Target Top1, acceptance, "
         "second GDR and committed states; prefill/Draft/block preparation and "
         "host publication outside";
}
}  // namespace

void ValidateProfileOptions(const ProfileOptions& p) {
  Require(p.mode == "ordinary" || p.mode == "dflash", "invalid profile mode");
  const auto stages = Stages(p.mode);
  Require(p.stage == "all" ||
              std::find(stages.begin(), stages.end(), p.stage) != stages.end(),
          "invalid C++ stage for profile mode");
  Require(p.metrics == "PipeUtilization" || p.metrics == "Memory" ||
              p.metrics == "MemoryUB",
          "invalid profiling metrics");
  Require(!p.output.empty() && !std::filesystem::exists(p.output) &&
              !std::filesystem::is_symlink(p.output),
          "profile output must be a new directory");
  const auto mode = std::getenv("PROFILING_MODE");
  Require(mode && std::string(mode) == "dynamic" &&
              std::getenv("DFLASH_MSPROF_CONTROL_FD"),
          "use tools/run_msprof.sh --profile-backend cpp --profile-stage");
  const auto simulation = std::getenv("ASCEND310P_SIMULATION_ONLY");
  Require(!simulation || std::string(simulation) != "1",
          "simulation-only profile cannot measure NPU performance");
}

std::string ProfileChunk(AclChunkExecutor& executor,
                         const std::vector<std::int64_t>& prompt,
                         const GenerationOptions& generation,
                         const ProfileOptions& p) {
  Require(!prompt.empty() && prompt.size() <= executor.sequence_length(),
          "invalid profile prompt capacity");
  const auto token_ok = [&](std::int64_t t) {
    Require(t >= 0 && t < executor.vocabulary_size(),
            "profile token outside vocabulary");
  };
  for (auto token : prompt) token_ok(token);
  token_ok(generation.pad_token_id);
  const std::set<std::int64_t> eos(generation.eos_token_ids.begin(),
                                   generation.eos_token_ids.end());
  const auto stages =
      p.stage == "all" ? Stages(p.mode) : std::vector<std::string>{p.stage};
  const auto proposal_count = std::min({
      generation.max_draft_tokens, executor.draft_width(),
      generation.max_new_tokens > 0 ? generation.max_new_tokens - 1 : 0});
  if (p.mode == "dflash" && p.stage != "prefill")
    Require(proposal_count > 0, "Draft/verify profiling requires a nonempty proposal budget");
  const std::size_t extra = (p.stage == "prefill" || p.stage == "draft")
                                ? 0
                                : (p.mode == "ordinary" ? 1 : proposal_count + 1);
  Require(prompt.size() + extra <= executor.sequence_length(),
          "profile requires prompt plus full stage input capacity");
  Channel channel;
  std::vector<std::string> reports;
  for (const auto& stage : stages) {
    const auto output = p.stage == "all" ? p.output / stage : p.output;
    std::cerr << "[stage-profile] preparing " << p.mode << " stage=" << stage
              << " warmup=" << p.warmup << '\n';
    double elapsed = 0;
    std::map<std::string, std::size_t> captured;
    auto invoke = [&](bool measured, const std::function<void()>& work) {
      if (!measured) {
        work();
        return;
      }
      std::map<std::string, std::size_t> before;
      for (const auto& item : executor.stage_ms())
        before[item.first] = item.second.size();
      executor.Synchronize();
      channel.Exchange(
          "{\"event\":\"ready\",\"pid\":" + std::to_string(getpid()) +
              ",\"stage\":" + Quote(stage) +
              ",\"output\":" + Quote(output.string()) +
              ",\"device_id\":" + std::to_string(p.device_id) +
              ",\"metrics\":" + Quote(p.metrics) + "}",
          "started");
      const auto start = Clock::now();
      try {
        work();
        executor.Synchronize();
      } catch (...) {
        executor.Abort();
        channel.Exchange("{\"event\":\"done\",\"success\":false}", "stopped");
        throw;
      }
      elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start)
                    .count();
      channel.Exchange("{\"event\":\"done\",\"success\":true}", "stopped");
      for (const auto& item : executor.stage_ms()) {
        const auto delta = item.second.size() - before[item.first];
        if (delta) captured[item.first] = delta;
      }
    };
    auto once = [&](bool measured) {
      executor.Reset(generation.pad_token_id);
      std::vector<std::int64_t> result;
      if (stage == "prefill") {
        invoke(measured,
               [&] { result.push_back(executor.Prefill(prompt, false)); });
      } else {
        const auto anchor = executor.Prefill(prompt, p.mode == "dflash");
        token_ok(anchor);
        Require(!eos.count(anchor),
                "prefill anchor is EOS; no decode round to profile");
        if (stage == "decode") {
          invoke(measured, [&] { result.push_back(executor.Decode(anchor)); });
        } else if (stage == "draft") {
          invoke(measured, [&] {
            result = executor.Propose(anchor, proposal_count);
            result.resize(proposal_count);
          });
        } else {
          auto proposals = executor.Propose(anchor, proposal_count);
          proposals.resize(proposal_count);
          std::vector<std::int64_t> block{anchor};
          for (auto t : proposals) {
            token_ok(t);
            block.push_back(t);
            if (eos.count(t)) break;
          }
          invoke(measured, [&] { result = executor.Verify(block); });
          std::size_t accepted = 0;
          while (accepted + 1 < block.size() &&
                 block[accepted + 1] == result.at(accepted))
            ++accepted;
          executor.Commit(accepted + 1);
        }
      }
      for (auto token : result) token_ok(token);
      return result;
    };
    std::vector<std::int64_t> reference;
    try {
      for (std::size_t i = 0; i < p.warmup; ++i) reference = once(false);
      const auto measured = once(true);
      Require(!p.warmup || measured == reference,
              "profile output differs from warmup");
    } catch (...) {
      executor.Abort();
      throw;
    }
    const auto graph = stage == "draft" ? "draft" : "target_" + stage;
    const std::size_t expected =
        stage == "prefill" ? (prompt.size() + 63) / 64 : 1;
    Require(captured.size() == 1 && captured[graph] == expected,
            "capture executed unexpected OM calls");
    std::ostringstream report;
    report << std::setprecision(17)
           << "{\"status\":\"PASS_CAPTURE\",\"collector\":\"msprof dynamic "
              "CLI\",\"profile_backend\":\"cpp\",\"profile_mode\":"
           << Quote(p.mode) << ",\"profile_stage\":" << Quote(stage)
           << ",\"profile_output\":" << Quote(output.string())
           << ",\"capture_windows\":1,\"operator_rows_required\":true,"
              "\"operator_fallback_enabled\":false,\"profiled_elapsed_ms\":"
           << elapsed
           << ",\"captured_calls\":{\"prefill\":" << (stage == "prefill")
           << ",\"draft\":" << (stage == "draft")
           << ",\"target_verify\":" << (stage == "verify");
    if (p.mode == "ordinary")
      report << ",\"target_decode\":" << (stage == "decode");
    report << "},\"captured_graph_calls\":{" << Quote(graph) << ':' << expected
           << "},\"stage_scope\":" << Quote(Scope(stage))
           << ",\"warmup_iterations\":" << p.warmup
           << ",\"proposal_count\":" << ((stage == "draft" || stage == "verify") ? proposal_count : 0)
           << ",\"warmup_output_match\":" << (p.warmup ? "true" : "null")
           << ",\"formal_latency_evidence\":false,\"correctness_gate\":{"
              "\"status\":\"NOT_RUN_STAGE_DIAGNOSTIC\"}}";
    reports.push_back(report.str());
  }
  std::ostringstream out;
  out << "{\"schema_version\":1,\"status\":\"PASS_CAPTURE\",\"collector\":"
         "\"msprof dynamic CLI\",\"profile_backend\":\"cpp\",\"profile_mode\":"
      << Quote(p.mode) << ",\"profile_stage\":" << Quote(p.stage)
      << ",\"profile_output\":" << Quote(p.output.string())
      << ",\"capture_windows\":" << reports.size()
      << ",\"operator_fallback_enabled\":false,\"device_id\":" << p.device_id
      << ",\"runtime\":\"AscendCL\",\"prompt_tokens\":" << prompt.size()
      << ",\"formal_latency_evidence\":false,\"correctness_gate\":{\"status\":"
         "\"NOT_RUN_STAGE_DIAGNOSTIC\"},\"stages\":[";
  for (std::size_t i = 0; i < stages.size(); ++i) {
    if (i) out << ',';
    out << Quote(stages[i]);
  }
  out << "],\"captures\":[";
  for (std::size_t i = 0; i < reports.size(); ++i) {
    if (i) out << ',';
    out << reports[i];
  }
  out << "],\"aic_metrics\":" << Quote(p.metrics)
      << ",\"timing_scope\":\"synchronized stage including small H2D/D2H; "
         "excludes model load, request reset and msprof control waits\"}";
  // Single reports keep the same shape as the Python entry for CSV validation.
  if (p.stage != "all") {
    auto single = reports[0];
    single.pop_back();
    return single + ",\"runtime\":\"AscendCL\",\"device_id\":" +
           std::to_string(p.device_id) +
           ",\"prompt_tokens\":" + std::to_string(prompt.size()) + "}";
  }
  return out.str();
}
}  // namespace qwen35::dflash
