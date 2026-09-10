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
#include <optional>
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
std::string TokensJson(const std::vector<std::int64_t>& tokens) {
  std::ostringstream out;
  out << '[';
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    if (i) out << ',';
    out << tokens[i];
  }
  return out.str() + ']';
}
std::string NumberJson(std::optional<std::int64_t> value) {
  return value ? std::to_string(*value) : "null";
}
struct Sample {
  std::size_t iteration = 0;
  bool measured = false;
  std::vector<std::int64_t> input, output, raw_output;
  std::optional<std::int64_t> accepted;
};
struct Difference {
  std::string field;
  std::size_t index = 0;
  std::optional<std::int64_t> reference, actual;
};
Difference Compare(const Sample& reference, const Sample& actual) {
  for (bool input : {true, false}) {
    const auto& a = input ? reference.input : reference.output;
    const auto& b = input ? actual.input : actual.output;
    for (std::size_t i = 0; i < std::max(a.size(), b.size()); ++i) {
      const auto av = i < a.size() ? std::optional<std::int64_t>(a[i]) : std::nullopt;
      const auto bv = i < b.size() ? std::optional<std::int64_t>(b[i]) : std::nullopt;
      if (av != bv)
        return {input ? "input_token_ids" : "output_token_ids", i, av, bv};
    }
  }
  if (reference.accepted != actual.accepted)
    return {"accepted_draft_tokens", 0, reference.accepted, actual.accepted};
  return {};
}
class IterationTrace {
 public:
  explicit IterationTrace(const std::filesystem::path& raw_output)
      : path(raw_output.string() + ".iterations.jsonl") {
    // Keep the trace beside the raw directory: msprof must create that directory
    // itself, and the trace must survive even if the final PASS report is absent.
    if (!path.parent_path().empty())
      std::filesystem::create_directories(path.parent_path());
    fd_ = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd_ < 0) throw std::runtime_error("cannot create profile iteration trace: " + path.string());
    std::cerr << "[stage-profile] iteration_trace=" << path << '\n';
  }
  ~IterationTrace() { if (fd_ >= 0) close(fd_); }
  IterationTrace(const IterationTrace&) = delete;
  IterationTrace& operator=(const IterationTrace&) = delete;

  void Record(const std::string& mode, const std::string& stage,
              const Sample& sample, const Sample* reference,
              const char* event, const char* status, const Difference& difference = {}) {
    const auto match = [&](bool value) {
      return reference ? (value ? "true" : "false") : "null";
    };
    const auto tail = [](const Sample& value) {
      return std::vector<std::int64_t>(value.raw_output.begin() + value.output.size(),
                                       value.raw_output.end());
    };
    std::ostringstream out;
    out << "{\"schema_version\":1,\"event\":" << Quote(event)
        << ",\"profile_mode\":" << Quote(mode) << ",\"profile_stage\":" << Quote(stage)
        << ",\"iteration\":" << sample.iteration
        << ",\"measured\":" << (sample.measured ? "true" : "false")
        << ",\"input_token_ids\":" << TokensJson(sample.input)
        << ",\"output_token_ids\":" << TokensJson(sample.output)
        << ",\"raw_output_token_ids\":" << TokensJson(sample.raw_output)
        << ",\"verify_valid_rows\":" << (stage == "verify" ? std::to_string(sample.input.size()) : "null")
        << ",\"padding_output_rows\":" << sample.raw_output.size() - sample.output.size()
        << ",\"accepted_draft_tokens\":" << NumberJson(sample.accepted)
        << ",\"input_state_comparison\":\"NOT_RUN\",\"check\":{\"status\":" << Quote(status)
        << ",\"reference_iteration\":" << (reference ? std::to_string(reference->iteration) : "null")
        << ",\"input_token_ids_match\":" << match(reference && sample.input == reference->input)
        << ",\"output_token_ids_match\":" << match(reference && sample.output == reference->output)
        << ",\"padding_token_ids_match\":" << match(reference && tail(sample) == tail(*reference))
        << ",\"first_difference\":";
    if (difference.field.empty()) out << "null";
    else out << "{\"field\":" << Quote(difference.field) << ",\"index\":" << difference.index
             << ",\"reference\":" << NumberJson(difference.reference)
             << ",\"actual\":" << NumberJson(difference.actual) << '}';
    const auto line = out.str() + "}}\n";
    std::size_t offset = 0;
    while (offset < line.size()) {
      const auto n = write(fd_, line.data() + offset, line.size() - offset);
      if (n < 0 && errno == EINTR) continue;
      Require(n > 0, "cannot write profile iteration trace");
      offset += static_cast<std::size_t>(n);
    }
  }
  const std::filesystem::path path;
 private:
  int fd_ = -1;
};
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
  IterationTrace trace(p.output);
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
    auto once = [&](bool measured, std::size_t iteration) {
      executor.Reset(generation.pad_token_id);
      Sample sample;
      sample.iteration = iteration;
      sample.measured = measured;
      auto prepared = [&] { trace.Record(p.mode, stage, sample, nullptr, "prepared", "PREPARED"); };
      if (stage == "prefill") {
        sample.input = prompt;
        prepared();
        invoke(measured,
               [&] { sample.raw_output.push_back(executor.Prefill(prompt, false)); });
        sample.output = sample.raw_output;
      } else {
        const auto anchor = executor.Prefill(prompt, p.mode == "dflash");
        token_ok(anchor);
        Require(!eos.count(anchor),
                "prefill anchor is EOS; no decode round to profile");
        if (stage == "decode") {
          sample.input = {anchor};
          prepared();
          invoke(measured, [&] { sample.raw_output.push_back(executor.Decode(anchor)); });
          sample.output = sample.raw_output;
        } else if (stage == "draft") {
          sample.input = {anchor};
          prepared();
          invoke(measured, [&] {
            sample.raw_output = executor.Propose(anchor, proposal_count);
          });
          Require(sample.raw_output.size() >= proposal_count, "Draft output is shorter than proposal count");
          sample.output.assign(sample.raw_output.begin(), sample.raw_output.begin() + proposal_count);
        } else {
          auto proposals = executor.Propose(anchor, proposal_count);
          proposals.resize(proposal_count);
          std::vector<std::int64_t> block{anchor};
          for (auto t : proposals) {
            token_ok(t);
            block.push_back(t);
            if (eos.count(t)) break;
          }
          sample.input = block;
          prepared();
          invoke(measured, [&] { sample.raw_output = executor.Verify(block); });
          Require(sample.raw_output.size() >= block.size(), "verify output is shorter than valid rows");
          // Match GenerateChunk's logical row contract. Padding is retained in
          // the trace for diagnosis but never treated as generated token data.
          sample.output.assign(sample.raw_output.begin(), sample.raw_output.begin() + block.size());
          std::size_t accepted = 0;
          while (accepted + 1 < block.size() &&
                 block[accepted + 1] == sample.output.at(accepted))
            ++accepted;
          executor.Commit(accepted + 1);
          sample.accepted = static_cast<std::int64_t>(accepted);
        }
      }
      for (auto token : sample.output) token_ok(token);
      return sample;
    };
    std::optional<Sample> reference;
    Sample measured;
    auto check = [&](const Sample& sample) {
      const auto difference = reference ? Compare(*reference, sample) : Difference{};
      const auto* status = !difference.field.empty() ? "FAIL" : reference ? "PASS"
                                       : sample.measured ? "NO_WARMUP" : "REFERENCE";
      trace.Record(p.mode, stage, sample, reference ? &*reference : nullptr,
                   "completed", status, difference);
      if (!difference.field.empty()) {
        throw std::runtime_error(
            std::string(difference.field == "input_token_ids" ? "profile input token IDs differ" : "profile output differs") +
            " from warmup stage=" + stage + " iteration=" + std::to_string(sample.iteration) +
            " measured=" + (sample.measured ? "true" : "false") + " field=" + difference.field +
            " index=" + std::to_string(difference.index) + " reference=" + NumberJson(difference.reference) +
            " actual=" + NumberJson(difference.actual) + "; trace=" + trace.path.string());
      }
    };
    try {
      for (std::size_t i = 0; i < p.warmup; ++i) {
        const auto sample = once(false, i);
        check(sample);
        if (!reference) reference = sample;
      }
      measured = once(true, p.warmup);
      check(measured);
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
           << ",\"warmup_input_token_ids_match\":" << (p.warmup ? "true" : "null")
           << ",\"compared_output_rows\":" << measured.output.size()
           << ",\"output_comparison_scope\":\"valid rows only; padding retained in iteration trace\""
           << ",\"input_state_comparison\":\"NOT_RUN\",\"iteration_trace\":" << Quote(trace.path.string())
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
      << ",\"iteration_trace\":" << Quote(trace.path.string())
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
