#include "qwen35_dflash/target_parity_diagnostic.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace qwen35::dflash {
namespace {

template <class T>
void Array(std::ostream& out, const std::vector<T>& items) {
  out << '[';
  for (std::size_t i = 0; i < items.size(); ++i) {
    if (i) out << ',';
    out << items[i];
  }
  out << ']';
}

void Number(std::ostream& out, double value) {
  if (std::isfinite(value)) out << std::setprecision(17) << value;
  else out << "null";
}

std::int64_t Cursor(const TargetStateSnapshot& state) {
  if (state.size() != 5 || state[4].dtype != "int64" ||
      state[4].shape != std::vector<std::int64_t>{1} ||
      state[4].data.size() < sizeof(std::int64_t)) {
    throw std::runtime_error("diagnostic cursor ABI differs");
  }
  std::int64_t cursor;
  std::memcpy(&cursor, state[4].data.data(), sizeof(cursor));
  if (cursor < 0) throw std::runtime_error("negative diagnostic cursor");
  return cursor;
}

std::size_t ElementBytes(const std::string& dtype) {
  if (dtype == "float16") return 2;
  if (dtype == "float32") return 4;
  if (dtype == "int64") return 8;
  throw std::runtime_error("unsupported diagnostic dtype");
}

double Value(const TargetTensorSnapshot& tensor, std::size_t index) {
  const auto* data = tensor.data.data() + index * ElementBytes(tensor.dtype);
  if (tensor.dtype == "float32") {
    float value;
    std::memcpy(&value, data, sizeof(value));
    return value;
  }
  if (tensor.dtype == "int64") {
    std::int64_t value;
    std::memcpy(&value, data, sizeof(value));
    return static_cast<double>(value);
  }
  std::uint16_t bits;
  std::memcpy(&bits, data, sizeof(bits));
  const unsigned exponent = (bits >> 10) & 31;
  const unsigned fraction = bits & 1023;
  double value = exponent == 0 ? std::ldexp(static_cast<double>(fraction), -24)
      : exponent == 31 ? (fraction ? std::numeric_limits<double>::quiet_NaN()
                                  : std::numeric_limits<double>::infinity())
      : std::ldexp(static_cast<double>(1024 + fraction), static_cast<int>(exponent) - 25);
  return (bits & 0x8000) ? -value : value;
}

std::vector<std::size_t> Coordinates(
    const std::vector<std::int64_t>& shape, std::size_t index) {
  std::vector<std::size_t> coordinates(shape.size());
  for (std::size_t axis = shape.size(); axis-- > 0;) {
    coordinates[axis] = index % static_cast<std::size_t>(shape[axis]);
    index /= static_cast<std::size_t>(shape[axis]);
  }
  return coordinates;
}

// Compare all conv/GDR elements, but only logically live paged KV entries.
// Rejected speculative KV tail writes are expected and not committed state.
void CompareState(std::ostream& out, const TargetStateSnapshot& reference,
                  const TargetStateSnapshot& actual) {
  const std::array<const char*, 5> names{
      "target_conv_state", "target_recurrent_state",
      "target_key_cache", "target_value_cache", "logical_target_cursor"};
  const auto reference_cursor = Cursor(reference), actual_cursor = Cursor(actual);
  const auto live = static_cast<std::size_t>(std::min(reference_cursor, actual_cursor));
  out << "{\"reference_cursor\":" << reference_cursor
      << ",\"actual_cursor\":" << actual_cursor
      << ",\"cursor_equal\":" << (reference_cursor == actual_cursor ? "true" : "false")
      << ",\"kv_comparison_prefix\":" << live
      << ",\"layer_index_scope\":\"grouped state axis 0, not decoder layer ID\""
      << ",\"tensors\":[";
  for (std::size_t t = 0; t < names.size(); ++t) {
    const auto& a = reference.at(t);
    const auto& b = actual.at(t);
    if (a.dtype != b.dtype || a.shape != b.shape || a.data.size() != b.data.size() ||
        a.shape.empty() || a.shape[0] <= 0) {
      throw std::runtime_error("incompatible diagnostic state snapshots");
    }
    const auto bytes = ElementBytes(a.dtype);
    std::size_t elements = 1;
    for (auto dim : a.shape) {
      if (dim <= 0 || static_cast<std::size_t>(dim) > a.data.size() / bytes / elements) {
        throw std::runtime_error("diagnostic state shape exceeds payload");
      }
      elements *= static_cast<std::size_t>(dim);
    }
    const auto layers = t == 4 ? 1 : static_cast<std::size_t>(a.shape[0]);
    const auto per_layer = elements / layers;
    const bool kv = t == 2 || t == 3;
    if (kv && (a.shape.size() != 5 || a.shape[3] != 64 ||
               live > static_cast<std::size_t>(a.shape[1]) * 64)) {
      throw std::runtime_error("diagnostic paged KV layout/cursor differs");
    }
    if (t) out << ',';
    out << "{\"name\":\"" << names[t] << "\",\"dtype\":\"" << a.dtype
        << "\",\"shape\":";
    Array(out, a.shape);
    out << ",\"layers\":[";
    for (std::size_t layer = 0; layer < layers; ++layer) {
      std::size_t compared = 0, bit_different = 0, numeric_different = 0;
      std::size_t reference_nonfinite = 0, actual_nonfinite = 0;
      std::size_t first = elements;
      double max_abs = 0;
      for (std::size_t j = 0; j < per_layer; ++j) {
        if (kv) {
          const auto head_dim = static_cast<std::size_t>(a.shape[4]);
          const auto heads = static_cast<std::size_t>(a.shape[2]);
          const auto token = (j / (heads * 64 * head_dim)) * 64 + (j / head_dim) % 64;
          if (token >= live) continue;
        }
        ++compared;
        const auto index = layer * per_layer + j;
        const bool bit_diff = std::memcmp(
            a.data.data() + index * bytes, b.data.data() + index * bytes, bytes) != 0;
        if (bit_diff) {
          ++bit_different;
          if (first == elements) first = index;
        }
        const double av = Value(a, index), bv = Value(b, index);
        if (!std::isfinite(av)) ++reference_nonfinite;
        if (!std::isfinite(bv)) ++actual_nonfinite;
        if (av != bv) ++numeric_different;
        if (std::isfinite(av) && std::isfinite(bv)) max_abs = std::max(max_abs, std::abs(av - bv));
      }
      if (layer) out << ',';
      out << "{\"state_layer_index\":" << layer << ",\"compared_elements\":" << compared
          << ",\"excluded_elements\":" << per_layer - compared
          << ",\"bitwise_different_elements\":" << bit_different
          << ",\"numeric_different_elements\":" << numeric_different
          << ",\"reference_nonfinite\":" << reference_nonfinite
          << ",\"actual_nonfinite\":" << actual_nonfinite
          << ",\"max_finite_abs_error\":";
      Number(out, max_abs);
      out << ",\"first_bitwise_difference\":";
      if (first == elements) out << "null";
      else {
        out << "{\"flat_index\":" << first << ",\"coordinates\":";
        Array(out, Coordinates(a.shape, first));
        out << ",\"reference\":";
        Number(out, Value(a, first));
        out << ",\"actual\":";
        Number(out, Value(b, first));
        out << '}';
      }
      out << '}';
    }
    out << "]}";
  }
  out << "]}";
}

struct CapturedTransaction {
  std::string path;
  std::size_t proposal_limit = 0;
  TargetStateSnapshot before, after;
  std::vector<std::int64_t> verify_ids;
  StatefulStep result;
};
struct CaptureLimit {};

// Delegate scheduling unchanged to GenerateStatefulOnce. The diagnostic stops
// BEFORE the next model call, not by changing max_new_tokens or Draft's K.
class CaptureExecutor final : public StatefulGraphExecutor {
 public:
  AclIncrementalExecutor& executor;
  std::size_t limit;
  std::vector<CapturedTransaction> transactions;
  StatefulStep prefill;
  const std::function<void(const std::string&)>& progress;

  CaptureExecutor(AclIncrementalExecutor& e, std::size_t n,
                  const std::function<void(const std::string&)>& p)
      : executor(e), limit(n), progress(p) {}
  std::size_t sequence_length() const noexcept override { return executor.sequence_length(); }
  std::size_t prefill_width() const noexcept override { return executor.prefill_width(); }
  std::size_t proposal_width() const noexcept override { return executor.proposal_width(); }
  std::size_t eos_table_width() const noexcept override { return executor.eos_table_width(); }
  void ValidateRequest(std::size_t p, std::size_t n) const override { executor.ValidateRequest(p, n); }
  void Reset(std::int64_t pad, const std::vector<std::int64_t>& eos) override { executor.Reset(pad, eos); }
  StatefulStep PrefillChunk(const std::vector<std::int64_t>& ids, bool draft, std::size_t k) override {
    prefill = executor.PrefillChunk(ids, draft, k);
    return prefill;
  }
  std::size_t PrefillChunkDeferred(
      const std::vector<std::int64_t>& ids, bool draft, std::size_t k) override {
    return executor.PrefillChunkDeferred(ids, draft, k);
  }
  StatefulStep DecodeOne(std::int64_t) override {
    throw std::logic_error("unexpected ordinary call in diagnostic DFlash capture");
  }
  StatefulStep VerifyOne(std::int64_t id) override {
    return Capture("target-only-verify", 0, [&] { return executor.VerifyOne(id); });
  }
  StatefulStep SpeculativeStep(std::size_t k) override {
    return Capture("speculative-verify", k, [&] { return executor.SpeculativeStep(k); });
  }

 private:
  StatefulStep Capture(const char* path, std::size_t k, const std::function<StatefulStep()>& call) {
    if (transactions.size() == limit) throw CaptureLimit{};
    if (progress) progress("stage=target-parity-capture-start transaction=" +
                           std::to_string(transactions.size() + 1));
    CapturedTransaction t;
    t.path = path;
    t.proposal_limit = k;
    t.before = executor.CaptureTargetState();
    t.result = call();
    t.verify_ids = executor.CaptureVerifyInputIds();
    t.after = executor.CaptureTargetState();
    transactions.push_back(std::move(t));
    if (progress) progress("stage=target-parity-capture-done transaction=" +
                           std::to_string(transactions.size()));
    return transactions.back().result;
  }
};

std::vector<std::int64_t> Replay(
    AclIncrementalExecutor& executor, const TargetStateSnapshot& initial,
    const std::vector<std::int64_t>& inputs) {
  executor.RestoreTargetStateForDiagnostic(initial);
  std::vector<std::int64_t> predictions;
  for (auto id : inputs) {
    const auto step = executor.DecodeOne(id);
    if (step.token_ids.size() != 1) throw std::runtime_error("diagnostic Decode1 did not return one token");
    predictions.push_back(step.token_ids[0]);
  }
  return predictions;
}

bool CompareTokens(std::ostream& out, const std::vector<std::int64_t>& ordinary,
                   const CapturedTransaction& t, std::size_t generated_begin,
                   std::size_t prompt_size, std::size_t max_new_tokens,
                   std::size_t transaction, const char* replay,
                   std::string& first_mismatch) {
  const auto& actual = t.result.token_ids;
  if (ordinary.size() < actual.size()) throw std::runtime_error("diagnostic replay has too few rows");
  bool match = true;
  out << "{\"ordinary_top1\":";
  Array(out, ordinary);
  out << ",\"verify_compact_token_ids\":";
  Array(out, actual);
  out << ",\"compared_rows\":" << actual.size() << ",\"mismatches\":[";
  for (std::size_t row = 0; row < actual.size(); ++row) {
    if (ordinary[row] == actual[row]) continue;
    if (!match) out << ',';
    match = false;
    const char* role = row < t.result.accepted_draft_tokens ? "accepted-proposal"
        : t.proposal_limit == 0 ? "target-only"
        : t.result.accepted_draft_tokens == t.result.drafted_tokens ? "target-bonus"
        : "target-correction";
    std::ostringstream mismatch;
    mismatch << "{\"transaction\":" << transaction << ",\"replay\":\"" << replay
        << "\",\"verify_row\":" << row
        << ",\"generated_token_index\":" << generated_begin + row
        << ",\"absolute_token_index\":" << prompt_size + generated_begin + row
        << ",\"device_prediction_index\":" << Cursor(t.before) + row + 1
        << ",\"within_generation_budget\":" << (generated_begin + row < max_new_tokens ? "true" : "false")
        << ",\"token_role\":\"" << role
        << "\",\"input_token_id\":" << t.verify_ids.at(row)
        << ",\"expected_token\":" << ordinary[row] << ",\"actual_token\":" << actual[row] << '}';
    if (first_mismatch == "null") first_mismatch = mismatch.str();
    out << mismatch.str();
  }
  out << "],\"equal\":" << (match ? "true" : "false") << '}';
  return match;
}

}  // namespace

TargetParityDiagnostic DiagnoseTargetParity(
    AclIncrementalExecutor& executor, const std::vector<std::int64_t>& prompt,
    const GenerationOptions& options, std::size_t max_transactions,
    const std::function<void(const std::string&)>& progress) {
  if (!executor.merged_prefill() || executor.unified_target_step() ||
      executor.fused_speculative_step() || executor.execution_stats().target_step_dynamic_shape ||
      options.dflash_sync_window != 1 || options.coalesce_prefill_with_first_verify ||
      max_transactions == 0 || max_transactions > 4) {
    throw std::invalid_argument(
        "target parity diagnostic requires four static split OMs, sync-window=1, "
        "separate prefill, and 1..4 capture transactions");
  }
  // Includes capture snapshots, replay copies, pinned staging, and comparison
  // temporaries conservatively. Admission happens before Reset/prefill.
  const std::size_t state_bytes = executor.execution_stats().state_reset_bytes_per_request;
  constexpr std::size_t host_budget = std::size_t{2} * 1024 * 1024 * 1024;
  if (state_bytes == 0 || state_bytes > host_budget / (2 * max_transactions + 6)) {
    throw std::invalid_argument("target parity diagnostic exceeds 2 GiB host snapshot budget; reduce diagnostic-max-transactions");
  }
  CaptureExecutor capture(executor, max_transactions, progress);
  std::string capture_stop;
  try {
    capture_stop = GenerateStatefulOnce(capture, prompt, GenerationMode::kDFlash, options).stop_reason;
  } catch (const CaptureLimit&) {
    capture_stop = "transaction-limit";
  }
  TargetParityDiagnostic result;
  std::ostringstream out;
  out << "{\"max_transactions\":" << max_transactions
      << ",\"captured_transactions\":" << capture.transactions.size()
      << ",\"capture_stop_reason\":\"" << capture_stop << "\""
      << ",\"host_snapshot_budget_bytes\":" << host_budget
      << ",\"host_snapshot_estimate_upper_bound_bytes\":" << state_bytes * (2 * max_transactions + 6)
      << ",\"extra_device_state_bytes\":0,\"raw_logits_available\":false"
      << ",\"verify_top1_scope\":\"inferred from accepted prefix and correction/bonus in compact output, including any terminal bonus beyond generation budget; not raw logits\""
      << ",\"replay_scope\":\"teacher-forced DFlash consumed inputs; not free-running ordinary after a divergence\""
      << ",\"state_scope\":\"all conv/GDR; paged KV below common logical cursor, rejected tail excluded\""
      << ",\"prefill_token_ids\":";
  Array(out, capture.prefill.token_ids);
  out << ",\"transactions\":[";
  TargetStateSnapshot chained;
  if (!capture.transactions.empty()) chained = capture.transactions.front().before;
  std::size_t generated_begin = capture.prefill.token_ids.size();
  for (std::size_t i = 0; i < capture.transactions.size(); ++i) {
    const auto& t = capture.transactions[i];
    const std::size_t rows = t.result.accepted_draft_tokens + 1;
    if (rows > t.verify_ids.size() || rows == 0 ||
        Cursor(t.before) + static_cast<std::int64_t>(rows) >
            static_cast<std::int64_t>(executor.sequence_length())) {
      throw std::runtime_error("diagnostic committed input extent is invalid");
    }
    const std::vector<std::int64_t> inputs(t.verify_ids.begin(), t.verify_ids.begin() + rows);
    if (progress) progress("stage=target-parity-replay-start transaction=" + std::to_string(i + 1));
    if (i) out << ',';
    out << "{\"transaction\":" << i + 1 << ",\"path\":\"" << t.path
        << "\",\"generated_begin\":" << generated_begin
        << ",\"proposal_limit\":" << t.proposal_limit
        << ",\"accepted_draft_tokens\":" << t.result.accepted_draft_tokens
        << ",\"rejected_draft_tokens\":" << t.result.rejected_draft_tokens
        << ",\"finished\":" << (t.result.finished ? "true" : "false")
        << ",\"verify_input_ids\":";
    Array(out, t.verify_ids);
    out << ",\"replayed_input_ids\":";
    Array(out, inputs);
    const bool cursor_ok = Cursor(t.after) == Cursor(t.before) + static_cast<std::int64_t>(rows);
    result.cursor_parity &= cursor_ok && Cursor(chained) == Cursor(t.before);
    out << ",\"verify_cursor_transition_consistent\":" << (cursor_ok ? "true" : "false")
        << ",\"incoming_state_vs_chained_decode1\":";
    CompareState(out, chained, t.before);
    const auto chained_ids = Replay(executor, chained, inputs);
    chained = executor.CaptureTargetState();
    out << ",\"chained_decode1_tokens\":";
    result.token_parity &= CompareTokens(out, chained_ids, t, generated_begin,
        prompt.size(), options.max_new_tokens, i + 1, "chained-decode1",
        result.first_token_mismatch_json);
    out << ",\"chained_decode1_state_vs_verify\":";
    CompareState(out, chained, t.after);
    const auto isolated_ids = Replay(executor, t.before, inputs);
    out << ",\"same_input_state_decode1_tokens\":";
    result.token_parity &= CompareTokens(out, isolated_ids, t, generated_begin,
        prompt.size(), options.max_new_tokens, i + 1, "same-input-state-decode1",
        result.first_token_mismatch_json);
    out << ",\"same_input_state_decode1_state_vs_verify\":";
    const auto isolated_state = executor.CaptureTargetState();
    CompareState(out, isolated_state, t.after);
    result.cursor_parity &= Cursor(chained) == Cursor(t.after) && Cursor(isolated_state) == Cursor(t.after);
    out << '}';
    generated_begin += std::min(t.result.token_ids.size(), options.max_new_tokens - generated_begin);
    if (progress) progress("stage=target-parity-replay-done transaction=" + std::to_string(i + 1));
  }
  out << "],\"first_token_mismatch\":" << result.first_token_mismatch_json
      << ",\"first_token_mismatch_order\":\"transaction order, chained replay before same-input-state replay\""
      << ",\"token_parity\":\"" << (capture.transactions.empty() ? "NOT_CHECKED" : result.token_parity ? "PASS" : "FAIL")
      << "\",\"cursor_parity\":\"" << (capture.transactions.empty() ? "NOT_CHECKED" : result.cursor_parity ? "PASS" : "FAIL")
      << "\",\"claim_boundary\":\"bounded diagnostic only; no complete generation, torch_npu parity, or performance pass\"}";
  result.detail_json = out.str();
  return result;
}

}  // namespace qwen35::dflash
