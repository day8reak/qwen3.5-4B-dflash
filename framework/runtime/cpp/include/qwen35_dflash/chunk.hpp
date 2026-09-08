#pragma once

#include <filesystem>
#include <map>
#include <memory>

#include "qwen35_dflash/generation.hpp"

namespace qwen35::dflash {

struct TensorSpec {
  std::string name;
  std::string dtype;
  std::vector<std::int64_t> shape;
  std::size_t bytes() const;
};
struct ChunkGraph {
  std::string name;
  std::filesystem::path model;
  std::string sha256;
  std::vector<TensorSpec> inputs, outputs;
};
struct ChunkPlan {
  std::size_t capacity = 0;
  std::int64_t vocabulary = 0;
  std::map<std::string, ChunkGraph> graphs;
};
ChunkPlan ReadChunkPlan(const std::filesystem::path& path,
                        const std::string& mode = "paired");

// The same scheduler is exercised by host fixtures and the real AscendCL path.
class ChunkExecutor : public GraphExecutor {
 public:
  std::size_t draft_width() const noexcept override { return 15; }
  const GraphOutputs& Execute(const std::vector<std::int64_t>&,
                              std::int64_t) override;
  virtual std::int64_t vocabulary_size() const noexcept = 0;
  virtual void Reset(std::int64_t pad) = 0;
  virtual void Abort() noexcept = 0;
  virtual std::int64_t Prefill(const std::vector<std::int64_t>& ids,
                               bool draft) = 0;
  virtual std::vector<std::int64_t> Propose(std::int64_t anchor) = 0;
  virtual std::vector<std::int64_t> Verify(
      const std::vector<std::int64_t>& block) = 0;
  virtual void Commit(std::size_t rows) = 0;
  virtual std::int64_t Decode(std::int64_t anchor) = 0;
  virtual bool HasOrdinaryDecode() const noexcept { return false; }
  virtual std::size_t graph_calls() const noexcept = 0;
  virtual const std::map<std::string, std::vector<double>>& stage_ms()
      const = 0;
};
GenerationMeasurement GenerateChunk(ChunkExecutor&,
                                    const std::vector<std::int64_t>&,
                                    GenerationMode, const GenerationOptions&);

class AclChunkExecutor final : public ChunkExecutor {
 public:
  explicit AclChunkExecutor(const std::filesystem::path& plan,
                            int device_id = 0,
                            const std::string& mode = "paired");
  ~AclChunkExecutor() override;
  void Synchronize();
  std::size_t sequence_length() const noexcept override;
  std::int64_t vocabulary_size() const noexcept override;
  void Reset(std::int64_t) override;
  void Abort() noexcept override;
  std::int64_t Prefill(const std::vector<std::int64_t>&, bool) override;
  std::vector<std::int64_t> Propose(std::int64_t) override;
  std::vector<std::int64_t> Verify(const std::vector<std::int64_t>&) override;
  void Commit(std::size_t) override;
  std::int64_t Decode(std::int64_t) override;
  bool HasOrdinaryDecode() const noexcept override;
  std::size_t graph_calls() const noexcept override;
  const std::map<std::string, std::vector<double>>& stage_ms() const override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace qwen35::dflash
