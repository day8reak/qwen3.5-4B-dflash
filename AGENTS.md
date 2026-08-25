# Qwen3.5-4B MTP project rules

Read the workspace `AGENTS.md`, then resolve context for `qwen3.5-4b` and the
`ascend310p` target before work.

- The checkpoint is external and locked by `specs/data.lock.json`; never copy
  weights, caches, ONNX/OM artifacts, logs, or releases into this repository.
- Keep the ordinary target model authoritative. MTP may change how many target
  tokens are verified per call, but strict greedy output must have zero token-ID
  mismatch against ordinary generation.
- The draft structure and 15 tensor names are locked to the official Qwen3.5
  checkpoint. Do not substitute a small dense model or an invented head.
- CPU fallback is valid simulation evidence only. A target run must disable
  fallback and record its runtime/device identity.
- The initial target route may recompute committed prefixes. Incremental cache
  commit/rollback is a later optimization and needs its own state-branch gates.
