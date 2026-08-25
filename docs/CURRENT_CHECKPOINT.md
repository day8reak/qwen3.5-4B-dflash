# Current checkpoint

- Run: `20260819T015109Z-5594-199c17`
- Target/profile: `ascend310p / ascend310p-local (simulation-only)`
- Active phase: `reference`
- Last passing gates: official config/15-tensor audit; 19-test CPU/ONNX suite; real BF16
  main/MTP smoke; FP16 ordinary+MTP CPU admission; FP16 fixed-gear ONNX full checker;
  real-text S1/P1 fixture; ONNX Runtime CPU parity; DFlash 69-tensor remote-header audit;
  deterministic cache-free DFlash draft-core exact parity with the locked official source;
  DFlash target-feature portable and overlay gates; 34-test combined suite plus 3 passing
  subtests.
- Device: unavailable in the declared profile; ATC and CAModel also unavailable.
- Precision: FP16 experiment approved; CPU candidate is eligible for target testing but
  not promoted. Internal ordinary graph, ATC/CAModel and 310P device gates remain pending.
- DFlash: the cache-free six-layer draft core and strict replaceable-op ABI are implemented.
  End-to-end verifier, accept/reject, cache/state transaction and scheduler remain pending;
  no acceptance-rate or speed claim is available yet.
- DFlash target phase 1: a selective post-layer feature collector, output contract and an
  additive archive rooted exactly at the user's `qwen3_5/config`, `qwen3_5/models`,
  `qwen3_5/dflash` and `qwen3_5/tests` structure are implemented. Its SHA-256 is
  `ebcd634da992142c30271c0cee7228117f50224f0accdc80bcebbea91941649e`. The
  user's modified torch_npu target source is not present here, so real logits/cache equality
  and all-hidden-state feature parity remain pending.
- GitHub blocker: this host has no `gh`, GitHub token, configured credential helper,
  or destination repository.

Next internal command after unpacking the candidate and case:

```bash
PYTHONPATH=model python targets/ascend310p/scripts/run_onnx_case.py \
  --model /path/to/qwen35-4b-mtp-core-s1-p1-fp16.onnx \
  --case-dir /path/to/real-text-nihao-comma-s1-p1 \
  --output /tmp/onnxruntime-compare.json
```

Stop conditions: any checkpoint audit failure, non-finite main/MTP tensor, ordinary/MTP
token mismatch, frozen threshold failure, silent fallback on target, or artifact hash mismatch.
