"""Copy-and-fill template for a proprietary Ascend 310P runtime adapter.

For ``qwen35_mtp benchmark`` the concrete backend must expose a no-argument
``synchronize()`` hook.  A ``reset_benchmark_state()`` hook is also consumed
before every warmup and measured iteration so cached generation state cannot
leak between samples.
"""

from __future__ import annotations

import torch

from qwen35_mtp.backends import MainEvaluation


class InternalMainBackend:
    backend_id = "REPLACE_WITH_RUNTIME_AND_ARTIFACT_ID"

    def __init__(self, runtime):
        self.runtime = runtime

    def synchronize(self):
        self.runtime.synchronize()

    def reset_benchmark_state(self):
        self.runtime.reset_benchmark_state(role="main")

    def benchmark_metadata(self):
        # Must include the concrete device ID plus CANN/driver/firmware/runtime
        # identities used for this loaded artifact.
        return self.runtime.benchmark_metadata()

    def evaluate(self, input_ids, top1_positions):
        # Required runtime result:
        #   hidden: [1,S,2560] BF16 after final main RMSNorm
        #   top1:   [1,R] INT64 using the tied 248320-row LM head
        result = self.runtime.run_main(
            input_ids=input_ids.cpu().numpy(),
            top1_positions=list(top1_positions),
            recompute_committed_prefix=True,
        )
        hidden = torch.as_tensor(result["hidden_states"])
        top1 = torch.as_tensor(result["top1_token_ids"], dtype=torch.long)
        return MainEvaluation(hidden_states=hidden, top1_token_ids=top1)


class InternalDraftBackend:
    backend_id = "REPLACE_WITH_MTP_ARTIFACT_ID"

    def __init__(self, runtime):
        self.runtime = runtime

    def synchronize(self):
        self.runtime.synchronize()

    def reset_benchmark_state(self):
        self.runtime.reset_benchmark_state(role="draft")

    def benchmark_metadata(self):
        return self.runtime.benchmark_metadata()

    def propose(
        self,
        prefix_ids,
        main_hidden_states,
        max_draft_tokens,
        *,
        eos_token_ids=(),
    ):
        return list(
            self.runtime.run_official_mtp(
                prefix_ids=prefix_ids.cpu().numpy(),
                main_hidden_states=main_hidden_states.cpu().numpy(),
                max_draft_tokens=int(max_draft_tokens),
                eos_token_ids=list(eos_token_ids),
                rebuild_draft_cache=True,
            )
        )


def create_backend(*, role, model_dir, options):
    del model_dir, options
    # Replace this import/factory with the internal framework entry point.
    raise RuntimeError(
        f"template only: construct the internal runtime and return role={role!r} backend"
    )
