"""Strict-greedy DFlash scheduler for the integrated recompute OM graph."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .acl_runtime import AclOmRuntime
from .contracts import GenerationStep


class RecomputeDFlashOmBackend:
    """Run proposal and ordinary verification with no speculative state mutation."""

    def __init__(
        self,
        runtime: Any,
        *,
        graph_name: str,
        pad_token_id: int,
        device: Mapping[str, Any],
        cann: str,
        driver: str,
        firmware: str,
        runtime_identity: str,
        ordinary_only: bool = False,
    ) -> None:
        self.runtime = runtime
        self.graph_name = str(graph_name)
        self.pad_token_id = int(pad_token_id)
        self.ordinary_only = bool(ordinary_only)
        inputs = {item["name"]: item for item in runtime.graph_inputs(self.graph_name)}
        outputs = {item["name"]: item for item in runtime.graph_outputs(self.graph_name)}
        if set(inputs) != {"input_ids", "attention_mask"}:
            raise ValueError(f"recompute OM input ABI differs: {sorted(inputs)}")
        if set(outputs) != {"target_top1", "draft_top1"}:
            raise ValueError(f"recompute OM output ABI differs: {sorted(outputs)}")
        for name, descriptor in {**inputs, **outputs}.items():
            if str(descriptor.get("dtype", "")).lower() != "int64":
                raise ValueError(f"recompute OM tensor {name} must use int64")
        input_shape = tuple(int(item) for item in inputs["input_ids"]["shape"])
        mask_shape = tuple(int(item) for item in inputs["attention_mask"]["shape"])
        if input_shape != mask_shape or len(input_shape) != 2 or input_shape[0] != 1:
            raise ValueError("recompute OM inputs must share static shape [1,S]")
        target_shape = tuple(int(item) for item in outputs["target_top1"]["shape"])
        draft_shape = tuple(int(item) for item in outputs["draft_top1"]["shape"])
        if target_shape != input_shape:
            raise ValueError("target_top1 must cover every fixed-gear input row")
        if len(draft_shape) != 2 or draft_shape[0] != 1 or draft_shape[1] < 1:
            raise ValueError("draft_top1 must have shape [1,K] with K >= 1")
        self.max_sequence_length = input_shape[1]
        self.available_draft_tokens = draft_shape[1]
        self._metadata = {
            "cpu_fallback": False,
            "artifacts": runtime.artifact_hashes(),
            "device": dict(device),
            "cann": str(cann),
            "driver": str(driver),
            "firmware": str(firmware),
            "runtime": str(runtime_identity),
            "graph_name": self.graph_name,
            "max_sequence_length": self.max_sequence_length,
            "available_draft_tokens": self.available_draft_tokens,
            "state_policy": "recompute committed prefixes",
            "generation_mode": (
                "ordinary-greedy" if self.ordinary_only else "dflash-strict-greedy"
            ),
        }
        graph_hash = self._metadata["artifacts"].get(self.graph_name, "unknown")
        self.backend_id = f"qwen35-dflash-recompute-om:{graph_hash[:16]}"

    def metadata(self) -> Mapping[str, Any]:
        return self._metadata

    def synchronize(self) -> None:
        self.runtime.synchronize()

    def reset(self) -> None:
        # This route has no persistent model state by construction.
        return None

    def _run(self, token_ids: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        values = [int(token) for token in token_ids]
        if not values:
            raise ValueError("recompute OM needs at least one committed token")
        if len(values) > self.max_sequence_length:
            raise ValueError("committed prefix exceeds the recompute OM gear")
        input_ids = np.full(
            (1, self.max_sequence_length), self.pad_token_id, dtype=np.int64
        )
        attention_mask = np.zeros((1, self.max_sequence_length), dtype=np.int64)
        input_ids[0, : len(values)] = values
        attention_mask[0, : len(values)] = 1
        outputs = self.runtime.run_graph(
            self.graph_name,
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )
        target = np.asarray(outputs["target_top1"])
        draft = np.asarray(outputs["draft_top1"])
        if target.shape != (1, self.max_sequence_length):
            raise RuntimeError("target_top1 runtime shape differs from the OM ABI")
        if draft.shape != (1, self.available_draft_tokens):
            raise RuntimeError("draft_top1 runtime shape differs from the OM ABI")
        return target.astype(np.int64, copy=False), draft.astype(np.int64, copy=False)

    def _check_capacity(self, prefix_length: int, max_new_tokens: int) -> None:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        # The last generated token is returned from the logits of the last
        # processed row and never needs to be fed back into this run.
        if prefix_length + max_new_tokens - 1 > self.max_sequence_length:
            raise ValueError(
                "prompt plus requested generation exceeds the fixed recompute OM gear"
            )

    def prefill(
        self,
        prompt_token_ids: Sequence[int],
        *,
        max_new_tokens: int,
        eos_token_ids: Sequence[int],
    ) -> GenerationStep:
        prompt = [int(token) for token in prompt_token_ids]
        self._check_capacity(len(prompt), max_new_tokens)
        target, _draft = self._run(prompt)
        token = int(target[0, len(prompt) - 1])
        if token < 0:
            raise RuntimeError("target OM returned a negative token ID")
        return GenerationStep(
            token_ids=(token,),
            finished=token in set(int(item) for item in eos_token_ids),
            metadata={"graph_calls": 1, "mode": "ordinary-prefill"},
        )

    def decode(
        self,
        committed_token_ids: Sequence[int],
        *,
        max_new_tokens: int,
        max_draft_tokens: int,
        eos_token_ids: Sequence[int],
    ) -> GenerationStep:
        prefix = [int(token) for token in committed_token_ids]
        self._check_capacity(len(prefix), max_new_tokens)
        if max_draft_tokens <= 0:
            raise ValueError("max_draft_tokens must be positive")
        target, draft = self._run(prefix)
        ordinary_next = int(target[0, len(prefix) - 1])
        if ordinary_next < 0:
            raise RuntimeError("target OM returned a negative token ID")
        if self.ordinary_only:
            return GenerationStep(
                token_ids=(ordinary_next,),
                finished=ordinary_next in set(eos_token_ids),
                metadata={"graph_calls": 1, "mode": "ordinary-greedy"},
            )
        if max_new_tokens == 1:
            return GenerationStep(
                token_ids=(ordinary_next,),
                finished=ordinary_next in set(eos_token_ids),
                metadata={"graph_calls": 1, "mode": "ordinary-tail"},
            )

        proposal_count = min(
            int(max_draft_tokens),
            self.available_draft_tokens,
            max_new_tokens - 1,
        )
        proposals = [int(item) for item in draft[0, :proposal_count]]
        eos = set(int(item) for item in eos_token_ids)
        if any(token < 0 for token in proposals):
            raise RuntimeError("draft OM returned a negative token ID")
        for index, token in enumerate(proposals):
            if token in eos:
                proposals = proposals[: index + 1]
                break
        proposal_count = len(proposals)
        if proposal_count == 0:
            return GenerationStep(
                token_ids=(ordinary_next,),
                finished=ordinary_next in eos,
                metadata={"graph_calls": 1, "mode": "ordinary-no-proposal"},
            )

        verify_target, _next_draft = self._run([*prefix, *proposals])
        base = len(prefix)
        target_predictions = [
            int(verify_target[0, base - 1 + index])
            for index in range(proposal_count)
        ]
        if target_predictions[0] != ordinary_next:
            raise RuntimeError(
                "target OM changed its next token between proposal and verify recomputes"
            )
        accepted = 0
        for proposal, prediction in zip(proposals, target_predictions):
            if proposal != prediction:
                break
            accepted += 1

        if accepted < proposal_count:
            correction = target_predictions[accepted]
            committed = [*proposals[:accepted], correction]
            return GenerationStep(
                token_ids=tuple(committed),
                drafted_tokens=proposal_count,
                accepted_draft_tokens=accepted,
                rejected_draft_tokens=proposal_count - accepted,
                finished=correction in eos,
                metadata={
                    "graph_calls": 2,
                    "mode": "draft-verify-correction",
                    "target_predictions": target_predictions,
                },
            )

        if proposals[-1] in eos:
            return GenerationStep(
                token_ids=tuple(proposals),
                drafted_tokens=proposal_count,
                accepted_draft_tokens=accepted,
                finished=True,
                metadata={
                    "graph_calls": 2,
                    "mode": "draft-verify-eos",
                    "target_predictions": target_predictions,
                },
            )
        bonus = int(verify_target[0, base + proposal_count - 1])
        if bonus < 0:
            raise RuntimeError("target OM returned a negative bonus token ID")
        committed = [*proposals, bonus]
        return GenerationStep(
            token_ids=tuple(committed),
            drafted_tokens=proposal_count,
            accepted_draft_tokens=accepted,
            finished=bonus in eos,
            metadata={
                "graph_calls": 2,
                "mode": "draft-verify-bonus",
                "target_predictions": target_predictions,
            },
        )

    def close(self) -> None:
        self.runtime.close()


def create_backend(
    *,
    bundle_dir: Path,
    manifest: Mapping[str, Any],
    device_id: int,
    options: Mapping[str, Any],
) -> RecomputeDFlashOmBackend:
    """Factory consumed directly by ``infer-om --backend``."""

    del manifest
    required = ("device_model", "cann", "driver", "firmware", "runtime")
    missing = [name for name in required if not str(options.get(name, "")).strip()]
    if missing:
        raise ValueError(f"recompute backend options are missing identities: {missing}")
    ordinary_only = options.get("ordinary_only", False)
    if not isinstance(ordinary_only, bool):
        raise TypeError("ordinary_only must be a JSON boolean")
    runtime = AclOmRuntime(bundle_dir / "deployment-manifest.json", device_id=device_id)
    try:
        return RecomputeDFlashOmBackend(
            runtime,
            graph_name=str(options.get("graph_name", "dflash_recompute")),
            pad_token_id=int(options.get("pad_token_id", 0)),
            device={
                "target_id": "ascend310p",
                "model": str(options["device_model"]),
                "device_id": int(device_id),
            },
            cann=str(options["cann"]),
            driver=str(options["driver"]),
            firmware=str(options["firmware"]),
            runtime_identity=str(options["runtime"]),
            ordinary_only=ordinary_only,
        )
    except BaseException:
        runtime.close()
        raise
