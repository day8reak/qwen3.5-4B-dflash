"""Typed boundaries shared by the AIR exporter and OM generation runner."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import torch


_GRAPH_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_TORCH_OPERATOR_NAME = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*::[A-Za-z_][A-Za-z0-9_]*$"
)
_TORCH_OVERLOAD_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_GE_OPERATOR_TYPE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class CustomOpExportSpec:
    """One front-end custom operator that must remain one GE graph node."""

    torch_op: str
    ge_op_type: str
    overload: str = "default"
    minimum_occurrences: int = 1

    def __post_init__(self) -> None:
        if not _TORCH_OPERATOR_NAME.fullmatch(self.torch_op):
            raise ValueError(f"invalid torch custom-operator name: {self.torch_op!r}")
        if not _TORCH_OVERLOAD_NAME.fullmatch(self.overload):
            raise ValueError(f"invalid torch overload name: {self.overload!r}")
        if not _GE_OPERATOR_TYPE.fullmatch(self.ge_op_type):
            raise ValueError(f"invalid GE operator type: {self.ge_op_type!r}")
        if (
            isinstance(self.minimum_occurrences, bool)
            or not isinstance(self.minimum_occurrences, int)
            or self.minimum_occurrences < 0
        ):
            raise ValueError("minimum_occurrences must be a non-negative integer")

    @property
    def torch_target(self) -> str:
        namespace, name = self.torch_op.split("::", 1)
        return f"{namespace}.{name}.{self.overload}"


@dataclass(frozen=True)
class AirGraphSpec:
    """One Torch module and representative call to export as one AIR graph.

    The built-in first route returns one integrated target+DFlash recompute
    graph.  A later incremental deployment may instead return separate target,
    verify and draft graphs while preserving its frozen cache/state ABI.
    """

    name: str
    role: str
    model: torch.nn.Module
    example_args: tuple[Any, ...]
    example_kwargs: Mapping[str, Any] = field(default_factory=dict)
    input_names: tuple[str, ...] = ()
    output_names: tuple[str, ...] = ()
    dynamic: bool = False
    input_dim_gears: Mapping[int, Mapping[int, tuple[int, ...]]] = field(
        default_factory=dict
    )
    compiler_config: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    custom_ops: tuple[CustomOpExportSpec, ...] = ()

    def __post_init__(self) -> None:
        if not _GRAPH_NAME.fullmatch(self.name):
            raise ValueError(f"invalid AIR graph name: {self.name!r}")
        if not self.role or not _GRAPH_NAME.fullmatch(self.role):
            raise ValueError(f"invalid AIR graph role: {self.role!r}")
        if not isinstance(self.model, torch.nn.Module):
            raise TypeError("AirGraphSpec.model must be a torch.nn.Module")
        if not isinstance(self.example_args, tuple):
            raise TypeError("AirGraphSpec.example_args must be a tuple")
        reserved = {
            "model",
            "export_path",
            "export_name",
            "dynamic",
            "config",
        }
        overlap = sorted(reserved.intersection(self.example_kwargs))
        if overlap:
            raise ValueError(f"example kwargs use TorchAir control names: {overlap}")
        if len(set(self.input_names)) != len(self.input_names):
            raise ValueError("AIR input names must be unique")
        if len(set(self.output_names)) != len(self.output_names):
            raise ValueError("AIR output names must be unique")
        if not isinstance(self.input_dim_gears, Mapping):
            raise TypeError("AirGraphSpec.input_dim_gears must be a mapping")
        if self.input_dim_gears and not self.dynamic:
            raise ValueError("AIR dimension gears require dynamic=True")
        for input_index, dimensions in self.input_dim_gears.items():
            if (
                isinstance(input_index, bool)
                or not isinstance(input_index, int)
                or input_index < 0
                or input_index >= len(self.example_args)
            ):
                raise ValueError("AIR dimension gear input index is invalid")
            value = self.example_args[input_index]
            if not isinstance(value, torch.Tensor):
                raise TypeError("AIR dimension gears require a Tensor argument")
            if not isinstance(dimensions, Mapping) or not dimensions:
                raise TypeError("AIR dimension gear entries must be non-empty mappings")
            for dimension, gears in dimensions.items():
                if (
                    isinstance(dimension, bool)
                    or not isinstance(dimension, int)
                    or dimension < 0
                    or dimension >= value.dim()
                ):
                    raise ValueError("AIR dimension gear axis is invalid")
                if not isinstance(gears, tuple) or not 2 <= len(gears) <= 100:
                    raise ValueError("AIR dimension gears must contain 2..100 values")
                if any(
                    isinstance(gear, bool)
                    or not isinstance(gear, int)
                    or gear <= 0
                    for gear in gears
                ):
                    raise ValueError("AIR dimension gears must be positive integers")
                if len(set(gears)) != len(gears):
                    raise ValueError("AIR dimension gears must be unique")
                if int(value.shape[dimension]) not in gears:
                    raise ValueError("AIR example shape must select one declared gear")
        if not isinstance(self.custom_ops, tuple):
            raise TypeError("AirGraphSpec.custom_ops must be a tuple")
        if not all(isinstance(item, CustomOpExportSpec) for item in self.custom_ops):
            raise TypeError("AirGraphSpec.custom_ops contains an invalid item")
        targets = [item.torch_target for item in self.custom_ops]
        if len(set(targets)) != len(targets):
            raise ValueError("AIR custom-operator targets must be unique")


@dataclass(frozen=True)
class GenerationStep:
    """Tokens committed by one synchronized prefill or decode invocation."""

    token_ids: tuple[int, ...]
    drafted_tokens: int = 0
    accepted_draft_tokens: int = 0
    rejected_draft_tokens: int = 0
    finished: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        tokens = tuple(int(token) for token in self.token_ids)
        object.__setattr__(self, "token_ids", tokens)
        if not tokens:
            raise ValueError("a generation step must commit at least one token")
        if any(token < 0 for token in tokens):
            raise ValueError("token IDs must be non-negative")
        counters = (
            self.drafted_tokens,
            self.accepted_draft_tokens,
            self.rejected_draft_tokens,
        )
        if any(int(value) < 0 for value in counters):
            raise ValueError("generation counters must be non-negative")
        if self.accepted_draft_tokens > self.drafted_tokens:
            raise ValueError("accepted draft count exceeds drafted count")
        if self.rejected_draft_tokens > self.drafted_tokens:
            raise ValueError("rejected draft count exceeds drafted count")


@runtime_checkable
class DFlashOmBackend(Protocol):
    """High-level adapter around the deployment's concrete OM graph suite.

    The backend owns graph-specific tensors and DFlash acceptance/cache rules.
    The framework owns strict fallback checks, tokenizer I/O, synchronization,
    generation limits, reproducibility checks, and stage timing.
    """

    backend_id: str

    def metadata(self) -> Mapping[str, Any]: ...

    def synchronize(self) -> None: ...

    def reset(self) -> None: ...

    def prefill(
        self,
        prompt_token_ids: Sequence[int],
        *,
        max_new_tokens: int,
        eos_token_ids: Sequence[int],
    ) -> GenerationStep: ...

    def decode(
        self,
        committed_token_ids: Sequence[int],
        *,
        max_new_tokens: int,
        max_draft_tokens: int,
        eos_token_ids: Sequence[int],
    ) -> GenerationStep: ...

    def close(self) -> None: ...
