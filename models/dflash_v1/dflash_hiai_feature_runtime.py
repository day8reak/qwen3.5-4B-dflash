"""Legacy runtime sidecar for HIAI DFlash feature outputs.

Feature-enabled forwards need one extra tensor without discarding private HIAI
output fields.  This proxy delegates every existing field/index operation to
the original output and adds only ``dflash_features``.  Ordinary forwards do
not use this type and retain their exact receiver ABI.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from torch import Tensor


class DFlashPassthroughOutput(Mapping[str, Any]):
    """Read-only feature sidecar that preserves a receiver output by delegation."""

    __slots__ = ("_base_output", "dflash_features")

    def __init__(
        self,
        base_output: object,
        dflash_features: Tensor,
        *,
        required_fields: Sequence[str] = (),
    ) -> None:
        if base_output is None:
            raise TypeError("base_output must not be None")
        if isinstance(base_output, DFlashPassthroughOutput):
            raise TypeError("refusing to wrap an existing DFlash feature output")
        if not isinstance(dflash_features, Tensor):
            raise TypeError("dflash_features must be a Tensor")
        for field in required_fields:
            if not _has_field(base_output, field):
                raise TypeError(f"receiver output does not expose required field {field!r}")
        existing = _get_optional_field(base_output, "dflash_features")
        if existing is not None:
            raise TypeError("receiver output already exposes non-empty dflash_features")
        object.__setattr__(self, "_base_output", base_output)
        object.__setattr__(self, "dflash_features", dflash_features)

    @property
    def base_output(self) -> object:
        """The untouched receiver object, available for identity-sensitive code."""

        return self._base_output

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_output, name)

    def __getitem__(self, key: str | int | slice) -> Any:
        if key == "dflash_features":
            return self.dflash_features
        base = self._base_output
        if isinstance(key, str) and not isinstance(base, Mapping):
            return getattr(base, key)
        try:
            return base[key]  # type: ignore[index]
        except (KeyError, IndexError, TypeError):
            if isinstance(key, str):
                return getattr(base, key)
            raise

    def __iter__(self) -> Iterator[str]:
        yielded: set[str] = set()
        for key in _field_names(self._base_output):
            if key == "dflash_features":
                continue
            yielded.add(key)
            yield key
        if "dflash_features" not in yielded:
            yield "dflash_features"

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def to_tuple(self) -> tuple[Any, ...]:
        """Match Transformers-style tuple conversion and append the sidecar."""

        converter = getattr(self._base_output, "to_tuple", None)
        if callable(converter):
            base_values = tuple(converter())
        elif isinstance(self._base_output, Mapping):
            base_values = tuple(self._base_output.values())
        elif isinstance(self._base_output, (tuple, list)):
            base_values = tuple(self._base_output)
        else:
            base_values = tuple(
                getattr(self._base_output, name)
                for name in _field_names(self._base_output)
            )
        return (*base_values, self.dflash_features)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(base_output={self._base_output!r}, "
            f"dflash_features={self.dflash_features!r})"
        )


def _field_names(output: object) -> tuple[str, ...]:
    if isinstance(output, Mapping):
        return tuple(str(key) for key in output.keys())
    values = getattr(output, "__dict__", None)
    if isinstance(values, dict):
        return tuple(name for name in values if not name.startswith("_"))
    return ()


def _has_field(output: object, name: str) -> bool:
    if isinstance(output, Mapping):
        return name in output
    return hasattr(output, name)


def _get_optional_field(output: object, name: str) -> object | None:
    if isinstance(output, Mapping):
        return output.get(name)
    return getattr(output, name, None)


def attach_dflash_features(
    base_output: object,
    dflash_features: Tensor,
    *,
    required_fields: Sequence[str] = (),
) -> DFlashPassthroughOutput:
    """Attach the feature tensor while retaining every target output field."""

    return DFlashPassthroughOutput(
        base_output,
        dflash_features,
        required_fields=required_fields,
    )
