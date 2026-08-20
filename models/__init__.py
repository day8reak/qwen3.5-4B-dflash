"""Qwen3.5-4B model integrations shipped by this repository.

The implementation now lives in :mod:`models.dflash_v1`.  Extending this
package's search path keeps historical imports such as ``models.dflash_weights``
working without leaving a directory full of compatibility files.
"""

from pathlib import Path


_DFLASH_V1_DIRECTORY = str(Path(__file__).resolve().parent / "dflash_v1")
if _DFLASH_V1_DIRECTORY not in __path__:
    __path__.append(_DFLASH_V1_DIRECTORY)
