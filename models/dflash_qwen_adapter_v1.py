"""Compatibility entry point for the layered DFlash V1 package.

New code should import ``models.dflash_v1.dflash_qwen_adapter_v1``.  This
module keeps the previously documented ``python -m models.dflash_qwen_adapter_v1``
command working.
"""

from .dflash_v1.dflash_qwen_adapter_v1 import *  # noqa: F401,F403
from .dflash_v1.dflash_qwen_adapter_v1 import main as _main


if __name__ == "__main__":
    _main()

