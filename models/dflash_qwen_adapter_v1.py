"""Compatibility entry point for the layered DFlash package.

New code should import ``models.dflash_v1.dflash_qwen_adapter_v1``.  This
module keeps the documented ``python -m models.dflash_qwen_adapter_v1``
command working while making persistent rollback the default execution route.
The old full-prefix implementation remains importable as a correctness oracle.
"""

from .dflash_v1.dflash_qwen_adapter_v1 import *  # noqa: F401,F403
from .dflash_v1.dflash_rollback_adapter import *  # noqa: F401,F403
from .dflash_v1.dflash_rollback_decode import *  # noqa: F401,F403
from .dflash_v1.run_rollback import main as _main


if __name__ == "__main__":
    _main()
