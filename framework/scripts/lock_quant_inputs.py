#!/usr/bin/env python3
"""Create a hash-complete manifest for one quant AIR/OM build."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
FRAMEWORK_PYTHON = ROOT / "framework" / "python"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(FRAMEWORK_PYTHON) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_PYTHON))

from qwen35_dflash.ascend310p.input_manifest import build_quant_input_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--draft-dir", type=Path, required=True)
    parser.add_argument("--quant-config", type=Path, required=True)
    parser.add_argument("--receiver-models-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_quant_input_manifest(
        target_dir=args.target_dir,
        draft_dir=args.draft_dir,
        quant_config=args.quant_config,
        receiver_models_dir=args.receiver_models_dir,
        output=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
