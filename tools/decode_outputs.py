#!/usr/bin/env python3
"""Print saved ordinary/DFlash output text; does not load OMs or run inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "framework/python"), str(REPO)]

from tools.benchmark_prompts import decode_outputs, render_outputs


def saved_cases(path, prompt_ids=None):
    path = path.expanduser().resolve()
    if path.is_dir():
        path = path / "runner-batch.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if "cases" in report:
        cases = []
        directory = Path(str(path) + ".cases")
        for item in report["cases"]:
            name = item["id"]
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
                raise ValueError("unsafe case id in batch index")
            case_path = directory / (name + ".json")
            if Path(item["report"]).resolve() != case_path.resolve():
                raise ValueError("unexpected case report path in batch index")
            cases.append((name, case_path))
    else:
        cases = [(path.stem.removesuffix(".ordinary"), path)]
    if prompt_ids:
        unknown = set(prompt_ids) - {name for name, _ in cases}
        if unknown:
            raise ValueError(f"unknown prompt id(s): {', '.join(sorted(unknown))}")
        cases = [(name, case_path) for name, case_path in cases if name in prompt_ids]
    result = []
    for name, case_path in cases:
        value = json.loads(case_path.read_text(encoding="utf-8"))
        # Older parity failures kept only an error plus a separate ordinary
        # report. Recover that output; never invent the missing DFlash tokens.
        ordinary_path = case_path.with_name(name + ".ordinary.json")
        if "ordinary" not in value and ordinary_path != case_path and ordinary_path.is_file():
            value["ordinary"] = json.loads(ordinary_path.read_text(encoding="utf-8"))
        result.append((name, value))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True,
                        help="paired report, *.ordinary.json, runner-batch.json, or prompt-suite directory")
    parser.add_argument("--model-dir", type=Path, required=True, help="the same tokenizer used for generation")
    parser.add_argument("--prompt-id", action="append", help="decode only these batch IDs; repeatable")
    args = parser.parse_args()
    try:
        from qwen35_dflash.ascend310p.workflow import load_tokenizer

        cases = saved_cases(args.report, args.prompt_id)
        tokenizer, _ = load_tokenizer(model_dir=args.model_dir)
        rows = [{"id": name, "status": value.get("status", "UNKNOWN"),
                 "prompt": tokenizer.decode(value["prompt_token_ids"], skip_special_tokens=False)
                           if value.get("prompt_token_ids") else None,
                 "error": value.get("error"),
                 "first_difference": value.get("ordinary_parity", {}).get("first_difference"),
                 "decoded_outputs": decode_outputs(value, tokenizer)} for name, value in cases]
        print(render_outputs(rows), end="")
        return 0  # Decoding succeeded; this does not change the saved parity status.
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(2, f"decode-outputs: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
