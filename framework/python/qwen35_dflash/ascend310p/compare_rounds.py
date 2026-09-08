"""Compare native and C++ DFlash rounds at the same committed token prefix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .utils import atomic_write_json, load_json_object, require_run_output, sha256_file


ROUND_FIELDS = (
    "proposed_token_ids", "target_token_ids", "accepted_draft_token_ids",
    "emitted_token_ids", "fallback_token_id",
)


def _rounds_by_prefix(
    rounds: list[dict[str, Any]], prompt: list[int], generated: list[int],
) -> dict[int, dict[str, Any]]:
    """Validate the trace before using prefix length as its lookup key."""
    output: list[int] = []
    indexed = {}
    for index, row in enumerate(rounds):
        prefix = len(prompt) + len(output)
        if row.get("committed_prefix_length") != prefix:
            raise ValueError(f"round {index}: committed_prefix_length disagrees with emitted tokens")
        if any(field not in row for field in ROUND_FIELDS):
            raise ValueError(f"round {index}: incomplete token trace")
        proposal, target = row["proposed_token_ids"], row["target_token_ids"]
        accepted, emitted = row["accepted_draft_token_ids"], row["emitted_token_ids"]
        count = len(accepted)
        if len(target) != len(proposal) + 1:
            raise ValueError(f"round {index}: verify rows disagree with proposal length")
        if accepted != proposal[:count] or accepted != target[:count]:
            raise ValueError(f"round {index}: accepted tokens are not a matching prefix")
        fallback = row["fallback_token_id"]
        if fallback is not None and fallback != target[count]:
            raise ValueError(f"round {index}: fallback is not the next Target token")
        if not emitted or emitted != accepted + ([] if fallback is None else [fallback]):
            raise ValueError(f"round {index}: emitted tokens disagree with acceptance")
        indexed[prefix] = {"index": index, "tokens": row}
        output.extend(emitted)
    if output != generated:
        raise ValueError("round trace does not reconstruct generated_token_ids")
    return indexed


def _compare_rounds(native, measurement, native_prompt, cpp_prompt):
    native_rows, cpp_rows = native.get("rounds"), measurement.get("rounds")
    if not native_rows or not cpp_rows:
        return {"status": "NOT_AVAILABLE",
                "reason": "both reports need complete rounds; run C++ with --trace-rounds"}
    if native_prompt != cpp_prompt:
        return {"status": "NOT_COMPARABLE", "reason": "prompt_token_ids differ"}
    native_tokens = native["generated_token_ids"]
    cpp_tokens = measurement["generated_token_ids"]
    lhs = _rounds_by_prefix(native_rows, native_prompt, native_tokens)
    rhs = _rounds_by_prefix(cpp_rows, cpp_prompt, cpp_tokens)
    shared = sorted(lhs.keys() & rhs.keys())
    # Equal lengths alone do not mean equal contexts after a token divergence.
    comparable = [prefix for prefix in shared if
                  native_tokens[:prefix - len(native_prompt)] ==
                  cpp_tokens[:prefix - len(cpp_prompt)]]
    differences = []
    for prefix in comparable:
        left, right = lhs[prefix], rhs[prefix]
        fields = {field: {"native": left["tokens"][field], "cpp": right["tokens"][field]}
                  for field in ROUND_FIELDS if left["tokens"][field] != right["tokens"][field]}
        if fields:
            differences.append({
                "committed_prefix_length": prefix,
                "native_round": left["index"], "cpp_round": right["index"],
                "fields": fields,
            })
    exact = not differences and len(comparable) == len(lhs) == len(rhs)
    return {
        "status": "MATCH" if exact else "DIFFERENT",
        "native_round_count": len(lhs), "cpp_round_count": len(rhs),
        "same_prefix_rounds_compared": len(comparable),
        "native_only_prefix_lengths": sorted(lhs.keys() - rhs.keys()),
        "cpp_only_prefix_lengths": sorted(rhs.keys() - lhs.keys()),
        "different_context_prefix_lengths": sorted(set(shared) - set(comparable)),
        "first_difference": differences[0] if differences else None,
        "differences": differences,
    }


def compare_npu_cpp_rounds(native: dict, cpp: dict) -> dict:
    native_dflash, cpp_dflash = native["dflash"], cpp["dflash"]
    native_prompt = native_dflash["prompt_token_ids"]
    cpp_prompt = cpp["prompt_token_ids"]
    native_eos = native.get("request", {}).get("eos_token_ids")
    cpp_eos = cpp.get("eos_token_ids")
    eos_status = ("NOT_AVAILABLE" if native_eos is None or cpp_eos is None else
                  "MATCH" if set(native_eos) == set(cpp_eos) else "DIFFERENT")
    measurements = []
    for index, measurement in enumerate(cpp_dflash["measurements"]):
        tokens_match = measurement["generated_token_ids"] == native_dflash["generated_token_ids"]
        stop_match = measurement["stop_reason"] == native_dflash["stop_reason"]
        measurements.append({
            "repetition": measurement.get("repetition", index),
            "final_tokens": "MATCH" if tokens_match else "DIFFERENT",
            "stop_reason": "MATCH" if stop_match else "DIFFERENT",
            "rounds": _compare_rounds(native_dflash, measurement, native_prompt, cpp_prompt),
        })
    if not measurements:
        raise ValueError("C++ report contains no DFlash measurements")
    prompt_status = "MATCH" if native_prompt == cpp_prompt else "DIFFERENT"
    exact = prompt_status == eos_status == "MATCH" and all(
        row["final_tokens"] == row["stop_reason"] == row["rounds"]["status"] == "MATCH"
        for row in measurements)
    return {
        "schema_version": 1,
        "status": "MATCH" if exact else "NOT_MATCHED",
        "scope": "token traces only; no timing or intermediate tensor parity claim",
        "prompt_tokens": prompt_status,
        "eos_policy": {"status": eos_status, "native": native_eos, "cpp": cpp_eos},
        "measurements": measurements,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--cpp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = require_run_output(args.output)
    if output.exists():
        raise FileExistsError(f"comparison report already exists: {output}")
    report = compare_npu_cpp_rounds(load_json_object(args.native), load_json_object(args.cpp))
    report["inputs"] = {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in (("native", args.native), ("cpp", args.cpp))
    }
    atomic_write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "MATCH" else 1


if __name__ == "__main__":
    raise SystemExit(main())
