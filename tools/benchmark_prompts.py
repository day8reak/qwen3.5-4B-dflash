#!/usr/bin/env python3
"""Paired multi-prompt acceptance/latency test with models reused in one C++ process."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "framework/python"), str(REPO)]

DEFAULT_PROMPTS = [
    {"id": "zh_explain", "category": "中文解释", "prompt": "请用通俗的中文解释什么是机器学习，并用一个生活中的例子说明训练和推理的区别。"},
    {"id": "zh_plan", "category": "中文规划", "prompt": "请给刚开始学习 Python 的初学者制定一周学习计划，每天列出学习内容和一个练习。"},
    {"id": "math", "category": "数学推理", "prompt": "一辆汽车前半程以每小时40公里行驶，后半程以每小时60公里行驶，全程120公里。求总用时和平均速度，并逐步解释。"},
    {"id": "code", "category": "代码生成", "prompt": "用 Python 编写二分查找函数：在升序数组中查找目标值，找到返回下标，否则返回 -1。给出代码、两个测试例子和时间复杂度。"},
    {"id": "translate", "category": "英译中", "prompt": "将下面英文翻译成自然的中文，再解释其中的技术术语：Speculative decoding uses a smaller model to propose several tokens. A larger model checks these proposals in parallel and accepts a matching prefix. The benefit depends on both acceptance and verification cost."},
    {"id": "summary", "category": "摘要", "prompt": "请将以下内容概括为三点：某社区将闲置空地改成公共花园。居民轮流浇水，孩子参与种植。花园设有雨水收集桶以节约用水。社区还每月举办一次交流活动，但需要解决工具保管和长期维护经费的问题。"},
    {"id": "en_explain", "category": "英文解释", "prompt": "Explain the difference between a process and a thread to a beginner. Include an example and discuss memory sharing and isolation."},
    {"id": "creative", "category": "中文创作", "prompt": "写一个约200字的小故事：一位图书管理员在旧书中发现一张没有署名的地图。故事要有清晰的开头、转折和结尾。"},
]


def load_prompts(path, prompt_ids=None):
    values = json.loads(path.read_text()) if path else DEFAULT_PROMPTS
    if not isinstance(values, list) or not 1 <= len(values) <= 64:
        raise ValueError("prompts must be a JSON list of 1..64 strings or {id, prompt, category} objects")
    result, seen = [], set()
    for i, value in enumerate(values):
        if not isinstance(value, (str, dict)):
            raise ValueError("each prompt must be a string or an object")
        item = {"prompt": value} if isinstance(value, str) else dict(value)
        name = item.get("id", f"prompt_{i + 1:02d}")
        prompt = item.get("prompt")
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name)
                or name in seen or not isinstance(prompt, str) or not prompt.strip()):
            raise ValueError("each prompt needs nonempty text and a unique safe id")
        seen.add(name)
        result.append({"id": name, "category": str(item.get("category", "custom")), "prompt": prompt})
    if prompt_ids:
        unknown = set(prompt_ids) - seen
        if unknown:
            raise ValueError(f"unknown prompt id(s): {', '.join(sorted(unknown))}")
        result = [item for item in result if item["id"] in prompt_ids]
    return result


def decode_outputs(report, tokenizer):
    """Decode saved tokens, including specials, without changing parity results."""
    outputs = {}
    for mode in ("ordinary", "dflash"):
        benchmark = report.get(mode)
        # Also accept a legacy per-mode *.ordinary.json or --mode report.
        standalone = report.get("benchmark", report)
        if benchmark is None and standalone.get("generation_mode", "").startswith(mode):
            benchmark = standalone
        tokens = benchmark.get("stable_generated_token_ids") if benchmark else None
        if not isinstance(tokens, list):
            outputs[mode] = {"available": False}
            continue
        if any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError("invalid saved generated token IDs")
        outputs[mode] = {"available": True, "token_count": len(tokens),
                         "stop_reason": benchmark.get("stable_stop_reason"),
                         "text": tokenizer.decode(tokens, skip_special_tokens=False)}
        difference = report.get("ordinary_parity", {}).get("first_difference")
        if difference and difference.get("field") == "generated_token_ids":
            index = difference["index"]
            begin, end = max(0, index - 12), min(len(tokens), index + 13)
            outputs[mode]["difference_context"] = {
                "token_begin": begin, "token_end": end,
                "token_ids": tokens[begin:end],
                "text": tokenizer.decode(tokens[begin:end], skip_special_tokens=False),
            }
    return outputs


def render_outputs(rows):
    lines = []
    for row in rows:
        lines += [f"=== {row.get('id', 'report')} | {row.get('status', 'UNKNOWN')} ==="]
        if row.get("prompt"):
            lines += ["Prompt:", row["prompt"]]
        if row.get("error"):
            lines += ["Error: " + row["error"]]
        difference = row.get("first_difference")
        if difference:
            if difference.get("field") == "generated_token_ids":
                lines += [f"First different generated token (zero-based): {difference['index']}; "
                          f"ordinary={difference['ordinary_token_id']}, dflash={difference['dflash_token_id']}"]
            else:
                lines += ["First difference: " + json.dumps(difference, ensure_ascii=False)]
        for mode in ("ordinary", "dflash"):
            value = row.get("decoded_outputs", {}).get(mode, {})
            if not value.get("available"):
                lines += [f"--- {mode}: output token IDs unavailable in saved report ---"]
            else:
                lines += [f"--- {mode}: {value['token_count']} tokens, stop={value['stop_reason']} ---", value["text"]]
                context = value.get("difference_context")
                if context:
                    lines += [f"First-difference context, tokens [{context['token_begin']}, {context['token_end']}):",
                              context["text"]]
        lines += [""]
    return "\n".join(lines) + "\n"


def observed_acceptance(report):
    """Read current-verifier acceptance, independently of the pair's PASS/FAIL."""
    draft = report.get("dflash")
    standalone = report.get("benchmark", report)
    if draft is None and standalone.get("generation_mode", "").startswith("dflash"):
        draft = standalone
    if not isinstance(draft, dict):
        return {"available": False}
    measurements = draft.get("measurements")
    if measurements:
        # WriteBenchmark stores only measured calls here, not the warmup calls.
        counters = [m["counters"] for m in measurements]
        source = "measurements"
    elif isinstance(draft.get("totals"), dict):
        counters = [draft["totals"]]
        source = "totals"
    else:
        return {"available": False}
    for count in counters:
        proposed, accepted = count["drafted_tokens"], count["accepted_draft_tokens"]
        if (type(proposed) is not int or type(accepted) is not int
                or not 0 <= accepted <= proposed):
            raise ValueError("invalid saved DFlash acceptance counters")
    proposed = sum(c["drafted_tokens"] for c in counters)
    accepted = sum(c["accepted_draft_tokens"] for c in counters)
    return {"available": True, "source": source,
            "drafted_tokens": proposed, "accepted_draft_tokens": accepted,
            "acceptance_rate": accepted / proposed if proposed else None,
            "scope": "current verify decisions; measured repetitions only, warmups excluded"}


def render_acceptance(rows):
    """Display failed runs too, without admitting them into validated statistics."""
    lines = ["Observed DFlash acceptance (current verify):", "",
             "| Prompt | Status | Accepted / proposed | Acceptance |",
             "|---|---|---:|---:|"]
    available = []
    for row in rows:
        stats = row.get("observed_acceptance", {})
        if stats.get("available"):
            available.append(stats)
            ratio = stats["acceptance_rate"]
            rate = f"{ratio:.2%}" if ratio is not None else "N/A (no proposals)"
            counts = f'{stats["accepted_draft_tokens"]} / {stats["drafted_tokens"]}'
        else:
            counts, rate = "—", "N/A (not recorded)"
        lines.append(f"| {row.get('id', 'report')} | {row.get('status', 'UNKNOWN')} | {counts} | {rate} |")
    proposed = sum(s["drafted_tokens"] for s in available)
    accepted = sum(s["accepted_draft_tokens"] for s in available)
    rate = f"{accepted / proposed:.2%}" if proposed else "N/A"
    lines += ["", f"Observed weighted acceptance: {rate}; accepted={accepted}, proposed={proposed}; "
              f"reports with counters={len(available)}/{len(rows)}.",
              "Counts sum measured repetitions and exclude warmups.",
              "FAIL remains FAIL. These are current verify decisions, including failed parity runs; "
              "they do not establish ordinary parity or speedup."]
    return "\n".join(lines) + "\n"


def summarize_prompt(report):
    ordinary, draft = report["ordinary"], report["dflash"]
    measurements = draft["measurements"]
    drafted = sum(m["counters"]["drafted_tokens"] for m in measurements)
    accepted = sum(m["counters"]["accepted_draft_tokens"] for m in measurements)
    rounds = [r for m in measurements for r in m["rounds"] if r["proposed_token_ids"]]
    emitted = sum(len(r["emitted_token_ids"]) for r in rounds)
    ordinary_ms = sum(m["latency_ms"]["model_total"] for m in ordinary["measurements"])
    dflash_ms = sum(m["latency_ms"]["model_total"] for m in measurements)
    if not 0 <= accepted <= drafted or ordinary_ms <= 0 or dflash_ms <= 0:
        raise ValueError("invalid acceptance counters or model latency")
    return {
        "drafted_tokens": drafted, "accepted_draft_tokens": accepted,
        "acceptance_rate": accepted / drafted if drafted else None,
        "speculative_rounds": len(rounds), "tokens_emitted_in_speculative_rounds": emitted,
        "tokens_per_speculative_round": emitted / len(rounds) if rounds else None,
        "target_only_fallback_rounds": sum(m["counters"]["target_only_fallback_rounds"] for m in measurements),
        "speculation_disable_events": sum(m["counters"]["speculation_disable_events"] for m in measurements),
        "generated_tokens": len(draft["stable_generated_token_ids"]),
        "stop_reason": draft["stable_stop_reason"],
        "ordinary_total_measured_ms": ordinary_ms, "dflash_total_measured_ms": dflash_ms,
        "ordinary_median_ms": ordinary["latency_ms"]["model_total"]["median"],
        "dflash_median_ms": draft["latency_ms"]["model_total"]["median"],
        "speedup": ordinary["latency_ms"]["model_total"]["median"] / draft["latency_ms"]["model_total"]["median"],
        "dflash_tokens_per_second": draft["generated_tokens_per_second"],
    }


def aggregate(rows):
    good = [r for r in rows if r["status"] == "PASS"]
    drafted = sum(r["drafted_tokens"] for r in good)
    accepted = sum(r["accepted_draft_tokens"] for r in good)
    latency = sum(r["dflash_total_measured_ms"] for r in good)
    return {"scope": "passing prompts only; inspect failures before interpreting the suite",
            "passed_prompts": len(good), "failed_prompts": sum(r["status"] == "FAIL" for r in rows),
            "not_run_prompts": sum(r["status"] == "NOT_RUN" for r in rows),
            "drafted_tokens": drafted, "accepted_draft_tokens": accepted,
            "weighted_acceptance_rate": accepted / drafted if drafted else None,
            "total_model_time_speedup": sum(r["ordinary_total_measured_ms"] for r in good) / latency if latency else None}


def markdown(summary):
    def value(number, percent=False):
        return "N/A" if number is None else f"{number * 100:.2f}%" if percent else f"{number:.2f}"
    lines = ["| Prompt | Status | Acceptance | Tokens / speculative round | Draft tok/s | Speedup | Generated |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in summary["cases"]:
        if row["status"] != "PASS":
            lines.append(f"| {row['id']} | {row['status']} | — | — | — | — | — |")
        else:
            lines.append(f"| {row['id']} | PASS | {value(row['acceptance_rate'], True)} | "
                         f"{value(row['tokens_per_speculative_round'])} | {value(row['dflash_tokens_per_second'])} | "
                         f"{value(row['speedup'])}x | {row['generated_tokens']} |")
    totals = summary["aggregate"]
    lines += ["", f"Passed: {totals['passed_prompts']}; failed: {totals['failed_prompts']}; not run: {totals['not_run_prompts']}.",
              f"Weighted acceptance: {value(totals['weighted_acceptance_rate'], True)}.",
              "Acceptance = accepted draft tokens / proposed draft tokens; warmups excluded.",
              "Speedup > 1 means faster than ordinary generation; acceptance alone does not establish speedup."]
    if totals["passed_prompts"]:
        lines += ["Each passing prompt passed token/EOS parity and the existing 3+10 repeatability gates."]
    else:
        lines += ["No prompt has passed the complete checks; no validated acceptance or speedup is available."]
    if any(r.get("observed_acceptance", {}).get("available") for r in summary["cases"]):
        lines += ["", render_acceptance(summary["cases"]).rstrip()]
    for row in summary["cases"]:
        if row.get("error"):
            lines += ["", f"- {row['id']} ({row.get('failure_stage', 'report_validation')}): "
                      + row["error"].replace("\n", " ")]
    return "\n".join(lines) + "\n"


def run_runner(command, log):
    with log.open("x") as stream:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                stream.write(line)
                if line.startswith("[prompt-batch]"):
                    print(line.rstrip(), flush=True)
            return process.wait()


def run(args):
    from qwen35_dflash.ascend310p.cpp_runtime import resolve_cpp_runner, validate_cpp_runner_options, validate_cpp_runner_report
    from qwen35_dflash.ascend310p.generation import tokenize_prompt
    from qwen35_dflash.ascend310p.incremental_plan import write_incremental_plan
    from qwen35_dflash.ascend310p.utils import atomic_write_json, require_run_output, sha256_file
    from qwen35_dflash.ascend310p.workflow import load_tokenizer

    if os.environ.get("ASCEND310P_SIMULATION_ONLY") == "1" or os.environ.get("PROFILING_MODE") not in (None, "", "false"):
        raise ValueError("batch target test requires real Ascend310P outside msprof")
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir() or run_dir.is_relative_to(REPO):
        raise ValueError("run-dir must be an existing directory outside the repository")
    os.environ["AI_RUN_DIR"] = str(run_dir)
    if args.device_id < 0 or not 1 <= args.max_draft_tokens <= 15 or args.max_new_tokens <= 0:
        raise ValueError("invalid device/token limits")
    prompts = load_prompts(args.prompts, getattr(args, "prompt_id", None))
    manifest = (args.deployment_manifest or run_dir / "artifacts/deployment-manifest.json").resolve()
    config = json.loads((args.runner_config or run_dir / "runner.json").read_text())
    identity = validate_cpp_runner_options(config, args.device_id)
    executable = resolve_cpp_runner(args.runner)
    help_result = subprocess.run([str(executable), "--help"], capture_output=True, text=True)
    if help_result.returncode or "--prompt-batch" not in help_result.stdout:
        raise RuntimeError("rebuild the C++ runner: --prompt-batch support is required")
    root = require_run_output(Path(tempfile.mkdtemp(prefix="prompt-suite-", dir=run_dir)))
    print(f"Output: {root}", flush=True)
    plan, deployment, contract = write_incremental_plan(manifest, root / "chunk-plan.txt")
    tokenizer, tokenizer_source = load_tokenizer(model_dir=args.model_dir)
    eos = args.eos_token_id or [248044]
    if any(token < 0 or token >= contract["vocab_size"] for token in eos):
        raise ValueError("EOS token outside model vocabulary")
    for item in prompts:
        tokens = tokenize_prompt(tokenizer, item["prompt"], chat=args.chat)
        if (not tokens or len(tokens) + args.max_new_tokens > contract["capacity"]
                or any(token < 0 or token >= contract["vocab_size"] for token in tokens)):
            raise ValueError(f"prompt {item['id']} exceeds vocabulary/context limit; shorten it or reduce max-new-tokens")
        item["prompt_token_ids"] = tokens
    batch = root / "prompts.txt"
    batch.write_text("QWEN35_PROMPT_BATCH_V1\n" + "".join(
        f"{p['id']} \"{','.join(map(str, p['prompt_token_ids']))}\"\n" for p in prompts))
    index = root / "runner-batch.json"
    plan_hash, batch_hash = sha256_file(plan), sha256_file(batch)
    command = [str(executable), "--model-kind", "chunk", "--mode", "paired", "--model", str(plan),
        "--model-sha256", plan_hash, "--prompt-batch", str(batch), "--prompt-batch-sha256", batch_hash,
        "--output", str(index), "--eos-token-ids", ",".join(map(str, eos)),
        "--pad-token-id", str(identity["pad_token_id"]), "--device-id", str(args.device_id),
        "--max-new-tokens", str(args.max_new_tokens), "--max-draft-tokens", str(args.max_draft_tokens),
        "--warmup", "3", "--repetitions", "10", "--trace-rounds"]
    if args.low_memory:
        command.append("--low-memory")
    request = {"schema_version": 1, "prompts": prompts, "chat": args.chat, "eos_token_ids": eos,
        "max_new_tokens": args.max_new_tokens, "max_draft_tokens": args.max_draft_tokens,
        "runtime_identity": identity, "tokenizer_source": tokenizer_source,
        "script_sha256": sha256_file(Path(__file__)),
        "runner": {"path": str(executable), "sha256": sha256_file(executable)},
        "deployment_manifest": {"path": str(manifest), "sha256": sha256_file(manifest)},
        "om_sha256": {g["name"]: g["om"]["sha256"] for g in deployment["graphs"]},
        "draft_atc_command": next(g["atc_command"] for g in deployment["graphs"] if g["name"] == "draft"),
        "command": command, "model_load_policy": "reuse models across all prompts"}
    atomic_write_json(root / "request.json", request)
    start = time.monotonic()
    exit_code = run_runner(command, root / "runner.log")
    process_wall_seconds = time.monotonic() - start
    raw = json.loads(index.read_text()) if index.exists() else {}
    index_error = None
    if (exit_code not in (0, 1) or raw.get("fake_acl") is not False
            or raw.get("prompt_batch_sha256") != batch_hash or raw.get("model_sha256") != plan_hash
            or [c["id"] for c in raw.get("cases", [])] != [p["id"] for p in prompts]):
        index_error = "missing/invalid batch index or fake ACL; inspect runner.log"
    rows = []
    for i, prompt in enumerate(prompts):
        row = {**prompt, "status": "FAIL"}
        try:
            if index_error:
                raise RuntimeError(index_error)
            case = raw["cases"][i]
            path = Path(case["report"]).resolve()
            if path != (Path(str(index) + ".cases") / (prompt["id"] + ".json")).resolve():
                raise ValueError("unexpected case report path")
            report = json.loads(path.read_text())
            row.update(raw_report=str(path), decoded_outputs=decode_outputs(report, tokenizer),
                       observed_acceptance=observed_acceptance(report))
            if case["status"] != "PASS":
                # Failed parity reports retain both mode measurements. They
                # are diagnostic output, never feed them into PASS statistics.
                row.update(status="NOT_RUN" if case["status"] == "NOT_RUN" else "FAIL",
                    failure_stage=report.get("failure_stage", "unknown"),
                    first_difference=report.get("ordinary_parity", {}).get("first_difference"),
                    error=report.get("error", "C++ prompt failed"))
                rows.append(row)
                continue
            validate_cpp_runner_report(report, prompt_token_ids=prompt["prompt_token_ids"],
                om_sha256=plan_hash, device_id=args.device_id, max_new_tokens=args.max_new_tokens,
                max_draft_tokens=args.max_draft_tokens, chunk_abi=True, low_memory=args.low_memory)
            if report["eos_token_ids"] != eos or report["protocol"].get("round_trace_enabled") is not True:
                raise ValueError("EOS or trace settings differ")
            row.update(summarize_prompt(report))
            row.update(status="PASS", raw_report=str(path), generated_text=tokenizer.decode(
                report["dflash"]["stable_generated_token_ids"], skip_special_tokens=True))
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            row["error"] = str(error)
        rows.append(row)
    ok = exit_code == 0 and raw.get("status") == "PASS" and all(r["status"] == "PASS" for r in rows)
    summary = {"schema_version": 1, "status": "PASS" if ok else "FAIL_OR_INCOMPLETE",
        "cases": rows, "aggregate": aggregate(rows), "protocol": {
            "warmup": 3, "repetitions": 10, "low_memory": args.low_memory,
            "dflash_speculation_policy": raw.get("dflash_speculation_policy", "not_recorded"),
            "models_reused_across_prompts": raw.get("models_reused_across_prompts"), "order": raw.get("order")},
        "startup_ms": raw.get("startup_ms"), "process_wall_seconds": process_wall_seconds,
        "request": str(root / "request.json"), "runner_index": str(index), "runner_exit_code": exit_code,
        "ordinary_parity": "PASS" if ok else "FAIL_OR_INCOMPLETE",
        "scope": "This selected prompt suite; no general acceptance-rate or speedup claim beyond it."}
    atomic_write_json(root / "summary.json", summary)
    (root / "summary.md").write_text(markdown(summary))
    (root / "generations.txt").write_text(render_outputs(rows), encoding="utf-8")
    print(markdown(summary), flush=True)
    print(f"Summary: {root / 'summary.json'}", flush=True)
    print(f"Decoded outputs: {root / 'generations.txt'}", flush=True)
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=os.environ.get("AI_RUN_DIR"), required=not os.environ.get("AI_RUN_DIR"))
    parser.add_argument("--runner", type=Path, default=os.environ.get("CPP_RUNNER"), required=not os.environ.get("CPP_RUNNER"))
    parser.add_argument("--deployment-manifest", type=Path)
    parser.add_argument("--runner-config", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, help="optional JSON list; defaults to eight varied prompts")
    parser.add_argument("--prompt-id", action="append", help="run only selected ID(s); repeatable")
    parser.add_argument("--chat", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eos-token-id", type=int, action="append", help="repeatable; default 248044")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-draft-tokens", type=int, default=15)
    parser.add_argument("--low-memory", action="store_true")
    try:
        return run(parser.parse_args())
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(2, f"prompt-suite: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
