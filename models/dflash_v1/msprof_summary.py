"""Standard-library summaries of independent msprof stage exports."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import re


def _tasks(path: Path, capture: Path, window: dict) -> tuple[list[dict], int]:
    tasks, skipped = [], 0
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        names = {"".join(c for c in key.lower() if c.isalnum()): key
                 for key in reader.fieldnames or []}
        if "taskdurationus" not in names:
            raise ValueError(f"Missing Task Duration(us) column in {path}: {reader.fieldnames}")
        for row in reader:
            try:
                duration = float(row[names["taskdurationus"]])
            except (ValueError, TypeError):
                skipped += 1
                continue
            if not math.isfinite(duration) or duration < 0:
                skipped += 1
                continue

            def field(key):
                return (row.get(names.get(key, "")) or "N/A").strip()

            tasks.append({
                "stage": window.get("stage", ""),
                "profile_mode": window.get("profile_mode", ""),
                "profile_backend": window.get("profile_backend", ""),
                "source_csv": str(path.relative_to(capture)),
                "device_id": field("deviceid"), "model_id": field("modelid"),
                "stream_id": field("streamid"), "task_id": field("taskid"),
                "op_name": field("opname"), "op_type": field("optype"),
                "task_type": field("tasktype"), "op_state": field("opstate"),
                "duration_ms": duration / 1000,
                "input_shapes": field("inputshapes"),
                "input_data_types": field("inputdatatypes"),
                "output_shapes": field("outputshapes"),
                "output_data_types": field("outputdatatypes"),
            })
    if not tasks:
        raise ValueError(f"No finite, non-negative task durations in {path}")
    tasks.sort(key=lambda item: item["duration_ms"], reverse=True)
    return tasks, skipped


def summarize_windows(windows: list[dict], output: Path, *, prefix="", top=20) -> str:
    """Keep stages, devices/exports and input/output precision groups separate."""
    if not windows or top < 1 or (prefix and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", prefix)):
        raise ValueError("invalid summary windows, prefix or top count")
    stem = prefix + "-" if prefix else ""
    paths = [output / (stem + name) for name in
             ("operator-types.csv", "operator-tasks.csv", "hotspots.txt")]
    if any(path.exists() or path.is_symlink() for path in paths):
        raise ValueError("operator summary output already exists")
    type_rows, task_rows, lines = [], [], []
    for window in windows:
        capture = Path(window["profile_output"])
        files = sorted(capture.rglob("op_summary*.csv"))
        if not files:
            raise ValueError(f"No op_summary CSV found below {capture}; inspect capture/log")
        for path in files:
            tasks, skipped = _tasks(path, capture, window)
            task_rows.extend(tasks)
            # FLOAT/FLOAT matmul must not be hidden in the FP16 matmul group.
            keys = ("device_id", "model_id", "op_type", "task_type", "op_state",
                    "input_data_types", "output_data_types")
            groups = {}
            for task in tasks:
                groups.setdefault(tuple(task[key] for key in keys), []).append(task["duration_ms"])
            identity = {key: tasks[0][key] for key in
                        ("stage", "profile_mode", "profile_backend", "source_csv")}
            types = [{
                **identity, **dict(zip(keys, key)),
                "count": len(times), "total_ms": math.fsum(times),
                "mean_ms": math.fsum(times) / len(times), "max_ms": max(times),
            } for key, times in groups.items()]
            types.sort(key=lambda item: item["total_ms"], reverse=True)
            type_rows.extend(types)
            lines.extend([
                f"\nStage: {window.get('stage', 'unspecified')} / "
                f"{window.get('profile_mode', '')} / {window.get('profile_backend', '')}",
                f"CSV: {path}", f"Valid tasks: {len(tasks)}; skipped invalid durations: {skipped}",
                "Operator types: count  total_ms  mean_ms  max_ms  OP Type / Task Type / OP State / Input Dtypes -> Output Dtypes",
            ])
            for item in types[:top]:
                lines.append(f"{item['count']:6d} {item['total_ms']:11.3f} {item['mean_ms']:10.3f} "
                             f"{item['max_ms']:10.3f}  {item['op_type']} / {item['task_type']} / "
                             f"{item['op_state']} / {item['input_data_types']} -> {item['output_data_types']}")
            lines.append("Individual tasks: duration_ms  Op Name / OP Type / Task Type / OP State / Input Shapes")
            for item in tasks[:top]:
                lines.append(f"{item['duration_ms']:11.3f}  {item['op_name']} / {item['op_type']} / "
                             f"{item['task_type']} / {item['op_state']} / {item['input_shapes']}")
    lines.append("\nTask-duration sums are not stage wall time: streams can overlap. "
                 "Each CSV is ranked separately; exports are not added together. "
                 "Independent stage windows, including combined windows, must not be added as request latency.")
    text = "\n".join(lines) + "\n"
    for path, rows in zip(paths[:2], (type_rows, task_rows)):
        with path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with paths[2].open("x", encoding="utf-8") as stream:
        stream.write(text)
    return text


def summarize(capture: Path, output: Path, top: int = 20) -> str:
    return summarize_windows([{"profile_output": str(capture)}], output, top=top)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="")
    args = parser.parse_args(argv)
    with args.stage_summary.open(encoding="utf-8", newline="") as stream:
        windows = list(csv.DictReader(stream))
    summarize_windows(windows, args.output_dir, prefix=args.prefix)
    stem = args.prefix + "-" if args.prefix else ""
    print(f"Per-stage operator timings: {args.output_dir / (stem + 'operator-types.csv')}")
    print(f"Per-stage hotspots: {args.output_dir / (stem + 'hotspots.txt')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
