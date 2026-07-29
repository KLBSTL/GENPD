#!/usr/bin/env python3
"""Validate a rendered XPBD stage timer diagnostic and issue a gather Go/No-Go."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def quantile(values, q):
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    index = (len(values) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def stats(values):
    return {
        "mean_ms": statistics.fmean(values),
        "p50_ms": quantile(values, 0.5),
        "p95_ms": quantile(values, 0.95),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    manifest = read_json(root / "manifest.json")
    if manifest.get("protocol_version") != "xpbd-stage-profile-v1":
        raise RuntimeError("Unexpected stage-profile protocol.")
    run_dir = root / "profile"
    metadata = read_json(run_dir / "run_metadata.json")
    if metadata["benchmark"]["no_render"] or metadata["benchmark"]["hide_window"] or not metadata["profile_gpu_queries"]:
        raise RuntimeError("Stage profile must be rendered with GPU queries enabled.")
    controls = metadata["solver_controls"]
    if controls.get("xpbd_fuse_apply_collision") != "0" or controls.get("xpbd_cached_pins") != "1":
        raise RuntimeError("Stage profile controls do not match the manifest.")
    if not (run_dir / "frame_presentation.csv").is_file():
        raise RuntimeError("Rendered presentation profile is missing.")

    warmup = int(manifest["warmup"])
    expected = int(manifest["frames"])
    rows = [row for row in read_csv(run_dir / "frame_profile_experiment.csv") if int(row["frame"]) >= warmup]
    validity = [row for row in read_csv(run_dir / "frame_profile_extended.csv") if int(row["frame"]) >= warmup]
    if len(rows) != expected or len(validity) != expected or any(row["frame_valid"] != "1" for row in validity):
        raise RuntimeError("Stage profile has missing or invalid measured frames.")

    constraint = [float(row["xpbd_constraint_gpu_ms"]) for row in rows]
    vertex = [float(row["xpbd_vertex_gpu_ms"]) for row in rows]
    total = [float(row["xpbd_gpu_ms"]) for row in rows]
    if any(a < 0.0 or b < 0.0 or c <= 0.0 for a, b, c in zip(constraint, vertex, total)):
        raise RuntimeError("Stage timer contains invalid values.")
    if max(abs((a + b) - c) for a, b, c in zip(constraint, vertex, total)) > 1.0e-4:
        raise RuntimeError("XPBD total timer does not equal the two stage timers.")

    constraint_stats = stats(constraint)
    vertex_stats = stats(vertex)
    total_stats = stats(total)
    constraint_share = constraint_stats["mean_ms"] / total_stats["mean_ms"]
    threshold = float(manifest["go_threshold_constraint_share"])
    decision = "GO_VERTEX_GATHER_PROTOTYPE" if constraint_share >= threshold else "NO_GO_PROFILE_OTHER_STAGE"
    summary = {
        "frames": expected,
        "constraint": constraint_stats,
        "vertex_side": vertex_stats,
        "total": total_stats,
        "constraint_share": constraint_share,
        "go_threshold_constraint_share": threshold,
        "decision": decision,
        "diagnostic_only": True,
    }
    (root / "xpbd_stage_profile_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (root / "xpbd_stage_profile_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "mean_ms", "p50_ms", "p95_ms", "mean_share"])
        writer.writeheader()
        writer.writerow({"stage": "constraint_scatter", **constraint_stats, "mean_share": constraint_share})
        writer.writerow({"stage": "vertex_apply_collision", **vertex_stats, "mean_share": 1.0 - constraint_share})
        writer.writerow({"stage": "xpbd_total", **total_stats, "mean_share": 1.0})
    report = f"""# XPBD Stage Profile

- Rendered measured frames: {expected}
- Constraint Jacobi scatter: {constraint_stats['mean_ms']:.3f} ms ({constraint_share * 100.0:.1f}% of XPBD GPU time)
- Vertex apply plus collision: {vertex_stats['mean_ms']:.3f} ms ({(1.0 - constraint_share) * 100.0:.1f}%)
- XPBD total: {total_stats['mean_ms']:.3f} ms
- Decision threshold: constraint share >= {threshold * 100.0:.0f}%
- Decision: **{decision}**

These stage queries are diagnostic only: each query result is read after its
substage, so the run is rendered and valid but is not used as end-to-end frame
performance evidence. The decision only selects the next implementation step.
"""
    (root / "xpbd_stage_profile_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
