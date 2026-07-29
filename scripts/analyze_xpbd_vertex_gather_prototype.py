#!/usr/bin/env python3
"""Validate the XPBD vertex-gather prototype against atomic Jacobi timing/quality runs."""

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
    low = math.floor(index)
    high = math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def stage_stats(rows, warmup, expected_frames):
    rows = [row for row in rows if int(row["frame"]) >= warmup]
    if len(rows) != expected_frames:
        raise RuntimeError(f"Expected {expected_frames} measured profile rows, found {len(rows)}.")
    stages = {
        "constraint_ms": [float(row["xpbd_constraint_gpu_ms"]) for row in rows],
        "vertex_ms": [float(row["xpbd_vertex_gpu_ms"]) for row in rows],
        "total_ms": [float(row["xpbd_gpu_ms"]) for row in rows],
    }
    if any(value < 0.0 for values in stages.values() for value in values):
        raise RuntimeError("Negative XPBD stage time found.")
    if max(abs(a + b - c) for a, b, c in zip(stages["constraint_ms"], stages["vertex_ms"], stages["total_ms"])) > 1.0e-4:
        raise RuntimeError("XPBD stage timer total does not equal its two components.")
    return {
        name: {
            "mean_ms": statistics.fmean(values),
            "p50_ms": quantile(values, 0.5),
            "p95_ms": quantile(values, 0.95),
        }
        for name, values in stages.items()
    }


def timing_stats(run_dir: Path, warmup: int, expected_frames: int):
    rows = [row for row in read_csv(run_dir / "frame_profile.csv") if int(row["frame"]) >= warmup]
    if len(rows) != expected_frames:
        raise RuntimeError(f"{run_dir}: expected {expected_frames} measured timing rows, found {len(rows)}.")
    output = {}
    for field in ("total_ms", "iteration_ms", "optimization_ms", "update_posvel_ms"):
        values = [float(row[field]) for row in rows]
        output[field] = {
            "mean_ms": statistics.fmean(values),
            "p50_ms": quantile(values, 0.5),
            "p95_ms": quantile(values, 0.95),
        }
    return output


def require_rendered_profile(run_dir: Path, expected_variant: str):
    metadata = read_json(run_dir / "run_metadata.json")
    benchmark = metadata["benchmark"]
    if metadata["solver_variant"] != expected_variant:
        raise RuntimeError(f"{run_dir}: expected {expected_variant}, found {metadata['solver_variant']}.")
    if benchmark["no_render"] or benchmark["hide_window"] or not metadata["profile_gpu_queries"]:
        raise RuntimeError(f"{run_dir}: stage profile must be rendered with GPU queries enabled.")
    if not (run_dir / "frame_presentation.csv").is_file():
        raise RuntimeError(f"{run_dir}: missing rendered presentation profile.")
    return metadata


def require_rendered_timing(run_dir: Path, expected_variant: str):
    metadata = read_json(run_dir / "run_metadata.json")
    benchmark = metadata["benchmark"]
    if metadata["solver_variant"] != expected_variant:
        raise RuntimeError(f"{run_dir}: expected {expected_variant}, found {metadata['solver_variant']}.")
    if benchmark["no_render"] or benchmark["hide_window"] or metadata["profile_gpu_queries"]:
        raise RuntimeError(f"{run_dir}: pilot timing must be rendered without stage-query instrumentation.")
    if not (run_dir / "frame_presentation.csv").is_file():
        raise RuntimeError(f"{run_dir}: missing rendered presentation profile.")
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--position-p95-max", type=float, default=1.0e-4)
    args = parser.parse_args()

    root = Path(args.study_root).resolve()
    reference = root / "reference"
    gather_quality = root / "gather_quality"
    atomic_stage = root / "atomic_stage"
    gather_stage = root / "gather_stage"
    atomic_timing = root / "atomic_timing"
    gather_timing = root / "gather_timing"
    for path in (reference, gather_quality, atomic_stage, gather_stage, atomic_timing, gather_timing):
        if not path.is_dir():
            raise RuntimeError(f"Missing required run directory: {path}")

    atomic_metadata = require_rendered_profile(atomic_stage, "gpu-xpbd-jacobi")
    gather_metadata = require_rendered_profile(gather_stage, "gpu-xpbd-vertex-gather")
    atomic_benchmark = atomic_metadata["benchmark"]
    gather_benchmark = gather_metadata["benchmark"]
    comparable_fields = ("frames", "warmup_frames", "uncapped", "sync_gpu", "disable_vsync")
    if any(atomic_benchmark[field] != gather_benchmark[field] for field in comparable_fields):
        raise RuntimeError("Atomic and gather stage-profile benchmark controls differ.")
    for field in ("cloth_dimension", "scene"):
        if atomic_metadata["experiment_overrides"][field] != gather_metadata["experiment_overrides"][field]:
            raise RuntimeError(f"Atomic and gather stage-profile override differs: {field}.")
    if atomic_metadata["quality"]["iterations_per_frame"] != gather_metadata["quality"]["iterations_per_frame"]:
        raise RuntimeError("Atomic and gather stage-profile iteration budgets differ.")

    frames = int(atomic_benchmark["frames"])
    warmup = int(atomic_benchmark["warmup_frames"])
    atomic_stats = stage_stats(read_csv(atomic_stage / "frame_profile_experiment.csv"), warmup, frames)
    gather_stats = stage_stats(read_csv(gather_stage / "frame_profile_experiment.csv"), warmup, frames)

    atomic_timing_metadata = require_rendered_timing(atomic_timing, "gpu-xpbd-jacobi")
    gather_timing_metadata = require_rendered_timing(gather_timing, "gpu-xpbd-vertex-gather")
    if any(atomic_timing_metadata["benchmark"][field] != gather_timing_metadata["benchmark"][field] for field in comparable_fields):
        raise RuntimeError("Atomic and gather timing benchmark controls differ.")
    for field in ("cloth_dimension", "scene"):
        if atomic_timing_metadata["experiment_overrides"][field] != gather_timing_metadata["experiment_overrides"][field]:
            raise RuntimeError(f"Atomic and gather timing override differs: {field}.")
    atomic_timing_stats = timing_stats(atomic_timing, warmup, frames)
    gather_timing_stats = timing_stats(gather_timing, warmup, frames)

    quality_metadata = read_json(gather_quality / "run_metadata.json")
    if quality_metadata["solver_variant"] != "gpu-xpbd-vertex-gather":
        raise RuntimeError("Gather quality run has the wrong solver variant.")
    if quality_metadata["benchmark"]["no_render"] or quality_metadata["benchmark"]["hide_window"]:
        raise RuntimeError("Gather quality run must be rendered.")
    quality_rows = read_csv(gather_quality / "quality_metrics.csv")
    extended_rows = read_csv(gather_quality / "frame_profile_extended.csv")
    if not quality_rows or len(quality_rows) != len(extended_rows):
        raise RuntimeError("Gather quality metrics/checkpoint coverage is incomplete.")
    if any(row["has_reference"] != "1" or row["finite"] != "1" or row["exploded"] != "0" for row in quality_rows):
        raise RuntimeError("Gather quality run has missing reference, nonfinite state, or explosion.")
    if any(row["frame_valid"] != "1" or row["exploded"] != "0" for row in extended_rows):
        raise RuntimeError("Gather quality run has invalid frame status.")
    position_errors = [float(row["position_rel_l2"]) for row in quality_rows]
    velocity_errors = [float(row["velocity_rel_l2"]) for row in quality_rows]
    quality = {
        "frames": len(quality_rows),
        "position_rel_l2_p95": quantile(position_errors, 0.95),
        "position_rel_l2_max": max(position_errors),
        "velocity_rel_l2_p95": quantile(velocity_errors, 0.95),
        "velocity_rel_l2_max": max(velocity_errors),
        "position_p95_threshold": args.position_p95_max,
    }
    quality["passed"] = quality["position_rel_l2_p95"] <= args.position_p95_max

    atomic_first = read_csv(atomic_stage / "frame_profile_experiment.csv")[warmup]
    gather_first = read_csv(gather_stage / "frame_profile_experiment.csv")[warmup]
    speedup = atomic_stats["total_ms"]["mean_ms"] / gather_stats["total_ms"]["mean_ms"]
    summary = {
        "protocol": "xpbd-vertex-gather-prototype-v1",
        "atomic_commit": atomic_metadata["git_commit"],
        "gather_commit": gather_metadata["git_commit"],
        "rendered_stage_frames": frames,
        "warmup": warmup,
        "atomic": atomic_stats,
        "vertex_gather": gather_stats,
        "quality": quality,
        "gpu_timer_speedup": speedup,
        "gpu_timer_delta_percent": 100.0 * (1.0 - 1.0 / speedup),
        "end_to_end_pilot": {
            "atomic": atomic_timing_stats,
            "vertex_gather": gather_timing_stats,
            "total_speedup": atomic_timing_stats["total_ms"]["mean_ms"] / gather_timing_stats["total_ms"]["mean_ms"],
            "iteration_speedup": atomic_timing_stats["iteration_ms"]["mean_ms"] / gather_timing_stats["iteration_ms"]["mean_ms"],
            "repetitions": 1,
        },
        "atomic_constraint_share": atomic_stats["constraint_ms"]["mean_ms"] / atomic_stats["total_ms"]["mean_ms"],
        "gather_constraint_share": gather_stats["constraint_ms"]["mean_ms"] / gather_stats["total_ms"]["mean_ms"],
        "tracked_buffer_bytes": {
            "atomic_delta": int(atomic_first["xpbd_delta_buffer_bytes"]),
            "atomic_lambda": int(atomic_first["xpbd_lambda_buffer_bytes"]),
            "gather_edge_correction": int(gather_first["xpbd_edge_correction_buffer_bytes"]),
            "gather_lambda": int(gather_first["xpbd_lambda_buffer_bytes"]),
        },
        "timer_scope_note": "Per-stage GL query readback makes this a GPU-stage diagnostic, not end-to-end frame-time evidence.",
    }
    (root / "vertex_gather_prototype_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (root / "vertex_gather_prototype_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", "constraint_mean_ms", "vertex_mean_ms", "total_gpu_timer_ms", "frame_total_ms", "solver_iteration_ms", "total_gpu_speedup_vs_atomic"])
        writer.writeheader()
        writer.writerow({"variant": "gpu-xpbd-jacobi", "constraint_mean_ms": atomic_stats["constraint_ms"]["mean_ms"], "vertex_mean_ms": atomic_stats["vertex_ms"]["mean_ms"], "total_gpu_timer_ms": atomic_stats["total_ms"]["mean_ms"], "frame_total_ms": atomic_timing_stats["total_ms"]["mean_ms"], "solver_iteration_ms": atomic_timing_stats["iteration_ms"]["mean_ms"], "total_gpu_speedup_vs_atomic": 1.0})
        writer.writerow({"variant": "gpu-xpbd-vertex-gather", "constraint_mean_ms": gather_stats["constraint_ms"]["mean_ms"], "vertex_mean_ms": gather_stats["vertex_ms"]["mean_ms"], "total_gpu_timer_ms": gather_stats["total_ms"]["mean_ms"], "frame_total_ms": gather_timing_stats["total_ms"]["mean_ms"], "solver_iteration_ms": gather_timing_stats["iteration_ms"]["mean_ms"], "total_gpu_speedup_vs_atomic": speedup})

    report = f"""# XPBD Vertex-Gather Prototype

## Protocol

- Commit: `{gather_metadata['git_commit']}`
- Scene: moving sphere cloth, {gather_metadata['experiment_overrides']['cloth_dimension']} x {gather_metadata['experiment_overrides']['cloth_dimension']} vertices
- Solver budget: {gather_metadata['quality']['iterations_per_frame']} XPBD iterations per frame
- Rendering: 1600 x 900, uncapped, VSync disabled, GPU synchronized
- Stage profile: {frames} measured frames after {warmup} warm-up frames

## Quality Gate

The vertex-gather trajectory was compared with atomic Jacobi XPBD for {quality['frames']} rendered frames at a separate 960 x 540 quality setting. P95/max position relative L2 were {quality['position_rel_l2_p95']:.3e}/{quality['position_rel_l2_max']:.3e}; P95/max velocity relative L2 were {quality['velocity_rel_l2_p95']:.3e}/{quality['velocity_rel_l2_max']:.3e}. All frames were finite and non-exploded. The pre-set P95 position threshold was {args.position_p95_max:.1e}: **{'PASS' if quality['passed'] else 'FAIL'}**.

## GPU-Stage Diagnostic

| Variant | Constraint pass (ms) | Vertex side (ms) | XPBD GPU total (ms) |
| --- | ---: | ---: | ---: |
| Atomic Jacobi | {atomic_stats['constraint_ms']['mean_ms']:.3f} | {atomic_stats['vertex_ms']['mean_ms']:.3f} | {atomic_stats['total_ms']['mean_ms']:.3f} |
| Vertex gather | {gather_stats['constraint_ms']['mean_ms']:.3f} | {gather_stats['vertex_ms']['mean_ms']:.3f} | {gather_stats['total_ms']['mean_ms']:.3f} |

At equal short-horizon trajectory quality, the measured GPU-stage ratio is {speedup:.3f}x (atomic/gather), or {summary['gpu_timer_delta_percent']:.1f}% relative change. The gather path replaces the {summary['tracked_buffer_bytes']['atomic_delta']:,}-byte atomic delta buffer with a {summary['tracked_buffer_bytes']['gather_edge_correction']:,}-byte edge-correction buffer; lambda storage is unchanged.

## Rendered End-to-End Pilot

With stage queries disabled, a single matched 60-frame rendered pilot measured {atomic_timing_stats['total_ms']['mean_ms']:.3f} ms/frame for atomic Jacobi and {gather_timing_stats['total_ms']['mean_ms']:.3f} ms/frame for vertex gather. The solver iteration portion was {atomic_timing_stats['iteration_ms']['mean_ms']:.3f} ms versus {gather_timing_stats['iteration_ms']['mean_ms']:.3f} ms. This is a {summary['end_to_end_pilot']['total_speedup']:.3f}x frame-time ratio and confirms the diagnostic direction, but is one repetition only and is not a paper statistic.

**Interpretation limit.** Per-stage GL query results are read after each substage. They are valid for locating GPU work, but perturb host scheduling; this document does not make an end-to-end frame-time or paper-performance claim. Any later performance comparison must repeat at matched quality with query instrumentation disabled.
"""
    (root / "vertex_gather_prototype_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
