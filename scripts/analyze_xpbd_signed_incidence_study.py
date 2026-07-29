#!/usr/bin/env python3
"""Compare raw-CSR and signed-incidence XPBD vertex-gather diagnostics."""

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


def stats(values):
    return {
        "mean_ms": statistics.fmean(values),
        "p50_ms": quantile(values, 0.5),
        "p95_ms": quantile(values, 0.95),
    }


def profile_rows(run_dir: Path, name: str, metadata, require_queries: bool):
    benchmark = metadata["benchmark"]
    if benchmark["no_render"] or benchmark["hide_window"] or metadata["profile_gpu_queries"] != require_queries:
        raise RuntimeError(f"{run_dir}: rendered/query contract mismatch.")
    rows = [row for row in read_csv(run_dir / name) if int(row["frame"]) >= int(benchmark["warmup_frames"])]
    if len(rows) != int(benchmark["frames"]):
        raise RuntimeError(f"{run_dir}: incomplete measured profile rows.")
    return rows


def stage_summary(run_dir: Path):
    metadata = read_json(run_dir / "run_metadata.json")
    rows = profile_rows(run_dir, "frame_profile_experiment.csv", metadata, True)
    output = {
        "constraint_ms": stats([float(row["xpbd_constraint_gpu_ms"]) for row in rows]),
        "vertex_ms": stats([float(row["xpbd_vertex_gpu_ms"]) for row in rows]),
        "total_ms": stats([float(row["xpbd_gpu_ms"]) for row in rows]),
    }
    if max(abs(a + b - c) for a, b, c in zip(
        [float(row["xpbd_constraint_gpu_ms"]) for row in rows],
        [float(row["xpbd_vertex_gpu_ms"]) for row in rows],
        [float(row["xpbd_gpu_ms"]) for row in rows],
    )) > 1.0e-4:
        raise RuntimeError(f"{run_dir}: invalid stage-timer sum.")
    return metadata, output


def timing_summary(run_dir: Path):
    metadata = read_json(run_dir / "run_metadata.json")
    rows = profile_rows(run_dir, "frame_profile.csv", metadata, False)
    return metadata, {
        "frame_ms": stats([float(row["total_ms"]) for row in rows]),
        "iteration_ms": stats([float(row["iteration_ms"]) for row in rows]),
    }


def validate_matching_controls(left, right):
    for key in ("frames", "warmup_frames", "uncapped", "sync_gpu", "disable_vsync"):
        if left["benchmark"][key] != right["benchmark"][key]:
            raise RuntimeError(f"Mismatched benchmark control: {key}")
    for key in ("cloth_dimension", "scene"):
        if left["experiment_overrides"][key] != right["experiment_overrides"][key]:
            raise RuntimeError(f"Mismatched scene control: {key}")
    if left["quality"]["iterations_per_frame"] != right["quality"]["iterations_per_frame"]:
        raise RuntimeError("Mismatched XPBD iteration budget.")


def quality_summary(run_dir: Path):
    metadata = read_json(run_dir / "run_metadata.json")
    if metadata["solver_variant"] != "gpu-xpbd-vertex-gather":
        raise RuntimeError("Quality run is not vertex gather.")
    if metadata["benchmark"]["no_render"] or metadata["benchmark"]["hide_window"]:
        raise RuntimeError("Quality run must be rendered.")
    rows = read_csv(run_dir / "quality_metrics.csv")
    validity = read_csv(run_dir / "frame_profile_extended.csv")
    if not rows or len(rows) != len(validity):
        raise RuntimeError("Missing quality/validity rows.")
    if any(row["has_reference"] != "1" or row["finite"] != "1" or row["exploded"] != "0" for row in rows):
        raise RuntimeError("Quality gate has invalid or exploded state.")
    if any(row["frame_valid"] != "1" or row["exploded"] != "0" for row in validity):
        raise RuntimeError("Quality validity gate failed.")
    positions = [float(row["position_rel_l2"]) for row in rows]
    velocities = [float(row["velocity_rel_l2"]) for row in rows]
    return metadata, {
        "frames": len(rows),
        "position_rel_l2_p95": quantile(positions, 0.95),
        "position_rel_l2_max": max(positions),
        "velocity_rel_l2_p95": quantile(velocities, 0.95),
        "velocity_rel_l2_max": max(velocities),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True, help="a955299 raw-CSR gather study root")
    parser.add_argument("--signed-root", required=True, help="signed-incidence gather study root")
    args = parser.parse_args()
    raw_root = Path(args.raw_root).resolve()
    signed_root = Path(args.signed_root).resolve()

    raw_stage_metadata, raw_stage = stage_summary(raw_root / "gather_stage")
    signed_stage_metadata, signed_stage = stage_summary(signed_root / "gather_stage")
    raw_timing_metadata, raw_timing = timing_summary(raw_root / "gather_timing")
    signed_timing_metadata, signed_timing = timing_summary(signed_root / "gather_timing")
    signed_quality_metadata, quality = quality_summary(signed_root / "gather_quality")
    validate_matching_controls(raw_stage_metadata, signed_stage_metadata)
    validate_matching_controls(raw_timing_metadata, signed_timing_metadata)

    atomic_summary = read_json(raw_root / "vertex_gather_prototype_summary.json")
    if atomic_summary["atomic_commit"] != raw_stage_metadata["git_commit"]:
        raise RuntimeError("Raw study atomic summary does not match raw gather commit.")
    if signed_stage_metadata["git_commit"] != signed_quality_metadata["git_commit"]:
        raise RuntimeError("Signed quality and stage commits differ.")

    stage_speedup = raw_stage["total_ms"]["mean_ms"] / signed_stage["total_ms"]["mean_ms"]
    vertex_speedup = raw_stage["vertex_ms"]["mean_ms"] / signed_stage["vertex_ms"]["mean_ms"]
    frame_speedup = raw_timing["frame_ms"]["mean_ms"] / signed_timing["frame_ms"]["mean_ms"]
    iteration_speedup = raw_timing["iteration_ms"]["mean_ms"] / signed_timing["iteration_ms"]["mean_ms"]
    summary = {
        "protocol": "xpbd-signed-incidence-v1",
        "raw_commit": raw_stage_metadata["git_commit"],
        "signed_commit": signed_stage_metadata["git_commit"],
        "quality": quality,
        "raw_csr": {"stage": raw_stage, "timing": raw_timing},
        "signed_incidence": {"stage": signed_stage, "timing": signed_timing},
        "atomic_reference": atomic_summary["atomic"],
        "speedup_raw_over_signed": {
            "stage_total": stage_speedup,
            "vertex_stage": vertex_speedup,
            "frame": frame_speedup,
            "iteration": iteration_speedup,
        },
        "instrumentation_note": "Stage queries are diagnostic only; the rendered frame-time pilot disables them and has one repetition.",
    }
    (signed_root / "signed_incidence_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (signed_root / "signed_incidence_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", "constraint_stage_ms", "vertex_stage_ms", "xpbd_stage_total_ms", "rendered_frame_ms", "iteration_ms"])
        writer.writeheader()
        writer.writerow({"variant": "atomic-jacobi", "constraint_stage_ms": atomic_summary["atomic"]["constraint_ms"]["mean_ms"], "vertex_stage_ms": atomic_summary["atomic"]["vertex_ms"]["mean_ms"], "xpbd_stage_total_ms": atomic_summary["atomic"]["total_ms"]["mean_ms"], "rendered_frame_ms": atomic_summary["end_to_end_pilot"]["atomic"]["total_ms"]["mean_ms"], "iteration_ms": atomic_summary["end_to_end_pilot"]["atomic"]["iteration_ms"]["mean_ms"]})
        for label, stage, timing in (("raw-csr-gather", raw_stage, raw_timing), ("signed-incidence-gather", signed_stage, signed_timing)):
            writer.writerow({"variant": label, "constraint_stage_ms": stage["constraint_ms"]["mean_ms"], "vertex_stage_ms": stage["vertex_ms"]["mean_ms"], "xpbd_stage_total_ms": stage["total_ms"]["mean_ms"], "rendered_frame_ms": timing["frame_ms"]["mean_ms"], "iteration_ms": timing["iteration_ms"]["mean_ms"]})

    report = f"""# XPBD Signed-Incidence Gather Study

- Raw gather commit: `{summary['raw_commit']}`
- Signed-incidence commit: `{summary['signed_commit']}`
- Scene: 256 x 256 moving-sphere cloth, 32 XPBD iterations/frame
- Quality: 12 rendered frames against atomic XPBD; P95/max position relative L2 = {quality['position_rel_l2_p95']:.3e}/{quality['position_rel_l2_max']:.3e}, velocity = {quality['velocity_rel_l2_p95']:.3e}/{quality['velocity_rel_l2_max']:.3e}; all frames finite and non-exploded.
- Profiles: 5 warm-up plus 60 rendered frames at 1600 x 900, uncapped, VSync disabled, GPU synchronized.

| Variant | Constraint stage | Vertex stage | XPBD GPU stage | Rendered frame pilot |
| --- | ---: | ---: | ---: | ---: |
| Atomic Jacobi | {atomic_summary['atomic']['constraint_ms']['mean_ms']:.3f} ms | {atomic_summary['atomic']['vertex_ms']['mean_ms']:.3f} ms | {atomic_summary['atomic']['total_ms']['mean_ms']:.3f} ms | {atomic_summary['end_to_end_pilot']['atomic']['total_ms']['mean_ms']:.3f} ms |
| Raw CSR gather | {raw_stage['constraint_ms']['mean_ms']:.3f} ms | {raw_stage['vertex_ms']['mean_ms']:.3f} ms | {raw_stage['total_ms']['mean_ms']:.3f} ms | {raw_timing['frame_ms']['mean_ms']:.3f} ms |
| Signed-incidence gather | {signed_stage['constraint_ms']['mean_ms']:.3f} ms | {signed_stage['vertex_ms']['mean_ms']:.3f} ms | {signed_stage['total_ms']['mean_ms']:.3f} ms | {signed_timing['frame_ms']['mean_ms']:.3f} ms |

Relative to raw CSR gather, signed incidence is {vertex_speedup:.3f}x in the vertex stage and {stage_speedup:.3f}x in total XPBD GPU-stage time. Its one-repeat rendered pilot is {frame_speedup:.3f}x in frame time and {iteration_speedup:.3f}x in solver iteration time. The compact encoding preserves the short reference trajectory exactly, but it should not enter a paper performance table unless a separately pre-registered, multi-repeat equal-quality protocol confirms the result.
"""
    (signed_root / "signed_incidence_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
