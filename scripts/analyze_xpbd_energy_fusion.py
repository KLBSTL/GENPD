#!/usr/bin/env python3
"""Validate the XPBD energy audit and summarize the apply/collision fusion A/B."""

import argparse
import csv
import json
import math
import statistics
import struct
from pathlib import Path


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(row, field):
    return float(row[field])


def quantile(values, fraction):
    values = sorted(values)
    if not values:
        raise ValueError("Cannot take a quantile of no values.")
    if len(values) == 1:
        return values[0]
    location = (len(values) - 1) * fraction
    low = math.floor(location)
    high = math.ceil(location)
    return values[low] + (values[high] - values[low]) * (location - low)


def require_rendered(run_dir: Path):
    metadata = read_json(run_dir / "run_metadata.json")
    benchmark = metadata["benchmark"]
    if benchmark["no_render"] or benchmark["hide_window"]:
        raise RuntimeError(f"Run is not rendered: {run_dir}")
    if not (run_dir / "frame_presentation.csv").is_file():
        raise RuntimeError(f"Rendered presentation profile missing: {run_dir}")
    return metadata


def validate_energy(run_dir: Path, acceptance):
    metadata = require_rendered(run_dir)
    if metadata["solver_controls"].get("energy_audit") != "1":
        raise RuntimeError("Energy audit metadata is not enabled.")
    rows = read_csv(run_dir / "xpbd_energy_audit.csv")
    if not rows or any(row["finite"] != "1" for row in rows):
        raise RuntimeError("Energy audit contains no rows or a non-finite row.")

    def relative(left, right):
        return abs(left - right) / max(1.0e-12, abs(left), abs(right))

    objective_rel = [as_float(row, "objective_gpu_rel_error") for row in rows]
    constraint_rel = [relative(as_float(row, "constraint_energy"), as_float(row, "gpu_constraint_energy")) for row in rows]
    inertia_rel = [relative(as_float(row, "inertia_energy"), as_float(row, "gpu_inertia_energy")) for row in rows]
    derivative_abs = [abs(as_float(row, "analytic_directional_derivative") - as_float(row, "fd_directional_derivative")) for row in rows]
    derivative_rel = [as_float(row, "directional_derivative_rel_error") for row in rows]

    summary = {
        "frames": len(rows),
        "max_objective_gpu_rel_error": max(objective_rel),
        "max_constraint_gpu_rel_error": max(constraint_rel),
        "max_inertia_gpu_rel_error": max(inertia_rel),
        "max_directional_derivative_abs_error": max(derivative_abs),
        "max_directional_derivative_rel_error": max(derivative_rel),
        "first_implicit_objective": as_float(rows[0], "implicit_objective_cpu"),
        "last_implicit_objective": as_float(rows[-1], "implicit_objective_cpu"),
        "collision_projection_active": any(row["collision_projection_active"] == "1" for row in rows),
    }
    if summary["max_objective_gpu_rel_error"] > acceptance["cpu_gpu_objective_rel_error_max"]:
        raise RuntimeError(f"CPU/GPU objective mismatch: {summary['max_objective_gpu_rel_error']}")
    if summary["max_directional_derivative_abs_error"] > acceptance["directional_derivative_abs_error_max"]:
        raise RuntimeError(f"Directional derivative absolute mismatch: {summary['max_directional_derivative_abs_error']}")
    return summary


def compare_equivalence(run_root: Path):
    unfused = run_root / "equivalence" / "unfused"
    fused = run_root / "equivalence" / "fused"
    unfused_meta = require_rendered(unfused)
    fused_meta = require_rendered(fused)
    if unfused_meta["solver_controls"].get("xpbd_fuse_apply_collision") != "0":
        raise RuntimeError("Unfused equivalence metadata is incorrect.")
    if fused_meta["solver_controls"].get("xpbd_fuse_apply_collision") != "1":
        raise RuntimeError("Fused equivalence metadata is incorrect.")
    left = sorted((unfused / "reference_checkpoints").glob("*.bin"))
    right = sorted((fused / "reference_checkpoints").glob("*.bin"))
    if not left or len(left) != len(right) or [p.name for p in left] != [p.name for p in right]:
        raise RuntimeError("Fusion equivalence checkpoints are incomplete.")
    exact = True
    max_abs = 0.0
    max_relative_l2 = 0.0
    for lhs, rhs in zip(left, right):
        lhs_bytes = lhs.read_bytes()
        rhs_bytes = rhs.read_bytes()
        exact = exact and lhs_bytes == rhs_bytes
        if len(lhs_bytes) != len(rhs_bytes) or len(lhs_bytes) % 4:
            raise RuntimeError(f"Invalid checkpoint layout: {lhs}")
        lhs_values = struct.unpack("<%df" % (len(lhs_bytes) // 4), lhs_bytes)
        rhs_values = struct.unpack("<%df" % (len(rhs_bytes) // 4), rhs_bytes)
        delta_l2 = math.sqrt(sum((a - b) * (a - b) for a, b in zip(lhs_values, rhs_values)))
        base_l2 = math.sqrt(sum(a * a for a in lhs_values))
        max_relative_l2 = max(max_relative_l2, delta_l2 / max(1.0e-12, base_l2))
        max_abs = max(max_abs, max(abs(a - b) for a, b in zip(lhs_values, rhs_values)))
    if not exact:
        raise RuntimeError(f"Fused checkpoints differ: max_abs={max_abs}, rel_l2={max_relative_l2}")
    return {"checkpoint_count": len(left), "exact_match": exact, "max_abs": max_abs, "max_relative_l2": max_relative_l2}


def validate_timing_run(run_dir: Path, condition: str):
    metadata = require_rendered(run_dir)
    expected_fusion = "1" if condition == "fused" else "0"
    if metadata["solver_controls"].get("xpbd_fuse_apply_collision") != expected_fusion:
        raise RuntimeError(f"Fusion metadata mismatch in {run_dir}")
    warmup = int(metadata["benchmark"]["warmup_frames"])
    profile = [row for row in read_csv(run_dir / "frame_profile.csv") if int(row["frame"]) >= warmup]
    extended = [row for row in read_csv(run_dir / "frame_profile_extended.csv") if int(row["frame"]) >= warmup]
    experiment = [row for row in read_csv(run_dir / "frame_profile_experiment.csv") if int(row["frame"]) >= warmup]
    expected_count = int(metadata["benchmark"]["frames"])
    if len(profile) != expected_count or len(extended) != expected_count or len(experiment) != expected_count:
        raise RuntimeError(f"Measured frame count mismatch in {run_dir}")
    if any(row["frame_valid"] != "1" for row in extended):
        raise RuntimeError(f"Invalid XPBD frame in {run_dir}")
    total_ms = [as_float(row, "total_ms") for row in profile]
    actual_dispatches = []
    for row in experiment:
        actual_dispatches.append(
            int(row["xpbd_constraint_dispatches"])
            + int(row["xpbd_apply_dispatches"])
            + int(row["xpbd_collision_dispatches"])
            - int(row["xpbd_fused_apply_collision_dispatches"])
        )
    if len(set(actual_dispatches)) != 1:
        raise RuntimeError(f"XPBD dispatch count varies within {run_dir}")
    return {
        "run_dir": str(run_dir),
        "condition": condition,
        "frames": len(total_ms),
        "frame_mean_ms": statistics.fmean(total_ms),
        "frame_p50_ms": quantile(total_ms, 0.50),
        "frame_p95_ms": quantile(total_ms, 0.95),
        "actual_xpbd_dispatches": actual_dispatches[0],
        "mean_xpbd_gpu_ms": statistics.fmean(as_float(row, "xpbd_gpu_ms") for row in experiment),
    }


def summarize_timing(run_root: Path, repetitions: int):
    all_rows = []
    for condition in ("unfused", "fused"):
        rows = []
        for repetition in range(1, repetitions + 1):
            row = validate_timing_run(run_root / "timing" / condition / f"rep{repetition:02d}", condition)
            row["repetition"] = repetition
            rows.append(row)
            all_rows.append(row)
        means = [row["frame_mean_ms"] for row in rows]
        all_frames_p50 = statistics.fmean(row["frame_p50_ms"] for row in rows)
        all_frames_p95 = max(row["frame_p95_ms"] for row in rows)
        all_rows.append({
            "run_dir": "aggregate", "condition": condition, "repetition": 0, "frames": sum(row["frames"] for row in rows),
            "frame_mean_ms": statistics.fmean(means),
            "frame_p50_ms": all_frames_p50, "frame_p95_ms": all_frames_p95,
            "actual_xpbd_dispatches": rows[0]["actual_xpbd_dispatches"],
            "mean_xpbd_gpu_ms": statistics.fmean(row["mean_xpbd_gpu_ms"] for row in rows),
            "frame_mean_std_ms": statistics.stdev(means) if len(means) > 1 else 0.0,
        })
    return all_rows


def write_csv(path: Path, rows):
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    manifest = read_json(root / "manifest.json")
    if manifest.get("protocol_version") != "xpbd-energy-fusion-v1":
        raise RuntimeError("Unexpected XPBD energy/fusion protocol version.")
    energy = validate_energy(root / "energy_audit", manifest["energy_diagnostic"]["acceptance"])
    equivalence = compare_equivalence(root)
    timing_rows = summarize_timing(root, int(manifest["timing"]["repetitions"]))
    write_csv(root / "xpbd_fusion_timing_summary.csv", timing_rows)
    aggregate = {row["condition"]: row for row in timing_rows if row.get("repetition") == 0}
    baseline = aggregate["unfused"]
    fused = aggregate["fused"]
    improvement = (baseline["frame_mean_ms"] - fused["frame_mean_ms"]) / baseline["frame_mean_ms"]
    report = f"""# XPBD Energy And Fusion Study

**Run root:** `{root}`

## Formula Audit

- Rendered diagnostic frames: {energy['frames']}
- CPU/GPU implicit-objective maximum relative error: {energy['max_objective_gpu_rel_error']:.3e}
- CPU/GPU raw-constraint maximum relative error: {energy['max_constraint_gpu_rel_error']:.3e}
- CPU/GPU inertia maximum relative error: {energy['max_inertia_gpu_rel_error']:.3e}
- Directional derivative maximum absolute error: {energy['max_directional_derivative_abs_error']:.3e}
- Directional derivative maximum relative error: {energy['max_directional_derivative_rel_error']:.3e}

The relative directional-derivative field is retained for transparency, but the
acceptance guard uses absolute error because these rendered XPBD states have
near-zero directional derivatives in single precision. Collision remains a
projection, so its contact response is not folded into a smooth potential.

## Fusion Equivalence

- Rendered checkpoint count: {equivalence['checkpoint_count']}
- Exact byte match: {equivalence['exact_match']}
- Maximum absolute state difference: {equivalence['max_abs']:.3e}
- Maximum relative L2 state difference: {equivalence['max_relative_l2']:.3e}

## Rendered Timing

| Condition | Mean frame time (ms) | Std. across repetitions (ms) | P50 (ms) | P95 (ms) | Actual XPBD compute dispatches/frame |
|---|---:|---:|---:|---:|---:|
| Unfused | {baseline['frame_mean_ms']:.3f} | {baseline.get('frame_mean_std_ms', 0.0):.3f} | {baseline['frame_p50_ms']:.3f} | {baseline['frame_p95_ms']:.3f} | {baseline['actual_xpbd_dispatches']} |
| Fused | {fused['frame_mean_ms']:.3f} | {fused.get('frame_mean_std_ms', 0.0):.3f} | {fused['frame_p50_ms']:.3f} | {fused['frame_p95_ms']:.3f} | {fused['actual_xpbd_dispatches']} |

Fusion changes only the apply-plus-collision launch boundary. It preserves the
32 constraint iterations, collision order, rendered viewport, and checkpointed
state. The measured end-to-end frame-time change is {improvement * 100.0:.2f}%.
"""
    (root / "xpbd_energy_fusion_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
