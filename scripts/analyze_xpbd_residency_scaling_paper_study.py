#!/usr/bin/env python3
"""Aggregate three formal rendered XPBD residency studies across mesh sizes."""

import argparse
import csv
import json
import math
from pathlib import Path


CONDITIONS = ("resident", "forced-cpu-state-roundtrip")


def fail(message):
    raise RuntimeError(message)


def load_json(path):
    if not path.is_file():
        fail("Missing JSON: {0}".format(path))
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_csv(path):
    if not path.is_file():
        fail("Missing CSV: {0}".format(path))
    with path.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        fail("Empty CSV: {0}".format(path))
    return rows


def number(row, field, path):
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError):
        fail("Missing numeric field '{0}' in {1}".format(field, path))
    if not math.isfinite(value):
        fail("Non-finite field '{0}' in {1}".format(field, path))
    return value


def mean(values):
    return sum(values) / float(len(values))


def sample_std(values):
    if len(values) < 2:
        return 0.0
    average = mean(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / float(len(values) - 1))


def percentile(values, fraction=0.95):
    if not values:
        fail("Cannot calculate a percentile from no values.")
    ordered = sorted(values)
    index = int(math.ceil(fraction * len(ordered))) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def measured(rows, warmup, path):
    selected = [row for row in rows if int(number(row, "frame", path)) >= warmup]
    if not selected:
        fail("No measured rows in {0}".format(path))
    return selected


def run_directory(run_root, dimension):
    return run_root / "dimensions" / "d{0}".format(dimension)


def timing_samples(dimension_root, condition, manifest):
    samples = []
    repetition_means = []
    timing = manifest["timing"]
    for repetition in range(1, timing["repetitions"] + 1):
        directory = dimension_root / "timing" / condition / "rep{0:02d}".format(repetition)
        metadata = load_json(directory / "run_metadata.json")
        benchmark = metadata.get("benchmark", {})
        if benchmark.get("no_render") or benchmark.get("hide_window") or not benchmark.get("sync_gpu") or not benchmark.get("disable_vsync"):
            fail("Timing run is not rendered, synchronized, and vsync-disabled: {0}".format(directory))
        expected_forced = "1" if condition == "forced-cpu-state-roundtrip" else "0"
        if metadata.get("solver_variant") != "gpu-xpbd-jacobi" or metadata.get("solver_controls", {}).get("force_cpu_state_roundtrip") != expected_forced:
            fail("Timing metadata mismatch: {0}".format(directory))
        profile_path = directory / "frame_profile.csv"
        extended_path = directory / "frame_profile_extended.csv"
        experiment_path = directory / "frame_profile_experiment.csv"
        presentation_path = directory / "frame_presentation.csv"
        profile = measured(load_csv(profile_path), timing["warmup"], profile_path)
        extended = measured(load_csv(extended_path), timing["warmup"], extended_path)
        experiment = measured(load_csv(experiment_path), timing["warmup"], experiment_path)
        presentation = measured(load_csv(presentation_path), timing["warmup"], presentation_path)
        if any(len(rows) != timing["frames"] for rows in (profile, extended, experiment, presentation)):
            fail("Incomplete formal timing run: {0}".format(directory))
        if any(row.get("frame_valid") != "1" or row.get("termination_reason") != "none" for row in extended):
            fail("Invalid timing frame: {0}".format(directory))
        if any(row.get("rendered") != "1" or row.get("gpu_sync_enabled") != "1" for row in presentation):
            fail("Unrendered timing frame: {0}".format(directory))
        if any(int(number(row, "iterations", profile_path)) != manifest["iterations_per_frame"] for row in profile):
            fail("Unexpected XPBD iteration count: {0}".format(directory))
        for row, exp, present in zip(profile, experiment, presentation):
            dispatches = number(exp, "xpbd_constraint_dispatches", experiment_path) + number(exp, "xpbd_apply_dispatches", experiment_path) + number(exp, "xpbd_collision_dispatches", experiment_path)
            samples.append({
                "frame_wall_ms": number(present, "frame_wall_ms", presentation_path),
                "state_h2d_bytes": number(exp, "state_h2d_bytes", experiment_path),
                "state_d2h_bytes": number(exp, "state_d2h_bytes", experiment_path),
                "state_upload_calls": number(exp, "state_upload_calls", experiment_path),
                "state_readback_calls": number(exp, "state_readback_calls", experiment_path),
                "xpbd_dispatches": dispatches,
            })
        repetition_means.append(mean([number(row, "frame_wall_ms", presentation_path) for row in presentation]))
    return samples, repetition_means


def trajectory_quality(dimension_root, condition, manifest):
    directory = dimension_root / "trajectory" / condition
    quality_path = directory / "quality_metrics.csv"
    quality = measured(load_csv(quality_path), manifest["trajectory"]["warmup"], quality_path)
    if len(quality) != manifest["trajectory"]["frames"]:
        fail("Incomplete trajectory quality rows: {0}".format(directory))
    required = ("finite", "exploded", "mean_stretch_strain", "max_stretch_strain", "max_penetration_depth")
    if any(field not in quality[0] for field in required):
        fail("Trajectory quality schema mismatch: {0}".format(directory))
    if any(row.get("finite") != "1" or row.get("exploded") != "0" for row in quality):
        fail("Invalid trajectory quality row: {0}".format(directory))
    return {
        "p95_mean_stretch_strain": percentile([number(row, "mean_stretch_strain", quality_path) for row in quality]),
        "p95_max_stretch_strain": percentile([number(row, "max_stretch_strain", quality_path) for row in quality]),
        "max_penetration_depth": max(number(row, "max_penetration_depth", quality_path) for row in quality),
    }


def trajectory_errors(dimension_root):
    rows = load_csv(dimension_root / "xpbd_residency_trajectory.csv")
    return {
        "p95_position_rel_l2": percentile([number(row, "position_rel_l2", dimension_root / "xpbd_residency_trajectory.csv") for row in rows]),
        "p95_velocity_rel_l2": percentile([number(row, "velocity_rel_l2", dimension_root / "xpbd_residency_trajectory.csv") for row in rows]),
    }


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    manifest = load_json(run_root / "manifest.json")
    if manifest.get("protocol_version") != "xpbd-residency-scaling-v1":
        fail("Unexpected formal XPBD residency scaling protocol.")
    if manifest.get("timing", {}).get("frames") != 300 or manifest.get("timing", {}).get("warmup") != 30 or manifest.get("timing", {}).get("repetitions") != 3:
        fail("Formal XPBD residency scaling must use 30 warm-up + 300 measured frames and three repetitions.")

    rows = []
    comparisons = []
    expected_commit = manifest.get("git_commit", "")
    for dimension in manifest["cloth_dimensions"]:
        dimension_root = run_directory(run_root, dimension)
        child_manifest = load_json(dimension_root / "manifest.json")
        if child_manifest.get("git_commit") != expected_commit or child_manifest.get("cloth_dimension") != dimension:
            fail("Child residency manifest mismatch at {0}.".format(dimension))
        errors = trajectory_errors(dimension_root)
        if errors["p95_position_rel_l2"] > manifest["acceptance"]["position_checkpoint_rel_l2_p95"]:
            fail("Position trajectory consistency gate failed at {0}.".format(dimension))
        entries = {}
        for condition in CONDITIONS:
            samples, repetition_means = timing_samples(dimension_root, condition, manifest)
            quality = trajectory_quality(dimension_root, condition, manifest)
            dispatches = {sample["xpbd_dispatches"] for sample in samples}
            if len(dispatches) != 1:
                fail("XPBD dispatch count varies at {0} {1}.".format(dimension, condition))
            row = {
                "cloth_dimension": dimension,
                "vertices": dimension * dimension,
                "condition": condition,
                "repetitions": len(repetition_means),
                "measured_frames_per_repetition": manifest["timing"]["frames"],
                "frame_wall_ms_mean": mean(repetition_means),
                "frame_wall_ms_std": sample_std(repetition_means),
                "frame_wall_ms_p50": percentile([sample["frame_wall_ms"] for sample in samples], 0.50),
                "frame_wall_ms_p95": percentile([sample["frame_wall_ms"] for sample in samples], 0.95),
                "state_h2d_mib_per_frame": mean([sample["state_h2d_bytes"] for sample in samples]) / 1048576.0,
                "state_d2h_mib_per_frame": mean([sample["state_d2h_bytes"] for sample in samples]) / 1048576.0,
                "state_upload_calls_per_frame": mean([sample["state_upload_calls"] for sample in samples]),
                "state_readback_calls_per_frame": mean([sample["state_readback_calls"] for sample in samples]),
                "xpbd_dispatches_per_frame": next(iter(dispatches)),
                "p95_position_rel_l2": errors["p95_position_rel_l2"],
                "p95_velocity_rel_l2": errors["p95_velocity_rel_l2"],
                "p95_mean_stretch_strain": quality["p95_mean_stretch_strain"],
                "p95_max_stretch_strain": quality["p95_max_stretch_strain"],
                "max_penetration_depth": quality["max_penetration_depth"],
                "git_commit": expected_commit,
            }
            entries[condition] = row
            rows.append(row)
        resident = entries["resident"]
        forced = entries["forced-cpu-state-roundtrip"]
        if resident["xpbd_dispatches_per_frame"] != forced["xpbd_dispatches_per_frame"]:
            fail("Residency conditions changed XPBD dispatch count at {0}.".format(dimension))
        if resident["state_h2d_mib_per_frame"] != 0.0 or resident["state_d2h_mib_per_frame"] != 0.0:
            fail("Resident XPBD has state traffic at {0}.".format(dimension))
        if forced["state_h2d_mib_per_frame"] <= 0.0 or forced["state_d2h_mib_per_frame"] <= 0.0:
            fail("Forced XPBD lacks roundtrip traffic at {0}.".format(dimension))
        comparisons.append({
            "cloth_dimension": dimension,
            "resident_vs_forced_mean_speedup": forced["frame_wall_ms_mean"] / resident["frame_wall_ms_mean"],
            "resident_vs_forced_p50_speedup": forced["frame_wall_ms_p50"] / resident["frame_wall_ms_p50"],
            "resident_mean_ms": resident["frame_wall_ms_mean"],
            "forced_mean_ms": forced["frame_wall_ms_mean"],
            "resident_p50_ms": resident["frame_wall_ms_p50"],
            "forced_p50_ms": forced["frame_wall_ms_p50"],
        })

    write_csv(run_root / "xpbd_residency_scaling_summary.csv", rows)
    write_csv(run_root / "xpbd_residency_scaling_comparisons.csv", comparisons)
    report = [
        "# Formal XPBD Residency Scaling Study",
        "",
        "- Commit `{0}`; GPU `{1}`; driver `{2}`.".format(expected_commit, manifest["hardware"]["gpu_name"], manifest["hardware"]["nvidia_driver_version"]),
        "- Moving-sphere cloth, rendered {0}x{1}, 30 warm-up + 300 measured frames, three independent repetitions per condition.".format(manifest["measurement"]["render_width"], manifest["measurement"]["render_height"]),
        "- Resident and forced runs use `gpu-xpbd-jacobi`, {0} iterations/frame, and equal per-frame XPBD dispatch counts. Timing omits trajectory and quality readback.".format(manifest["iterations_per_frame"]),
        "",
        "| Mesh | Condition | Frame mean +/- std (ms) | P50 / P95 (ms) | H2D / D2H MiB | Upload / readback calls | XPBD dispatches |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        report.append("| {0}x{0} | {1} | {2:.3f} +/- {3:.3f} | {4:.3f} / {5:.3f} | {6:.3f} / {7:.3f} | {8:.1f} / {9:.1f} | {10:.0f} |".format(
            row["cloth_dimension"], row["condition"], row["frame_wall_ms_mean"], row["frame_wall_ms_std"], row["frame_wall_ms_p50"], row["frame_wall_ms_p95"],
            row["state_h2d_mib_per_frame"], row["state_d2h_mib_per_frame"], row["state_upload_calls_per_frame"], row["state_readback_calls_per_frame"], row["xpbd_dispatches_per_frame"]))
    report += ["", "| Mesh | Resident / forced mean speedup | Resident / forced P50 speedup | P95 position / velocity relative L2 |", "| --- | ---: | ---: | ---: |"]
    for comparison in comparisons:
        resident = next(row for row in rows if row["cloth_dimension"] == comparison["cloth_dimension"] and row["condition"] == "resident")
        report.append("| {0}x{0} | {1:.2f}x | {2:.2f}x | {3:.3e} / {4:.3e} |".format(
            comparison["cloth_dimension"], comparison["resident_vs_forced_mean_speedup"], comparison["resident_vs_forced_p50_speedup"], resident["p95_position_rel_l2"], resident["p95_velocity_rel_l2"]))
    report += ["", "## Trajectory-quality audit", "", "| Mesh | Condition | P95 mean stretch | P95 max stretch | Max penetration |", "| --- | --- | ---: | ---: | ---: |"]
    for row in rows:
        report.append("| {0}x{0} | {1} | {2:.6g} | {3:.6g} | {4:.6g} |".format(row["cloth_dimension"], row["condition"], row["p95_mean_stretch_strain"], row["p95_max_stretch_strain"], row["max_penetration_depth"]))
    report += [
        "",
        "## Interpretation boundary",
        "",
        "This is a controlled full-state coherence experiment. It establishes that the same XPBD dispatch sequence avoids host/device state traffic when the simulation state is GPU-resident. CPU still schedules frame iterations and dispatches; the result does not claim GPU-autonomous simulation. The trajectory audit accepts accumulated representation roundoff below the registered position and velocity gates, not bitwise identity.",
    ]
    (run_root / "xpbd_residency_scaling_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("Formal XPBD residency scaling summary: {0}".format(run_root / "xpbd_residency_scaling_summary.csv"))


if __name__ == "__main__":
    main()
