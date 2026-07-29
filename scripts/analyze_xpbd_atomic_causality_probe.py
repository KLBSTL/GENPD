#!/usr/bin/env python3
"""Validate the rendered A/B/B/A XPBD atomic-causality probe."""

import argparse
import csv
import hashlib
import json
import math
import statistics
import struct
from pathlib import Path


RUNS = (
    ("atomic-a", "gpu-xpbd-jacobi"),
    ("gather-a", "gpu-xpbd-vertex-gather"),
    ("gather-b", "gpu-xpbd-vertex-gather"),
    ("atomic-b", "gpu-xpbd-jacobi"),
)


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


def measured(rows, warmup, path):
    selected = [row for row in rows if int(number(row, "frame", path)) >= warmup]
    if not selected:
        fail("No measured rows in {0}".format(path))
    return selected


def percentile(values, fraction=0.95):
    if not values:
        fail("Cannot take a percentile of no values.")
    values = sorted(values)
    index = max(0, min(len(values) - 1, int(math.ceil(fraction * len(values))) - 1))
    return values[index]


def rms(values):
    if not values:
        fail("Cannot take an RMS of no values.")
    return math.sqrt(sum(value * value for value in values) / float(len(values)))


def relative_l2(values, reference):
    if len(values) != len(reference):
        fail("Checkpoint vector sizes differ.")
    difference_sq = sum((left - right) ** 2 for left, right in zip(values, reference))
    reference_sq = sum(value * value for value in reference)
    return math.sqrt(difference_sq) / max(math.sqrt(reference_sq), 1.0e-12)


def read_checkpoint(path):
    raw = path.read_bytes()
    if len(raw) < 24:
        fail("Truncated checkpoint: {0}".format(path))
    magic, version, frame, scalar_bytes, count = struct.unpack_from("<8sIIII", raw, 0)
    if magic != b"GPDQREF1" or version != 1 or scalar_bytes != 4 or count == 0:
        fail("Unsupported checkpoint format: {0}".format(path))
    if len(raw) != 24 + 2 * count * scalar_bytes:
        fail("Checkpoint size mismatch: {0}".format(path))
    values = struct.unpack_from("<{0}f".format(2 * count), raw, 24)
    if not all(math.isfinite(value) for value in values):
        fail("Non-finite checkpoint: {0}".format(path))
    return frame, values[:count], values[count:], hashlib.sha256(raw).hexdigest()


def load_checkpoints(run_dir, warmup, minimum):
    records = {}
    directory = run_dir / "reference_checkpoints"
    for path in sorted(directory.glob("reference_state_*.bin")):
        frame, positions, velocities, digest = read_checkpoint(path)
        if frame in records:
            fail("Duplicate checkpoint frame: {0}".format(path))
        records[frame] = (positions, velocities, digest)
    records = {frame: state for frame, state in records.items() if frame >= warmup}
    if len(records) < minimum:
        fail("Too few measured checkpoints in {0}".format(directory))
    return records


def summarize_run(run_dir, run_id, expected_variant, manifest):
    metadata = load_json(run_dir / "run_metadata.json")
    benchmark = metadata.get("benchmark", {})
    if metadata.get("solver_variant") != expected_variant:
        fail("Unexpected solver variant in {0}".format(run_dir))
    if benchmark.get("no_render") or benchmark.get("hide_window") or not benchmark.get("sync_gpu") or not benchmark.get("disable_vsync"):
        fail("Run is not rendered, synchronized, and vsync-disabled: {0}".format(run_dir))
    if metadata.get("quality", {}).get("iterations_per_frame") != str(manifest["iterations_per_frame"]):
        fail("Iteration-budget mismatch in {0}".format(run_dir))
    if metadata.get("experiment_overrides", {}).get("cloth_dimension") != str(manifest["cloth_dimension"]):
        fail("Cloth-dimension mismatch in {0}".format(run_dir))
    if Path(metadata.get("experiment_overrides", {}).get("scene", "")).name.lower() != Path(manifest["scene_path"]).name.lower():
        fail("Scene mismatch in {0}".format(run_dir))
    controls = metadata.get("solver_controls", {})
    if controls.get("force_cpu_state_roundtrip") != "0" or controls.get("xpbd_fuse_apply_collision") != "0" or controls.get("xpbd_cached_pins") != "1":
        fail("Unexpected XPBD controls in {0}".format(run_dir))

    warmup = manifest["trajectory"]["warmup"]
    expected_frames = manifest["trajectory"]["frames"]
    profile_path = run_dir / "frame_profile.csv"
    extended_path = run_dir / "frame_profile_extended.csv"
    experiment_path = run_dir / "frame_profile_experiment.csv"
    presentation_path = run_dir / "frame_presentation.csv"
    quality_path = run_dir / "quality_metrics.csv"
    profile = measured(load_csv(profile_path), warmup, profile_path)
    extended = measured(load_csv(extended_path), warmup, extended_path)
    experiment = measured(load_csv(experiment_path), warmup, experiment_path)
    presentation = measured(load_csv(presentation_path), warmup, presentation_path)
    quality = measured(load_csv(quality_path), warmup, quality_path)
    if any(len(rows) != expected_frames for rows in (profile, extended, experiment, presentation, quality)):
        fail("Incomplete measured frame count in {0}".format(run_dir))
    if any(row.get("frame_valid") != "1" or row.get("termination_reason") != "none" for row in extended):
        fail("Invalid solver frame in {0}".format(run_dir))
    if any(row.get("rendered") != "1" or row.get("gpu_sync_enabled") != "1" for row in presentation):
        fail("Unrendered frame in {0}".format(run_dir))
    if any(row.get("finite") != "1" or row.get("exploded") != "0" for row in quality):
        fail("Invalid quality metric in {0}".format(run_dir))
    if any(int(number(row, "iterations", profile_path)) != manifest["iterations_per_frame"] for row in profile):
        fail("Unexpected XPBD iteration count in {0}".format(run_dir))
    for row in experiment:
        if int(number(row, "xpbd_constraint_dispatches", experiment_path)) != manifest["iterations_per_frame"]:
            fail("Constraint dispatch count mismatch in {0}".format(run_dir))
        if int(number(row, "xpbd_apply_dispatches", experiment_path)) != manifest["iterations_per_frame"]:
            fail("Apply dispatch count mismatch in {0}".format(run_dir))
        if int(number(row, "xpbd_collision_dispatches", experiment_path)) != manifest["iterations_per_frame"]:
            fail("Collision dispatch count mismatch in {0}".format(run_dir))

    checkpoints = load_checkpoints(run_dir, warmup, manifest["acceptance"]["min_matched_measured_checkpoints"])
    return {
        "run_id": run_id,
        "solver_variant": expected_variant,
        "reduction": "float-atomic-scatter" if expected_variant == "gpu-xpbd-jacobi" else "fixed-order-vertex-gather",
        "measured_frames": expected_frames,
        "measured_checkpoints": len(checkpoints),
        "p95_mean_stretch_strain": percentile([number(row, "mean_stretch_strain", quality_path) for row in quality]),
        "p95_max_stretch_strain": percentile([number(row, "max_stretch_strain", quality_path) for row in quality]),
        "max_penetration_depth": max(number(row, "max_penetration_depth", quality_path) for row in quality),
        "git_commit": metadata.get("git_commit", ""),
        "gpu_name": metadata.get("gpu_name", ""),
        "driver": metadata.get("nvidia_driver_version", ""),
        "result_dir": str(run_dir),
        "checkpoints": checkpoints,
    }


def compare_pair(pair_type, encoded_pair, left, right, minimum):
    left_name, right_name = encoded_pair.split(":", 1)
    shared = sorted(set(left["checkpoints"]).intersection(right["checkpoints"]))
    if len(shared) < minimum:
        fail("Too few matched checkpoints for {0}".format(encoded_pair))
    rows = []
    for frame in shared:
        left_position, left_velocity, left_digest = left["checkpoints"][frame]
        right_position, right_velocity, right_digest = right["checkpoints"][frame]
        rows.append({
            "pair_type": pair_type,
            "pair": encoded_pair,
            "frame": frame,
            "checkpoint_sha256_equal": int(left_digest == right_digest),
            "position_rel_l2": relative_l2(right_position, left_position),
            "velocity_rel_l2": relative_l2(right_velocity, left_velocity),
            "velocity_difference_rms": rms([right_value - left_value for left_value, right_value in zip(left_velocity, right_velocity)]),
            "velocity_reference_rms": rms(left_velocity),
        })
    return rows


def pair_summary(pair_type, pair, rows):
    return {
        "pair_type": pair_type,
        "pair": pair,
        "matched_checkpoints": len(rows),
        "checkpoint_sha256_all_equal": int(all(row["checkpoint_sha256_equal"] == 1 for row in rows)),
        "position_rel_l2_p95": percentile([row["position_rel_l2"] for row in rows]),
        "position_rel_l2_max": max(row["position_rel_l2"] for row in rows),
        "velocity_rel_l2_p95": percentile([row["velocity_rel_l2"] for row in rows]),
        "velocity_rel_l2_max": max(row["velocity_rel_l2"] for row in rows),
        "velocity_difference_rms_p95": percentile([row["velocity_difference_rms"] for row in rows]),
        "velocity_reference_rms_p50": percentile([row["velocity_reference_rms"] for row in rows], 0.50),
    }


def relative_difference(left, right):
    return abs(left - right) / max(abs(left), 1.0e-12)


def write_csv(path, rows):
    if not rows:
        fail("Refusing to write empty CSV: {0}".format(path))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    manifest = load_json(root / "manifest.json")
    if manifest.get("protocol_version") != "xpbd-atomic-causality-probe-v1":
        fail("Unexpected atomic-causality protocol.")
    if manifest.get("measurement", {}).get("mode") != "rendered-trajectory-diagnostic" or manifest.get("measurement", {}).get("timing_claim") != "none":
        fail("Causality probe must remain a rendered trajectory diagnostic.")
    if [(entry.get("id"), entry.get("solver_variant")) for entry in manifest.get("runs", [])] != list(RUNS):
        fail("Unexpected A/B/B/A run set.")

    records = {}
    quality_rows = []
    for run_id, variant in RUNS:
        record = summarize_run(root / "runs" / run_id, run_id, variant, manifest)
        records[run_id] = record
        quality_rows.append({key: value for key, value in record.items() if key != "checkpoints"})

    expected_commit = manifest.get("git_commit", "")
    commits = {row["git_commit"] for row in quality_rows}
    hardware = {(row["gpu_name"], row["driver"]) for row in quality_rows}
    if len(commits) != 1 or not next(iter(commits)) or not expected_commit.startswith(next(iter(commits))):
        fail("Probe runs do not share a commit compatible with the manifest.")
    if len(hardware) != 1:
        fail("Probe runs do not share one GPU and driver.")

    causality = manifest["causality"]
    pairs = (
        ("atomic-repeat", causality["atomic_repeat"]),
        ("gather-repeat", causality["gather_repeat"]),
    )
    all_pair_rows = []
    summaries = {}
    for pair_type, encoded_pair in pairs:
        left_name, right_name = encoded_pair.split(":", 1)
        rows = compare_pair(pair_type, encoded_pair, records[left_name], records[right_name], manifest["acceptance"]["min_matched_measured_checkpoints"])
        all_pair_rows.extend(rows)
        summaries[pair_type] = pair_summary(pair_type, encoded_pair, rows)

    atomic_values = [row for row in quality_rows if row["solver_variant"] == "gpu-xpbd-jacobi"]
    gather_values = [row for row in quality_rows if row["solver_variant"] == "gpu-xpbd-vertex-gather"]
    physical = {
        "atomic_p95_mean_stretch": statistics.fmean(row["p95_mean_stretch_strain"] for row in atomic_values),
        "gather_p95_mean_stretch": statistics.fmean(row["p95_mean_stretch_strain"] for row in gather_values),
        "atomic_p95_max_stretch": statistics.fmean(row["p95_max_stretch_strain"] for row in atomic_values),
        "gather_p95_max_stretch": statistics.fmean(row["p95_max_stretch_strain"] for row in gather_values),
        "atomic_max_penetration": max(row["max_penetration_depth"] for row in atomic_values),
        "gather_max_penetration": max(row["max_penetration_depth"] for row in gather_values),
    }
    physical["mean_stretch_relative_difference"] = relative_difference(physical["atomic_p95_mean_stretch"], physical["gather_p95_mean_stretch"])
    physical["max_stretch_relative_difference"] = relative_difference(physical["atomic_p95_max_stretch"], physical["gather_p95_max_stretch"])
    physical["penetration_absolute_difference"] = abs(physical["atomic_max_penetration"] - physical["gather_max_penetration"])
    physical["comparable"] = int(
        physical["mean_stretch_relative_difference"] <= causality["physical_mean_stretch_relative_difference_max"]
        and physical["max_stretch_relative_difference"] <= causality["physical_max_stretch_relative_difference_max"]
        and physical["penetration_absolute_difference"] <= causality["physical_penetration_absolute_difference_max"]
    )

    atomic = summaries["atomic-repeat"]
    gather = summaries["gather-repeat"]
    gather_exact = gather["checkpoint_sha256_all_equal"] == 1
    atomic_nonidentical = atomic["checkpoint_sha256_all_equal"] == 0
    strong_support = (
        physical["comparable"] == 1
        and atomic_nonidentical
        and atomic["velocity_rel_l2_p95"] >= causality["atomic_signal_velocity_rel_l2_p95"]
        and gather_exact
        and gather["velocity_rel_l2_p95"] <= causality["gather_repeat_velocity_rel_l2_p95_max"]
    )
    if strong_support:
        verdict = "strong-support-for-atomic-scatter-causality"
    elif gather_exact and atomic_nonidentical:
        verdict = "atomic-associated-but-below-pre-registered-signal-or-quality-gate"
    elif not gather_exact:
        verdict = "inconclusive-gather-path-is-not-repeat-exact"
    else:
        verdict = "inconclusive-atomic-repeat-did-not-reproduce"

    write_csv(root / "run_quality_summary.csv", quality_rows)
    write_csv(root / "trajectory_pair_errors.csv", all_pair_rows)
    write_csv(root / "causality_summary.csv", [atomic, gather])
    summary = {
        "protocol": manifest["protocol_version"],
        "git_commit": expected_commit,
        "verdict": verdict,
        "strong_support": strong_support,
        "atomic_repeat": atomic,
        "gather_repeat": gather,
        "physical_comparability": physical,
    }
    (root / "causality_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    report = [
        "# XPBD Atomic-Scatter Causality Probe",
        "",
        "## Protocol",
        "",
        "- Commit `{0}`; GPU `{1}`; driver `{2}`.".format(expected_commit, next(iter(hardware))[0], next(iter(hardware))[1]),
        "- Moving-sphere cloth, {0}x{0}, rendered {1}x{2}; 20 warm-up + 120 measured frames; checkpoint stride 10.".format(manifest["cloth_dimension"], manifest["measurement"]["render_width"], manifest["measurement"]["render_height"]),
        "- All runs use 32 XPBD iterations/frame, the same collision settings, GPU synchronization, disabled VSync, and no forced CPU state roundtrip.",
        "- A/B/B/A process order: atomic Jacobi, fixed-order vertex gather, fixed-order vertex gather, atomic Jacobi.",
        "- This is a trajectory diagnostic, not a timing comparison.",
        "",
        "## Repeat Results",
        "",
        "| Pair | Checkpoint SHA-256 all equal | Position P95 relative L2 | Velocity P95 relative L2 | Velocity-difference P95 RMS |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in (atomic, gather):
        report.append("| {0} | {1} | {2:.3e} | {3:.3e} | {4:.3e} |".format(
            row["pair"], "yes" if row["checkpoint_sha256_all_equal"] else "no", row["position_rel_l2_p95"],
            row["velocity_rel_l2_p95"], row["velocity_difference_rms_p95"]))
    report += [
        "",
        "## Physical Comparability",
        "",
        "| Metric | Atomic mean/max | Gather mean/max | Difference | Gate |",
        "| --- | ---: | ---: | ---: | ---: |",
        "| P95 mean stretch | {0:.6f} | {1:.6f} | {2:.3e} relative | <= {3:.3e} |".format(
            physical["atomic_p95_mean_stretch"], physical["gather_p95_mean_stretch"], physical["mean_stretch_relative_difference"], causality["physical_mean_stretch_relative_difference_max"]),
        "| P95 max stretch | {0:.6f} | {1:.6f} | {2:.3e} relative | <= {3:.3e} |".format(
            physical["atomic_p95_max_stretch"], physical["gather_p95_max_stretch"], physical["max_stretch_relative_difference"], causality["physical_max_stretch_relative_difference_max"]),
        "| Max penetration | {0:.6f} | {1:.6f} | {2:.3e} absolute | <= {3:.3e} |".format(
            physical["atomic_max_penetration"], physical["gather_max_penetration"], physical["penetration_absolute_difference"], causality["physical_penetration_absolute_difference_max"]),
        "",
        "Physical comparability: **{0}**.".format("PASS" if physical["comparable"] else "FAIL"),
        "",
        "## Verdict",
        "",
        "**{0}**".format(verdict),
        "",
    ]
    if strong_support:
        report += [
            "The atomic repeat is nonidentical and above the registered velocity signal, while the no-atomic vertex-gather repeat is checkpoint-hash identical and physically comparable. This is strong causal evidence that the float atomic scatter, rather than CPU roundtrip or moving contact alone, is the dominant source of the observed 128x128 run-to-run divergence.",
        ]
    else:
        report += [
            "The registered condition for a strong atomic-scatter conclusion was not fully met. Do not attribute the long-horizon difference uniquely to atomics from this run; inspect the exact pair data and extend the probe before making that claim.",
        ]
    report += [
        "",
        "## Artifacts",
        "",
        "- `manifest.json`: pre-registered controls and conclusion rule.",
        "- `run_quality_summary.csv`: finite-state and physical-quality checks.",
        "- `trajectory_pair_errors.csv`: matched checkpoint errors and exact SHA-256 equality flags.",
        "- `causality_summary.csv` and `causality_summary.json`: compact verdict inputs.",
        "- `runs/<id>/`: raw rendered profiles, metadata, logs, and checkpoint binaries.",
    ]
    (root / "xpbd_atomic_causality_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("XPBD atomic-causality summary: {0}".format(root / "causality_summary.json"))


if __name__ == "__main__":
    main()
