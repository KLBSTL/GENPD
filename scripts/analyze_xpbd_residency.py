#!/usr/bin/env python3
"""Validate a rendered GPU-resident versus forced-roundtrip XPBD study."""

import argparse
import csv
import json
import math
import struct
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONDITIONS = ("resident", "forced-cpu-state-roundtrip")
LABELS = {"resident": "GPU-resident", "forced-cpu-state-roundtrip": "Forced roundtrip"}
COLORS = {"resident": "#4E79A7", "forced-cpu-state-roundtrip": "#E15759"}


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


def timing_run(run_dir, condition, manifest):
    metadata = load_json(run_dir / "run_metadata.json")
    benchmark = metadata.get("benchmark", {})
    if benchmark.get("no_render") or benchmark.get("hide_window") or not benchmark.get("sync_gpu") or not benchmark.get("disable_vsync"):
        fail("Timing run is not rendered, synchronized, and vsync-disabled: {0}".format(run_dir))
    if metadata.get("solver_variant") != "gpu-xpbd-jacobi":
        fail("Unexpected solver variant: {0}".format(run_dir))
    expected_forced = "1" if condition == "forced-cpu-state-roundtrip" else "0"
    if metadata.get("solver_controls", {}).get("force_cpu_state_roundtrip") != expected_forced:
        fail("Forced-roundtrip metadata mismatch: {0}".format(run_dir))

    profile_path = run_dir / "frame_profile.csv"
    extended_path = run_dir / "frame_profile_extended.csv"
    experiment_path = run_dir / "frame_profile_experiment.csv"
    presentation_path = run_dir / "frame_presentation.csv"
    profile = measured(load_csv(profile_path), manifest["timing"]["warmup"], profile_path)
    extended = measured(load_csv(extended_path), manifest["timing"]["warmup"], extended_path)
    experiment = measured(load_csv(experiment_path), manifest["timing"]["warmup"], experiment_path)
    presentation = measured(load_csv(presentation_path), manifest["timing"]["warmup"], presentation_path)
    expected_frames = manifest["timing"]["frames"]
    if any(len(rows) != expected_frames for rows in (profile, extended, experiment, presentation)):
        fail("Incomplete timing rows in {0}".format(run_dir))
    if any(row.get("frame_valid") != "1" or row.get("termination_reason") != "none" for row in extended):
        fail("Invalid timing frame in {0}".format(run_dir))
    if any(row.get("rendered") != "1" or row.get("gpu_sync_enabled") != "1" for row in presentation):
        fail("Unrendered timing frame in {0}".format(run_dir))
    if any(int(number(row, "iterations", profile_path)) != manifest["iterations_per_frame"] for row in profile):
        fail("Unexpected XPBD iteration budget in {0}".format(run_dir))

    required = (
        "persistent_buffers_active", "forced_cpu_state_roundtrip", "state_h2d_bytes", "state_d2h_bytes",
        "state_upload_calls", "state_readback_calls", "xpbd_constraint_dispatches",
        "xpbd_apply_dispatches", "xpbd_collision_dispatches",
    )
    for field in required:
        if field not in experiment[0]:
            fail("Missing telemetry field '{0}' in {1}".format(field, experiment_path))

    expected_active = 1 if condition == "resident" else 0
    if {int(number(row, "persistent_buffers_active", experiment_path)) for row in experiment} != {expected_active}:
        fail("Resident-state activity mismatch in {0}".format(run_dir))
    if {int(number(row, "forced_cpu_state_roundtrip", experiment_path)) for row in experiment} != {int(expected_forced)}:
        fail("Roundtrip telemetry mismatch in {0}".format(run_dir))

    h2d = [number(row, "state_h2d_bytes", experiment_path) for row in experiment]
    d2h = [number(row, "state_d2h_bytes", experiment_path) for row in experiment]
    if condition == "resident":
        if any(value != 0.0 for value in h2d + d2h):
            fail("Resident XPBD timing has per-frame full-state traffic in {0}".format(run_dir))
    elif any(value <= 0.0 for value in h2d) or any(value <= 0.0 for value in d2h):
        fail("Forced XPBD roundtrip lacks per-frame state traffic in {0}".format(run_dir))

    dispatches = [
        number(row, "xpbd_constraint_dispatches", experiment_path)
        + number(row, "xpbd_apply_dispatches", experiment_path)
        + number(row, "xpbd_collision_dispatches", experiment_path)
        for row in experiment
    ]
    return {
        "frame_wall_ms": mean([number(row, "frame_wall_ms", presentation_path) for row in presentation]),
        "total_ms": mean([number(row, "total_ms", profile_path) for row in profile]),
        "state_h2d_bytes": mean(h2d),
        "state_d2h_bytes": mean(d2h),
        "state_upload_calls": mean([number(row, "state_upload_calls", experiment_path) for row in experiment]),
        "state_readback_calls": mean([number(row, "state_readback_calls", experiment_path) for row in experiment]),
        "host_readbacks": mean([number(row, "host_readbacks", experiment_path) for row in experiment]),
        "xpbd_dispatches": mean(dispatches),
        "git_commit": metadata.get("git_commit", ""),
        "gpu_name": metadata.get("gpu_name", ""),
        "driver": metadata.get("nvidia_driver_version", ""),
    }


def read_checkpoint(path):
    raw = path.read_bytes()
    if len(raw) < 24:
        fail("Truncated checkpoint: {0}".format(path))
    magic, version, frame, scalar_bytes, count = struct.unpack_from("<8sIIII", raw, 0)
    if magic != b"GPDQREF1" or version != 1 or scalar_bytes != 4 or count == 0:
        fail("Unsupported checkpoint format: {0}".format(path))
    expected_bytes = 24 + 2 * count * scalar_bytes
    if len(raw) != expected_bytes:
        fail("Checkpoint size mismatch: {0}".format(path))
    values = struct.unpack_from("<{0}f".format(2 * count), raw, 24)
    positions = values[:count]
    velocities = values[count:]
    if not all(math.isfinite(value) for value in values):
        fail("Non-finite checkpoint: {0}".format(path))
    return frame, positions, velocities


def relative_l2(values, reference):
    if len(values) != len(reference):
        fail("Checkpoint vector sizes differ.")
    difference_sq = sum((left - right) ** 2 for left, right in zip(values, reference))
    reference_sq = sum(value * value for value in reference)
    return math.sqrt(difference_sq) / max(math.sqrt(reference_sq), 1.0e-12)


def trajectory_errors(run_root, manifest):
    checkpoints = {}
    for condition in CONDITIONS:
        directory = run_root / "trajectory" / condition / "reference_checkpoints"
        files = sorted(directory.glob("reference_state_*.bin"))
        if not files:
            fail("No trajectory checkpoints for {0}".format(condition))
        records = {}
        for path in files:
            frame, positions, velocities = read_checkpoint(path)
            records[frame] = (positions, velocities)
        checkpoints[condition] = records
    shared = sorted(set(checkpoints["resident"]).intersection(checkpoints["forced-cpu-state-roundtrip"]))
    shared = [frame for frame in shared if frame >= manifest["trajectory"]["warmup"]]
    if len(shared) < 3:
        fail("Trajectory audit has fewer than three matched measured checkpoints.")
    position_errors = []
    velocity_errors = []
    rows = []
    for frame in shared:
        resident_position, resident_velocity = checkpoints["resident"][frame]
        forced_position, forced_velocity = checkpoints["forced-cpu-state-roundtrip"][frame]
        position_error = relative_l2(forced_position, resident_position)
        velocity_error = relative_l2(forced_velocity, resident_velocity)
        position_errors.append(position_error)
        velocity_errors.append(velocity_error)
        rows.append({"frame": frame, "position_rel_l2": position_error, "velocity_rel_l2": velocity_error})
    if percentile(position_errors) > manifest["acceptance"]["position_checkpoint_rel_l2_p95"]:
        fail("XPBD position trajectory mismatch exceeds the pre-registered threshold.")
    if percentile(velocity_errors) > manifest["acceptance"]["velocity_checkpoint_rel_l2_p95"]:
        fail("XPBD velocity trajectory mismatch exceeds the pre-registered threshold.")
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot(summary, output_dir):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    metrics = [
        ("frame_wall_ms_mean", "frame_wall_ms_std", "Rendered frame time (ms)"),
        ("state_h2d_bytes_mean", None, "State H2D traffic (MB/frame)"),
        ("state_d2h_bytes_mean", None, "State D2H traffic (MB/frame)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.1))
    for axis, (field, std_field, label) in zip(axes, metrics):
        values = [float(summary[condition][field]) for condition in CONDITIONS]
        if "bytes" in field:
            values = [value / (1024.0 * 1024.0) for value in values]
        errors = [float(summary[condition][std_field]) for condition in CONDITIONS] if std_field else None
        axis.bar(range(2), values, color=[COLORS[condition] for condition in CONDITIONS], yerr=errors, capsize=3)
        axis.set_xticks(range(2), ["Resident", "Roundtrip"])
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout(pad=0.45)
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / "xpbd_residency.{0}".format(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    manifest = load_json(run_root / "manifest.json")
    if manifest.get("protocol_version") != "xpbd-residency-v1":
        fail("Unexpected XPBD residency protocol.")
    if manifest.get("measurement", {}).get("mode") != "rendered-end-to-end" or manifest.get("timing", {}).get("repetitions") != 3:
        fail("XPBD residency study is not the formal rendered three-repetition protocol.")

    per_repetition = []
    summary = {}
    for condition in CONDITIONS:
        entries = []
        for repetition in range(1, manifest["timing"]["repetitions"] + 1):
            run_dir = run_root / "timing" / condition / "rep{0:02d}".format(repetition)
            entry = timing_run(run_dir, condition, manifest)
            entry["condition"] = condition
            entry["repetition"] = repetition
            entry["result_dir"] = str(run_dir)
            entries.append(entry)
            per_repetition.append(entry)
        row = {"condition": condition, "repetitions": len(entries)}
        for field in ("frame_wall_ms", "total_ms", "state_h2d_bytes", "state_d2h_bytes", "state_upload_calls", "state_readback_calls", "host_readbacks", "xpbd_dispatches"):
            values = [entry[field] for entry in entries]
            row[field + "_mean"] = mean(values)
            row[field + "_std"] = sample_std(values)
        for field in ("git_commit", "gpu_name", "driver"):
            values = {entry[field] for entry in entries}
            if len(values) != 1 or not next(iter(values)):
                fail("Inconsistent metadata for {0}".format(condition))
            row[field] = next(iter(values))
        summary[condition] = row

    if abs(summary["resident"]["xpbd_dispatches_mean"] - summary["forced-cpu-state-roundtrip"]["xpbd_dispatches_mean"]) > 1.0e-9:
        fail("XPBD dispatch counts differ between residency conditions.")
    commits = {summary[condition]["git_commit"] for condition in CONDITIONS}
    if len(commits) != 1:
        fail("Residency conditions were not run from one commit.")

    trajectory_rows = trajectory_errors(run_root, manifest)
    write_csv(run_root / "xpbd_residency_per_repetition.csv", per_repetition)
    write_csv(run_root / "xpbd_residency_summary.csv", [summary[condition] for condition in CONDITIONS])
    write_csv(run_root / "xpbd_residency_trajectory.csv", trajectory_rows)
    plot(summary, run_root)

    ratio = summary["forced-cpu-state-roundtrip"]["frame_wall_ms_mean"] / summary["resident"]["frame_wall_ms_mean"]
    report = [
        "# GPU-Resident XPBD Study",
        "",
        "## Controlled protocol",
        "",
        "- Same `gpu-xpbd-jacobi` scene, resolution, iterations, rendered viewport, and tracked XPBD dispatch count.",
        "- The forced condition alone reads finalized position/velocity state to CPU, invalidates it, then reuploads the same state on the next frame.",
        "- Timing excludes checkpoint export and uses {0} warm-up plus {1} measured frames across three repetitions.".format(manifest["timing"]["warmup"], manifest["timing"]["frames"]),
        "",
        "## Result",
        "",
        "- Rendered frame time: {0:.3f} +/- {1:.3f} ms resident versus {2:.3f} +/- {3:.3f} ms forced roundtrip ({4:.2f}x).".format(
            summary["resident"]["frame_wall_ms_mean"], summary["resident"]["frame_wall_ms_std"],
            summary["forced-cpu-state-roundtrip"]["frame_wall_ms_mean"], summary["forced-cpu-state-roundtrip"]["frame_wall_ms_std"], ratio),
        "- State traffic resident versus forced H2D/D2H: {0:.3f}/{1:.3f} versus {2:.3f}/{3:.3f} MiB/frame.".format(
            summary["resident"]["state_h2d_bytes_mean"] / 1048576.0, summary["resident"]["state_d2h_bytes_mean"] / 1048576.0,
            summary["forced-cpu-state-roundtrip"]["state_h2d_bytes_mean"] / 1048576.0, summary["forced-cpu-state-roundtrip"]["state_d2h_bytes_mean"] / 1048576.0),
        "- Matched checkpoint P95 position/velocity relative L2: {0:.3e}/{1:.3e}.".format(
            percentile([row["position_rel_l2"] for row in trajectory_rows]), percentile([row["velocity_rel_l2"] for row in trajectory_rows])),
        "",
        "## Interpretation boundary",
        "",
        "This is evidence that the same XPBD implementation avoids measured full-state host traffic when its simulation state remains GPU-resident. CPU still controls frame dispatch and reads compact state statistics; the study does not claim fully GPU-autonomous execution.",
    ]
    (run_root / "xpbd_residency_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("XPBD residency summary: {0}".format(run_root / "xpbd_residency_summary.csv"))


if __name__ == "__main__":
    main()
