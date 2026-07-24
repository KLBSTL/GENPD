#!/usr/bin/env python3
"""Validate a rendered persistent-state versus forced-roundtrip experiment."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONDITIONS = ("persistent", "roundtrip")
LABELS = {"persistent": "GPU-resident state", "roundtrip": "Forced state roundtrip"}
COLORS = {"persistent": "#4E79A7", "roundtrip": "#E15759"}


def fail(message):
    raise RuntimeError(message)


def read_csv(path):
    if not path.is_file():
        fail("Missing required file: {0}".format(path))
    with path.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        fail("CSV contains no rows: {0}".format(path))
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


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        fail("Cannot calculate a percentile from no values.")
    index = int(math.ceil(fraction * len(ordered))) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def measured(rows, warmup, path):
    selected = [row for row in rows if int(number(row, "frame", path)) >= warmup]
    if not selected:
        fail("No measured rows after warmup in {0}".format(path))
    return selected


def validate_run(run_dir, condition, expected_frames, warmup, expected_iterations):
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.is_file():
        fail("Missing run metadata: {0}".format(metadata_path))
    metadata = json.loads(metadata_path.read_text())
    benchmark = metadata.get("benchmark", {})
    if benchmark.get("no_render") or benchmark.get("hide_window"):
        fail("Run is not rendered: {0}".format(run_dir))
    if metadata.get("solver_variant") != "gpu-gather-fusion-batched-ls-persistent":
        fail("Unexpected solver variant in {0}".format(run_dir))
    controls = metadata.get("solver_controls", {})
    forced_metadata = controls.get("force_cpu_state_roundtrip")
    expected_forced = "1" if condition == "roundtrip" else "0"
    if forced_metadata != expected_forced:
        fail("Metadata roundtrip flag mismatch in {0}: {1}".format(run_dir, forced_metadata))

    profile_path = run_dir / "frame_profile.csv"
    experiment_path = run_dir / "frame_profile_experiment.csv"
    extended_path = run_dir / "frame_profile_extended.csv"
    presentation_path = run_dir / "frame_presentation.csv"
    profile = measured(read_csv(profile_path), warmup, profile_path)
    experiment = measured(read_csv(experiment_path), warmup, experiment_path)
    extended = measured(read_csv(extended_path), warmup, extended_path)
    presentation = measured(read_csv(presentation_path), warmup, presentation_path)
    if len(profile) != expected_frames or len(experiment) != expected_frames or len(extended) != expected_frames:
        fail("Measured profile count does not equal {0} in {1}".format(expected_frames, run_dir))
    if len(presentation) < expected_frames:
        fail("Rendered presentation rows are incomplete in {0}".format(run_dir))
    if any(int(number(row, "rendered", presentation_path)) != 1 for row in presentation):
        fail("A measured presentation row was not rendered in {0}".format(run_dir))
    if any(int(number(row, "frame_valid", extended_path)) != 1 for row in extended):
        fail("An invalid simulation frame occurred in {0}".format(run_dir))
    if any(int(number(row, "exploded", extended_path)) != 0 for row in extended):
        fail("An exploded simulation frame occurred in {0}".format(run_dir))
    if any(int(number(row, "iterations", profile_path)) != expected_iterations for row in profile):
        fail("Unexpected selected iteration count in {0}".format(run_dir))
    forced_values = {int(number(row, "forced_cpu_state_roundtrip", experiment_path)) for row in experiment}
    if forced_values != {int(expected_forced)}:
        fail("Experiment-profile roundtrip flag mismatch in {0}".format(run_dir))

    return {
        "frame_wall_ms": mean([number(row, "frame_wall_ms", presentation_path) for row in presentation]),
        "total_ms": mean([number(row, "total_ms", profile_path) for row in profile]),
        "transfer_ms": mean([number(row, "transfer_ms", profile_path) for row in profile]),
        "host_readbacks": mean([number(row, "host_readbacks", experiment_path) for row in experiment]),
        "dispatches": mean([
            number(row, "gradient_dispatches", experiment_path)
            + number(row, "stats_dispatches", experiment_path)
            + number(row, "reduction_dispatches", experiment_path)
            + number(row, "xupdate_dispatches", experiment_path)
            + number(row, "descent_dispatches", experiment_path)
            for row in experiment
        ]),
        "p95_max_position": percentile([number(row, "max_position", profile_path) for row in profile], 0.95),
        "persistent_active": mean([number(row, "persistent_buffers_active", experiment_path) for row in experiment]),
        "git_commit": metadata.get("git_commit", ""),
        "gpu_name": metadata.get("gpu_name", ""),
        "driver": metadata.get("nvidia_driver_version", ""),
    }


def write_csv(path, rows):
    fields = [
        "condition", "repetitions", "frame_wall_ms_mean", "frame_wall_ms_std",
        "total_ms_mean", "total_ms_std", "transfer_ms_mean", "transfer_ms_std",
        "host_readbacks_mean", "dispatches_mean", "p95_max_position_mean",
        "persistent_active_mean", "git_commit", "gpu_name", "driver",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(summary, output_dir):
    metrics = [
        ("frame_wall_ms_mean", "frame_wall_ms_std", "Rendered frame time (ms)"),
        ("transfer_ms_mean", "transfer_ms_std", "CPU transfer time (ms)"),
        ("host_readbacks_mean", None, "Host readbacks / frame"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.45))
    for axis, (field, std_field, ylabel) in zip(axes, metrics):
        values = [float(summary[condition][field]) for condition in CONDITIONS]
        errors = [float(summary[condition][std_field]) for condition in CONDITIONS] if std_field else None
        axis.bar(range(2), values, color=[COLORS[condition] for condition in CONDITIONS], yerr=errors, capsize=3)
        axis.set_xticks(range(2))
        axis.set_xticklabels(["Resident", "Roundtrip"], fontsize=8)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_title("End-to-end rendered cost")
    axes[1].set_title("Explicit state-transfer cost")
    axes[2].set_title("Synchronization pressure")
    fig.tight_layout(pad=0.45)
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / "roundtrip_validation.{0}".format(suffix), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--repetitions", type=int, required=True)
    args = parser.parse_args()
    if args.expected_frames < 1 or args.warmup < 0 or args.repetitions < 2:
        fail("Invalid expected experiment dimensions.")

    run_root = Path(args.run_root).resolve()
    manifest_path = run_root / "roundtrip_manifest.json"
    if not manifest_path.is_file():
        fail("Missing roundtrip manifest: {0}".format(manifest_path))
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("protocol") != "persistent-state-roundtrip-v1":
        fail("Unexpected roundtrip manifest protocol.")
    if manifest.get("frames") != args.expected_frames or manifest.get("warmup_frames") != args.warmup:
        fail("CLI expectations disagree with the manifest.")
    expected_iterations = int(manifest["iterations_per_frame"])

    all_runs = {}
    for condition in CONDITIONS:
        entries = []
        for repetition in range(1, args.repetitions + 1):
            run_name = "{0}-rep{1:02d}".format(condition, repetition)
            entries.append(validate_run(run_root / run_name, condition, args.expected_frames, args.warmup, expected_iterations))
        all_runs[condition] = entries

    commits = {entry["git_commit"] for entries in all_runs.values() for entry in entries}
    if len(commits) != 1 or not next(iter(commits)):
        fail("All repetitions must report one non-empty Git commit.")
    gpu_names = {entry["gpu_name"] for entries in all_runs.values() for entry in entries}
    drivers = {entry["driver"] for entries in all_runs.values() for entry in entries}
    if len(gpu_names) != 1 or len(drivers) != 1:
        fail("All repetitions must use one GPU and driver.")

    summary = {}
    summary_rows = []
    for condition, entries in all_runs.items():
        row = {"condition": condition, "repetitions": len(entries)}
        for metric in ("frame_wall_ms", "total_ms", "transfer_ms"):
            values = [entry[metric] for entry in entries]
            row[metric + "_mean"] = mean(values)
            row[metric + "_std"] = sample_std(values)
        for metric in ("host_readbacks", "dispatches", "p95_max_position", "persistent_active"):
            row[metric + "_mean"] = mean([entry[metric] for entry in entries])
        row["git_commit"] = next(iter(commits))
        row["gpu_name"] = next(iter(gpu_names))
        row["driver"] = next(iter(drivers))
        summary[condition] = row
        summary_rows.append(row)

    if abs(summary["persistent"]["dispatches_mean"] - summary["roundtrip"]["dispatches_mean"]) > 1e-9:
        fail("The two conditions have different tracked solver-dispatch counts.")
    resident_pos = summary["persistent"]["p95_max_position_mean"]
    roundtrip_pos = summary["roundtrip"]["p95_max_position_mean"]
    relative_position_delta = abs(resident_pos - roundtrip_pos) / max(abs(resident_pos), 1e-8)
    if relative_position_delta > 1e-5:
        fail("State roundtrip changed P95 maximum position by {0:.3e}.".format(relative_position_delta))

    write_csv(run_root / "roundtrip_summary.csv", summary_rows)
    plot(summary, run_root)
    resident = summary["persistent"]
    roundtrip = summary["roundtrip"]
    ratio = roundtrip["frame_wall_ms_mean"] / resident["frame_wall_ms_mean"]
    report = [
        "# Persistent-State Roundtrip Validation",
        "",
        "## Controlled protocol",
        "",
        "- Same persistent NCG variant, scene, cloth resolution, selected iterations, rendered viewport, and tracked dispatch count.",
        "- The roundtrip condition synchronizes GPU position and velocity to CPU after every finalized frame, invalidates the resident state, and therefore reuploads it at the next frame.",
        "- All runs are rendered, GPU-synchronized, finite, and repeated {0} times after {1} warm-up frames.".format(args.repetitions, args.warmup),
        "",
        "## Result",
        "",
        "- Rendered frame time: {0:.3f} +/- {1:.3f} ms resident versus {2:.3f} +/- {3:.3f} ms forced roundtrip ({4:.2f}x).".format(
            resident["frame_wall_ms_mean"], resident["frame_wall_ms_std"], roundtrip["frame_wall_ms_mean"], roundtrip["frame_wall_ms_std"], ratio),
        "- CPU transfer time: {0:.3f} versus {1:.3f} ms; host readbacks: {2:.1f} versus {3:.1f} per frame.".format(
            resident["transfer_ms_mean"], roundtrip["transfer_ms_mean"], resident["host_readbacks_mean"], roundtrip["host_readbacks_mean"]),
        "- P95 maximum-position relative difference: {0:.3e}; both conditions completed without invalid frames.".format(relative_position_delta),
        "",
        "## Interpretation boundary",
        "",
        "This is direct evidence for the cost of abandoning per-frame GPU-resident position/velocity state while retaining the same persistent compute-shader solver. It does not attribute the separate batched-LS-to-persistent variant gap, because that broader variant comparison also changes prediction, state finalization, and solver-control placement.",
    ]
    (run_root / "roundtrip_report.md").write_text("\n".join(report) + "\n")
    metadata = {
        "protocol": manifest["protocol"],
        "run_root": str(run_root),
        "git_commit": next(iter(commits)),
        "gpu_name": next(iter(gpu_names)),
        "driver": next(iter(drivers)),
        "relative_p95_max_position_delta": relative_position_delta,
        "frame_time_ratio_roundtrip_over_resident": ratio,
    }
    (run_root / "roundtrip_analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
