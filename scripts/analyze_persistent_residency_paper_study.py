#!/usr/bin/env python3
"""Aggregate the formal rendered persistent-state residency counterfactual."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONDITIONS = ("resident", "forced-cpu-state-roundtrip")
LABELS = {"resident": "GPU-resident", "forced-cpu-state-roundtrip": "Forced roundtrip"}
COLORS = {"resident": "#4E79A7", "forced-cpu-state-roundtrip": "#E15759"}
TRAJECTORY_TOLERANCE = 1.0e-4


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
    ordered = sorted(values)
    if not ordered:
        fail("Cannot compute a percentile from no values.")
    index = int(math.ceil(fraction * len(ordered))) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def measured(rows, warmup, path):
    result = [row for row in rows if int(number(row, "frame", path)) >= warmup]
    if not result:
        fail("No measured rows in {0}".format(path))
    return result


def timing_metrics(run_dir, condition, manifest):
    metadata = load_json(run_dir / "run_metadata.json")
    benchmark = metadata.get("benchmark", {})
    if benchmark.get("no_render") or not benchmark.get("sync_gpu") or not benchmark.get("disable_vsync"):
        fail("Timing run is not rendered, GPU-synchronised, and vsync-disabled: {0}".format(run_dir))
    if metadata.get("solver_variant") != manifest["solver_variant"]:
        fail("Unexpected solver variant: {0}".format(run_dir))
    expected_forced = "1" if condition == "forced-cpu-state-roundtrip" else "0"
    if metadata.get("solver_controls", {}).get("force_cpu_state_roundtrip") != expected_forced:
        fail("Roundtrip metadata mismatch: {0}".format(run_dir))

    profile_path = run_dir / "frame_profile.csv"
    extended_path = run_dir / "frame_profile_extended.csv"
    experiment_path = run_dir / "frame_profile_experiment.csv"
    presentation_path = run_dir / "frame_presentation.csv"
    profile = measured(load_csv(profile_path), manifest["timing"]["warmup"], profile_path)
    extended = measured(load_csv(extended_path), manifest["timing"]["warmup"], extended_path)
    experiment = measured(load_csv(experiment_path), manifest["timing"]["warmup"], experiment_path)
    presentation = measured(load_csv(presentation_path), manifest["timing"]["warmup"], presentation_path)
    expected = manifest["timing"]["frames"]
    if any(len(rows) != expected for rows in (profile, extended, experiment, presentation)):
        fail("Incomplete timing frame count in {0}".format(run_dir))
    if any(row.get("frame_valid") != "1" or row.get("termination_reason") != "none" for row in extended):
        fail("Invalid timing frame in {0}".format(run_dir))
    if any(row.get("rendered") != "1" or row.get("gpu_sync_enabled") != "1" for row in presentation):
        fail("Unrendered presentation frame in {0}".format(run_dir))
    required = ("state_h2d_bytes", "state_d2h_bytes", "state_upload_calls", "state_readback_calls")
    for field in required:
        if field not in experiment[0]:
            fail("Missing state-traffic field '{0}' in {1}".format(field, experiment_path))

    def average_rows(rows, field, path):
        return mean([number(row, field, path) for row in rows])

    dispatches = mean([
        number(row, "gradient_dispatches", experiment_path)
        + number(row, "stats_dispatches", experiment_path)
        + number(row, "reduction_dispatches", experiment_path)
        + number(row, "xupdate_dispatches", experiment_path)
        + number(row, "descent_dispatches", experiment_path)
        for row in experiment
    ])
    return {
        "frame_wall_ms": average_rows(presentation, "frame_wall_ms", presentation_path),
        "total_ms": average_rows(profile, "total_ms", profile_path),
        "transfer_ms": average_rows(profile, "transfer_ms", profile_path),
        "readback_wait_ms": average_rows(profile, "cs_x_readback_wait_ms", profile_path),
        "readback_copy_ms": average_rows(profile, "cs_x_readback_copy_ms", profile_path),
        "state_h2d_bytes": average_rows(experiment, "state_h2d_bytes", experiment_path),
        "state_d2h_bytes": average_rows(experiment, "state_d2h_bytes", experiment_path),
        "state_upload_calls": average_rows(experiment, "state_upload_calls", experiment_path),
        "state_readback_calls": average_rows(experiment, "state_readback_calls", experiment_path),
        "host_readbacks": average_rows(experiment, "host_readbacks", experiment_path),
        "dispatches": dispatches,
        "p95_max_position": percentile([number(row, "max_position", profile_path) for row in profile]),
        "git_commit": metadata.get("git_commit", ""),
        "gpu_name": metadata.get("gpu_name", ""),
        "driver": metadata.get("nvidia_driver_version", ""),
    }


def quality_metrics(run_dir, condition, manifest):
    metadata = load_json(run_dir / "run_metadata.json")
    benchmark = metadata.get("benchmark", {})
    if benchmark.get("no_render") or not benchmark.get("sync_gpu"):
        fail("Quality run is not rendered and GPU-synchronised: {0}".format(run_dir))
    expected_forced = "1" if condition == "forced-cpu-state-roundtrip" else "0"
    if metadata.get("solver_controls", {}).get("force_cpu_state_roundtrip") != expected_forced:
        fail("Quality roundtrip metadata mismatch: {0}".format(run_dir))
    quality_path = run_dir / "quality_metrics.csv"
    extended_path = run_dir / "frame_profile_extended.csv"
    quality = measured(load_csv(quality_path), manifest["quality"]["warmup"], quality_path)
    extended = measured(load_csv(extended_path), manifest["quality"]["warmup"], extended_path)
    if len(quality) != manifest["quality"]["frames"] or len(extended) != manifest["quality"]["frames"]:
        fail("Incomplete quality run: {0}".format(run_dir))
    if any(row.get("finite") != "1" or row.get("exploded") != "0" for row in quality):
        fail("Quality gate input is invalid: {0}".format(run_dir))
    if not any(row.get("has_reference") == "1" for row in quality):
        fail("No reference-aligned quality checkpoint in {0}".format(run_dir))
    if any(row.get("frame_valid") != "1" for row in extended):
        fail("Invalid quality frame: {0}".format(run_dir))
    return percentile([number(row, "position_rel_l2", quality_path) for row in quality])


def aggregate_case(run_root, manifest, case):
    case_id = "{0}-d{1}".format(case["scene_id"], case["cloth_dimension"])
    rows = []
    for condition in CONDITIONS:
        repetitions = []
        for repetition in range(1, manifest["timing"]["repetitions"] + 1):
            run_dir = run_root / "timing" / case_id / condition / "rep{0:02d}".format(repetition)
            repetitions.append(timing_metrics(run_dir, condition, manifest))
        quality_dir = run_root / "quality" / case_id / condition
        quality_p95 = quality_metrics(quality_dir, condition, manifest)
        row = {"scene_id": case["scene_id"], "cloth_dimension": int(case["cloth_dimension"]), "condition": condition}
        for key in repetitions[0]:
            values = [entry[key] for entry in repetitions]
            if key in ("git_commit", "gpu_name", "driver"):
                if len(set(values)) != 1 or not values[0]:
                    fail("Inconsistent metadata for {0} {1}".format(case_id, condition))
                row[key] = values[0]
            else:
                row[key + "_mean"] = mean(values)
                row[key + "_std"] = sample_std(values)
        row["p95_position_rel_l2"] = quality_p95
        rows.append(row)
    resident, roundtrip = rows
    if abs(resident["dispatches_mean"] - roundtrip["dispatches_mean"]) > 1.0e-9:
        fail("Tracked dispatches differ in {0}".format(case_id))
    max_delta = abs(resident["p95_max_position_mean"] - roundtrip["p95_max_position_mean"])
    relative_delta = max_delta / max(abs(resident["p95_max_position_mean"]), 1.0e-8)
    if relative_delta > TRAJECTORY_TOLERANCE:
        fail("State trajectory proxy differs by {0:.3e} in {1}".format(relative_delta, case_id))
    if max(resident["p95_position_rel_l2"], roundtrip["p95_position_rel_l2"]) > 1.0e-3:
        fail("Position quality gate failed in {0}".format(case_id))
    return rows


def write_csv(path, rows):
    fields = [
        "scene_id", "cloth_dimension", "condition", "frame_wall_ms_mean", "frame_wall_ms_std",
        "total_ms_mean", "transfer_ms_mean", "readback_wait_ms_mean", "readback_copy_ms_mean",
        "state_h2d_bytes_mean", "state_d2h_bytes_mean", "state_upload_calls_mean", "state_readback_calls_mean",
        "host_readbacks_mean", "dispatches_mean", "p95_max_position_mean", "p95_position_rel_l2",
        "git_commit", "gpu_name", "driver",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, output_dir):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    cases = [("hanging", 256), ("hanging", 386), ("moving-sphere", 256), ("moving-sphere", 386)]
    index = {(row["scene_id"], row["cloth_dimension"], row["condition"]): row for row in rows}
    labels = ["Hang 256", "Hang 386", "Sphere 256", "Sphere 386"]
    x = list(range(len(cases)))
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.45))
    metrics = [
        ("frame_wall_ms_mean", "frame_wall_ms_std", "Rendered frame time (ms)"),
        ("state_h2d_bytes_mean", None, "State H2D traffic (MB/frame)"),
        ("state_d2h_bytes_mean", None, "State D2H traffic (MB/frame)"),
        ("state_readback_calls_mean", None, "State readbacks / frame"),
    ]
    width = 0.34
    for axis, (field, std_field, ylabel) in zip(axes.flat, metrics):
        for offset, condition in zip((-width / 2.0, width / 2.0), CONDITIONS):
            values = [float(index[(scene, dimension, condition)][field]) for scene, dimension in cases]
            if "bytes" in field:
                values = [value / (1024.0 * 1024.0) for value in values]
            errors = None
            if std_field:
                errors = [float(index[(scene, dimension, condition)][std_field]) for scene, dimension in cases]
            axis.bar([value + offset for value in x], values, width=width, color=COLORS[condition], label=LABELS[condition], yerr=errors, capsize=2)
        axis.set_xticks(x, labels, rotation=18, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.tight_layout(pad=0.5)
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / "persistent_residency.pdf" if suffix == "pdf" else output_dir / "persistent_residency.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(path, manifest, rows):
    index = {(row["scene_id"], row["cloth_dimension"], row["condition"]): row for row in rows}
    lines = [
        "# Persistent GPU-Resident State Study",
        "",
        "## Protocol",
        "",
        "- Commit `{0}`; rendered 1600x900 timing with {1} warm-up + {2} measured frames and three repetitions.".format(manifest["git_commit"], manifest["timing"]["warmup"], manifest["timing"]["frames"]),
        "- Conditions differ only by the per-frame position/velocity CPU roundtrip; timing excludes quality readback and capture.",
        "- Separate rendered quality runs use the archived CPU-NCG reference checkpoints from `{0}`.".format(manifest["calibration_commit"]),
        "",
        "## Results",
        "",
        "| Scene | Grid | Resident ms | Roundtrip ms | Ratio | H2D/D2H resident MB | H2D/D2H roundtrip MB | Dispatches | P95 position error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scene, dimension in (("hanging", 256), ("hanging", 386), ("moving-sphere", 256), ("moving-sphere", 386)):
        resident = index[(scene, dimension, "resident")]
        roundtrip = index[(scene, dimension, "forced-cpu-state-roundtrip")]
        ratio = roundtrip["frame_wall_ms_mean"] / resident["frame_wall_ms_mean"]
        lines.append("| {0} | {1}^2 | {2:.3f} +/- {3:.3f} | {4:.3f} +/- {5:.3f} | {6:.2f}x | {7:.3f}/{8:.3f} | {9:.3f}/{10:.3f} | {11:.0f} | {12:.3e} |".format(
            "Moving sphere" if scene == "moving-sphere" else "Hanging", dimension,
            resident["frame_wall_ms_mean"], resident["frame_wall_ms_std"], roundtrip["frame_wall_ms_mean"], roundtrip["frame_wall_ms_std"], ratio,
            resident["state_h2d_bytes_mean"] / 1048576.0, resident["state_d2h_bytes_mean"] / 1048576.0,
            roundtrip["state_h2d_bytes_mean"] / 1048576.0, roundtrip["state_d2h_bytes_mean"] / 1048576.0,
            resident["dispatches_mean"], max(resident["p95_position_rel_l2"], roundtrip["p95_position_rel_l2"])))
    lines.extend(["", "The result supports an implementation-level claim about avoiding the measured full-state host roundtrip in these configurations. It does not claim zero synchronization or a cross-API comparison."])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--report-path", required=True)
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    manifest = load_json(run_root / "manifest.json")
    if manifest.get("protocol_version") != "persistent-residency-paper-v1":
        fail("Unexpected residency-study protocol.")
    if manifest.get("measurement", {}).get("mode") != "rendered-end-to-end" or manifest.get("timing", {}).get("repetitions") != 3:
        fail("Residency study is not a complete rendered three-repetition protocol.")
    cases = load_csv(run_root / "planned_cases.csv")
    if len(cases) != 4:
        fail("Residency study must contain four scene-resolution cases.")
    rows = []
    for case in cases:
        rows.extend(aggregate_case(run_root, manifest, case))
    write_csv(run_root / "residency_summary.csv", rows)
    plot(rows, run_root)
    write_report(Path(args.report_path).resolve(), manifest, rows)
    print("Persistent residency summary: {0}".format(run_root / "residency_summary.csv"))


if __name__ == "__main__":
    main()
