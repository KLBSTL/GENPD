#!/usr/bin/env python3
"""Aggregate the rendered edge-scatter versus vertex-owned gather microstudy."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


VARIANTS = ("gpu-edge-scatter", "gpu-gather-no-fusion", "gpu-gather-fusion")
LABELS = {
    "gpu-edge-scatter": "Edge scatter",
    "gpu-gather-no-fusion": "Vertex gather",
    "gpu-gather-fusion": "Gather + fusion",
}
COLORS = {
    "gpu-edge-scatter": "#E15759",
    "gpu-gather-no-fusion": "#4E79A7",
    "gpu-gather-fusion": "#59A14F",
}
QUALITY_TOLERANCE = 1.0e-3


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
    selected = [row for row in rows if int(number(row, "frame", path)) >= warmup]
    if not selected:
        fail("No measured rows in {0}".format(path))
    return selected


def average(rows, field, path):
    return mean([number(row, field, path) for row in rows])


def timing_metrics(run_dir, variant, manifest):
    metadata = load_json(run_dir / "run_metadata.json")
    benchmark = metadata.get("benchmark", {})
    if benchmark.get("no_render") or not benchmark.get("sync_gpu") or not benchmark.get("disable_vsync"):
        fail("Timing run is not rendered, GPU-synchronised, and vsync-disabled: {0}".format(run_dir))
    if metadata.get("solver_variant") != variant:
        fail("Unexpected solver variant in {0}".format(run_dir))

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
    for field in ("constraint_spring_count", "constraint_attachment_count", "gradient_dispatches", "stats_dispatches"):
        if field not in experiment[0]:
            fail("Missing structural field '{0}' in {1}".format(field, experiment_path))

    spring_count = average(experiment, "constraint_spring_count", experiment_path)
    attachment_count = average(experiment, "constraint_attachment_count", experiment_path)
    if spring_count <= 0 or attachment_count <= 0:
        fail("Expected positive spring and attachment topology counts in {0}".format(run_dir))
    gradient_dispatches = average(experiment, "gradient_dispatches", experiment_path)
    stats_dispatches = average(experiment, "stats_dispatches", experiment_path)
    full_vector_passes = 2.0 if variant != "gpu-gather-fusion" else 1.0
    expected_gradient_dispatches = 2.0 if variant == "gpu-edge-scatter" else 1.0
    expected_stats_dispatches = 1.0 if variant != "gpu-gather-fusion" else 0.0
    if abs(gradient_dispatches - expected_gradient_dispatches) > 1.0e-9 or abs(stats_dispatches - expected_stats_dispatches) > 1.0e-9:
        fail("Unexpected gradient/stat dispatch structure in {0}".format(run_dir))
    atomic_writes = 6.0 * spring_count + 3.0 * attachment_count if variant == "gpu-edge-scatter" else 0.0

    return {
        "frame_wall_ms": average(presentation, "frame_wall_ms", presentation_path),
        "total_ms": average(profile, "total_ms", profile_path),
        "gradient_path_gpu_ms": average(profile, "cs_gradient_gpu_ms", profile_path) + average(profile, "cs_gradstats_ms", profile_path),
        "constraint_spring_count": spring_count,
        "constraint_attachment_count": attachment_count,
        "gradient_dispatches": gradient_dispatches,
        "stats_dispatches": stats_dispatches,
        "gradient_related_barriers": gradient_dispatches + stats_dispatches,
        "full_vector_gradient_passes": full_vector_passes,
        "theoretical_scalar_atomic_writes": atomic_writes,
        "git_commit": metadata.get("git_commit", ""),
        "gpu_name": metadata.get("gpu_name", ""),
        "driver": metadata.get("nvidia_driver_version", ""),
    }


def quality_p95(run_dir, variant, manifest):
    metadata = load_json(run_dir / "run_metadata.json")
    benchmark = metadata.get("benchmark", {})
    if benchmark.get("no_render") or not benchmark.get("sync_gpu") or metadata.get("solver_variant") != variant:
        fail("Quality run is not rendered, synchronised, and variant-matched: {0}".format(run_dir))
    quality_path = run_dir / "quality_metrics.csv"
    extended_path = run_dir / "frame_profile_extended.csv"
    quality = measured(load_csv(quality_path), manifest["quality"]["warmup"], quality_path)
    extended = measured(load_csv(extended_path), manifest["quality"]["warmup"], extended_path)
    if len(quality) != manifest["quality"]["frames"] or len(extended) != manifest["quality"]["frames"]:
        fail("Incomplete quality run: {0}".format(run_dir))
    if any(row.get("finite") != "1" or row.get("exploded") != "0" for row in quality):
        fail("Invalid quality input in {0}".format(run_dir))
    if not any(row.get("has_reference") == "1" for row in quality):
        fail("No reference-aligned quality checkpoint in {0}".format(run_dir))
    if any(row.get("frame_valid") != "1" or row.get("termination_reason") != "none" for row in extended):
        fail("Invalid quality frame in {0}".format(run_dir))
    value = percentile([number(row, "position_rel_l2", quality_path) for row in quality])
    if value > manifest["quality"]["position_gate_p95"]:
        fail("Quality gate failed in {0}: {1:.3e}".format(run_dir, value))
    return value


def aggregate(run_root, manifest, cases):
    rows = []
    for case in cases:
        scene = case["scene_id"]
        dimension = int(case["cloth_dimension"])
        case_id = "{0}-d{1}".format(scene, dimension)
        for variant in VARIANTS:
            repetitions = []
            for repetition in range(1, manifest["timing"]["repetitions"] + 1):
                repetitions.append(timing_metrics(run_root / "timing" / case_id / variant / "rep{0:02d}".format(repetition), variant, manifest))
            row = {"scene_id": scene, "cloth_dimension": dimension, "solver_variant": variant}
            for key in repetitions[0]:
                values = [entry[key] for entry in repetitions]
                if key in ("git_commit", "gpu_name", "driver"):
                    if not values[0] or len(set(values)) != 1:
                        fail("Inconsistent metadata for {0} {1}".format(case_id, variant))
                    row[key] = values[0]
                else:
                    row[key + "_mean"] = mean(values)
                    row[key + "_std"] = sample_std(values)
            row["p95_position_rel_l2"] = quality_p95(run_root / "quality" / case_id / variant, variant, manifest)
            rows.append(row)
    return rows


def write_csv(path, rows):
    fields = [
        "scene_id", "cloth_dimension", "solver_variant", "frame_wall_ms_mean", "frame_wall_ms_std",
        "total_ms_mean", "total_ms_std", "gradient_path_gpu_ms_mean", "gradient_path_gpu_ms_std",
        "constraint_spring_count_mean", "constraint_attachment_count_mean", "gradient_dispatches_mean",
        "stats_dispatches_mean", "gradient_related_barriers_mean", "full_vector_gradient_passes_mean",
        "theoretical_scalar_atomic_writes_mean", "p95_position_rel_l2", "git_commit", "gpu_name", "driver",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, output_dir):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    cases = [("hanging", 256), ("hanging", 386), ("moving-sphere", 256), ("moving-sphere", 386)]
    labels = ["Hang 256", "Hang 386", "Sphere 256", "Sphere 386"]
    index = {(row["scene_id"], row["cloth_dimension"], row["solver_variant"]): row for row in rows}
    x = list(range(len(cases)))
    width = 0.23
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.45))
    metrics = [
        ("frame_wall_ms_mean", "frame_wall_ms_std", "Rendered frame time (ms)", 1.0),
        ("gradient_path_gpu_ms_mean", "gradient_path_gpu_ms_std", "Gradient + stats GPU time (ms)", 1.0),
        ("gradient_related_barriers_mean", None, "Gradient-related barriers / iteration", 1.0),
        ("theoretical_scalar_atomic_writes_mean", None, "Theoretical scalar atomics / iteration (M)", 1.0e-6),
    ]
    for axis, (field, std_field, ylabel, scale) in zip(axes.flat, metrics):
        for index_variant, variant in enumerate(VARIANTS):
            values = [float(index[(scene, dimension, variant)][field]) * scale for scene, dimension in cases]
            errors = None
            if std_field:
                errors = [float(index[(scene, dimension, variant)][std_field]) * scale for scene, dimension in cases]
            positions = [value + (index_variant - 1) * width for value in x]
            axis.bar(positions, values, width=width, color=COLORS[variant], label=LABELS[variant], yerr=errors, capsize=2)
        axis.set_xticks(x, labels, rotation=18, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.tight_layout(pad=0.5)
    fig.savefig(output_dir / "vertex_owned_microstudy.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "vertex_owned_microstudy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(path, manifest, rows):
    index = {(row["scene_id"], row["cloth_dimension"], row["solver_variant"]): row for row in rows}
    lines = [
        "# Vertex-Owned Gather Microstudy",
        "",
        "## Protocol",
        "",
        "- Commit `{0}`; rendered 1600x900 timing with {1} warm-up + {2} measured frames and three repetitions.".format(manifest["git_commit"], manifest["timing"]["warmup"], manifest["timing"]["frames"]),
        "- The three variants share the same scene, topology, iteration budget, rendering, GPU synchronisation, and CPU-NCG reference checkpoints. Quality readback is a separate rendered run.",
        "- The scalar atomic count is structural: `6 * spring constraints + 3 * attachment constraints` for edge scatter, and zero for gather. It is not a hardware transaction counter.",
        "",
        "## Results",
        "",
        "| Scene | Grid | Variant | Frame ms | Gradient+stats ms | Dispatches | Barriers | Atomics (M) | P95 position error |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scene, dimension in (("hanging", 256), ("hanging", 386), ("moving-sphere", 256), ("moving-sphere", 386)):
        for variant in VARIANTS:
            row = index[(scene, dimension, variant)]
            lines.append("| {0} | {1} | {2} | {3:.3f} +/- {4:.3f} | {5:.3f} | {6:.1f} | {7:.1f} | {8:.3f} | {9:.2e} |".format(
                scene, dimension, LABELS[variant], row["frame_wall_ms_mean"], row["frame_wall_ms_std"], row["gradient_path_gpu_ms_mean"],
                row["gradient_dispatches_mean"] + row["stats_dispatches_mean"], row["gradient_related_barriers_mean"],
                row["theoretical_scalar_atomic_writes_mean"] * 1.0e-6, row["p95_position_rel_l2"]
            ))
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "This isolates the gradient dataflow. It demonstrates that gather removes the edge-scatter atomic accumulation and fusion removes the standalone stats pass; it does not by itself attribute every frame-time difference to atomics because rendering, reductions, and line search remain in the end-to-end timing scope.",
        "",
        "Figure: `results/{0}/vertex_owned_microstudy.pdf`.".format(manifest["label"]),
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--report-path", required=True)
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    manifest = load_json(run_root / "manifest.json")
    if manifest.get("protocol_version") != "vertex-owned-microstudy-v1":
        fail("Unexpected protocol version.")
    if manifest.get("measurement", {}).get("mode") != "rendered-end-to-end":
        fail("Vertex-owned study must be rendered end-to-end.")
    if tuple(manifest.get("variants", [])) != VARIANTS or manifest.get("timing", {}).get("repetitions") != 3:
        fail("Incomplete vertex-owned study manifest.")
    cases = load_csv(run_root / "planned_cases.csv")
    if len(cases) != 4:
        fail("Expected four vertex-owned cases.")
    rows = aggregate(run_root, manifest, cases)
    write_csv(run_root / "microstudy_summary.csv", rows)
    plot(rows, run_root)
    write_report(Path(args.report_path).resolve(), manifest, rows)
    print("Vertex-owned microstudy analysis complete: {0}".format(run_root))


if __name__ == "__main__":
    main()
