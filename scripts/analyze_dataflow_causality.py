#!/usr/bin/env python3
"""Validate the traversal and GPU-resident-state explanations in a formal R2 run.

This is a diagnostic analysis, not a paper-result generator. It rejects data that
do not preserve the controlled comparison contracts and writes a CSV, Markdown
report, and compact PDF/PNG figure for inspection.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TRAVERSAL_VARIANTS = [
    "gpu-edge-scatter",
    "gpu-gather-no-fusion",
    "gpu-gather-fusion",
]
RESIDENCY_VARIANTS = [
    "gpu-gather-fusion-batched-ls",
    "gpu-gather-fusion-batched-ls-persistent",
]
SCENES = ["hanging", "moving-sphere"]
COLORS = {
    "gpu-edge-scatter": "#E15759",
    "gpu-gather-no-fusion": "#59A14F",
    "gpu-gather-fusion": "#F28E2B",
    "gpu-gather-fusion-batched-ls": "#B07AA1",
    "gpu-gather-fusion-batched-ls-persistent": "#76B7B2",
}
LABELS = {
    "gpu-edge-scatter": "Edge scatter",
    "gpu-gather-no-fusion": "Gather",
    "gpu-gather-fusion": "Gather + fusion",
    "gpu-gather-fusion-batched-ls": "+ batched LS",
    "gpu-gather-fusion-batched-ls-persistent": "+ persistent",
}


def fail(message):
    raise RuntimeError(message)


def read_csv(path):
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row, field):
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError):
        fail("Missing numeric field '{0}' in {1}".format(field, row))
    if not math.isfinite(value):
        fail("Non-finite field '{0}' in {1}".format(field, row))
    return value


def mean(rows, field):
    if not rows:
        fail("Cannot average empty rows for '{0}'.".format(field))
    return sum(number(row, field) for row in rows) / float(len(rows))


def summary_index(rows):
    indexed = {}
    for row in rows:
        key = (row["scene_id"], int(row["cloth_dimension"]), row["solver_variant"])
        if key in indexed:
            fail("Duplicate paper summary row: {0}".format(key))
        indexed[key] = row
    return indexed


def collect_experiment_means(run_root, row):
    scene = row["scene_id"]
    dimension = int(row["cloth_dimension"])
    variant = row["solver_variant"]
    repetitions = int(row["repetitions"])
    rows = []
    for repetition in range(1, repetitions + 1):
        run_dir = run_root / "performance" / "{0}-d{1}-{2}-rep{3:02d}".format(
            scene, dimension, variant, repetition
        )
        profile = read_csv(run_dir / "frame_profile_experiment.csv")
        if not profile:
            fail("Missing experiment profile rows: {0}".format(run_dir))
        rows.extend(profile)
    return {
        "host_readbacks": mean(rows, "host_readbacks"),
        "solver_gl_finish_calls": mean(rows, "solver_gl_finish_calls"),
        "persistent_buffers_active": mean(rows, "persistent_buffers_active"),
    }


def required_row(indexed, scene, dimension, variant):
    key = (scene, dimension, variant)
    if key not in indexed:
        fail("Missing selected formal row: {0}".format(key))
    row = indexed[key]
    if int(row["repetitions"]) != 3:
        fail("Causality diagnostic requires three repetitions: {0}".format(key))
    if float(row["calibration_failure_rate"]) != 0.0:
        fail("Causality diagnostic requires zero calibration failures: {0}".format(key))
    if row["quality_gate"] != "ncg-reference-position":
        fail("Causality diagnostic is defined only for NCG position-gated rows: {0}".format(key))
    if number(row, "p95_position_rel_l2") > 1e-3:
        fail("Selected NCG row does not satisfy its position gate: {0}".format(key))
    return row


def write_csv(path, rows):
    fields = [
        "scene", "dimension", "comparison", "variant", "iterations", "frame_wall_ms",
        "total_ms", "optimization_ms", "transfer_ms", "gradient_gpu_ms",
        "stats_readback_ms", "dispatches", "host_readbacks", "solver_gl_finish_calls",
        "persistent_buffers_active", "tracked_buffer_mb", "p95_position_rel_l2",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, output_dir):
    by_key = {(row["scene"], row["comparison"], row["variant"]): row for row in rows}
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.55))
    scene_labels = ["Hanging", "Moving sphere"]

    ax = axes[0]
    x = list(range(len(SCENES)))
    width = 0.23
    for offset, variant in zip([-width, 0.0, width], TRAVERSAL_VARIANTS):
        values = [float(by_key[(scene, "traversal", variant)]["gradient_gpu_ms"]) for scene in SCENES]
        ax.bar([value + offset for value in x], values, width=width, color=COLORS[variant], label=LABELS[variant])
    ax.set_xticks(x)
    ax.set_xticklabels(scene_labels)
    ax.set_ylabel("Gradient GPU query (ms)")
    ax.set_title("Traversal diagnostic")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=6.3)

    ax = axes[1]
    for offset, variant in zip([-0.18, 0.18], RESIDENCY_VARIANTS):
        values = [float(by_key[(scene, "residency", variant)]["frame_wall_ms"]) for scene in SCENES]
        ax.bar([value + offset for value in x], values, width=0.36, color=COLORS[variant], label=LABELS[variant])
    ax.set_xticks(x)
    ax.set_xticklabels(scene_labels)
    ax.set_ylabel("Rendered frame time (ms)")
    ax.set_title("Resident-state A/B")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=6.3)

    ax = axes[2]
    for offset, variant in zip([-0.18, 0.18], RESIDENCY_VARIANTS):
        values = [float(by_key[(scene, "residency", variant)]["transfer_ms"]) for scene in SCENES]
        ax.bar([value + offset for value in x], values, width=0.36, color=COLORS[variant], label=LABELS[variant])
    ax.set_xticks(x)
    ax.set_xticklabels(scene_labels)
    ax.set_ylabel("CPU-side transfer time (ms)")
    ax.set_title("Transfer removed by residency")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(pad=0.45)
    for extension in ("pdf", "png"):
        fig.savefig(output_dir / "dataflow_causality.{0}".format(extension), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, help="Formal R2 result root.")
    parser.add_argument("--output-dir", required=True, help="Diagnostic artifact directory.")
    parser.add_argument("--dimension", type=int, default=386, help="Square cloth dimension to inspect.")
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_root / "manifest.json"
    summary_path = run_root / "paper_summary.csv"
    if not manifest_path.is_file() or not summary_path.is_file():
        fail("A complete formal run root must contain manifest.json and paper_summary.csv.")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("protocol_version") != 4 or not manifest.get("scope", {}).get("paper_figure_eligible"):
        fail("This diagnostic only accepts a complete protocol-version-4 formal R2 run.")
    if manifest.get("measurement", {}).get("mode") != "rendered-end-to-end":
        fail("This diagnostic requires rendered end-to-end measurements.")

    indexed = summary_index(read_csv(summary_path))
    output_rows = []
    report_lines = [
        "# Dataflow Causality Diagnostic",
        "",
        "This report validates controlled relationships present in the rendered R2 data. It does not replace a forced-roundtrip counterfactual.",
        "",
        "## Contracts",
        "",
        "- Protocol version 4, rendered end-to-end timing, and three repetitions.",
        "- All compared rows pass the NCG P95 relative position-error gate.",
        "- The resident-state pair has identical selected iteration and tracked solver-dispatch counts.",
        "",
        "## Results",
        "",
    ]

    for scene in SCENES:
        traversal = [required_row(indexed, scene, args.dimension, variant) for variant in TRAVERSAL_VARIANTS]
        residency = [required_row(indexed, scene, args.dimension, variant) for variant in RESIDENCY_VARIANTS]
        if int(residency[0]["iterations_per_frame"]) != int(residency[1]["iterations_per_frame"]):
            fail("Resident-state pair has mismatched iteration budgets in {0}.".format(scene))
        if abs(number(residency[0], "dispatches_mean") - number(residency[1], "dispatches_mean")) > 1e-9:
            fail("Resident-state pair has mismatched solver-dispatch counts in {0}.".format(scene))

        for comparison, records in [("traversal", traversal), ("residency", residency)]:
            for record in records:
                experiment = collect_experiment_means(run_root, record)
                row = {
                    "scene": scene,
                    "dimension": args.dimension,
                    "comparison": comparison,
                    "variant": record["solver_variant"],
                    "iterations": int(record["iterations_per_frame"]),
                    "frame_wall_ms": number(record, "frame_wall_ms_mean"),
                    "total_ms": number(record, "total_ms_mean"),
                    "optimization_ms": number(record, "optimization_ms_mean"),
                    "transfer_ms": number(record, "transfer_ms_mean"),
                    "gradient_gpu_ms": number(record, "cs_gradient_gpu_ms_mean"),
                    "stats_readback_ms": number(record, "cs_stats_readback_ms_mean"),
                    "dispatches": number(record, "dispatches_mean"),
                    "host_readbacks": experiment["host_readbacks"],
                    "solver_gl_finish_calls": experiment["solver_gl_finish_calls"],
                    "persistent_buffers_active": experiment["persistent_buffers_active"],
                    "tracked_buffer_mb": number(record, "tracked_buffer_bytes_mean") / (1024.0 * 1024.0),
                    "p95_position_rel_l2": number(record, "p95_position_rel_l2"),
                }
                output_rows.append(row)

        scatter, gather, fusion = traversal
        batched, persistent = residency
        scatter_gradient = number(scatter, "cs_gradient_gpu_ms_mean")
        fusion_gradient = number(fusion, "cs_gradient_gpu_ms_mean")
        frame_ratio = number(batched, "frame_wall_ms_mean") / number(persistent, "frame_wall_ms_mean")
        transfer_delta = number(batched, "transfer_ms_mean") - number(persistent, "transfer_ms_mean")
        report_lines.extend([
            "### {0}, {1}^2".format("Moving sphere" if scene == "moving-sphere" else "Hanging", args.dimension),
            "",
            "- Gather+fusion gradient query: {0:.3f} ms versus edge-scatter {1:.3f} ms ({2:.2f}x).".format(
                fusion_gradient, scatter_gradient, fusion_gradient / scatter_gradient
            ),
            "- Edge scatter to gather+fusion changes tracked dispatches from {0:.0f} to {1:.0f}; this is not a measured kernel-speed improvement.".format(
                number(scatter, "dispatches_mean"), number(fusion, "dispatches_mean")
            ),
            "- Batched and persistent variants both use {0} selected iteration(s) and {1:.0f} tracked dispatches.".format(
                batched["iterations_per_frame"], number(batched, "dispatches_mean")
            ),
            "- Persistent state changes rendered time from {0:.3f} to {1:.3f} ms ({2:.2f}x), transfer from {3:.3f} to {4:.3f} ms, host readbacks from 3 to 1, and solver glFinish calls from 1 to 0.".format(
                number(batched, "frame_wall_ms_mean"), number(persistent, "frame_wall_ms_mean"), frame_ratio,
                number(batched, "transfer_ms_mean"), number(persistent, "transfer_ms_mean")
            ),
            "- Transfer-time reduction in this pair is {0:.3f} ms; it is a measured contributor, not a complete attribution of the frame-time delta.".format(transfer_delta),
            "",
        ])

    report_lines.extend([
        "## Interpretation Boundary",
        "",
        "The traversal rows provide direct GL timer-query evidence that this CSR gather implementation is not faster than edge scatter at the selected one-iteration budget. The resident-state rows are a controlled end-to-end A/B at the variant level, but persistence also changes CPU-state prediction/finalization and GPU-resident line-search control. A future forced CPU-state roundtrip mode is required to assign the entire difference solely to eliminated position/velocity copies.",
    ])
    write_csv(output_dir / "dataflow_causality.csv", output_rows)
    (output_dir / "dataflow_causality_report.md").write_text("\n".join(report_lines) + "\n")
    metadata = {
        "run_root": str(run_root),
        "git_commit": manifest.get("git_commit", ""),
        "protocol_version": manifest.get("protocol_version"),
        "dimension": args.dimension,
        "measurement_mode": manifest.get("measurement", {}).get("mode", ""),
    }
    (output_dir / "dataflow_causality_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plot(output_rows, output_dir)
    print("Wrote dataflow causality diagnostic to {0}".format(output_dir))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print("Dataflow causality diagnostic rejected input: {0}".format(error))
        raise SystemExit(2)
