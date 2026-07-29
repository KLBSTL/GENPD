#!/usr/bin/env python3
"""Check whether CPU NCG 100 iterations is close to 200/400-iteration references."""

import argparse
import csv
import json
import math
import struct
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ITERATIONS = (100, 200, 400)


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


def percentile(values, fraction=0.95):
    if not values:
        fail("Cannot calculate a percentile from no values.")
    ordered = sorted(values)
    index = int(math.ceil(fraction * len(ordered))) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def mean(values):
    return sum(values) / float(len(values))


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
    return frame, values[:count], values[count:]


def relative_l2(values, reference):
    if len(values) != len(reference):
        fail("Checkpoint vector sizes differ.")
    difference_sq = sum((left - right) ** 2 for left, right in zip(values, reference))
    reference_sq = sum(value * value for value in reference)
    return math.sqrt(difference_sq) / max(math.sqrt(reference_sq), 1.0e-12)


def run_directory(run_root, scene_id, dimension, iterations):
    return run_root / "references" / "{0}-d{1}".format(scene_id, dimension) / "i{0:03d}".format(iterations)


def validate_reference_run(directory, scene_id, iteration, manifest):
    metadata = load_json(directory / "run_metadata.json")
    benchmark = metadata.get("benchmark", {})
    if benchmark.get("no_render") or benchmark.get("hide_window") or not benchmark.get("sync_gpu") or not benchmark.get("disable_vsync"):
        fail("Reference run is not rendered, synchronized, and vsync-disabled: {0}".format(directory))
    if metadata.get("solver_variant") != "cpu-ncg" or int(metadata.get("quality", {}).get("iterations_per_frame", 0)) != iteration:
        fail("CPU reference metadata mismatch: {0}".format(directory))
    expected_scene = manifest["scenes"][scene_id]
    actual_scene = str(metadata.get("experiment_overrides", {}).get("scene", ""))
    if not actual_scene.replace("/", "\\").endswith(expected_scene.replace("/", "\\")):
        fail("Reference scene mismatch: {0}".format(directory))

    extended_path = directory / "frame_profile_extended.csv"
    presentation_path = directory / "frame_presentation.csv"
    extended = [row for row in load_csv(extended_path) if int(number(row, "frame", extended_path)) >= manifest["timing"]["warmup"]]
    presentation = [row for row in load_csv(presentation_path) if int(number(row, "frame", presentation_path)) >= manifest["timing"]["warmup"]]
    if len(extended) != manifest["timing"]["frames"] or len(presentation) != manifest["timing"]["frames"]:
        fail("Reference measured-frame count mismatch: {0}".format(directory))
    if any(row.get("frame_valid") != "1" or row.get("termination_reason") != "none" for row in extended):
        fail("Invalid CPU reference frame: {0}".format(directory))
    if any(row.get("rendered") != "1" or row.get("gpu_sync_enabled") != "1" for row in presentation):
        fail("Unrendered CPU reference frame: {0}".format(directory))

    checkpoints = {}
    for path in sorted((directory / "reference_checkpoints").glob("reference_state_*.bin")):
        frame, positions, velocities = read_checkpoint(path)
        checkpoints[frame] = (positions, velocities)
    measured_frames = {int(number(row, "frame", extended_path)) for row in extended}
    if not measured_frames.issubset(checkpoints):
        fail("CPU reference is missing measured checkpoints: {0}".format(directory))
    return checkpoints, metadata


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, output_dir):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    scenes = ["hanging", "moving-sphere"]
    labels = ["Hanging", "Moving sphere"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.25))
    for axis, field, title in zip(axes, ("p95_position_rel_l2", "p95_velocity_rel_l2"), ("Position error to CPU 400", "Velocity error to CPU 400")):
        for iteration, color in ((100, "#4E79A7"), (200, "#F28E2B")):
            values = [next(float(row[field]) for row in rows if row["scene_id"] == scene and int(row["candidate_iterations"]) == iteration) for scene in scenes]
            offset = -0.18 if iteration == 100 else 0.18
            axis.bar([index + offset for index in range(len(scenes))], values, width=0.34, color=color, label="{0} vs 400".format(iteration))
        # Symlog retains exact-zero convergence results without inventing a floor.
        axis.set_yscale("symlog", linthresh=1.0e-12)
        axis.set_xticks(range(len(scenes)), labels)
        axis.set_ylabel("Relative L2")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].axhline(1.0e-3, color="#59A14F", linestyle="--", linewidth=1.0, label="Position gate")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(pad=0.5)
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / "cpu_reference_sanity.{0}".format(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    manifest = load_json(run_root / "manifest.json")
    if manifest.get("protocol_version") != "cpu-reference-sanity-v1" or tuple(manifest.get("iterations", ())) != ITERATIONS:
        fail("Unexpected CPU reference sanity protocol.")
    if manifest.get("measurement", {}).get("mode") != "rendered-checkpoint-export":
        fail("CPU reference sanity is not a rendered checkpoint protocol.")

    rows = []
    summary = {}
    for scene_id in manifest["scenes"]:
        trajectories = {}
        metadata = {}
        for iteration in ITERATIONS:
            trajectories[iteration], metadata[iteration] = validate_reference_run(
                run_directory(run_root, scene_id, manifest["cloth_dimension"], iteration), scene_id, iteration, manifest)
        common = set(trajectories[400])
        for iteration in (100, 200):
            common &= set(trajectories[iteration])
        common = sorted(frame for frame in common if frame >= manifest["timing"]["warmup"])
        if len(common) != manifest["timing"]["frames"]:
            fail("Reference trajectories do not share every measured frame for {0}.".format(scene_id))

        for iteration in (100, 200):
            position_errors = []
            velocity_errors = []
            for frame in common:
                candidate_position, candidate_velocity = trajectories[iteration][frame]
                reference_position, reference_velocity = trajectories[400][frame]
                position_errors.append(relative_l2(candidate_position, reference_position))
                velocity_errors.append(relative_l2(candidate_velocity, reference_velocity))
            row = {
                "scene_id": scene_id,
                "cloth_dimension": manifest["cloth_dimension"],
                "candidate_iterations": iteration,
                "reference_iterations": 400,
                "matched_measured_frames": len(common),
                "mean_position_rel_l2": mean(position_errors),
                "p95_position_rel_l2": percentile(position_errors),
                "max_position_rel_l2": max(position_errors),
                "mean_velocity_rel_l2": mean(velocity_errors),
                "p95_velocity_rel_l2": percentile(velocity_errors),
                "max_velocity_rel_l2": max(velocity_errors),
                "position_gate": int(percentile(position_errors) <= manifest["decision"]["p95_position_rel_l2"]),
                "git_commit": metadata[iteration].get("git_commit", ""),
                "gpu_name": metadata[iteration].get("gpu_name", ""),
                "driver": metadata[iteration].get("nvidia_driver_version", ""),
            }
            rows.append(row)
            summary[(scene_id, iteration)] = row

    required = [summary[(scene, 100)] for scene in manifest["scenes"]]
    if any(row["position_gate"] != 1 for row in required):
        fail("CPU NCG 100-iteration reference does not meet the pre-registered position convergence gate.")
    write_csv(run_root / "cpu_reference_sanity_summary.csv", rows)
    plot(rows, run_root)

    report = [
        "# CPU NCG Reference Sanity Check",
        "",
        "## Protocol",
        "",
        "- Rendered CPU NCG references at 100, 200, and 400 iterations per frame.",
        "- Two {0} x {0} cases, {1} warm-up plus {2} measured frames, with checkpoints on every frame.".format(manifest["cloth_dimension"], manifest["timing"]["warmup"], manifest["timing"]["frames"]),
        "- The 400-iteration trajectory is the comparison target; 100 versus 400 requires P95 position relative L2 <= {0:.1e}.".format(manifest["decision"]["p95_position_rel_l2"]),
        "",
        "## Result",
        "",
        "| Scene | 100 vs 400 P95 position | 100 vs 400 P95 velocity | 200 vs 400 P95 position | 200 vs 400 P95 velocity | 100 gate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scene_id in manifest["scenes"]:
        hundred = summary[(scene_id, 100)]
        two_hundred = summary[(scene_id, 200)]
        report.append("| {0} | {1:.3e} | {2:.3e} | {3:.3e} | {4:.3e} | {5} |".format(
            scene_id, hundred["p95_position_rel_l2"], hundred["p95_velocity_rel_l2"],
            two_hundred["p95_position_rel_l2"], two_hundred["p95_velocity_rel_l2"], "pass" if hundred["position_gate"] else "fail"))
    report.extend([
        "",
        "The decision addresses CPU-reference iteration adequacy for the measured scenarios only. It does not claim a global optimum or replace independent physical-quality validation.",
    ])
    (run_root / "cpu_reference_sanity_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("CPU reference sanity summary: {0}".format(run_root / "cpu_reference_sanity_summary.csv"))


if __name__ == "__main__":
    main()
