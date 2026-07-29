#!/usr/bin/env python3
"""Extract convergence behavior from an existing rendered CPU-reference ladder."""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


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


def mean(values):
    return sum(values) / float(len(values))


def percentile(values, fraction=0.95):
    ordered = sorted(values)
    index = int(math.ceil(fraction * len(ordered))) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def run_directory(run_root, scene_id, dimension, iterations):
    return run_root / "references" / "{0}-d{1}".format(scene_id, dimension) / "i{0:03d}".format(iterations)


def stop_reason(row, cap):
    if row.get("frame_valid") != "1" or row.get("termination_reason") != "none":
        return row.get("termination_reason") or "invalid"
    actual = int(number(row, "iterations", "frame_profile_extended.csv"))
    if row.get("converged") == "1" and actual < cap:
        return "converged-before-cap"
    if actual >= cap:
        return "iteration-cap"
    return "stopped-without-convergence"


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--report-path", default="")
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    manifest = load_json(run_root / "manifest.json")
    if manifest.get("protocol_version") != "cpu-reference-sanity-v1" or tuple(manifest.get("iterations", ())) != ITERATIONS:
        fail("Unexpected CPU-reference ladder protocol.")

    rows = []
    for scene_id in manifest["scenes"]:
        for cap in ITERATIONS:
            directory = run_directory(run_root, scene_id, manifest["cloth_dimension"], cap)
            metadata = load_json(directory / "run_metadata.json")
            if metadata.get("solver_variant") != "cpu-ncg" or int(metadata.get("quality", {}).get("iterations_per_frame", 0)) != cap:
                fail("CPU-reference metadata mismatch: {0}".format(directory))
            extended_path = directory / "frame_profile_extended.csv"
            extended = [row for row in load_csv(extended_path) if int(number(row, "frame", extended_path)) >= manifest["timing"]["warmup"]]
            if len(extended) != manifest["timing"]["frames"]:
                fail("CPU-reference measured-frame count mismatch: {0}".format(directory))
            if any(row.get("frame_valid") != "1" or row.get("termination_reason") != "none" for row in extended):
                fail("Invalid CPU-reference frame: {0}".format(directory))

            actual_iterations = [int(number(row, "iterations", extended_path)) for row in extended]
            gradient_norms = [number(row, "gradient_norm", extended_path) for row in extended]
            objectives = [number(row, "objective_energy", extended_path) for row in extended]
            reasons = Counter(stop_reason(row, cap) for row in extended)
            if any(actual < 1 or actual > cap for actual in actual_iterations):
                fail("CPU-reference iteration count lies outside the configured cap: {0}".format(directory))
            if reasons.get("stopped-without-convergence", 0) or reasons.get("iteration-cap", 0):
                fail("CPU-reference did not converge before the configured cap: {0}".format(directory))
            row = {
                "scene_id": scene_id,
                "cloth_dimension": manifest["cloth_dimension"],
                "iteration_cap": cap,
                "measured_frames": len(extended),
                "actual_iterations_mean": mean(actual_iterations),
                "actual_iterations_min": min(actual_iterations),
                "actual_iterations_max": max(actual_iterations),
                "converged_frames": sum(1 for row in extended if row.get("converged") == "1"),
                "converged_before_cap_frames": reasons.get("converged-before-cap", 0),
                "iteration_cap_frames": reasons.get("iteration-cap", 0),
                "termination_reason_values": ";".join(sorted({row.get("termination_reason", "") for row in extended})),
                "gradient_norm_mean": mean(gradient_norms),
                "gradient_norm_p95": percentile(gradient_norms),
                "gradient_norm_final": gradient_norms[-1],
                "objective_energy_mean": mean(objectives),
                "objective_energy_p95": percentile(objectives),
                "objective_energy_final": objectives[-1],
                "git_commit": metadata.get("git_commit", ""),
                "result_dir": str(directory),
            }
            rows.append(row)

    commits = {row["git_commit"] for row in rows}
    if len(commits) != 1 or not next(iter(commits)):
        fail("CPU-reference ladder spans inconsistent commits.")
    write_csv(run_root / "cpu_reference_convergence_summary.csv", rows)
    report = [
        "# CPU Reference Convergence Audit",
        "",
        "This audit explains the stored-state equality of the 100/200/400 iteration reference ladder. It reuses the rendered checkpoint runs and adds no timing claims.",
        "",
        "| Scene | Cap | Actual iter mean [min, max] | Frames converged before cap | P95 gradient norm | Final gradient norm | Final objective |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        report.append("| {0} | {1} | {2:.2f} [{3}, {4}] | {5}/{6} | {7:.3e} | {8:.3e} | {9:.3e} |".format(
            row["scene_id"], row["iteration_cap"], row["actual_iterations_mean"], row["actual_iterations_min"], row["actual_iterations_max"],
            row["converged_before_cap_frames"], row["measured_frames"], row["gradient_norm_p95"], row["gradient_norm_final"], row["objective_energy_final"]))
    report += [
        "",
        "All profiled termination-reason values are `none`: this field identifies invalid/exploded termination, not successful convergence. The derived stop classification is `converged-before-cap` whenever `converged=1` and the observed iteration count is below the configured cap.",
        "",
        "The audit only supports the existing two-scene, 256x256 CPU-reference protocol. It does not establish a global CPU convergence rate or an XPBD/NCG quality equivalence claim.",
    ]
    report_path = Path(args.report_path).resolve() if args.report_path else run_root / "cpu_reference_convergence_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("CPU reference convergence summary: {0}".format(run_root / "cpu_reference_convergence_summary.csv"))


if __name__ == "__main__":
    main()
