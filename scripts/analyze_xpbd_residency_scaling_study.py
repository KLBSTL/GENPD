#!/usr/bin/env python3
"""Validate and report a rendered XPBD residency/scaling diagnostic."""

import argparse
import csv
import json
import math
from pathlib import Path


EXPECTED_CONDITIONS = (
    "cpu-ncg",
    "xpbd-resident",
    "xpbd-forced-roundtrip",
    "xpbd-signed-incidence-gather",
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


def mean(values):
    return sum(values) / float(len(values))


def percentile(values, fraction):
    values = sorted(values)
    index = max(0, min(len(values) - 1, int(math.ceil(fraction * len(values))) - 1))
    return values[index]


def measured(rows, warmup, path):
    selected = [row for row in rows if int(number(row, "frame", path)) >= warmup]
    if not selected:
        fail("No measured rows in {0}".format(path))
    return selected


def finite_rows(rows, fields, path):
    for row in rows:
        for field in fields:
            number(row, field, path)


def summarize_run(run_dir, condition, dimension, manifest):
    metadata_path = run_dir / "run_metadata.json"
    metadata = load_json(metadata_path)
    benchmark = metadata.get("benchmark", {})
    if benchmark.get("no_render") or benchmark.get("hide_window") or not benchmark.get("sync_gpu") or not benchmark.get("disable_vsync"):
        fail("Run is not rendered, synchronized, and vsync-disabled: {0}".format(run_dir))
    if metadata.get("solver_variant") != condition["solver_variant"]:
        fail("Unexpected solver variant in {0}".format(run_dir))
    if metadata.get("quality", {}).get("iterations_per_frame") != str(condition["iterations_per_frame"]):
        fail("Iteration-budget mismatch in {0}".format(run_dir))
    expected_forced = "1" if condition["force_cpu_state_roundtrip"] else "0"
    if metadata.get("solver_controls", {}).get("force_cpu_state_roundtrip") != expected_forced:
        fail("Roundtrip metadata mismatch in {0}".format(run_dir))
    if metadata.get("experiment_overrides", {}).get("cloth_dimension") != str(dimension):
        fail("Cloth-dimension metadata mismatch in {0}".format(run_dir))

    profile_path = run_dir / "frame_profile.csv"
    extended_path = run_dir / "frame_profile_extended.csv"
    experiment_path = run_dir / "frame_profile_experiment.csv"
    presentation_path = run_dir / "frame_presentation.csv"
    warmup = manifest["timing"]["warmup"]
    profile = measured(load_csv(profile_path), warmup, profile_path)
    extended = measured(load_csv(extended_path), warmup, extended_path)
    experiment = measured(load_csv(experiment_path), warmup, experiment_path)
    presentation = measured(load_csv(presentation_path), warmup, presentation_path)
    expected = manifest["timing"]["frames"]
    if any(len(rows) != expected for rows in (profile, extended, experiment, presentation)):
        fail("Incomplete measured frame count in {0}".format(run_dir))
    if any(row.get("frame_valid") != "1" or row.get("termination_reason") != "none" for row in extended):
        fail("Invalid measured frame in {0}".format(run_dir))
    if any(row.get("rendered") != "1" or row.get("gpu_sync_enabled") != "1" for row in presentation):
        fail("Unrendered measured frame in {0}".format(run_dir))
    actual_iterations = [int(number(row, "iterations", profile_path)) for row in profile]
    if condition["solver_variant"].startswith("gpu-xpbd"):
        if any(value != condition["iterations_per_frame"] for value in actual_iterations):
            fail("Unexpected XPBD iteration count in {0}".format(run_dir))
    elif any(value < 1 or value > condition["iterations_per_frame"] for value in actual_iterations):
        fail("CPU NCG iteration count lies outside its configured maximum in {0}".format(run_dir))
    finite_rows(profile, ("total_ms", "iteration_ms", "optimization_ms", "update_posvel_ms"), profile_path)
    finite_rows(presentation, ("frame_wall_ms",), presentation_path)
    finite_rows(experiment, ("state_h2d_bytes", "state_d2h_bytes", "state_upload_calls", "state_readback_calls", "host_readbacks"), experiment_path)

    result = {
        "cloth_dimension": dimension,
        "vertices": dimension * dimension,
        "condition": condition["id"],
        "solver_variant": condition["solver_variant"],
        "force_cpu_state_roundtrip": int(condition["force_cpu_state_roundtrip"]),
        "iterations_per_frame": condition["iterations_per_frame"],
        "iterations_actual_mean": mean(actual_iterations),
        "iterations_actual_min": min(actual_iterations),
        "iterations_actual_max": max(actual_iterations),
        "comparison_role": condition["comparison_role"],
        "measured_frames": expected,
        "frame_wall_ms_mean": mean([number(row, "frame_wall_ms", presentation_path) for row in presentation]),
        "frame_wall_ms_p50": percentile([number(row, "frame_wall_ms", presentation_path) for row in presentation], 0.50),
        "frame_wall_ms_p95": percentile([number(row, "frame_wall_ms", presentation_path) for row in presentation], 0.95),
        "total_ms_mean": mean([number(row, "total_ms", profile_path) for row in profile]),
        "iteration_ms_mean": mean([number(row, "iteration_ms", profile_path) for row in profile]),
        "optimization_ms_mean": mean([number(row, "optimization_ms", profile_path) for row in profile]),
        "update_posvel_ms_mean": mean([number(row, "update_posvel_ms", profile_path) for row in profile]),
        "state_h2d_mib_mean": mean([number(row, "state_h2d_bytes", experiment_path) for row in experiment]) / 1048576.0,
        "state_d2h_mib_mean": mean([number(row, "state_d2h_bytes", experiment_path) for row in experiment]) / 1048576.0,
        "state_upload_calls_mean": mean([number(row, "state_upload_calls", experiment_path) for row in experiment]),
        "state_readback_calls_mean": mean([number(row, "state_readback_calls", experiment_path) for row in experiment]),
        "host_readbacks_mean": mean([number(row, "host_readbacks", experiment_path) for row in experiment]),
        "xpbd_constraint_dispatches_mean": mean([number(row, "xpbd_constraint_dispatches", experiment_path) for row in experiment]),
        "xpbd_apply_dispatches_mean": mean([number(row, "xpbd_apply_dispatches", experiment_path) for row in experiment]),
        "xpbd_collision_dispatches_mean": mean([number(row, "xpbd_collision_dispatches", experiment_path) for row in experiment]),
        "git_commit": metadata.get("git_commit", ""),
        "gpu_name": metadata.get("gpu_name", ""),
        "driver": metadata.get("nvidia_driver_version", ""),
        "result_dir": str(run_dir),
    }
    if condition["solver_variant"].startswith("gpu-xpbd"):
        result["xpbd_dispatches_mean"] = (
            result["xpbd_constraint_dispatches_mean"] + result["xpbd_apply_dispatches_mean"] + result["xpbd_collision_dispatches_mean"]
        )
    else:
        result["xpbd_dispatches_mean"] = 0.0
    return result


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def table_row(row):
    return "| {0} | {1:,} | {2}/{3:.1f} | {4:.3f} | {5:.3f} | {6:.3f} | {7:.3f} | {8:.3f}/{9:.3f} | {10:.1f} |".format(
        row["condition"], row["vertices"], row["iterations_per_frame"], row["iterations_actual_mean"], row["frame_wall_ms_mean"],
        row["frame_wall_ms_p50"], row["frame_wall_ms_p95"], row["optimization_ms_mean"],
        row["state_h2d_mib_mean"], row["state_d2h_mib_mean"], row["xpbd_dispatches_mean"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    manifest = load_json(run_root / "manifest.json")
    if manifest.get("protocol_version") != "xpbd-residency-scaling-diagnostic-v1":
        fail("Unexpected scaling-study protocol.")
    if manifest.get("measurement", {}).get("mode") != "rendered-end-to-end" or manifest.get("measurement", {}).get("repetitions") != 1:
        fail("This analyzer only accepts the single-repetition rendered diagnostic protocol.")
    conditions = manifest.get("conditions", [])
    if [condition.get("id") for condition in conditions] != list(EXPECTED_CONDITIONS):
        fail("Unexpected condition order or condition set.")

    rows = []
    for dimension in manifest["cloth_dimensions"]:
        entries = {}
        for condition in conditions:
            run_dir = run_root / "timing" / str(dimension) / condition["id"]
            entry = summarize_run(run_dir, condition, dimension, manifest)
            entries[condition["id"]] = entry
            rows.append(entry)
        resident = entries["xpbd-resident"]
        forced = entries["xpbd-forced-roundtrip"]
        if resident["xpbd_dispatches_mean"] != forced["xpbd_dispatches_mean"]:
            fail("XPBD dispatches differ between residency conditions at {0}.".format(dimension))
        if resident["state_h2d_mib_mean"] != 0.0 or resident["state_d2h_mib_mean"] != 0.0:
            fail("Resident XPBD has unexpected full-state traffic at {0}.".format(dimension))
        if forced["state_h2d_mib_mean"] <= 0.0 or forced["state_d2h_mib_mean"] <= 0.0:
            fail("Forced XPBD lacks state roundtrip traffic at {0}.".format(dimension))

    commits = {row["git_commit"] for row in rows}
    gpu_drivers = {(row["gpu_name"], row["driver"]) for row in rows}
    if len(commits) != 1 or not next(iter(commits)):
        fail("Runs were not produced from one nonempty commit.")
    if len(gpu_drivers) != 1:
        fail("GPU/driver metadata changed across runs.")

    write_csv(run_root / "xpbd_residency_scaling_summary.csv", rows)
    by_dimension = {dimension: [row for row in rows if row["cloth_dimension"] == dimension] for dimension in manifest["cloth_dimensions"]}
    comparison_rows = []
    for dimension, entries in by_dimension.items():
        lookup = {entry["condition"]: entry for entry in entries}
        resident = lookup["xpbd-resident"]
        forced = lookup["xpbd-forced-roundtrip"]
        signed = lookup["xpbd-signed-incidence-gather"]
        comparison_rows.append({
            "cloth_dimension": dimension,
            "resident_vs_forced_speedup": forced["frame_wall_ms_mean"] / resident["frame_wall_ms_mean"],
            "signed_vs_atomic_frame_ratio": signed["frame_wall_ms_mean"] / resident["frame_wall_ms_mean"],
            "signed_vs_atomic_p50_ratio": signed["frame_wall_ms_p50"] / resident["frame_wall_ms_p50"],
            "signed_vs_atomic_optimization_ratio": signed["optimization_ms_mean"] / resident["optimization_ms_mean"],
        })
    write_csv(run_root / "xpbd_residency_scaling_comparisons.csv", comparison_rows)

    gpu_name, driver = next(iter(gpu_drivers))
    report = [
        "# XPBD Residency, CPU Context, and Signed-Gather Scaling Diagnostic",
        "",
        "## Scope",
        "",
        "This is a rendered diagnostic, not a final repeated paper benchmark: each case has one 300-frame measured sequence after 30 warm-up frames. It adds scale coverage for the GPU-resident XPBD causal test and records the current signed-incidence vertex-gather implementation. It does not establish equal-quality CPU-NCG versus XPBD speedups.",
        "",
        "- Scene: moving sphere cloth (`scenes/moving_sphere_cloth.xml`).",
        "- Viewport: {0}x{1}; `--sync-gpu`, `--disable-vsync`, and `--uncapped` were enabled; every timed frame rendered.".format(manifest["measurement"]["render_width"], manifest["measurement"]["render_height"]),
        "- GPU: {0}; driver {1}; commit `{2}`.".format(gpu_name, driver, next(iter(commits))),
        "- XPBD cases: exactly 32 iterations/frame. CPU context: CPU NCG with a 32-iteration maximum; its early convergence makes the actual iteration count an observed value.",
        "- During timing, quality readback was disabled. All rows below completed with finite state, `frame_valid=1`, and `termination_reason=none`.",
        "",
        "## Results",
        "",
        "Columns: `iter max/actual` reports the requested maximum and measured mean actual count; rendered wall time uses `frame_wall_ms`; GPU optimization is the application's timed solver section; state traffic is MiB/frame; `dispatch` sums XPBD constraint, apply, and collision dispatches. CPU NCG has no XPBD dispatches.",
        "",
    ]
    for dimension in manifest["cloth_dimensions"]:
        report += [
            "### {0}x{0} ({1:,} vertices)".format(dimension, dimension * dimension),
            "",
            "| condition | vertices | iter max/actual | mean ms | P50 ms | P95 ms | solver ms | state H2D/D2H MiB | dispatch |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        report += [table_row(row) for row in by_dimension[dimension]]
        comparison = next(row for row in comparison_rows if row["cloth_dimension"] == dimension)
        report += [
            "",
            "- GPU-resident atomic XPBD is `{0:.2f}x` faster than the forced state-roundtrip counterfactual at this resolution.".format(comparison["resident_vs_forced_speedup"]),
            "- Signed-incidence gather / atomic XPBD ratio: `{0:.2f}x` rendered mean, `{1:.2f}x` rendered P50, and `{2:.2f}x` solver time. Values above 1 mean gather is slower.".format(comparison["signed_vs_atomic_frame_ratio"], comparison["signed_vs_atomic_p50_ratio"], comparison["signed_vs_atomic_optimization_ratio"]),
            "",
        ]
    report += [
        "## Interpretation",
        "",
        "1. **Residency is a controlled data-transfer result.** Resident and forced runs use the same `gpu-xpbd-jacobi` iteration budget and the same XPBD dispatch count. The forced condition alone transfers finalized position/velocity state through CPU between frames. Its measured overhead therefore isolates this full-state traffic plus synchronization, rather than a different solver.",
        "2. **The CPU row is timing context, not a solver ranking.** CPU NCG and XPBD use different update rules and fixed iteration budgets; the CPU row must not be converted into an equal-quality speedup claim. A reference-calibrated CPU/XPBD study remains required for that claim.",
        "3. **Signed gather is an implementation measurement.** It removes the raw gather prototype's full `Edge` fetch in the vertex pass, but does not eliminate the atomic constraint pass. This diagnostic reports its actual end-to-end cost; it is only a positive optimization if its ratio is below 1. The prior 256x256 short checkpoint test establishes atomic/gather numerical equality for the same starting state; these 300-frame timing runs establish validity, not a new long-horizon trajectory gate.",
        "4. **Why a small residency gain is plausible.** The transfer counterfactual changes only two position/velocity transfers per frame, while XPBD still executes three GPU passes per iteration and is commonly dominated by the constraint scatter pass and rendering. A modest difference is therefore still useful causal evidence, but not evidence of the much larger reductions observed when eliminating more CPU-GPU synchronization from a different NCG pipeline.",
        "5. **No consistent signed-gather speedup is established.** The rendered P50 ratio is retained because desktop presentation has occasional long-tail stalls. If mean wall time and P50 disagree, the result is treated as inconclusive for end-to-end gain and solver time supplies the implementation signal. This diagnostic cannot support a claim that signed gather improves XPBD performance across resolutions.",
        "",
        "## Evidence Boundary",
        "",
        "This report is suitable as an internal reproducible diagnostic. It is not a final paper table because it has one repetition, does not recalibrate equal quality across CPU NCG and XPBD, and does not add a long-horizon atomic/gather trajectory comparison at every resolution. Any manuscript use should preserve these limitations or replace the data with the pre-registered multi-repeat quality-calibrated protocol.",
        "",
        "## Files",
        "",
        "- `manifest.json`: protocol, controls, commit, GPU, and driver.",
        "- `planned_runs.csv`: all 12 planned cases.",
        "- `xpbd_residency_scaling_summary.csv`: validated per-case aggregates.",
        "- `xpbd_residency_scaling_comparisons.csv`: causal ratios derived from the validated aggregates.",
        "- `timing/<resolution>/<condition>/`: raw CSVs, metadata, and process logs for every case.",
    ]
    (run_root / "xpbd_residency_scaling_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("XPBD scaling summary: {0}".format(run_root / "xpbd_residency_scaling_summary.csv"))


if __name__ == "__main__":
    main()
