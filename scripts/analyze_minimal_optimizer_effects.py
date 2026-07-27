#!/usr/bin/env python3
"""Evidence-gated analysis for the bounded optimizer-effects recheck."""

import argparse
import csv
import json
import math
import random
from pathlib import Path


PROTOCOL_VERSION = "minimal-optimizer-effects-r1"
BOOTSTRAP_SAMPLES = 20000


def fail(message):
    raise RuntimeError(message)


def load_json(path):
    if not path.is_file():
        fail("Missing JSON: {0}".format(path))
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_csv(path):
    if not path.is_file():
        fail("Missing CSV: {0}".format(path))
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
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


def integer(row, field, path):
    return int(number(row, field, path))


def mean(values):
    if not values:
        fail("Cannot compute a mean from no values.")
    return sum(values) / float(len(values))


def sample_std(values):
    if len(values) < 2:
        return 0.0
    average = mean(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / float(len(values) - 1))


def percentile(values, fraction=0.95):
    if not values:
        fail("Cannot compute a percentile from no values.")
    ordered = sorted(values)
    index = int(math.ceil(fraction * len(ordered))) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def measured(rows, warmup, path):
    selected = [row for row in rows if integer(row, "frame", path) >= warmup]
    if not selected:
        fail("No measured rows in {0}".format(path))
    return selected


def condition_index(manifest):
    index = {}
    for suite_name in ("baseline", "stress"):
        suite = manifest.get(suite_name, {})
        for condition in suite.get("conditions", []):
            key = (suite_name, condition["condition_id"])
            if key in index:
                fail("Duplicate condition in manifest: {0}".format(key))
            index[key] = condition
    return index


def expected_dispatch_counts(condition_id, iterations_per_frame):
    per_iteration = {
        "edge-scatter-k8": (2.0, 1.0),
        "vertex-gather-k8": (1.0, 1.0),
        "gather-fusion-k8": (1.0, 0.0),
    }[condition_id]
    return tuple(value * float(iterations_per_frame) for value in per_iteration)


def quality_gate_passes(p95_position_rel_l2, gate):
    return math.isfinite(p95_position_rel_l2) and p95_position_rel_l2 <= gate


def requires_nonempty_adaptive_trace(condition):
    return condition.get("decision_trace", False) and condition.get("solver_variant") == "gpu-gather-fusion-adaptive-ls-persistent"


def expected_timing_dir(run_root, suite, condition_id, block):
    return run_root / "timing" / suite / condition_id / "block{0:02d}".format(block)


def validate_metadata(metadata, condition, manifest, run_dir, require_quality):
    benchmark = metadata.get("benchmark", {})
    if benchmark.get("no_render") or benchmark.get("hide_window") or not benchmark.get("sync_gpu") or not benchmark.get("disable_vsync"):
        fail("Run is not rendered, visible, synchronized, and vsync-disabled: {0}".format(run_dir))
    if metadata.get("solver_variant") != condition["solver_variant"]:
        fail("Solver variant mismatch in {0}".format(run_dir))
    if str(metadata.get("quality", {}).get("iterations_per_frame", "")) != str(condition["iterations_per_frame"]):
        fail("Iteration budget mismatch in {0}".format(run_dir))
    overrides = metadata.get("experiment_overrides", {})
    if str(overrides.get("cloth_dimension", "")) != str(manifest["baseline"]["cloth_dimension"]):
        fail("Cloth resolution mismatch in {0}".format(run_dir))
    if not str(overrides.get("scene", "")).replace("/", "\\").endswith("moving_sphere_cloth.xml"):
        fail("Scene mismatch in {0}".format(run_dir))
    controls = metadata.get("solver_controls", {})
    if str(controls.get("batched_ls_k", "")) != str(condition["batched_ls_k"]):
        fail("Batched K mismatch in {0}".format(run_dir))
    if str(controls.get("adaptive_ls_history", "")) != condition["adaptive_ls_history"]:
        fail("Adaptive history mismatch in {0}".format(run_dir))
    if str(controls.get("ncg_restart_mode", "")) != "non-descent":
        fail("Restart mode mismatch in {0}".format(run_dir))
    if require_quality and str(metadata.get("quality", {}).get("reference_dir", "")) != str(manifest["reference_dir"]):
        fail("Quality reference mismatch in {0}".format(run_dir))


def timing_metrics(run_dir, condition, manifest):
    metadata = load_json(run_dir / "run_metadata.json")
    validate_metadata(metadata, condition, manifest, run_dir, False)
    timing = manifest["timing"]
    presentation_path = run_dir / "frame_presentation.csv"
    profile_path = run_dir / "frame_profile.csv"
    extended_path = run_dir / "frame_profile_extended.csv"
    experiment_path = run_dir / "frame_profile_experiment.csv"
    presentation = measured(load_csv(presentation_path), timing["warmup"], presentation_path)
    profile = measured(load_csv(profile_path), timing["warmup"], profile_path)
    extended = measured(load_csv(extended_path), timing["warmup"], extended_path)
    experiment = measured(load_csv(experiment_path), timing["warmup"], experiment_path)
    if any(len(rows) != timing["frames"] for rows in (presentation, profile, extended, experiment)):
        fail("Incomplete timing run in {0}".format(run_dir))
    if any(row.get("rendered") != "1" or row.get("gpu_sync_enabled") != "1" for row in presentation):
        fail("Unrendered or unsynchronized presentation frame in {0}".format(run_dir))
    if any(row.get("frame_valid") != "1" or row.get("termination_reason") != "none" for row in extended):
        fail("Invalid timing frame in {0}".format(run_dir))

    gradient_dispatches = mean([number(row, "gradient_dispatches", experiment_path) for row in experiment])
    stats_dispatches = mean([number(row, "stats_dispatches", experiment_path) for row in experiment])
    if condition["comparison_group"] == "traversal":
        expected = expected_dispatch_counts(condition["condition_id"], condition["iterations_per_frame"])
        if abs(gradient_dispatches - expected[0]) > 1.0e-9 or abs(stats_dispatches - expected[1]) > 1.0e-9:
            fail("Gradient/stat dispatch structure changed in {0}".format(run_dir))

    frame_values = [number(row, "frame_wall_ms", presentation_path) for row in presentation]
    gradient_values = [number(row, "cs_gradient_gpu_ms", profile_path) + number(row, "cs_gradstats_ms", profile_path) for row in profile]
    return {
        "frame_wall_ms": mean(frame_values),
        "frame_wall_p95_ms": percentile(frame_values),
        "gradient_stats_ms": mean(gradient_values),
        "gradient_dispatches": gradient_dispatches,
        "stats_dispatches": stats_dispatches,
        "git_commit": metadata.get("git_commit", ""),
        "gpu_name": metadata.get("gpu_name", ""),
        "driver": metadata.get("nvidia_driver_version", ""),
    }


def quality_metrics(run_dir, condition, manifest):
    metadata = load_json(run_dir / "run_metadata.json")
    validate_metadata(metadata, condition, manifest, run_dir, True)
    quality_config = manifest["quality"]
    quality_path = run_dir / "quality_metrics.csv"
    extended_path = run_dir / "frame_profile_extended.csv"
    quality = measured(load_csv(quality_path), quality_config["warmup"], quality_path)
    extended = measured(load_csv(extended_path), quality_config["warmup"], extended_path)
    if len(quality) != quality_config["frames"] or len(extended) != quality_config["frames"]:
        fail("Incomplete quality run in {0}".format(run_dir))
    if any(row.get("finite") != "1" or row.get("exploded") != "0" for row in quality):
        fail("Non-finite quality record in {0}".format(run_dir))
    if any(row.get("frame_valid") != "1" or row.get("termination_reason") != "none" for row in extended):
        fail("Invalid quality frame in {0}".format(run_dir))
    reference_errors = [number(row, "position_rel_l2", quality_path) for row in quality if row.get("has_reference") == "1"]
    if not reference_errors:
        fail("No reference-aligned quality rows in {0}".format(run_dir))
    p95_error = percentile(reference_errors)

    def sum_field(field):
        return sum(number(row, field, extended_path) for row in extended)

    result = {
        "p95_position_rel_l2": p95_error,
        "quality_gate_pass": quality_gate_passes(p95_error, quality_config["position_gate_p95"]),
        "invalid_frames": 0,
        "full_line_searches": sum_field("cs_full_ls"),
        "adaptive_candidate_evaluations": sum_field("adaptive_candidate_evaluations"),
        "adaptive_history_uses": sum_field("adaptive_history_uses"),
        "adaptive_first_batch_accepts": sum_field("adaptive_first_batch_accepts"),
        "adaptive_second_batch_accepts": sum_field("adaptive_second_batch_accepts"),
        "armijo_fallbacks": sum_field("armijo_fallbacks"),
        "armijo_rejections": sum_field("armijo_rejections"),
        "armijo_failures": sum_field("armijo_failures"),
    }
    if requires_nonempty_adaptive_trace(condition):
        trace_path = run_dir / "line_search_trace.csv"
        trace = load_csv(trace_path)
    return result


def bootstrap_interval(values, seed):
    rng = random.Random(seed)
    count = len(values)
    samples = []
    for _ in range(BOOTSTRAP_SAMPLES):
        samples.append(mean([values[rng.randrange(count)] for _ in range(count)]))
    return percentile(samples, 0.025), percentile(samples, 0.975)


def classify_effect(improvements, threshold, comparison_id):
    average = mean(improvements)
    ci_low, ci_high = bootstrap_interval(improvements, comparison_id)
    faster_blocks = sum(1 for value in improvements if value > 0.0)
    slower_blocks = sum(1 for value in improvements if value < 0.0)
    if average >= threshold and ci_low > 0.0 and faster_blocks >= 5:
        verdict = "material-benefit"
    elif average <= -threshold and ci_high < 0.0 and slower_blocks >= 5:
        verdict = "material-regression"
    elif ci_low >= -threshold and ci_high <= threshold:
        verdict = "no-material-effect"
    else:
        verdict = "inconclusive"
    return average, ci_low, ci_high, faster_blocks, slower_blocks, verdict


def write_csv(path, rows):
    if not rows:
        fail("Refusing to write an empty CSV: {0}".format(path))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(run_root, manifest):
    conditions = condition_index(manifest)
    timing_rows = load_csv(run_root / "planned_timing_runs.csv")
    expected_runs = manifest["timing"]["repetitions"] * (len(manifest["baseline"]["conditions"]) + len(manifest["stress"]["conditions"]))
    if len(timing_rows) != expected_runs:
        fail("Timing plan has {0} rows, expected {1}.".format(len(timing_rows), expected_runs))

    timing_by_condition = {}
    for row in timing_rows:
        suite = row.get("suite", "")
        condition_id = row.get("condition_id", "")
        block = integer(row, "block", run_root / "planned_timing_runs.csv")
        condition = conditions.get((suite, condition_id))
        if condition is None:
            fail("Timing plan references unknown condition: {0}/{1}".format(suite, condition_id))
        key = (suite, condition_id)
        timing_by_condition.setdefault(key, {})[block] = timing_metrics(expected_timing_dir(run_root, suite, condition_id, block), condition, manifest)

    expected_blocks = set(range(1, manifest["timing"]["repetitions"] + 1))
    summary_rows = []
    for key, condition in conditions.items():
        timing_by_block = timing_by_condition.get(key, {})
        if set(timing_by_block) != expected_blocks:
            fail("Missing or duplicate timing blocks for {0}".format(key))
        run_metrics = [timing_by_block[block] for block in sorted(expected_blocks)]
        quality_dir = run_root / "quality" / key[0] / key[1] / "trace"
        quality = quality_metrics(quality_dir, condition, manifest)
        commits = {entry["git_commit"] for entry in run_metrics}
        gpus = {entry["gpu_name"] for entry in run_metrics}
        drivers = {entry["driver"] for entry in run_metrics}
        if len(commits) != 1 or len(gpus) != 1 or len(drivers) != 1 or not next(iter(commits)):
            fail("Inconsistent timing metadata for {0}".format(key))
        frame_values = [entry["frame_wall_ms"] for entry in run_metrics]
        gradient_values = [entry["gradient_stats_ms"] for entry in run_metrics]
        summary_rows.append({
            "suite": key[0],
            "condition_id": key[1],
            "solver_variant": condition["solver_variant"],
            "iterations_per_frame": condition["iterations_per_frame"],
            "batched_ls_k": condition["batched_ls_k"],
            "adaptive_ls_history": condition["adaptive_ls_history"],
            "timing_repetitions": len(frame_values),
            "frame_wall_ms_mean": mean(frame_values),
            "frame_wall_ms_std": sample_std(frame_values),
            "frame_wall_ms_p95": percentile(frame_values),
            "gradient_stats_ms_mean": mean(gradient_values),
            "gradient_stats_ms_std": sample_std(gradient_values),
            "gradient_dispatches": run_metrics[0]["gradient_dispatches"],
            "stats_dispatches": run_metrics[0]["stats_dispatches"],
            "p95_position_rel_l2": quality["p95_position_rel_l2"],
            "quality_gate_pass": quality["quality_gate_pass"],
            "full_line_searches": quality["full_line_searches"],
            "adaptive_candidate_evaluations": quality["adaptive_candidate_evaluations"],
            "adaptive_history_uses": quality["adaptive_history_uses"],
            "adaptive_first_batch_accepts": quality["adaptive_first_batch_accepts"],
            "adaptive_second_batch_accepts": quality["adaptive_second_batch_accepts"],
            "armijo_fallbacks": quality["armijo_fallbacks"],
            "armijo_rejections": quality["armijo_rejections"],
            "armijo_failures": quality["armijo_failures"],
            "git_commit": next(iter(commits)),
            "gpu_name": next(iter(gpus)),
            "driver": next(iter(drivers)),
        })

    summary_by_key = {(row["suite"], row["condition_id"]): row for row in summary_rows}
    comparisons = []
    threshold = float(manifest["practical_effect_threshold"])
    for spec in manifest["comparisons"]:
        suite = spec["suite"]
        control_key = (suite, spec["control"])
        treatment_key = (suite, spec["treatment"])
        controls = timing_by_condition.get(control_key, {})
        treatments = timing_by_condition.get(treatment_key, {})
        if set(controls) != expected_blocks or set(treatments) != expected_blocks:
            fail("Incomplete paired timing data for {0}".format(spec["id"]))
        improvements = [(controls[block]["frame_wall_ms"] - treatments[block]["frame_wall_ms"]) / controls[block]["frame_wall_ms"] for block in sorted(expected_blocks)]
        quality_qualified = bool(summary_by_key[control_key]["quality_gate_pass"]) and bool(summary_by_key[treatment_key]["quality_gate_pass"])
        if quality_qualified:
            average, ci_low, ci_high, faster_blocks, slower_blocks, verdict = classify_effect(improvements, threshold, spec["id"])
        else:
            average = mean(improvements)
            ci_low, ci_high = bootstrap_interval(improvements, spec["id"])
            faster_blocks = sum(1 for value in improvements if value > 0.0)
            slower_blocks = sum(1 for value in improvements if value < 0.0)
            verdict = "not-qualified-quality"
        comparisons.append({
            "comparison_id": spec["id"],
            "family": spec["family"],
            "suite": suite,
            "control": spec["control"],
            "treatment": spec["treatment"],
            "paired_blocks": len(improvements),
            "treatment_improvement_mean": average,
            "treatment_improvement_ci95_low": ci_low,
            "treatment_improvement_ci95_high": ci_high,
            "faster_blocks": faster_blocks,
            "slower_blocks": slower_blocks,
            "quality_qualified": quality_qualified,
            "practical_effect_threshold": threshold,
            "verdict": verdict,
        })
    return summary_rows, comparisons


def write_report(path, manifest, summaries, comparisons):
    rows_by_key = {(row["suite"], row["condition_id"]): row for row in summaries}
    lines = [
        "# Minimal Optimizer-Effects Recheck",
        "",
        "## Protocol",
        "",
        "- Commit `{0}` on `{1}` (driver `{2}`).".format(summaries[0]["git_commit"], summaries[0]["gpu_name"], summaries[0]["driver"]),
        "- Rendered 1600x900 end-to-end timing with GPU synchronization, vsync disabled, 150 measured + 30 warm-up frames, and six interleaved process repetitions.",
        "- Baseline: 386^2 moving sphere, default physics, one NCG iteration per frame. Stress: the same scene/mesh/physics with eight NCG iterations per frame.",
        "- Quality is a separate 120 measured + 20 warm-up rendered run against the archived CPU-NCG reference checkpoints; every condition must satisfy P95 position relative L2 <= 1e-3 with no invalid frame.",
        "- A treatment is a material benefit only when its paired mean frame-time improvement is at least 3%, the bootstrap 95% interval is entirely positive, and it is faster in at least five of six blocks. The symmetric rule marks a material regression; intervals contained in +/-3% are no-material-effect; all other outcomes are inconclusive. A comparison with either condition failing the quality gate is reported as raw timing only and receives no performance verdict.",
        "",
        "## Condition Summary",
        "",
        "| Suite | Condition | Frame ms | Gradient+stats ms | P95 position error | Gate | K | History | Armijo failures |",
        "|---|---|---:|---:|---:|---|---:|---|---:|",
    ]
    for row in sorted(summaries, key=lambda value: (value["suite"], value["condition_id"])):
        lines.append("| {0} | {1} | {2:.4f} +/- {3:.4f} | {4:.4f} | {5:.3e} | {6} | {7} | {8} | {9} |".format(
            row["suite"], row["condition_id"], row["frame_wall_ms_mean"], row["frame_wall_ms_std"], row["gradient_stats_ms_mean"],
            row["p95_position_rel_l2"], "pass" if row["quality_gate_pass"] else "fail", row["batched_ls_k"], row["adaptive_ls_history"], row["armijo_failures"]))
    lines += [
        "",
        "## Paired Comparisons",
        "",
        "Positive improvement means the treatment is faster than the control. Comparisons marked `not-qualified-quality` are diagnostic raw timings only.",
        "",
        "| Comparison | Treatment improvement | 95% bootstrap CI | Faster blocks | Quality | Verdict |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in comparisons:
        lines.append("| {0} | {1:+.2%} | [{2:+.2%}, {3:+.2%}] | {4}/{5} | {6} | {7} |".format(
            row["comparison_id"], row["treatment_improvement_mean"], row["treatment_improvement_ci95_low"],
            row["treatment_improvement_ci95_high"], row["faster_blocks"], row["paired_blocks"], "pass" if row["quality_qualified"] else "fail", row["verdict"]))
    lines += [
        "",
        "## Line-Search Diagnostics",
        "",
        "| Condition | Full searches | Candidate evaluations | History uses | First-batch accepts | Second-batch accepts | Fallbacks | Rejections |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition_id in ("fixed-k4", "adaptive-k4-no-history", "adaptive-k4-frame-history"):
        row = rows_by_key[("stress", condition_id)]
        lines.append("| {0} | {1:.0f} | {2:.0f} | {3:.0f} | {4:.0f} | {5:.0f} | {6:.0f} | {7:.0f} |".format(
            condition_id, row["full_line_searches"], row["adaptive_candidate_evaluations"], row["adaptive_history_uses"],
            row["adaptive_first_batch_accepts"], row["adaptive_second_batch_accepts"], row["armijo_fallbacks"], row["armijo_rejections"]))
    lines += [
        "",
        "## Interpretation Boundary",
        "",
        "This recheck tests one retained paper workload and one pre-registered higher-workload point. It can establish whether the current implementation has a material rendered end-to-end effect at these two points; it cannot establish a universal result across GPU architectures, mesh topologies, material parameters, or all Armijo workloads.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Analyze the bounded GenPD optimizer-effects recheck.")
    parser.add_argument("--run-root", required=True, help="Result directory written by run_minimal_optimizer_effects_recheck.ps1")
    parser.add_argument("--report-path", required=True, help="Markdown report path")
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    manifest = load_json(run_root / "manifest.json")
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        fail("Unexpected protocol version.")
    measurement = manifest.get("measurement", {})
    if measurement.get("mode") != "rendered-end-to-end" or not measurement.get("sync_gpu") or not measurement.get("disable_vsync"):
        fail("Minimal recheck must use rendered synchronized timing.")
    if manifest.get("practical_effect_threshold") != 0.03:
        fail("Minimal recheck must use a 3% practical effect threshold.")
    if len(manifest.get("baseline", {}).get("conditions", [])) != 3 or len(manifest.get("stress", {}).get("conditions", [])) != 6:
        fail("Minimal recheck condition matrix is incomplete.")

    summaries, comparisons = summarize(run_root, manifest)
    write_csv(run_root / "minimal_optimizer_effects_summary.csv", summaries)
    write_csv(run_root / "minimal_optimizer_effects_comparisons.csv", comparisons)
    write_report(Path(args.report_path).resolve(), manifest, summaries, comparisons)


if __name__ == "__main__":
    main()
