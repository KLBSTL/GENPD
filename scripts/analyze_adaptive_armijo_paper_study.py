#!/usr/bin/env python3
"""Aggregate the formal rendered adaptive Armijo core and sensitivity studies."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CORE_METHODS = (
    "serial",
    "fixed-k8",
    "adaptive-k4-history-none",
    "adaptive-k4-history-iteration",
    "adaptive-k4-history-frame",
)
METHOD_LABELS = {
    "serial": "Serial Armijo",
    "fixed-k8": "Fixed K=8",
    "adaptive-k4-history-none": "Adaptive K=4",
    "adaptive-k4-history-iteration": "Adaptive + iter. hist.",
    "adaptive-k4-history-frame": "Adaptive + frame hist.",
}
METHOD_COLORS = {
    "serial": "#79706E",
    "fixed-k8": "#4E79A7",
    "adaptive-k4-history-none": "#F28E2B",
    "adaptive-k4-history-iteration": "#B07AA1",
    "adaptive-k4-history-frame": "#59A14F",
}


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


def numeric(row, field, path, allow_blank=False):
    value = row.get(field, "")
    if allow_blank and (value is None or value == ""):
        return math.nan
    try:
        value = float(value)
    except (TypeError, ValueError):
        fail("Missing numeric field '{0}' in {1}".format(field, path))
    if not math.isfinite(value):
        fail("Non-finite field '{0}' in {1}".format(field, path))
    return value


def mean(values):
    return sum(values) / float(len(values))


def percentile(values, fraction=0.95):
    ordered = sorted(values)
    if not ordered:
        fail("Cannot compute percentile from no values.")
    index = int(math.ceil(fraction * len(ordered))) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def measured(rows, warmup, path):
    result = [row for row in rows if int(numeric(row, "frame", path)) >= warmup]
    if not result:
        fail("No measured frames in {0}".format(path))
    return result


def validate_rendered_run(directory, expected_variant, frames, warmup, trace):
    metadata = load_json(directory / "run_metadata.json")
    benchmark = metadata.get("benchmark", {})
    if benchmark.get("no_render") or not benchmark.get("sync_gpu") or not benchmark.get("disable_vsync"):
        fail("Run is not rendered, GPU-synchronised, and vsync-disabled: {0}".format(directory))
    if metadata.get("solver_variant") != expected_variant:
        fail("Unexpected variant in {0}".format(directory))
    extended_path = directory / "frame_profile_extended.csv"
    presentation_path = directory / "frame_presentation.csv"
    extended = measured(load_csv(extended_path), warmup, extended_path)
    presentation = measured(load_csv(presentation_path), warmup, presentation_path)
    if len(extended) != frames or len(presentation) != frames:
        fail("Incomplete frame count in {0}".format(directory))
    if any(row.get("frame_valid") != "1" or row.get("termination_reason") != "none" for row in extended):
        fail("Invalid frame in {0}".format(directory))
    if any(row.get("rendered") != "1" or row.get("gpu_sync_enabled") != "1" for row in presentation):
        fail("Unrendered presentation frame in {0}".format(directory))
    if trace:
        quality_path = directory / "quality_metrics.csv"
        quality = measured(load_csv(quality_path), warmup, quality_path)
        if len(quality) != frames:
            fail("Incomplete quality frame count in {0}".format(directory))
        if any(row.get("finite") != "1" or row.get("exploded") != "0" for row in quality):
            fail("Invalid quality data in {0}".format(directory))
        if not any(row.get("has_reference") == "1" for row in quality):
            fail("No reference-aligned quality checkpoint in {0}".format(directory))
        return extended, presentation, quality
    return extended, presentation, None


def schedule_variant(schedule):
    return {
        "serial": "gpu-gather-fusion-serial-ls-persistent",
        "fixed": "gpu-gather-fusion-batched-ls-persistent",
        "adaptive": "gpu-gather-fusion-adaptive-ls-persistent",
    }[schedule]


def validate_summary(summary_path, expected_schedule, expected_k, expected_beta, expected_history, manifest):
    rows = load_csv(summary_path)
    if len(rows) != 1:
        fail("Expected one line-search row in {0}".format(summary_path))
    row = rows[0]
    if row.get("schedule") != expected_schedule or row.get("solver_variant") != schedule_variant(expected_schedule):
        fail("Schedule/variant mismatch in {0}".format(summary_path))
    if int(numeric(row, "batched_ls_k", summary_path)) != expected_k:
        fail("K mismatch in {0}".format(summary_path))
    if abs(numeric(row, "armijo_beta", summary_path) - expected_beta) > 1.0e-12:
        fail("Beta mismatch in {0}".format(summary_path))
    expected_history = expected_history if expected_schedule == "adaptive" else "none"
    if row.get("adaptive_ls_history") != expected_history:
        fail("History mismatch in {0}".format(summary_path))
    if int(numeric(row, "timing_repetitions", summary_path)) != manifest["timing"]["repetitions"]:
        fail("Repetition mismatch in {0}".format(summary_path))
    if int(numeric(row, "timing_frames_per_repetition", summary_path)) != manifest["timing"]["frames"]:
        fail("Timing-frame mismatch in {0}".format(summary_path))
    if int(numeric(row, "trace_frame_samples", summary_path)) != manifest["trace"]["frames"]:
        fail("Trace-frame mismatch in {0}".format(summary_path))
    if int(numeric(row, "invalid_trace_frames", summary_path)) != 0 or numeric(row, "trace_failure_rate", summary_path) != 0.0:
        fail("Invalid trace row in {0}".format(summary_path))
    if int(numeric(row, "armijo_failures", summary_path)) != 0:
        fail("Armijo failure in {0}".format(summary_path))
    p95 = numeric(row, "p95_position_rel_l2", summary_path)
    if p95 > manifest["trace"]["position_gate_p95"]:
        fail("Quality gate failed in {0}: {1:.3e}".format(summary_path, p95))
    return row


def repetition_frame_means(summary_row, manifest):
    dirs = [Path(text) for text in summary_row["timing_dirs"].split(";") if text]
    if len(dirs) != manifest["timing"]["repetitions"]:
        fail("Unexpected timing directory count.")
    values = []
    for directory in dirs:
        _, presentation, _ = validate_rendered_run(
            directory,
            summary_row["solver_variant"],
            manifest["timing"]["frames"],
            manifest["timing"]["warmup"],
            trace=False,
        )
        values.append(mean([numeric(row, "frame_wall_ms", directory / "frame_presentation.csv") for row in presentation]))
    trace_dir = Path(summary_row["trace_dir"])
    validate_rendered_run(
        trace_dir,
        summary_row["solver_variant"],
        manifest["trace"]["frames"],
        manifest["trace"]["warmup"],
        trace=True,
    )
    return values


def core_row(case, method, row, repetition_means):
    summary_path = ""
    return {
        "scene_id": case["scene_id"],
        "cloth_dimension": int(case["cloth_dimension"]),
        "iterations_per_frame": int(case["iterations_per_frame"]),
        "method_id": method["method_id"],
        "schedule": row["schedule"],
        "solver_variant": row["solver_variant"],
        "adaptive_ls_history": row["adaptive_ls_history"],
        "batched_ls_k": int(numeric(row, "batched_ls_k", "core summary")),
        "armijo_beta": numeric(row, "armijo_beta", "core summary"),
        "frame_wall_ms_mean": numeric(row, "rendered_frame_wall_ms_mean", "core summary"),
        "frame_wall_ms_std": numeric(row, "rendered_frame_wall_ms_std", "core summary"),
        "frame_wall_ms_p95": numeric(row, "rendered_frame_wall_ms_p95", "core summary"),
        "total_ms_mean": numeric(row, "total_ms_mean", "core summary"),
        "line_search_ms": numeric(row, "line_search_ms", "core summary"),
        "candidates_per_search": numeric(row, "candidates_per_search", "core summary"),
        "history_use_ratio": numeric(row, "history_use_ratio", "core summary"),
        "first_batch_accept_ratio": numeric(row, "first_batch_accept_ratio", "core summary"),
        "second_batch_accept_ratio": numeric(row, "second_batch_accept_ratio", "core summary"),
        "fallback_ratio": numeric(row, "fallback_ratio", "core summary"),
        "armijo_rejections": numeric(row, "armijo_rejections", "core summary"),
        "armijo_failures": int(numeric(row, "armijo_failures", "core summary")),
        "p95_position_rel_l2": numeric(row, "p95_position_rel_l2", "core summary"),
        "invalid_trace_frames": int(numeric(row, "invalid_trace_frames", "core summary")),
        "repeat_frame_wall_ms": ";".join("{0:.12g}".format(value) for value in repetition_means),
        "timing_dirs": row["timing_dirs"],
        "trace_dir": row["trace_dir"],
    }


def read_core(run_root, manifest, cases):
    rows = []
    for case in cases:
        for method in manifest["core_methods"]:
            output = run_root / "core" / case["case_id"] / method["method_id"] / "line_search_summary.csv"
            summary = validate_summary(output, method["schedule"], int(method["k"]), float(method["beta"]), method["history"], manifest)
            repeats = repetition_frame_means(summary, manifest)
            rows.append(core_row(case, method, summary, repeats))
    return rows


def read_sensitivity(run_root, manifest, cases):
    rows = []
    for case in cases:
        beta_text = ("{0:g}".format(float(case["beta"]))).replace(".", "p")
        output = run_root / "sensitivity" / case["case_id"] / "adaptive-k{0}-b{1}-h{2}".format(case["k"], beta_text, case["history"]) / "line_search_summary.csv"
        summary = validate_summary(output, "adaptive", int(case["k"]), float(case["beta"]), case["history"], manifest)
        repeats = repetition_frame_means(summary, manifest)
        rows.append({
            "scene_id": case["scene_id"], "cloth_dimension": int(case["cloth_dimension"]), "iterations_per_frame": int(case["iterations_per_frame"]),
            "batched_ls_k": int(case["k"]), "armijo_beta": float(case["beta"]), "adaptive_ls_history": case["history"],
            "frame_wall_ms_mean": numeric(summary, "rendered_frame_wall_ms_mean", output),
            "frame_wall_ms_std": numeric(summary, "rendered_frame_wall_ms_std", output),
            "line_search_ms": numeric(summary, "line_search_ms", output),
            "candidates_per_search": numeric(summary, "candidates_per_search", output),
            "history_use_ratio": numeric(summary, "history_use_ratio", output),
            "fallback_ratio": numeric(summary, "fallback_ratio", output),
            "armijo_rejections": numeric(summary, "armijo_rejections", output),
            "armijo_failures": int(numeric(summary, "armijo_failures", output)),
            "p95_position_rel_l2": numeric(summary, "p95_position_rel_l2", output),
            "invalid_trace_frames": int(numeric(summary, "invalid_trace_frames", output)),
            "repeat_frame_wall_ms": ";".join("{0:.12g}".format(value) for value in repeats),
        })
    return rows


def write_csv(path, rows):
    if not rows:
        fail("No rows to write to {0}".format(path))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_go_no_go(core, manifest):
    rows = {(row["scene_id"], row["cloth_dimension"], row["method_id"]): row for row in core}
    checks = []
    improvements = []
    directions = []
    history_reductions = []
    for scene in ("hanging", "moving-sphere"):
        fixed = rows[(scene, 386, "fixed-k8")]
        adaptive = rows[(scene, 386, "adaptive-k4-history-frame")]
        no_history = rows[(scene, 386, "adaptive-k4-history-none")]
        improvements.append(1.0 - adaptive["frame_wall_ms_mean"] / fixed["frame_wall_ms_mean"])
        candidate_reduction = 1.0 - adaptive["candidates_per_search"] / fixed["candidates_per_search"]
        history_reductions.append(1.0 - adaptive["line_search_ms"] / no_history["line_search_ms"])
        fixed_reps = [float(value) for value in fixed["repeat_frame_wall_ms"].split(";")]
        adaptive_reps = [float(value) for value in adaptive["repeat_frame_wall_ms"].split(";")]
        directions.append(sum(1 for a, b in zip(adaptive_reps, fixed_reps) if a < b))
        checks.append(("{0}: adaptive candidate reduction".format(scene), candidate_reduction >= manifest["go_no_go"]["candidate_reduction_vs_fixed"], candidate_reduction))
        checks.append(("{0}: adaptive quality".format(scene), adaptive["p95_position_rel_l2"] <= manifest["go_no_go"]["p95_position_rel_l2"], adaptive["p95_position_rel_l2"]))
        checks.append(("{0}: adaptive valid/no Armijo failures".format(scene), adaptive["invalid_trace_frames"] == 0 and adaptive["armijo_failures"] == 0, adaptive["armijo_failures"]))
    checks.append(("at least one 386 case improves end-to-end by >=5%", max(improvements) >= manifest["go_no_go"]["adaptive_end_to_end_win_one_case"], max(improvements)))
    checks.append(("other 386 case regresses by <=3%", min(improvements) >= -manifest["go_no_go"]["adaptive_end_to_end_regression_other_case"], min(improvements)))
    checks.append(("frame history reduces line-search time by >=5% in both 386 cases", min(history_reductions) >= manifest["go_no_go"]["history_line_search_reduction"], min(history_reductions)))
    checks.append(("adaptive is faster in >=2/3 repetitions for both 386 cases", min(directions) >= 2, min(directions)))
    return checks, all(check[1] for check in checks), improvements, history_reductions, directions


def plot(core, sensitivity, output_dir):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.55))
    core_index = {(row["scene_id"], row["cloth_dimension"], row["method_id"]): row for row in core}
    scenes = ("hanging", "moving-sphere")
    positions = [0, 1]
    width = 0.15
    for method_index, method in enumerate(CORE_METHODS):
        values = [core_index[(scene, 386, method)]["frame_wall_ms_mean"] for scene in scenes]
        errors = [core_index[(scene, 386, method)]["frame_wall_ms_std"] for scene in scenes]
        axes[0, 0].bar([value + (method_index - 2) * width for value in positions], values, width=width, yerr=errors, capsize=2, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
    axes[0, 0].set_xticks(positions, ["Hanging", "Moving sphere"])
    axes[0, 0].set_ylabel("386^2 rendered frame time (ms)")
    axes[0, 0].grid(axis="y", alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=6, ncol=2)
    for method in ("fixed-k8", "adaptive-k4-history-none", "adaptive-k4-history-frame"):
        values = [core_index[(scene, 386, method)]["line_search_ms"] for scene in scenes]
        axes[0, 1].plot(positions, values, marker="o", linewidth=1.5, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
    axes[0, 1].set_xticks(positions, ["Hanging", "Moving sphere"])
    axes[0, 1].set_ylabel("Trace line-search time (ms)")
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend(frameon=False, fontsize=6)
    for dimension, axis in zip((256, 386), axes[1, :]):
        subset = [row for row in sensitivity if row["cloth_dimension"] == dimension and row["armijo_beta"] == 0.5]
        for history in ("none", "iteration", "frame"):
            series = sorted([row for row in subset if row["adaptive_ls_history"] == history], key=lambda row: row["batched_ls_k"])
            axis.plot([row["batched_ls_k"] for row in series], [row["frame_wall_ms_mean"] for row in series], marker="o", linewidth=1.5, label=history)
        axis.set_xticks([2, 4, 8])
        axis.set_xlabel("K (beta=0.5)")
        axis.set_ylabel("Rendered frame time (ms)")
        axis.set_title("Moving sphere {0}^2".format(dimension), fontsize=8)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=6)
    fig.tight_layout(pad=0.55)
    fig.savefig(output_dir / "adaptive_armijo.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "adaptive_armijo.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(path, manifest, core, sensitivity, checks, go, improvements, history_reductions, directions):
    lines = [
        "# Adaptive Armijo Study",
        "",
        "## Protocol",
        "",
        "- Commit `{0}`; all timing is rendered 1600x900 with {1} warm-up + {2} measured frames and three repetitions.".format(manifest["git_commit"], manifest["timing"]["warmup"], manifest["timing"]["frames"]),
        "- A separate rendered 100-iteration CPU-NCG reference produces checkpoints for the 120-frame quality/decision traces. Timing excludes quality readback and decision tracing.",
        "- Core: serial persistent Armijo, fixed K=8 persistent, and adaptive K=4 persistent with no/iteration/frame history. Sensitivity: moving sphere at 256^2 and 386^2, K={2,4,8}, beta={0.25,0.5,0.75}, and three history modes.",
        "",
        "## Core results",
        "",
        "| Scene | Grid | Method | Frame ms | LS ms | Candidates/search | History use | P95 position error |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in core:
        lines.append("| {0} | {1} | {2} | {3:.3f} +/- {4:.3f} | {5:.3f} | {6:.3f} | {7:.3f} | {8:.2e} |".format(
            row["scene_id"], row["cloth_dimension"], METHOD_LABELS[row["method_id"]], row["frame_wall_ms_mean"], row["frame_wall_ms_std"], row["line_search_ms"], row["candidates_per_search"], row["history_use_ratio"], row["p95_position_rel_l2"]
        ))
    lines.extend(["", "## Go / no-go", ""])
    for name, passed, value in checks:
        lines.append("- [{0}] {1}: `{2:.6g}`".format("x" if passed else " ", name, value))
    lines.extend([
        "",
        "Result: **{0}**.".format("GO" if go else "NO-GO"),
        "- 386^2 end-to-end improvements (hanging, moving sphere): {0}.".format(", ".join("{0:.2%}".format(value) for value in improvements)),
        "- Frame-history line-search reductions (hanging, moving sphere): {0}.".format(", ".join("{0:.2%}".format(value) for value in history_reductions)),
        "- Faster adaptive repetitions (hanging, moving sphere): {0}/3, {1}/3.".format(directions[0], directions[1]),
        "",
        "The sensitivity CSV is `adaptive_armijo_sensitivity.csv`; Figure: `results/{0}/adaptive_armijo.pdf`.".format(manifest["label"]),
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
    if manifest.get("protocol_version") != "adaptive-armijo-paper-v1" or manifest.get("measurement", {}).get("mode") != "rendered-end-to-end":
        fail("Unexpected adaptive Armijo study manifest.")
    if manifest.get("timing", {}).get("repetitions") != 3 or len(manifest.get("core_methods", [])) != 5:
        fail("Incomplete adaptive Armijo protocol.")
    core_cases = load_csv(run_root / "planned_core_cases.csv")
    sensitivity_cases = load_csv(run_root / "planned_sensitivity_cases.csv")
    if len(core_cases) != 6 or len(sensitivity_cases) != 54:
        fail("Unexpected planned adaptive Armijo case counts.")
    core = read_core(run_root, manifest, core_cases)
    sensitivity = read_sensitivity(run_root, manifest, sensitivity_cases)
    checks, go, improvements, history_reductions, directions = build_go_no_go(core, manifest)
    write_csv(run_root / "adaptive_armijo_core_summary.csv", core)
    write_csv(run_root / "adaptive_armijo_sensitivity.csv", sensitivity)
    plot(core, sensitivity, run_root)
    write_report(Path(args.report_path).resolve(), manifest, core, sensitivity, checks, go, improvements, history_reductions, directions)
    print("Adaptive Armijo analysis complete: {0}; {1}".format(run_root, "GO" if go else "NO-GO"))


if __name__ == "__main__":
    main()
