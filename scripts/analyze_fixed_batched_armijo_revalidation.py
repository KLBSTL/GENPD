#!/usr/bin/env python3
"""Validate the current-commit fixed K=8 versus serial Armijo revalidation."""

import argparse
import csv
import json
import math
from pathlib import Path


EXPECTED = {
    "serial": ("serial", "gpu-gather-fusion-serial-ls-persistent", 1),
    "fixed-k8": ("fixed", "gpu-gather-fusion-batched-ls-persistent", 8),
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


def number(row, field, path):
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError):
        fail("Missing numeric field '{0}' in {1}".format(field, path))
    if not math.isfinite(value):
        fail("Non-finite field '{0}' in {1}".format(field, path))
    return value


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    manifest = load_json(run_root / "manifest.json")
    if manifest.get("protocol_version") != "fixed-batched-armijo-revalidation-v1":
        fail("Unexpected fixed-batched Armijo protocol.")
    if manifest.get("timing", {}).get("frames") != 300 or manifest.get("timing", {}).get("warmup") != 30 or manifest.get("timing", {}).get("repetitions") != 3:
        fail("Armijo revalidation must have three 30-warm-up + 300-measured rendered repetitions.")

    rows = []
    comparisons = []
    for case in manifest["cases"]:
        summary_path = run_root / "cases" / case["case_id"] / "line_search_summary.csv"
        summary = load_csv(summary_path)
        entries = {}
        for method_id, (schedule, variant, k) in EXPECTED.items():
            matches = [row for row in summary if row.get("schedule") == schedule and row.get("solver_variant") == variant and int(number(row, "batched_ls_k", summary_path)) == k]
            if len(matches) != 1:
                fail("Expected one {0} row in {1}".format(method_id, summary_path))
            source = matches[0]
            if int(number(source, "timing_repetitions", summary_path)) != 3 or int(number(source, "timing_frames_per_repetition", summary_path)) != 300:
                fail("Timing protocol mismatch in {0}".format(summary_path))
            if int(number(source, "invalid_trace_frames", summary_path)) != 0 or int(number(source, "armijo_failures", summary_path)) != 0:
                fail("Invalid or failed Armijo trace in {0}".format(summary_path))
            quality = number(source, "p95_position_rel_l2", summary_path)
            if quality > manifest["acceptance"]["p95_position_rel_l2"]:
                fail("Position-quality gate failed in {0}".format(summary_path))
            row = {
                "case_id": case["case_id"],
                "scene_id": case["scene_id"],
                "cloth_dimension": int(case["cloth_dimension"]),
                "iterations_per_frame": int(case["iterations_per_frame"]),
                "method": method_id,
                "solver_variant": variant,
                "rendered_frame_wall_ms_mean": number(source, "rendered_frame_wall_ms_mean", summary_path),
                "rendered_frame_wall_ms_std": number(source, "rendered_frame_wall_ms_std", summary_path),
                "rendered_frame_wall_ms_p95": number(source, "rendered_frame_wall_ms_p95", summary_path),
                "total_ms_mean": number(source, "total_ms_mean", summary_path),
                "line_search_ms": number(source, "line_search_ms", summary_path),
                "candidates_per_search": number(source, "candidates_per_search", summary_path),
                "armijo_rejections": number(source, "armijo_rejections", summary_path),
                "armijo_failures": number(source, "armijo_failures", summary_path),
                "p95_position_rel_l2": quality,
                "timing_repetitions": int(number(source, "timing_repetitions", summary_path)),
                "trace_frames": int(number(source, "trace_frame_samples", summary_path)),
                "source_summary": str(summary_path),
            }
            entries[method_id] = row
            rows.append(row)
        serial = entries["serial"]
        fixed = entries["fixed-k8"]
        comparisons.append({
            "case_id": case["case_id"],
            "scene_id": case["scene_id"],
            "cloth_dimension": int(case["cloth_dimension"]),
            "fixed_vs_serial_frame_speedup": serial["rendered_frame_wall_ms_mean"] / fixed["rendered_frame_wall_ms_mean"],
            "fixed_vs_serial_line_search_speedup": serial["line_search_ms"] / fixed["line_search_ms"],
            "serial_frame_ms": serial["rendered_frame_wall_ms_mean"],
            "fixed_frame_ms": fixed["rendered_frame_wall_ms_mean"],
            "serial_line_search_ms": serial["line_search_ms"],
            "fixed_line_search_ms": fixed["line_search_ms"],
        })

    write_csv(run_root / "fixed_batched_armijo_revalidation_summary.csv", rows)
    write_csv(run_root / "fixed_batched_armijo_revalidation_comparisons.csv", comparisons)
    report = [
        "# Fixed K=8 versus Serial Armijo Revalidation",
        "",
        "- Current commit `{0}`; rendered 1600x900 timing with 30 warm-up + 300 measured frames and three repetitions per method.".format(manifest["git_commit"]),
        "- Each case reuses its archived CPU-NCG reference checkpoints only for a separate 120-frame quality trace; quality readback is absent from timing.",
        "",
        "| Case | Method | Frame mean +/- std (ms) | P95 (ms) | Line-search ms | P95 position relative L2 | Armijo failures |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        report.append("| {0} | {1} | {2:.3f} +/- {3:.3f} | {4:.3f} | {5:.3f} | {6:.3e} | {7:.0f} |".format(
            row["case_id"], row["method"], row["rendered_frame_wall_ms_mean"], row["rendered_frame_wall_ms_std"], row["rendered_frame_wall_ms_p95"], row["line_search_ms"], row["p95_position_rel_l2"], row["armijo_failures"]))
    report += ["", "| Case | Fixed / serial frame speedup | Fixed / serial line-search speedup |", "| --- | ---: | ---: |"]
    for row in comparisons:
        report.append("| {0} | {1:.2f}x | {2:.2f}x |".format(row["case_id"], row["fixed_vs_serial_frame_speedup"], row["fixed_vs_serial_line_search_speedup"]))
    report += [
        "",
        "The revalidation is limited to the two larger meshes where line-search work is material. It tests fixed K=8, not the previously rejected adaptive history variants. Values are a same-quality internal comparison because both methods use the same case-specific CPU reference checkpoints and the registered P95 position gate.",
    ]
    (run_root / "fixed_batched_armijo_revalidation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("Fixed-batched Armijo revalidation summary: {0}".format(run_root / "fixed_batched_armijo_revalidation_summary.csv"))


if __name__ == "__main__":
    main()
