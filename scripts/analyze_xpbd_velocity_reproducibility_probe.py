#!/usr/bin/env python3
"""Validate a rendered XPBD resident/roundtrip velocity reproducibility probe."""

import argparse
import csv
import json
import math
import statistics
import struct
from pathlib import Path


CONDITIONS = ("resident-a", "resident-b", "forced-a", "forced-b")


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
        fail("Cannot take a percentile of no values.")
    values = sorted(values)
    index = max(0, min(len(values) - 1, int(math.ceil(fraction * len(values))) - 1))
    return values[index]


def median(values):
    if not values:
        fail("Cannot take a median of no values.")
    return statistics.median(values)


def measured(rows, warmup, path):
    selected = [row for row in rows if int(number(row, "frame", path)) >= warmup]
    if not selected:
        fail("No measured rows in {0}".format(path))
    return selected


def read_checkpoint(path):
    raw = path.read_bytes()
    if len(raw) < 24:
        fail("Truncated checkpoint: {0}".format(path))
    magic, version, frame, scalar_bytes, count = struct.unpack_from("<8sIIII", raw, 0)
    if magic != b"GPDQREF1" or version != 1 or scalar_bytes != 4 or count == 0:
        fail("Unsupported checkpoint format: {0}".format(path))
    expected_bytes = 24 + 2 * count * scalar_bytes
    if len(raw) != expected_bytes:
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


def checkpoints(run_dir):
    directory = run_dir / "reference_checkpoints"
    records = {}
    for path in sorted(directory.glob("reference_state_*.bin")):
        frame, positions, velocities = read_checkpoint(path)
        if frame in records:
            fail("Duplicate checkpoint frame in {0}".format(directory))
        records[frame] = (positions, velocities)
    if not records:
        fail("No checkpoints in {0}".format(directory))
    return records


def summarize_run(run_dir, scene, condition, manifest):
    metadata = load_json(run_dir / "run_metadata.json")
    benchmark = metadata.get("benchmark", {})
    if benchmark.get("no_render") or benchmark.get("hide_window") or not benchmark.get("sync_gpu") or not benchmark.get("disable_vsync"):
        fail("Run is not rendered, synchronized, and vsync-disabled: {0}".format(run_dir))
    if metadata.get("solver_variant") != manifest["solver_variant"]:
        fail("Unexpected solver variant: {0}".format(run_dir))
    if metadata.get("quality", {}).get("iterations_per_frame") != str(manifest["iterations_per_frame"]):
        fail("Iteration-budget mismatch: {0}".format(run_dir))
    if metadata.get("experiment_overrides", {}).get("cloth_dimension") != str(manifest["cloth_dimension"]):
        fail("Cloth-dimension mismatch: {0}".format(run_dir))
    expected_scene = Path(scene["scene_path"]).name.lower()
    actual_scene = Path(metadata.get("experiment_overrides", {}).get("scene", "")).name.lower()
    if actual_scene != expected_scene:
        fail("Scene mismatch: {0}".format(run_dir))
    expected_forced = "1" if condition.startswith("forced-") else "0"
    if metadata.get("solver_controls", {}).get("force_cpu_state_roundtrip") != expected_forced:
        fail("Roundtrip-condition mismatch: {0}".format(run_dir))

    warmup = manifest["trajectory"]["warmup"]
    frames = manifest["trajectory"]["frames"]
    profile_path = run_dir / "frame_profile.csv"
    extended_path = run_dir / "frame_profile_extended.csv"
    presentation_path = run_dir / "frame_presentation.csv"
    quality_path = run_dir / "quality_metrics.csv"
    profile = measured(load_csv(profile_path), warmup, profile_path)
    extended = measured(load_csv(extended_path), warmup, extended_path)
    presentation = measured(load_csv(presentation_path), warmup, presentation_path)
    quality = measured(load_csv(quality_path), warmup, quality_path)
    if any(len(rows) != frames for rows in (profile, extended, presentation, quality)):
        fail("Incomplete measured frame count: {0}".format(run_dir))
    if any(row.get("frame_valid") != "1" or row.get("termination_reason") != "none" for row in extended):
        fail("Invalid solver frame: {0}".format(run_dir))
    if any(row.get("rendered") != "1" or row.get("gpu_sync_enabled") != "1" for row in presentation):
        fail("Unrendered measured frame: {0}".format(run_dir))
    if any(row.get("finite") != "1" or row.get("exploded") != "0" for row in quality):
        fail("Invalid quality frame: {0}".format(run_dir))
    if any(int(number(row, "iterations", profile_path)) != manifest["iterations_per_frame"] for row in profile):
        fail("Unexpected XPBD iteration count: {0}".format(run_dir))

    records = checkpoints(run_dir)
    measured_records = {frame: state for frame, state in records.items() if frame >= warmup}
    if len(measured_records) < manifest["acceptance"]["min_matched_measured_checkpoints"]:
        fail("Too few measured checkpoints: {0}".format(run_dir))
    return {
        "scene_id": scene["id"],
        "condition": condition,
        "force_cpu_state_roundtrip": int(expected_forced),
        "measured_frames": frames,
        "measured_checkpoints": len(measured_records),
        "p95_mean_stretch_strain": percentile([number(row, "mean_stretch_strain", quality_path) for row in quality]),
        "p95_max_stretch_strain": percentile([number(row, "max_stretch_strain", quality_path) for row in quality]),
        "max_penetration_depth": max(number(row, "max_penetration_depth", quality_path) for row in quality),
        "git_commit": metadata.get("git_commit", ""),
        "gpu_name": metadata.get("gpu_name", ""),
        "driver": metadata.get("nvidia_driver_version", ""),
        "result_dir": str(run_dir),
        "checkpoint_records": measured_records,
    }


def compare_pair(scene_id, pair_type, left_name, right_name, left, right, minimum):
    shared = sorted(set(left["checkpoint_records"]).intersection(right["checkpoint_records"]))
    if len(shared) < minimum:
        fail("Fewer than {0} matching checkpoints for {1}:{2}".format(minimum, left_name, right_name))
    rows = []
    for frame in shared:
        left_position, left_velocity = left["checkpoint_records"][frame]
        right_position, right_velocity = right["checkpoint_records"][frame]
        rows.append({
            "scene_id": scene_id,
            "pair_type": pair_type,
            "pair": left_name + ":" + right_name,
            "frame": frame,
            "position_rel_l2": relative_l2(right_position, left_position),
            "velocity_rel_l2": relative_l2(right_velocity, left_velocity),
        })
    return rows


def write_csv(path, rows):
    if not rows:
        fail("Refusing to write an empty CSV: {0}".format(path))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pair_summary(rows):
    return {
        "matched_checkpoints": len(rows),
        "position_rel_l2_p50": percentile([row["position_rel_l2"] for row in rows], 0.50),
        "position_rel_l2_p95": percentile([row["position_rel_l2"] for row in rows]),
        "position_rel_l2_max": max(row["position_rel_l2"] for row in rows),
        "velocity_rel_l2_p50": percentile([row["velocity_rel_l2"] for row in rows], 0.50),
        "velocity_rel_l2_p95": percentile([row["velocity_rel_l2"] for row in rows]),
        "velocity_rel_l2_max": max(row["velocity_rel_l2"] for row in rows),
    }


def interpretation(scene_id, summaries):
    repeat = [row["velocity_rel_l2_p95"] for row in summaries if row["scene_id"] == scene_id and row["pair_type"] == "within-condition-repeat"]
    cross = [row["velocity_rel_l2_p95"] for row in summaries if row["scene_id"] == scene_id and row["pair_type"] == "across-condition"]
    repeat_scale = max(repeat)
    cross_scale = median(cross)
    if cross_scale > max(4.0 * repeat_scale, 1.0e-7):
        verdict = "condition-sensitive"
    elif cross_scale <= max(2.0 * repeat_scale, 1.0e-7):
        verdict = "repeat-scale"
    else:
        verdict = "mixed"
    return repeat_scale, cross_scale, verdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    manifest = load_json(run_root / "manifest.json")
    if manifest.get("protocol_version") != "xpbd-velocity-reproducibility-probe-v1":
        fail("Unexpected XPBD velocity probe protocol.")
    if manifest.get("measurement", {}).get("mode") != "rendered-trajectory-diagnostic" or manifest.get("measurement", {}).get("timing_claim") != "none":
        fail("Velocity probe must remain a rendered trajectory diagnostic, not a timing study.")
    if [condition.get("id") for condition in manifest.get("conditions", [])] != list(CONDITIONS):
        fail("Unexpected condition order or set.")
    if [scene.get("id") for scene in manifest.get("scenes", [])] != ["hanging", "moving-sphere"]:
        fail("Unexpected scene order or set.")

    runs = {}
    run_rows = []
    for scene in manifest["scenes"]:
        runs[scene["id"]] = {}
        for condition in CONDITIONS:
            run_dir = run_root / "runs" / scene["id"] / condition
            record = summarize_run(run_dir, scene, condition, manifest)
            runs[scene["id"]][condition] = record
            run_rows.append({key: value for key, value in record.items() if key != "checkpoint_records"})

    expected_commit = manifest.get("git_commit", "")
    metadata = {(row["git_commit"], row["gpu_name"], row["driver"]) for row in run_rows}
    if len(metadata) != 1 or not expected_commit or next(iter(metadata))[0] != expected_commit:
        fail("Probe runs do not share the manifest commit, GPU, and driver.")

    definition = manifest["comparison_definition"]
    pair_sets = (("within-condition-repeat", definition["repeat_pairs"]), ("across-condition", definition["cross_condition_pairs"]))
    pair_rows = []
    summary_rows = []
    for scene in manifest["scenes"]:
        for pair_type, pairs in pair_sets:
            for encoded in pairs:
                left_name, right_name = encoded.split(":", 1)
                rows = compare_pair(
                    scene["id"], pair_type, left_name, right_name, runs[scene["id"]][left_name],
                    runs[scene["id"]][right_name], manifest["acceptance"]["min_matched_measured_checkpoints"])
                pair_rows.extend(rows)
                summary = {"scene_id": scene["id"], "pair_type": pair_type, "pair": encoded}
                summary.update(pair_summary(rows))
                summary_rows.append(summary)

    write_csv(run_root / "run_quality_summary.csv", run_rows)
    write_csv(run_root / "trajectory_pair_errors.csv", pair_rows)
    write_csv(run_root / "reproducibility_summary.csv", summary_rows)

    report = [
        "# XPBD 128x128 Velocity Reproducibility Probe",
        "",
        "## Scope",
        "",
        "This is a rendered trajectory diagnostic, not a performance benchmark. Every run uses atomic `gpu-xpbd-jacobi`, 32 iterations/frame, a 1600x900 rendered viewport, GPU synchronization, disabled VSync, 20 warm-up frames, and 120 measured frames. State checkpoints are exported every 10 frames and are excluded from any timing claim.",
        "",
        "The design distinguishes two sources of variation:",
        "",
        "- **Within-condition repeats:** `resident-a:resident-b` and `forced-a:forced-b` measure ordinary run-to-run variation with the same state-management condition.",
        "- **Across-condition pairs:** all four resident/forced combinations measure the effect associated with crossing the finalized position/velocity state through CPU between frames.",
        "- **Scene control:** `hanging` contains no collision primitives; `moving-sphere` includes the moving sphere and plane. A much larger moving-sphere cross-condition result implicates collision/contact amplification, but does not prove one individual shader instruction is at fault.",
        "",
        "## Checkpoint Comparison",
        "",
        "| Scene | Pair type | Pair | Matched checkpoints | Position P95 relative L2 | Velocity P95 relative L2 |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        report.append("| {0} | {1} | {2} | {3} | {4:.3e} | {5:.3e} |".format(
            row["scene_id"], row["pair_type"], row["pair"], row["matched_checkpoints"],
            row["position_rel_l2_p95"], row["velocity_rel_l2_p95"]))

    report += ["", "## Quality Sanity", "", "| Scene | Condition | P95 mean stretch | P95 max stretch | Maximum penetration |", "| --- | --- | ---: | ---: | ---: |"]
    for row in run_rows:
        report.append("| {0} | {1} | {2:.6f} | {3:.6f} | {4:.6f} |".format(
            row["scene_id"], row["condition"], row["p95_mean_stretch_strain"],
            row["p95_max_stretch_strain"], row["max_penetration_depth"]))

    report += ["", "## Interpretation", ""]
    for scene_id in ("hanging", "moving-sphere"):
        repeat_scale, cross_scale, verdict = interpretation(scene_id, summary_rows)
        report.append("- `{0}`: largest within-condition velocity P95 is `{1:.3e}`; median across-condition velocity P95 is `{2:.3e}`; classification: **{3}**.".format(
            scene_id, repeat_scale, cross_scale, verdict))
    hanging_repeat, hanging_cross, hanging_verdict = interpretation("hanging", summary_rows)
    moving_repeat, moving_cross, moving_verdict = interpretation("moving-sphere", summary_rows)
    if moving_cross > max(4.0 * hanging_cross, 1.0e-7):
        report.append("- The moving-sphere cross-condition scale is substantially larger than the hanging control. This supports contact-associated amplification of a small residency-condition perturbation.")
    elif moving_cross <= max(2.0 * hanging_cross, 1.0e-7):
        report.append("- The moving-sphere cross-condition scale is comparable to the hanging control. This probe does not isolate moving contact as the main amplifier.")
    else:
        report.append("- The moving-sphere and hanging cross-condition scales differ, but not enough for a clean contact-amplification attribution in this small probe.")
    report += [
        "- A `condition-sensitive` result is evidence against calling the 128x128 value a single-run accident. It still does not prove transfer corruption: the counterfactual preserves scalar state values but changes host/device crossings and next-frame buffer upload timing. XPBD also uses floating-point atomic accumulation, and moving collision introduces discontinuous projection branches; both can amplify a small state perturbation over many frames.",
        "- A `repeat-scale` result instead means the observed magnitude is comparable to ordinary run-to-run variation, so the prior resident/forced velocity discrepancy cannot be attributed confidently to the roundtrip condition alone.",
        "",
        "## Evidence Boundary",
        "",
        "This probe establishes only the source scale of the observed 128x128 velocity difference. It does not establish bitwise deterministic XPBD, an equal-quality performance ranking, or a general resolution-independent conclusion. The existing formal R3 study remains the source for rendered 300-frame residency timing and state-traffic measurements.",
        "",
        "## Artifacts",
        "",
        "- `manifest.json`: exact protocol, commit, GPU, and driver.",
        "- `planned_runs.csv`: the eight rendered trajectory runs.",
        "- `run_quality_summary.csv`: finite-state and physical-quality checks for every run.",
        "- `trajectory_pair_errors.csv`: all matched checkpoint errors.",
        "- `reproducibility_summary.csv`: compact per-pair P50/P95/max metrics.",
        "- `runs/<scene>/<condition>/`: raw CSVs, metadata, logs, and checkpoint binaries.",
    ]
    (run_root / "xpbd_velocity_reproducibility_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("XPBD velocity reproducibility summary: {0}".format(run_root / "reproducibility_summary.csv"))


if __name__ == "__main__":
    main()
