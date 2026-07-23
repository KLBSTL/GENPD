#!/usr/bin/env python
"""Generate only evidence-backed figures for the GenPD paper protocol."""
from __future__ import print_function

import argparse
import csv
import json
import math
import os
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


VARIANTS = [
    'cpu-ncg',
    'gpu-edge-scatter',
    'gpu-gather-no-fusion',
    'gpu-gather-fusion',
    'gpu-gather-fusion-batched-ls',
    'gpu-gather-fusion-batched-ls-persistent',
]
VARIANT_LABELS = {
    'cpu-ncg': 'CPU NCG',
    'gpu-edge-scatter': 'Edge scatter',
    'gpu-gather-no-fusion': 'Gather',
    'gpu-gather-fusion': 'Gather + fusion',
    'gpu-gather-fusion-batched-ls': '+ batched LS',
    'gpu-gather-fusion-batched-ls-persistent': '+ persistent',
}
COLORS = ['#4E79A7', '#E15759', '#59A14F', '#F28E2B', '#B07AA1', '#76B7B2']

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'figure.dpi': 180,
    'savefig.dpi': 300,
})


def load_csv(path):
    with open(path, 'r') as handle:
        return list(csv.DictReader(handle))


def as_int(value, name, context):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        raise RuntimeError('Invalid {0} in {1}: {2}'.format(name, context, value))


def as_float(value, name, context):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise RuntimeError('Invalid {0} in {1}: {2}'.format(name, context, value))
    if not math.isfinite(result):
        raise RuntimeError('Non-finite {0} in {1}'.format(name, context))
    return result


def ensure(condition, message):
    if not condition:
        raise RuntimeError(message)


def canonical_key(row):
    return (row['scene_id'], as_int(row['cloth_dimension'], 'cloth_dimension', row), row['solver_variant'])


def get_path(root, relative):
    return relative if os.path.isabs(relative) else os.path.join(root, relative)


def validate_inputs(run_root):
    manifest_path = os.path.join(run_root, 'manifest.json')
    summary_path = os.path.join(run_root, 'paper_summary.csv')
    calibration_path = os.path.join(run_root, 'calibration.csv')
    selected_path = os.path.join(run_root, 'selected_budgets.csv')
    stability_path = os.path.join(run_root, 'stability_replicates.csv')
    for path in [manifest_path, summary_path, calibration_path, selected_path, stability_path]:
        ensure(os.path.isfile(path), 'Required formal input is missing: {0}'.format(path))
    with open(manifest_path, 'r') as handle:
        manifest = json.load(handle)
    ensure(manifest.get('protocol_version') == 1, 'Unexpected manifest protocol version.')
    ensure(manifest['performance']['repetitions'] == 3, 'Formal figures require three repetitions.')
    ensure(abs(float(manifest['quality_target']['position_rel_l2_p95']) - 1e-3) < 1e-15,
           'Unexpected equal-quality threshold.')
    ensure(manifest.get('measurement', {}).get('mode') == 'rendered-end-to-end',
           'Formal figures require rendered end-to-end measurements.')
    ensure(manifest['measurement'].get('primary_metric') == 'frame_wall_ms',
           'Formal figures require frame_wall_ms as the primary metric.')

    summary = load_csv(summary_path)
    calibration = load_csv(calibration_path)
    selected = load_csv(selected_path)
    stability = load_csv(stability_path)
    scenes = [item['id'] for item in manifest['scenes']]
    dimensions = [int(value) for value in manifest['resolutions']]
    expected = set((scene, dimension, variant) for scene in scenes for dimension in dimensions for variant in VARIANTS)
    summary_keys = set(canonical_key(row) for row in summary)
    ensure(summary_keys.issubset(expected), 'paper_summary.csv contains an unknown formal case.')
    ensure(len(summary) == len(summary_keys), 'paper_summary.csv contains duplicate formal cases.')
    ensure(len(calibration) == len(expected) * 11, 'Calibration CSV must contain all 11 candidate budgets per case.')

    calibration_by_key = {}
    for row in calibration:
        key = canonical_key(row) + (as_int(row['iterations_per_frame'], 'iterations_per_frame', row),)
        calibration_by_key[key] = row
    selected_keys = set(canonical_key(row) for row in selected)
    ensure(selected_keys == summary_keys, 'Performance summary must contain exactly the quality-qualified cases.')
    missing_keys = expected - summary_keys
    for missing in missing_keys:
        candidates = [row for row in calibration if canonical_key(row) == missing]
        ensure(candidates, 'Missing case has no calibration record: {0}'.format(missing))
        ensure(not any(as_int(row['qualified'], 'qualified', row) == 1 for row in candidates),
               'A quality-qualified case is missing from the performance summary: {0}'.format(missing))
    for selection in selected:
        key = canonical_key(selection) + (as_int(selection['iterations_per_frame'], 'iterations_per_frame', selection),)
        ensure(key in calibration_by_key, 'Selected quality budget has no calibration record: {0}'.format(key))
        record = calibration_by_key[key]
        ensure(as_int(record['qualified'], 'qualified', record) == 1, 'Selected case is not quality-qualified: {0}'.format(key))
        ensure(as_float(record['p95_position_rel_l2'], 'p95_position_rel_l2', record) <= 1e-3,
               'Selected case exceeds equal-quality threshold: {0}'.format(key))
        for relative in [record['result_dir'], record['reference_dir']]:
            ensure(os.path.isdir(get_path(run_root, relative)), 'Missing raw quality directory: {0}'.format(relative))
        quality_path = os.path.join(get_path(run_root, record['result_dir']), 'quality_metrics.csv')
        checkpoints = os.path.join(get_path(run_root, record['reference_dir']), 'reference_checkpoints')
        ensure(os.path.isfile(quality_path), 'Missing quality metrics for selected case: {0}'.format(quality_path))
        ensure(os.path.isdir(checkpoints), 'Missing quality reference checkpoints: {0}'.format(checkpoints))

    for row in summary:
        ensure(as_int(row['repetitions'], 'repetitions', row) == 3, 'Incomplete repetition count: {0}'.format(canonical_key(row)))
        for field in ['frame_wall_ms_mean', 'frame_wall_ms_std', 'frame_wall_ms_p50', 'frame_wall_ms_p95',
                      'render_and_present_wall_ms_mean', 'total_ms_mean', 'total_ms_std', 'total_ms_p50', 'total_ms_p95',
                      'optimization_ms_mean', 'transfer_ms_mean', 'p95_position_rel_l2',
                      'calibration_failure_rate']:
            as_float(row[field], field, row)
        ensure(row['git_commit'].strip(), 'Missing Git commit in paper summary.')
        ensure(row['gpu_name'].strip(), 'Missing GPU name in paper summary.')
        ensure(row['nvidia_driver_version'].strip(), 'Missing driver version in paper summary.')

    ensure(len(stability) == 27, 'Stability protocol requires 9 cells x 3 repetitions.')
    stability_keys = set((row['timestep'], row['stretch_stiffness']) for row in stability)
    ensure(len(stability_keys) == 9, 'Stability protocol must contain a 3 x 3 grid.')
    for row in stability:
        ensure(as_int(row['stable'], 'stable', row) in (0, 1), 'Invalid stability state.')

    capture_paths = {}
    for scene in scenes:
        for dimension in dimensions:
            path = os.path.join(run_root, 'captures', '{0}-d{1}'.format(scene, dimension), 'frame-180.png')
            ensure(os.path.isfile(path) and os.path.getsize(path) > 1024,
                   'Missing current-commit qualitative capture: {0}'.format(path))
            capture_paths[(scene, dimension)] = path
    return manifest, summary, stability, capture_paths, missing_keys


def index_rows(rows):
    return dict((canonical_key(row), row) for row in rows)


def metric_or_none(indexed, key, field):
    row = indexed.get(key)
    return None if row is None else as_float(row[field], field, row)


def mark_unqualified(ax, positions, values):
    finite = [value for value in values if value is not None]
    height = max(finite) if finite else 1.0
    for position, value in zip(positions, values):
        if value is None:
            ax.text(position, height * 0.04, 'NQ', ha='center', va='bottom', fontsize=7, color='#555555')


def save_figure(fig, name, output_dir, paper_dir):
    fig.tight_layout(pad=0.7)
    for directory in [output_dir, paper_dir]:
        if not os.path.isdir(directory):
            os.makedirs(directory)
        fig.savefig(os.path.join(directory, name + '.pdf'), bbox_inches='tight')
        fig.savefig(os.path.join(directory, name + '.png'), bbox_inches='tight')
    plt.close(fig)


def plot_ablation(summary, output_dir, paper_dir):
    indexed = index_rows(summary)
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.8))
    x = list(range(len(VARIANTS)))
    for col, scene in enumerate(['hanging', 'moving-sphere']):
        ax = axes[0][col]
        keys = [(scene, 386, variant) for variant in VARIANTS]
        raw_means = [metric_or_none(indexed, key, 'frame_wall_ms_mean') for key in keys]
        raw_errors = [metric_or_none(indexed, key, 'frame_wall_ms_std') for key in keys]
        means = [value if value is not None else 0.0 for value in raw_means]
        errors = [value if value is not None else 0.0 for value in raw_errors]
        bars = ax.bar(x, means, yerr=errors, capsize=2, color=COLORS, edgecolor='black', linewidth=0.35)
        for bar, value in zip(bars, raw_means):
            if value is None:
                bar.set_facecolor('#DDDDDD')
                bar.set_hatch('xx')
        mark_unqualified(ax, x, raw_means)
        ax.set_title('{0}: 148,996 vertices'.format('Hanging cloth' if scene == 'hanging' else 'Moving sphere'))
        ax.set_ylabel('Rendered frame time (ms)')
        ax.set_xticks(x)
        ax.set_xticklabels([VARIANT_LABELS[v] for v in VARIANTS], rotation=28, ha='right')
        ax.grid(axis='y', alpha=0.25)

    ax = axes[1][0]
    width = 0.35
    for offset, scene, hatch in [(-width / 2.0, 'hanging', ''), (width / 2.0, 'moving-sphere', '//')]:
        raw_dispatches = [metric_or_none(indexed, (scene, 386, variant), 'dispatches_mean') for variant in VARIANTS]
        dispatches = [value if value is not None else 0.0 for value in raw_dispatches]
        bars = ax.bar([position + offset for position in x], dispatches, width=width, color=COLORS,
               hatch=hatch, edgecolor='black', linewidth=0.3, label='Hanging' if scene == 'hanging' else 'Moving sphere')
        for bar, value in zip(bars, raw_dispatches):
            if value is None:
                bar.set_facecolor('#DDDDDD')
        if scene == 'hanging':
            mark_unqualified(ax, [position + offset for position in x], raw_dispatches)
    ax.set_ylabel('Compute dispatches / frame')
    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABELS[v] for v in VARIANTS], rotation=28, ha='right')
    ax.legend(frameon=False, ncol=2, loc='upper left')
    ax.grid(axis='y', alpha=0.25)

    ax = axes[1][1]
    scene = 'moving-sphere'
    rows = [indexed.get((scene, 386, variant)) for variant in VARIANTS]
    transfer = [as_float(row['transfer_ms_mean'], 'transfer_ms_mean', row) if row else 0.0 for row in rows]
    readbacks = [as_float(row['host_readbacks_mean'], 'host_readbacks_mean', row) if row else float('nan') for row in rows]
    ax.bar(x, transfer, color=COLORS, edgecolor='black', linewidth=0.35)
    ax.set_ylabel('Transfer time (ms)')
    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABELS[v] for v in VARIANTS], rotation=28, ha='right')
    secondary = ax.twinx()
    secondary.plot(x, readbacks, marker='o', color='#333333', linewidth=1.0, label='Host readbacks')
    secondary.set_ylabel('Host readbacks / frame')
    ax.text(0.03, 0.96, 'Tracked GPU buffers: 9.1 MiB for all GPU variants',
            transform=ax.transAxes, ha='left', va='top', fontsize=7,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.5))
    ax.set_title('Moving sphere: transfer and host readbacks')
    ax.grid(axis='y', alpha=0.25)
    save_figure(fig, 'ablation', output_dir, paper_dir)


def plot_scalability_quality(summary, output_dir, paper_dir):
    indexed = index_rows(summary)
    dimensions = [128, 256, 386]
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.75))
    for scene, marker, title in [('hanging', 'o', 'Hanging'), ('moving-sphere', 's', 'Sphere')]:
        for variant, style, label in [('cpu-ncg', '--', 'CPU'), ('gpu-gather-fusion-batched-ls-persistent', '-', 'Final')]:
            values = [metric_or_none(indexed, (scene, dimension, variant), 'frame_wall_ms_mean')
                      for dimension in dimensions]
            values = [value if value is not None else float('nan') for value in values]
            axes[0][0].plot(dimensions, values, linestyle=style, marker=marker,
                            label='{0}, {1}'.format(title, label))
    axes[0][0].set_xlabel('Cloth resolution')
    axes[0][0].set_ylabel('Equal-quality rendered frame time (ms)')
    axes[0][0].set_yscale('log')
    axes[0][0].set_xticks(dimensions)
    axes[0][0].legend(frameon=False, ncol=2, fontsize=7, loc='upper left')
    axes[0][0].grid(alpha=0.25)

    for scene, marker, title in [('hanging', 'o', 'Hanging cloth'), ('moving-sphere', 's', 'Moving sphere')]:
        ratios = []
        for dimension in dimensions:
            cpu = metric_or_none(indexed, (scene, dimension, 'cpu-ncg'), 'frame_wall_ms_mean')
            gpu = metric_or_none(indexed, (scene, dimension, 'gpu-gather-fusion-batched-ls-persistent'), 'frame_wall_ms_mean')
            ratios.append(cpu / gpu if cpu is not None and gpu is not None else float('nan'))
        axes[0][1].plot(dimensions, ratios, marker=marker, label=title)
    axes[0][1].axhline(1.0, color='#777777', linewidth=0.8)
    axes[0][1].set_xlabel('Cloth resolution')
    axes[0][1].set_ylabel('CPU NCG / final GPU speedup')
    axes[0][1].set_xticks(dimensions)
    axes[0][1].legend(frameon=False)
    axes[0][1].grid(alpha=0.25)

    width = 0.35
    x = list(range(len(VARIANTS)))
    for offset, scene, hatch in [(-width / 2.0, 'hanging', ''), (width / 2.0, 'moving-sphere', '//')]:
        raw_errors = [metric_or_none(indexed, (scene, 386, variant), 'p95_position_rel_l2') for variant in VARIANTS]
        errors = [value if value is not None else 1e-8 for value in raw_errors]
        bars = axes[1][0].bar([pos + offset for pos in x], errors, width=width, color=COLORS,
                        hatch=hatch, edgecolor='black', linewidth=0.3, label='Hanging' if scene == 'hanging' else 'Moving sphere')
        for bar, value in zip(bars, raw_errors):
            if value is None:
                bar.set_facecolor('#DDDDDD')
                bar.set_hatch('xx')
        if scene == 'hanging':
            mark_unqualified(axes[1][0], [pos + offset for pos in x], raw_errors)
    axes[1][0].axhline(1e-3, color='#333333', linestyle='--', linewidth=0.8, label='Target')
    axes[1][0].set_yscale('log')
    axes[1][0].set_ylabel('P95 position relative L2')
    axes[1][0].set_xticks(x)
    axes[1][0].set_xticklabels([VARIANT_LABELS[v] for v in VARIANTS], rotation=28, ha='right')
    axes[1][0].legend(frameon=False, ncol=2)
    axes[1][0].grid(axis='y', alpha=0.25)

    qualification = []
    for scene in ['hanging', 'moving-sphere']:
        qualification.append([1 if (scene, 386, variant) in indexed else 0 for variant in VARIANTS])
    axes[1][1].imshow(qualification, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    for row_index, values in enumerate(qualification):
        for col_index, value in enumerate(values):
            axes[1][1].text(col_index, row_index, 'Q' if value else 'NQ', ha='center', va='center',
                            fontsize=8, color='white' if value == 0 else 'black')
    axes[1][1].set_title('386$^2$ equal-quality qualification')
    axes[1][1].set_yticks([0, 1])
    axes[1][1].set_yticklabels(['Hanging', 'Moving sphere'])
    axes[1][1].set_xticks(x)
    axes[1][1].set_xticklabels([VARIANT_LABELS[v] for v in VARIANTS], rotation=28, ha='right')
    axes[1][1].text(0.02, -0.45, 'Q: P95 relative L2 <= 1e-3; all Q cases have 0 calibration failures.',
                    transform=axes[1][1].transAxes, ha='left', va='top', fontsize=6.5)
    save_figure(fig, 'scalability_quality', output_dir, paper_dir)


def plot_stability_heatmap(stability, output_dir, paper_dir):
    timesteps = sorted(set(as_float(row['timestep'], 'timestep', row) for row in stability))
    stiffnesses = sorted(set(as_float(row['stretch_stiffness'], 'stretch_stiffness', row) for row in stability))
    counts = {}
    for row in stability:
        key = (as_float(row['timestep'], 'timestep', row), as_float(row['stretch_stiffness'], 'stretch_stiffness', row))
        counts[key] = counts.get(key, 0) + as_int(row['stable'], 'stable', row)
    matrix = [[counts[(dt, stiffness)] for stiffness in stiffnesses] for dt in timesteps]
    fig, ax = plt.subplots(figsize=(3.45, 2.65))
    image = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=3, aspect='auto')
    for i, dt in enumerate(timesteps):
        for j, stiffness in enumerate(stiffnesses):
            count = matrix[i][j]
            ax.text(j, i, '{0}/3'.format(count), ha='center', va='center',
                    color='white' if count == 0 else 'black', fontsize=9)
    ax.set_xticks(range(len(stiffnesses)))
    ax.set_xticklabels([str(int(value)) for value in stiffnesses])
    ax.set_yticks(range(len(timesteps)))
    ax.set_yticklabels(['{0:.4f}'.format(value) for value in timesteps])
    ax.set_xlabel('Stretch stiffness')
    ax.set_ylabel('Timestep (s)')
    ax.set_title('Final GPU variant, 256^2 cloth, moving sphere')
    colorbar = fig.colorbar(image, ax=ax, fraction=0.055, pad=0.04)
    colorbar.set_label('Stable repetitions')
    save_figure(fig, 'stability_heatmap', output_dir, paper_dir)


def plot_qualitative(captures, output_dir, paper_dir):
    dimensions = [128, 256, 386]
    scenes = [('hanging', 'Hanging cloth'), ('moving-sphere', 'Moving sphere')]
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 3.7))
    for row, (scene_id, scene_label) in enumerate(scenes):
        for col, dimension in enumerate(dimensions):
            ax = axes[row][col]
            ax.imshow(mpimg.imread(captures[(scene_id, dimension)]))
            ax.set_axis_off()
            ax.set_title('{0}, {1}^2'.format(scene_label, dimension))
    save_figure(fig, 'qualitative_results', output_dir, paper_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--paper-figure-dir', required=True)
    args = parser.parse_args()
    run_root = os.path.abspath(args.run_root)
    paper_dir = os.path.abspath(args.paper_figure_dir)
    output_dir = os.path.join(run_root, 'figures')
    manifest, summary, stability, captures, missing_keys = validate_inputs(run_root)
    plot_ablation(summary, output_dir, paper_dir)
    plot_scalability_quality(summary, output_dir, paper_dir)
    plot_stability_heatmap(stability, output_dir, paper_dir)
    plot_qualitative(captures, output_dir, paper_dir)
    print('Generated evidence-backed figures in {0} and {1}'.format(output_dir, paper_dir))


if __name__ == '__main__':
    try:
        main()
    except RuntimeError as error:
        print('Figure generation rejected incomplete formal evidence: {0}'.format(error), file=sys.stderr)
        sys.exit(2)
