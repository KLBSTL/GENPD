"""Generate evidence-gated scene and material matrix figures for the manuscript."""

import argparse
import csv
import json
import math
import os

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D


SCENE_COLORS = {'hanging': '#4E79A7', 'moving-sphere': '#E15759'}
MESH_MARKERS = {'square-256x256': 'o', 'rect-512x128': 's'}

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


def fail(message):
    raise RuntimeError(message)


def load_json(path):
    with open(path, 'r') as handle:
        return json.load(handle)


def load_csv(path):
    with open(path, 'r', newline='') as handle:
        return list(csv.DictReader(handle))


def as_int(value, name, row):
    try:
        return int(value)
    except (TypeError, ValueError):
        fail('Invalid {0} in {1}'.format(name, row))


def as_float(value, name, row):
    try:
        result = float(value)
    except (TypeError, ValueError):
        fail('Invalid {0} in {1}'.format(name, row))
    if not math.isfinite(result):
        fail('Non-finite {0} in {1}'.format(name, row))
    return result


def as_bool(value):
    return str(value).strip().lower() in ('1', 'true')


def row_key(row):
    return (
        row['scene_id'], row['mesh_id'], as_int(row['stretch_stiffness'], 'stretch_stiffness', row),
        as_int(row['bending_stiffness'], 'bending_stiffness', row), row['solver_variant'],
    )


def expected_keys(manifest):
    return set(
        (scene['id'], mesh['id'], int(stretch), int(bending), variant)
        for scene in manifest['scenes']
        for mesh in manifest['meshes']
        for stretch in manifest['stretch_stiffnesses']
        for bending in manifest['bending_stiffnesses']
        for variant in manifest['solver_variants']
    )


def validate_metadata(path, require_quality):
    metadata_path = os.path.join(path, 'run_metadata.json')
    if not os.path.isfile(metadata_path):
        fail('Missing run metadata: {0}'.format(path))
    metadata = load_json(metadata_path)
    if metadata['benchmark']['no_render'] or not metadata['benchmark']['sync_gpu']:
        fail('Run was not rendered with GPU synchronisation: {0}'.format(path))
    if require_quality and not os.path.isfile(os.path.join(path, 'quality_metrics.csv')):
        fail('Missing quality metrics: {0}'.format(path))
    return metadata


def validate_run(run_root):
    manifest_path = os.path.join(run_root, 'manifest.json')
    summary_path = os.path.join(run_root, 'scene_material_summary.csv')
    if not os.path.isfile(manifest_path) or not os.path.isfile(summary_path):
        fail('Formal scene/material manifest or summary is missing.')
    manifest = load_json(manifest_path)
    if manifest.get('protocol_version') != 2 or manifest.get('measurement') != 'rendered-end-to-end':
        fail('Scene/material figures require protocol 2 rendered evidence.')
    reference = manifest.get('reference', {})
    quality = manifest.get('quality', {})
    timing = manifest.get('timing', {})
    if reference != {'frames': 120, 'warmup': 20, 'iterations': 100}:
        fail('Unexpected CPU reference protocol.')
    if quality.get('frames') != 120 or quality.get('warmup') != 20 or quality.get('timing_quality_metrics'):
        fail('Unexpected quality protocol.')
    if timing.get('frames') != 300 or timing.get('warmup') != 30 or timing.get('repetitions') != 3:
        fail('Unexpected timing protocol.')
    if [scene['id'] for scene in manifest['scenes']] != ['hanging', 'moving-sphere']:
        fail('Formal scene matrix is incomplete.')
    if [(mesh['id'], mesh['width'], mesh['height']) for mesh in manifest['meshes']] != [
            ('square-256x256', 256, 256), ('rect-512x128', 512, 128)]:
        fail('Formal mesh matrix is incomplete.')
    if manifest.get('stretch_stiffnesses') != [40, 80, 160] or manifest.get('bending_stiffnesses') != [10, 20, 40]:
        fail('Formal material matrix is incomplete.')
    if manifest.get('solver_variants') != ['gpu-gather-fusion-batched-ls-persistent']:
        fail('Scene/material matrix must evaluate the final persistent variant only.')

    rows = load_csv(summary_path)
    index = {}
    fields = [
        'rendered_frame_wall_ms_mean', 'rendered_frame_wall_ms_std', 'rendered_frame_wall_ms_p95',
        'total_ms_mean', 'total_ms_std', 'optimization_ms_mean', 'optimization_ms_std',
        'transfer_ms_mean', 'transfer_ms_std', 'dispatches_mean', 'dispatches_std',
        'host_readbacks_mean', 'host_readbacks_std', 'tracked_buffer_bytes_mean', 'tracked_buffer_bytes_std',
        'p95_position_rel_l2', 'p95_velocity_rel_l2', 'p95_energy_rel_error',
        'p95_mean_stretch_strain', 'p95_max_stretch_strain', 'max_penetration_depth',
        'quality_failure_rate',
    ]
    for row in rows:
        key = row_key(row)
        if key in index:
            fail('Duplicate scene/material row: {0}'.format(key))
        if as_int(row['timing_repetitions'], 'timing_repetitions', row) != 3:
            fail('Incomplete timing repetitions: {0}'.format(key))
        if as_int(row['timing_frames_per_repetition'], 'timing_frames_per_repetition', row) != 300:
            fail('Unexpected timing frame budget: {0}'.format(key))
        if as_int(row['timing_frame_samples'], 'timing_frame_samples', row) != 900:
            fail('Incomplete timing samples: {0}'.format(key))
        if as_int(row['invalid_quality_frames'], 'invalid_quality_frames', row) != 0:
            fail('Invalid quality frame: {0}'.format(key))
        if as_float(row['quality_failure_rate'], 'quality_failure_rate', row) != 0.0:
            fail('Quality failure rate is nonzero: {0}'.format(key))
        for field in fields:
            as_float(row[field], field, row)
        if not row['git_commit'].strip() or not row['gpu_name'].strip() or not row['nvidia_driver_version'].strip():
            fail('Missing reproducibility metadata: {0}'.format(key))
        if not os.path.isdir(row['quality_dir']):
            fail('Missing quality directory: {0}'.format(key))
        validate_metadata(row['quality_dir'], require_quality=True)
        timing_dirs = row['timing_dirs'].split(';')
        if len(timing_dirs) != 3 or not all(os.path.isdir(path) for path in timing_dirs):
            fail('Missing timing directory: {0}'.format(key))
        for directory in timing_dirs:
            metadata = validate_metadata(directory, require_quality=False)
            if as_bool(metadata['solver_controls']['line_search_decisions_profiled']):
                fail('Timing run includes diagnostic line-search tracing: {0}'.format(directory))
        index[key] = row
    if set(index.keys()) != expected_keys(manifest):
        fail('Summary does not cover the complete pre-registered scene/material matrix.')
    return manifest, index


def matrix(index, scene, mesh, field):
    values = []
    for bending in [10, 20, 40]:
        row = []
        for stretch in [40, 80, 160]:
            key = (scene, mesh, stretch, bending, 'gpu-gather-fusion-batched-ls-persistent')
            row.append(as_float(index[key][field], field, index[key]))
        values.append(row)
    return values


def save(fig, run_root, paper_dir, name):
    output_dir = os.path.join(run_root, 'figures')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(paper_dir, exist_ok=True)
    for directory in [output_dir, paper_dir]:
        fig.savefig(os.path.join(directory, name + '.pdf'), bbox_inches='tight')
        fig.savefig(os.path.join(directory, name + '.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_performance(index, run_root, paper_dir):
    panels = [('hanging', 'square-256x256'), ('hanging', 'rect-512x128'),
              ('moving-sphere', 'square-256x256'), ('moving-sphere', 'rect-512x128')]
    all_values = [value for scene, mesh in panels for row in matrix(index, scene, mesh, 'rendered_frame_wall_ms_mean') for value in row]
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.55))
    image = None
    for ax, (scene, mesh) in zip(axes.flat, panels):
        values = matrix(index, scene, mesh, 'rendered_frame_wall_ms_mean')
        image = ax.imshow(values, cmap='viridis', vmin=min(all_values), vmax=max(all_values), aspect='auto')
        for row_index, row in enumerate(values):
            for column_index, value in enumerate(row):
                ax.text(column_index, row_index, '{0:.2f}'.format(value), ha='center', va='center', fontsize=7,
                        color='white' if value > (min(all_values) + max(all_values)) / 2.0 else 'black')
        ax.set_title('{0}, {1}'.format('Moving sphere' if scene == 'moving-sphere' else 'Hanging', mesh.replace('-', ' ')))
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['40', '80', '160'])
        ax.set_yticks([0, 1, 2]); ax.set_yticklabels(['10', '20', '40'])
        ax.set_xlabel('Stretch stiffness'); ax.set_ylabel('Bending stiffness')
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.028, pad=0.02)
    colorbar.set_label('Rendered frame time (ms)')
    fig.tight_layout(pad=0.7)
    save(fig, run_root, paper_dir, 'scene_material_performance')


def plot_quality(index, run_root, paper_dir):
    rows = list(index.values())
    panels = [
        ('p95_position_rel_l2', 'P95 position relative L2', True),
        ('p95_velocity_rel_l2', 'P95 velocity relative L2', True),
        ('p95_energy_rel_error', 'P95 constraint-energy relative error', True),
        ('p95_max_stretch_strain', 'P95 maximum stretch strain', False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.55))
    for ax, (field, ylabel, log_scale) in zip(axes.flat, panels):
        for row in rows:
            ax.scatter(as_float(row['rendered_frame_wall_ms_mean'], 'time', row), as_float(row[field], field, row),
                       color=SCENE_COLORS[row['scene_id']], marker=MESH_MARKERS[row['mesh_id']], s=22, alpha=0.9)
        ax.set_xlabel('Rendered frame time (ms)')
        ax.set_ylabel(ylabel)
        if log_scale:
            ax.set_yscale('log')
        ax.grid(alpha=0.25)
    legend_handles = [
        Line2D([0], [0], color=SCENE_COLORS['hanging'], marker='o', linestyle='None', label='Hanging'),
        Line2D([0], [0], color=SCENE_COLORS['moving-sphere'], marker='o', linestyle='None', label='Moving sphere'),
        Line2D([0], [0], color='#555555', marker='o', linestyle='None', label='Square 256x256'),
        Line2D([0], [0], color='#555555', marker='s', linestyle='None', label='Rectangle 512x128'),
    ]
    axes[0][0].legend(handles=legend_handles, frameon=False, ncol=2, loc='best')
    fig.tight_layout(pad=0.7)
    save(fig, run_root, paper_dir, 'scene_material_quality')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--paper-figure-dir', required=True)
    args = parser.parse_args()
    run_root = os.path.abspath(args.run_root)
    paper_dir = os.path.abspath(args.paper_figure_dir)
    _, index = validate_run(run_root)
    plot_performance(index, run_root, paper_dir)
    plot_quality(index, run_root, paper_dir)
    print('Generated evidence-gated scene/material figures in {0}'.format(os.path.join(run_root, 'figures')))


if __name__ == '__main__':
    main()
