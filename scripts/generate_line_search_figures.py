"""Generate evidence-gated line-search figures for the GenPD manuscript."""

import argparse
import csv
import json
import math
import os

import matplotlib.pyplot as plt


COLORS = {
    'gpu-gather-fusion': '#4E79A7',
    'gpu-gather-fusion-batched-ls': '#F28E2B',
    'gpu-gather-fusion-batched-ls-persistent': '#76B7B2',
}
LABELS = {
    'gpu-gather-fusion': 'Serial Armijo',
    'gpu-gather-fusion-batched-ls': 'Batched Armijo',
    'gpu-gather-fusion-batched-ls-persistent': 'Batched + persistent',
}

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
        row['solver_variant'],
        as_int(row['batched_ls_k'], 'batched_ls_k', row),
        round(as_float(row['armijo_beta'], 'armijo_beta', row), 8),
        row['ncg_restart_mode'],
    )


def expected_keys(manifest):
    result = set()
    for variant in manifest['solver_variants']:
        values_k = [1] if variant == 'gpu-gather-fusion' else manifest['k_values']
        for k_value in values_k:
            for beta in manifest['armijo_betas']:
                for restart in manifest['restart_modes']:
                    result.add((variant, int(k_value), round(float(beta), 8), restart))
    return result


def validate_run(run_root):
    manifest_path = os.path.join(run_root, 'manifest.json')
    summary_path = os.path.join(run_root, 'line_search_summary.csv')
    if not os.path.isfile(manifest_path) or not os.path.isfile(summary_path):
        fail('Formal line-search manifest or summary is missing.')
    manifest = load_json(manifest_path)
    if manifest.get('protocol_version') != 2:
        fail('Formal line-search figures require protocol version 2.')
    if manifest.get('measurement') != 'rendered-end-to-end':
        fail('Formal line-search figures require actual rendered measurements.')
    if manifest.get('timing_run_has_decision_tracing') or not manifest.get('trace_run_has_decision_tracing'):
        fail('Timing and diagnostic tracing are not correctly separated.')
    timing = manifest.get('timing', {})
    trace = manifest.get('trace', {})
    if timing.get('frames') != 300 or timing.get('warmup') != 30 or timing.get('repetitions') != 3:
        fail('Formal line-search timing must use 300 measured, 30 warm-up, and 3 repetitions.')
    if trace.get('frames') != 120 or trace.get('warmup') != 20 or not trace.get('quality_reference_dir'):
        fail('Formal line-search traces require 120 measured, 20 warm-up, and a quality reference.')
    if manifest.get('k_values') != [1, 2, 4, 8]:
        fail('Formal line-search figure requires K={1,2,4,8}.')
    if [round(float(value), 8) for value in manifest.get('armijo_betas', [])] != [0.25, 0.5, 0.75]:
        fail('Formal line-search figure requires beta={0.25,0.5,0.75}.')
    if manifest.get('restart_modes') != ['none', 'periodic', 'non-descent']:
        fail('Formal line-search figure requires all restart modes.')

    rows = load_csv(summary_path)
    index = {}
    for row in rows:
        key = row_key(row)
        if key in index:
            fail('Duplicate line-search row: {0}'.format(key))
        if as_int(row['timing_repetitions'], 'timing_repetitions', row) != 3:
            fail('Incomplete timing repetitions: {0}'.format(key))
        if as_int(row['timing_frames_per_repetition'], 'timing_frames_per_repetition', row) != 300:
            fail('Unexpected timing frame budget: {0}'.format(key))
        if as_int(row['timing_frame_samples'], 'timing_frame_samples', row) != 900:
            fail('Incomplete timing samples: {0}'.format(key))
        if as_int(row['trace_frame_samples'], 'trace_frame_samples', row) != 120:
            fail('Incomplete trace samples: {0}'.format(key))
        if as_int(row['timing_has_decision_tracing'], 'timing_has_decision_tracing', row) != 0:
            fail('Timing result contains decision tracing: {0}'.format(key))
        if as_int(row['trace_has_decision_tracing'], 'trace_has_decision_tracing', row) != 1:
            fail('Trace result lacks decision tracing: {0}'.format(key))
        if as_int(row['invalid_trace_frames'], 'invalid_trace_frames', row) != 0:
            fail('Trace contains invalid frames: {0}'.format(key))
        if as_float(row['trace_failure_rate'], 'trace_failure_rate', row) != 0.0:
            fail('Trace reports a failure rate: {0}'.format(key))
        for field in [
            'rendered_frame_wall_ms_mean', 'rendered_frame_wall_ms_std',
            'rendered_frame_wall_ms_p95', 'total_ms_mean', 'total_ms_std',
            'p95_position_rel_l2', 'p95_velocity_rel_l2',
            'p95_mean_stretch_strain', 'p95_max_stretch_strain',
            'max_penetration_depth',
        ]:
            as_float(row[field], field, row)
        timing_dirs = row['timing_dirs'].split(';')
        if len(timing_dirs) != 3 or not all(os.path.isdir(path) for path in timing_dirs):
            fail('Missing timing repetition directory: {0}'.format(key))
        if not os.path.isdir(row['trace_dir']):
            fail('Missing trace directory: {0}'.format(key))
        for path in timing_dirs:
            metadata = load_json(os.path.join(path, 'run_metadata.json'))
            if metadata['benchmark']['no_render'] or not metadata['benchmark']['sync_gpu']:
                fail('Timing metadata is not rendered and GPU-synchronised: {0}'.format(path))
            if as_bool(metadata['solver_controls']['line_search_decisions_profiled']):
                fail('Timing metadata enables decision tracing: {0}'.format(path))
        trace_metadata = load_json(os.path.join(row['trace_dir'], 'run_metadata.json'))
        if trace_metadata['benchmark']['no_render'] or not trace_metadata['benchmark']['sync_gpu']:
            fail('Trace metadata is not rendered and GPU-synchronised: {0}'.format(key))
        if not as_bool(trace_metadata['solver_controls']['line_search_decisions_profiled']):
            fail('Trace metadata lacks decision tracing: {0}'.format(key))
        index[key] = row
    required = expected_keys(manifest)
    if set(index.keys()) != required:
        fail('Line-search summary does not cover the pre-registered matrix.')
    return manifest, index


def get_row(index, variant, k_value, beta, restart):
    key = (variant, k_value, round(beta, 8), restart)
    if key not in index:
        fail('Missing required plotting row: {0}'.format(key))
    return index[key]


def save_figure(fig, run_root, paper_dir):
    output_dir = os.path.join(run_root, 'figures')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(paper_dir, exist_ok=True)
    for directory in [output_dir, paper_dir]:
        fig.savefig(os.path.join(directory, 'line_search.pdf'), bbox_inches='tight')
        fig.savefig(os.path.join(directory, 'line_search.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot(manifest, index, run_root, paper_dir):
    variants = manifest['solver_variants']
    k_values = manifest['k_values']
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.55))

    ax = axes[0][0]
    serial = get_row(index, 'gpu-gather-fusion', 1, 0.5, 'none')
    ax.errorbar([1], [as_float(serial['rendered_frame_wall_ms_mean'], 'time', serial)],
                yerr=[as_float(serial['rendered_frame_wall_ms_std'], 'std', serial)],
                fmt='o', color=COLORS['gpu-gather-fusion'], label=LABELS['gpu-gather-fusion'])
    for variant in variants:
        if variant == 'gpu-gather-fusion':
            continue
        rows = [get_row(index, variant, k_value, 0.5, 'none') for k_value in k_values]
        means = [as_float(row['rendered_frame_wall_ms_mean'], 'time', row) for row in rows]
        stds = [as_float(row['rendered_frame_wall_ms_std'], 'std', row) for row in rows]
        ax.errorbar(k_values, means, yerr=stds, marker='o', linewidth=1.15,
                    color=COLORS[variant], label=LABELS[variant])
    ax.set_xlabel('Batched candidates K')
    ax.set_ylabel('Rendered frame time (ms)')
    ax.set_title('K sensitivity (beta=0.5, no restart)')
    ax.set_xticks(k_values)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[0][1]
    beta_values = [0.25, 0.5, 0.75]
    for variant in variants:
        if variant == 'gpu-gather-fusion':
            continue
        rows = [get_row(index, variant, 4, beta, 'none') for beta in beta_values]
        means = [as_float(row['rendered_frame_wall_ms_mean'], 'time', row) for row in rows]
        stds = [as_float(row['rendered_frame_wall_ms_std'], 'std', row) for row in rows]
        ax.errorbar(beta_values, means, yerr=stds, marker='o', linewidth=1.15,
                    color=COLORS[variant], label=LABELS[variant])
    ax.set_xlabel('Armijo beta')
    ax.set_ylabel('Rendered frame time (ms)')
    ax.set_title('Beta sensitivity (K=4, no restart)')
    ax.set_xticks(beta_values)
    ax.grid(alpha=0.25)

    ax = axes[1][0]
    variant = 'gpu-gather-fusion-batched-ls'
    rows = [get_row(index, variant, 4, beta, 'none') for beta in beta_values]
    rejection_rate = [as_float(row['armijo_rejections'], 'armijo_rejections', row) /
                      as_int(row['trace_frame_samples'], 'trace_frame_samples', row) for row in rows]
    accepted_index = [as_float(row['accepted_candidate_index_mean'], 'accepted_candidate_index_mean', row) for row in rows]
    ax.bar(beta_values, rejection_rate, width=0.1, color=COLORS[variant], label='Armijo rejections / frame')
    ax.set_xlabel('Armijo beta')
    ax.set_ylabel('Rejections / trace frame')
    ax.set_title('Diagnostic Armijo decisions (K=4)')
    ax.set_xticks(beta_values)
    ax.grid(axis='y', alpha=0.25)
    secondary = ax.twinx()
    secondary.plot(beta_values, accepted_index, color='#333333', marker='o', linewidth=1.0,
                   label='Accepted candidate index')
    secondary.set_ylabel('Accepted candidate index')

    ax = axes[1][1]
    for variant in variants:
        if variant == 'gpu-gather-fusion':
            continue
        for restart in manifest['restart_modes']:
            row = get_row(index, variant, 4, 0.5, restart)
            time_value = as_float(row['rendered_frame_wall_ms_mean'], 'time', row)
            error_value = as_float(row['p95_position_rel_l2'], 'p95_position_rel_l2', row)
            ax.scatter([time_value], [error_value], color=COLORS[variant], s=24)
            ax.annotate('{0}: {1}'.format(LABELS[variant].replace('Batched ', ''), restart),
                        (time_value, error_value), xytext=(3, 3), textcoords='offset points', fontsize=6.2)
    ax.set_xlabel('Rendered frame time (ms)')
    ax.set_ylabel('P95 position relative L2')
    ax.set_yscale('log')
    ax.set_title('Restart quality/time Pareto (K=4, beta=0.5)')
    ax.grid(alpha=0.25)

    fig.tight_layout(pad=0.7)
    save_figure(fig, run_root, paper_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--paper-figure-dir', required=True)
    args = parser.parse_args()
    run_root = os.path.abspath(args.run_root)
    paper_dir = os.path.abspath(args.paper_figure_dir)
    manifest, index = validate_run(run_root)
    plot(manifest, index, run_root, paper_dir)
    print('Generated evidence-gated line-search figures in {0}'.format(os.path.join(run_root, 'figures')))


if __name__ == '__main__':
    main()
