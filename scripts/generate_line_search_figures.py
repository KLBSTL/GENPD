"""Generate evidence-gated adaptive Armijo figures for the GenPD manuscript."""

import argparse
import csv
import json
import math
import os

import matplotlib.pyplot as plt


COLORS = {
    ('fixed', 'none'): '#4E79A7',
    ('adaptive', 'none'): '#F28E2B',
    ('adaptive', 'iteration'): '#59A14F',
    ('adaptive', 'frame'): '#E15759',
}
LABELS = {
    ('fixed', 'none'): 'Fixed batched',
    ('adaptive', 'none'): 'Adaptive, no history',
    ('adaptive', 'iteration'): 'Adaptive, iteration history',
    ('adaptive', 'frame'): 'Adaptive, frame history',
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.labelsize': 8,
    'axes.titlesize': 9, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42,
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
        row['schedule'], row['adaptive_ls_history'],
        as_int(row['batched_ls_k'], 'batched_ls_k', row),
        round(as_float(row['armijo_beta'], 'armijo_beta', row), 8),
        row['ncg_restart_mode'],
    )


def expected_keys(manifest):
    result = set()
    for schedule in manifest['schedules']:
        histories = manifest['adaptive_history_modes'] if schedule == 'adaptive' else ['none']
        for history in histories:
            for k_value in manifest['k_values']:
                for beta in manifest['armijo_betas']:
                    for restart in manifest['restart_modes']:
                        result.add((schedule, history, int(k_value), round(float(beta), 8), restart))
    return result


def validate_run(run_root):
    manifest_path = os.path.join(run_root, 'manifest.json')
    summary_path = os.path.join(run_root, 'line_search_summary.csv')
    if not os.path.isfile(manifest_path) or not os.path.isfile(summary_path):
        fail('Formal adaptive Armijo manifest or summary is missing.')
    manifest = load_json(manifest_path)
    if manifest.get('protocol_version') != 3:
        fail('Adaptive Armijo figures require protocol version 3.')
    if manifest.get('measurement') != 'rendered-end-to-end':
        fail('Adaptive Armijo figures require rendered measurements.')
    if manifest.get('timing_run_has_decision_tracing') or not manifest.get('trace_run_has_decision_tracing'):
        fail('Timing and diagnostic tracing are not correctly separated.')
    if manifest.get('schedules') != ['fixed', 'adaptive']:
        fail('Adaptive Armijo figures require fixed and adaptive schedules.')
    if manifest.get('adaptive_history_modes') != ['none', 'iteration', 'frame']:
        fail('Adaptive Armijo figures require all history-scope controls.')
    timing = manifest.get('timing', {})
    trace = manifest.get('trace', {})
    if timing.get('frames') != 300 or timing.get('warmup') != 30 or timing.get('repetitions') != 3:
        fail('Formal timing must use 300 measured, 30 warm-up, and 3 repetitions.')
    if trace.get('frames') != 120 or trace.get('warmup') != 20 or not trace.get('quality_reference_dir'):
        fail('Formal traces require 120 measured, 20 warm-up, and a quality reference.')

    rows = load_csv(summary_path)
    index = {}
    required_fields = [
        'rendered_frame_wall_ms_mean', 'rendered_frame_wall_ms_std',
        'rendered_frame_wall_ms_p95', 'line_search_ms',
        'candidates_per_search', 'history_use_ratio',
        'first_batch_accept_ratio', 'second_batch_accept_ratio',
        'fallback_ratio', 'p95_position_rel_l2',
    ]
    for row in rows:
        key = row_key(row)
        if key in index:
            fail('Duplicate adaptive Armijo row: {0}'.format(key))
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
        for field in required_fields:
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
        trace_dir = row['trace_dir']
        trace_metadata = load_json(os.path.join(trace_dir, 'run_metadata.json'))
        if trace_metadata['benchmark']['no_render'] or not trace_metadata['benchmark']['sync_gpu']:
            fail('Trace metadata is not rendered and GPU-synchronised: {0}'.format(key))
        if not as_bool(trace_metadata['solver_controls']['line_search_decisions_profiled']):
            fail('Trace metadata lacks decision tracing: {0}'.format(key))
        if row['schedule'] == 'adaptive' and not os.path.isfile(os.path.join(trace_dir, 'line_search_trace.csv')):
            fail('Adaptive trace CSV is missing: {0}'.format(key))
        index[key] = row
    if set(index.keys()) != expected_keys(manifest):
        fail('Adaptive Armijo summary does not cover the pre-registered matrix.')
    return manifest, index


def get_row(index, schedule, history, k_value, beta, restart):
    key = (schedule, history, int(k_value), round(float(beta), 8), restart)
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
    k_values = manifest['k_values']
    beta = 0.5 if 0.5 in manifest['armijo_betas'] else manifest['armijo_betas'][0]
    k_focus = 4 if 4 in k_values else k_values[-1]
    restart = 'non-descent' if 'non-descent' in manifest['restart_modes'] else manifest['restart_modes'][0]
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.55))

    ax = axes[0][0]
    for schedule, history in [('fixed', 'none'), ('adaptive', 'frame')]:
        rows = [get_row(index, schedule, history, k, beta, restart) for k in k_values]
        ax.errorbar(k_values, [as_float(r['rendered_frame_wall_ms_mean'], 'frame time', r) for r in rows],
                    yerr=[as_float(r['rendered_frame_wall_ms_std'], 'frame std', r) for r in rows],
                    marker='o', linewidth=1.15, color=COLORS[(schedule, history)], label=LABELS[(schedule, history)])
    ax.set_xlabel('Batched candidates K'); ax.set_ylabel('Rendered frame time (ms)')
    ax.set_title('Candidate-count sensitivity'); ax.set_xticks(k_values); ax.grid(alpha=0.25); ax.legend(frameon=False)

    ax = axes[0][1]
    histories = ['none', 'iteration', 'frame']
    rows = [get_row(index, 'adaptive', history, k_focus, beta, restart) for history in histories]
    x = list(range(len(histories)))
    ax.bar(x, [as_float(r['rendered_frame_wall_ms_mean'], 'frame time', r) for r in rows], color=[COLORS[('adaptive', h)] for h in histories])
    ax.errorbar(x, [as_float(r['rendered_frame_wall_ms_mean'], 'frame time', r) for r in rows],
                yerr=[as_float(r['rendered_frame_wall_ms_std'], 'frame std', r) for r in rows], fmt='none', ecolor='#333333', capsize=2)
    ax.set_xticks(x, ['none', 'iteration', 'frame']); ax.set_ylabel('Rendered frame time (ms)')
    ax.set_title('History-scope ablation'); ax.grid(axis='y', alpha=0.25)

    ax = axes[1][0]
    candidates = [as_float(r['candidates_per_search'], 'candidates_per_search', r) for r in rows]
    history_use = [as_float(r['history_use_ratio'], 'history_use_ratio', r) for r in rows]
    ax.bar(x, candidates, color=[COLORS[('adaptive', h)] for h in histories], label='Candidates / search')
    ax.set_xticks(x, histories); ax.set_ylabel('Candidates / search')
    second = ax.twinx(); second.plot(x, history_use, color='#333333', marker='o', linewidth=1.1, label='History-use ratio')
    second.set_ylabel('History-use ratio'); ax.set_title('Temporal-coherence diagnostics'); ax.grid(axis='y', alpha=0.25)

    ax = axes[1][1]
    for schedule, history in [('fixed', 'none')] + [('adaptive', h) for h in histories]:
        row = get_row(index, schedule, history, k_focus, beta, restart)
        time_value = as_float(row['rendered_frame_wall_ms_mean'], 'frame time', row)
        error_value = as_float(row['p95_position_rel_l2'], 'p95_position_rel_l2', row)
        ax.scatter([time_value], [error_value], s=28, color=COLORS[(schedule, history)])
        ax.annotate(LABELS[(schedule, history)], (time_value, error_value), xytext=(3, 3), textcoords='offset points', fontsize=6.2)
    ax.set_xlabel('Rendered frame time (ms)'); ax.set_ylabel('P95 position relative L2')
    ax.set_yscale('log'); ax.set_title('Quality/time Pareto'); ax.grid(alpha=0.25)
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
    print('Generated evidence-gated adaptive Armijo figures in {0}'.format(os.path.join(run_root, 'figures')))


if __name__ == '__main__':
    main()
