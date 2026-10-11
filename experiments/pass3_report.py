"""Recompute immutable pass-3 measurements and gates; no model fitting."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    path = ROOT / 'experiments/results' / name
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['condition'], row['variant'])].append(row)
    return {key: dict(n=len(values), complete=sum(row['complete'] for row in values),
                     accuracy=statistics.mean(row['metrics']['accuracy'] for row in values),
                     minimum=min(row['metrics']['accuracy'] for row in values),
                     maximum=max(row['metrics']['accuracy'] for row in values),
                     seen=statistics.mean(row['seen_probe']['accuracy'] for row in values) if 'seen_probe' in values[0] else None,
                     frozen=statistics.mean(row['frozen_probe']['accuracy'] for row in values) if values[0].get('frozen_probe') else None,
                     births=statistics.mean(row['brain']['births'] for row in values),
                     deaths=statistics.mean(row['brain']['deaths'] for row in values),
                     hidden=statistics.mean(row['brain']['hidden'] for row in values))
            for key, values in groups.items()}


def gates(rows, selection, condition="interaction"):
    groups = aggregate(rows)
    candidate = groups.get((condition, 'candidate'))
    variants = selection['required_controls']
    seeds = selection['seeds']
    checks = dict(all_runs_complete=all(((condition, variant) in groups
                  and groups[(condition, variant)]['complete'] == len(seeds))
                 for variant in ['candidate'] + variants))
    if not checks['all_runs_complete']:
        return checks
    limits = selection['gates']
    checks['held_mean'] = candidate['accuracy'] >= limits['held_mean']
    checks['seen_mean'] = candidate['seen'] >= limits['seen_mean']
    checks['individual_seeds'] = sum(row['metrics']['accuracy'] >= limits['individual_min'] for row in rows if row['variant'] == 'candidate') >= limits['individual_passing_count']
    checks['frozen_transfer'] = candidate['frozen'] >= limits['frozen_transfer']
    for variant in variants:
        checks['advantage_' + variant] = candidate['accuracy'] - groups[(condition, variant)]['accuracy'] >= limits['control_advantage'] - 1e-12
    checks['frozen_advantage'] = candidate['frozen'] - groups[(condition, 'frozen')]['frozen'] >= limits['frozen_advantage'] - 1e-12
    checks['living_structure'] = sum(row['brain']['births'] > 0 and row['brain']['deaths'] > 0 for row in rows if row['variant'] == 'candidate') >= limits['living_structure_passing_count']
    checks['linear_controls_fail'] = all(groups[(condition, variant)]['accuracy'] <= limits['linear_max_mean'] for variant in ('pass2_linear', 'linear_memory'))
    checks['unique_seed_variants'] = len({(row['seed'], row['variant']) for row in rows}) == len(rows)
    checks['paired_streams'] = all(len({row['stream_hash'] for row in rows if row['seed'] == seed}) == 1 for seed in seeds)
    checks['paired_structure'] = all(len({row['schedule_hash'] for row in rows if row['seed'] == seed
                                           and row['variant'] in ('candidate', 'frozen', 'shuffled', 'temporal_shuffled', 'matched_capacity')}) == 1 for seed in seeds)
    return checks


def table(rows):
    lines = ['| Condition | Variant | Complete | Accuracy [range] | Seen | Frozen probe | Births | Deaths | Final size |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for (condition, variant), result in sorted(aggregate(rows).items()):
        def percent(value):
            return '—' if value is None else f'{100 * value:.2f}%'
        lines.append(f"| {condition} | {variant} | {result['complete']}/{result['n']} | {percent(result['accuracy'])} [{percent(result['minimum'])}, {percent(result['maximum'])}] | {percent(result['seen'])} | {percent(result['frozen'])} | {result['births']:.1f} | {result['deaths']:.1f} | {result['hidden']:.1f} |")
    return lines


def figure(records):
    groups = aggregate(records)
    variants = ['candidate', 'frozen', 'shuffled', 'temporal_shuffled',
                'matched_capacity', 'fixed_final', 'pass2_linear', 'linear_memory']
    if not all(('interaction', variant) in groups and groups[('interaction', variant)]['complete'] == 10 for variant in variants):
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), layout='constrained', width_ratios=(1.8, 1))
    means = [groups[('interaction', variant)]['accuracy'] for variant in variants]
    lower = [mean - groups[('interaction', variant)]['minimum'] for mean, variant in zip(means, variants)]
    upper = [groups[('interaction', variant)]['maximum'] - mean for mean, variant in zip(means, variants)]
    axes[0].bar(range(8), means, color=['#187d76'] + ['#b18446'] * 5 + ['#787878'] * 2,
                yerr=[lower, upper], capsize=3, label='Primary test: intrinsic learning continues')
    frozen = [(index, groups[('interaction', variant)]['frozen']) for index, variant in enumerate(variants)
              if groups[('interaction', variant)]['frozen'] is not None]
    axes[0].scatter([item[0] for item in frozen], [item[1] for item in frozen], marker='D', color='#1e3553', zorder=4,
                   label='Weights/statistics/structure frozen')
    axes[0].axhline(.9, color='#9a3333', linestyle='--', label='Registered 90% mean gate')
    axes[0].axhline(.5, color='#777777', linestyle=':')
    axes[0].set_xticks(range(8), ['Candidate', 'Frozen', 'Population\nshuffle', 'Episode\nshuffle', 'Oja', 'Fixed\nfinal size', 'Pass2\nlinear', 'Supervised\nlinear'], fontsize=9)
    axes[0].set(ylabel='Held-out accuracy', ylim=(0, 1.08), title='Confirmation: 10 seeds; bars show means and ranges')
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(fontsize=8, loc='lower left')
    names = ['pass3_structural_noise.jsonl', 'pass3_structural_noise_underfit.jsonl', 'pass3_structural_noise_covariance.jsonl']
    for shift, condition, color in ((-.18, 'independent', '#367ca1'), (.18, 'repeated', '#b18446')):
        samples = [[row['brain']['births'] for row in read(name) if row['condition'] == condition] for name in names]
        mean = [statistics.mean(values) for values in samples]
        lo = [value - min(values) for value, values in zip(mean, samples)]
        hi = [max(values) - value for value, values in zip(mean, samples)]
        axes[1].bar([index + shift for index in range(3)], mean, width=.36, color=color, yerr=[lo, hi], capsize=3, label=condition)
    axes[1].set_xticks(range(3), ['Absolute\nbranch covariance', 'Signed\nunderfit', 'Teacher minus\nforecast covariance'], fontsize=9)
    axes[1].set(ylabel='Births in 4000 unlabelled ticks', title='Noise regulation: 5 seeds; means and ranges')
    axes[1].legend(fontsize=9)
    axes[1].annotate('All four neurons died', xy=(.82, 0.), xytext=(.48, 2.7), fontsize=8,
                     arrowprops=dict(arrowstyle='->', color='#555555'))
    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
    fig.suptitle('SOMA pass 3: nonlinear task gates failed; noise recruitment improved')
    fig.savefig(ROOT / 'experiments/report/pass3_tradeoffs.png', dpi=150)
    plt.close(fig)


def hard_figure(records):
    condition = 'alternating_factor_property'
    groups = aggregate(records)
    variants = ['candidate', 'frozen', 'shuffled', 'temporal_shuffled', 'matched_capacity',
                'fixed_final', 'pass2_linear', 'linear_memory', 'nearest_exemplar']
    if not all((condition, variant) in groups and groups[(condition, variant)]['complete'] == 10 for variant in variants):
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    means = [groups[(condition, variant)]['accuracy'] for variant in variants]
    lower = [value - groups[(condition, variant)]['minimum'] for value, variant in zip(means, variants)]
    upper = [groups[(condition, variant)]['maximum'] - value for value, variant in zip(means, variants)]
    fig, axis = plt.subplots(figsize=(13, 5), layout='constrained')
    axis.bar(range(9), means, color=['#187d76'] + ['#b18446'] * 5 + ['#787878'] * 2 + ['#9a3333'],
             yerr=[lower, upper], capsize=3, label='Primary test; intrinsic learning continues')
    cold = [(index, groups[(condition, variant)]['frozen']) for index, variant in enumerate(variants)
            if groups[(condition, variant)]['frozen'] is not None]
    axis.scatter([item[0] for item in cold], [item[1] for item in cold], marker='D', color='#1e3553', zorder=4,
                 label='Weights/statistics/structure frozen')
    axis.axhline(.5, color='#777777', linestyle=':', label='Chance')
    axis.axhline(.9, color='#9a3333', linestyle='--', label='Unchanged 90% gate')
    axis.set_xticks(range(9), ['Candidate', 'Frozen', 'Population\nshuffle', 'Episode\nshuffle', 'Oja',
                             'Fixed\nfinal size', 'Pass2\nlinear', 'Supervised\nlinear', 'Nearest\nexemplar'], fontsize=9)
    axis.set(ylabel='Held-out accuracy', ylim=(0, 1.08), title='Stronger nonlinear task: frozen configuration, 10 seeds; means and ranges')
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.legend(fontsize=8, loc='upper right')
    axis.spines[['top', 'right']].set_visible(False)
    fig.savefig(ROOT / 'experiments/report/pass3_falsification.png', dpi=150)
    plt.close(fig)


def main():
    results = ROOT / 'experiments/results'
    lines = ['# Pass 3 measurements', '', 'Generated from immutable raw records. Ranges are seed ranges, not confidence intervals.', '']
    for path in sorted(results.glob('pass3_*.jsonl')):
        rows = read(path.name)
        if 'reproducibility' not in path.name and rows and all('metrics' in row and 'complete' in row and 'condition' in row and 'variant' in row for row in rows):
            lines.extend(['## ' + path.stem, ''] + table(rows) + [''])
    selection_path = ROOT / 'experiments/SELECTED_SOMA.json'
    if selection_path.exists():
        checks = gates(read('pass3_nonlinear_confirmation.jsonl'), json.loads(selection_path.read_text()))
        lines.extend(['## Registered confirmation checks', ''] + [f'- {key}: {value}' for key, value in checks.items()] + [''])
        hard = read('pass3_hard_confirmation.jsonl')
        if hard:
            condition = 'alternating_factor_property'
            hard_checks = gates(hard, json.loads(selection_path.read_text()), condition)
            nn = aggregate(hard).get((condition, 'nearest_exemplar'))
            hard_checks['nearest_exemplar_excluded'] = bool(nn and nn['accuracy'] <= .60)
            lines.extend(['## Stronger-task frozen-config checks', ''] + [f'- {key}: {value}' for key, value in hard_checks.items()] + [''])

    lifetimes = read('pass3_unified_comparison.jsonl') + read('pass3_protection_ablation.jsonl')
    lifetimes = [row for row in lifetimes if row['condition'] == 'context_lifetime']
    if lifetimes:
        lines.extend(['## Acquisition-first contextual lifetimes', '',
                      'Raw return accuracy above is not retention evidence for an unacquired mapping.', '',
                      '| Variant | Initial A | Acquired | New B | Raw return | Eligible-only return |',
                      '|---|---:|---:|---:|---:|---:|'])
        for variant in sorted({row['variant'] for row in lifetimes}):
            values = [row for row in lifetimes if row['variant'] == variant]
            eligible = [row for row in values if row['acquired']]
            a = statistics.mean(row['phases'][0]['last_400']['accuracy'] for row in values)
            b = statistics.mean(row['phases'][1]['last_400']['accuracy'] for row in values)
            raw = statistics.mean(row['metrics']['accuracy'] for row in values)
            recall = f"{100 * statistics.mean(row['metrics']['accuracy'] for row in eligible):.2f}%" if eligible else 'undefined'
            lines.append(f'| {variant} | {100*a:.2f}% | {len(eligible)}/{len(values)} | {100*b:.2f}% | {100*raw:.2f}% | {recall} |')
        lines.append('')
    lines.extend(['## Exact replay receipts', ''])
    for name in ('pass3_legacy_reproducibility.jsonl', 'pass3_pass2_reproducibility.jsonl', 'pass3_reproducibility.jsonl', 'pass3_reproducibility_verified.jsonl', 'pass3_hard_reproducibility.jsonl'):
        rows = read(name)
        if rows:
            if name == 'pass3_pass2_reproducibility.jsonl':
                for condition in ('current', 'pinned'):
                    values = [row for row in rows if row['condition'] == condition]
                    lines.append(f"- {name}, {condition}: {sum(row['exact_match'] for row in values)}/{len(values)} exact")
            else:
                lines.append(f"- {name}: {sum(row['exact_match'] for row in rows)}/{len(rows)} exact")
    lines.extend(['', '## Raw manifest', '', '| File | Records | SHA-256 |', '|---|---:|---|'])
    for path in sorted(results.glob('pass3_*.jsonl')):
        lines.append(f'| {path.name} | {len(path.read_text().splitlines())} | {hashlib.sha256(path.read_bytes()).hexdigest()} |')
    (ROOT / 'experiments/report/PASS3.md').write_text('\n'.join(lines) + '\n')
    figure(read('pass3_nonlinear_confirmation.jsonl'))
    hard_figure(read('pass3_hard_confirmation.jsonl'))
    print('\n'.join(lines[:20]))


if __name__ == '__main__':
    main()
