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


def gates(rows, selection):
    groups = aggregate(rows)
    candidate = groups.get(('interaction', 'candidate'))
    variants = selection['required_controls']
    seeds = selection['seeds']
    checks = dict(all_runs_complete=all((('interaction', variant) in groups
                  and groups[('interaction', variant)]['complete'] == len(seeds))
                 for variant in ['candidate'] + variants))
    if not checks['all_runs_complete']:
        return checks
    checks['held_mean'] = candidate['accuracy'] >= .9
    checks['seen_mean'] = candidate['seen'] >= .9
    checks['individual_seeds'] = sum(row['metrics']['accuracy'] >= .85 for row in rows if row['variant'] == 'candidate') >= 9
    checks['frozen_transfer'] = candidate['frozen'] >= .85
    for variant in variants:
        checks['advantage_' + variant] = candidate['accuracy'] - groups[('interaction', variant)]['accuracy'] >= .1 - 1e-12
    checks['frozen_advantage'] = candidate['frozen'] - groups[('interaction', 'frozen')]['frozen'] >= .1 - 1e-12
    checks['living_structure'] = sum(row['brain']['births'] > 0 and row['brain']['deaths'] > 0 for row in rows if row['variant'] == 'candidate') >= 9
    checks['linear_controls_fail'] = all(groups[('interaction', variant)]['accuracy'] <= .6 for variant in ('pass2_linear', 'linear_memory'))
    checks['unique_seed_variants'] = len({(row['seed'], row['variant']) for row in rows}) == len(rows)
    checks['paired_streams'] = all(len({row['stream_hash'] for row in rows if row['seed'] == seed}) == 1 for seed in seeds)
    checks['paired_structure'] = all(len({row['schedule_hash'] for row in rows if row['seed'] == seed
                                           and row['variant'] in ('candidate', 'frozen', 'shuffled', 'matched_capacity')}) == 1 for seed in seeds)
    return checks


def table(rows):
    lines = ['| Condition | Variant | Complete | Accuracy [range] | Seen | Frozen probe | Births | Deaths | Final size |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for (condition, variant), result in sorted(aggregate(rows).items()):
        def percent(value):
            return '—' if value is None else f'{100 * value:.2f}%'
        lines.append(f"| {condition} | {variant} | {result['complete']}/{result['n']} | {percent(result['accuracy'])} [{percent(result['minimum'])}, {percent(result['maximum'])}] | {percent(result['seen'])} | {percent(result['frozen'])} | {result['births']:.1f} | {result['deaths']:.1f} | {result['hidden']:.1f} |")
    return lines


def main():
    results = ROOT / 'experiments/results'
    lines = ['# Pass 3 measurements', '', 'Generated from immutable raw records. Ranges are seed ranges, not confidence intervals.', '']
    for path in sorted(results.glob('pass3_*.jsonl')):
        rows = read(path.name)
        if rows and all('metrics' in row and 'complete' in row and 'condition' in row and 'variant' in row for row in rows):
            lines.extend(['## ' + path.stem, ''] + table(rows) + [''])
    selection_path = ROOT / 'experiments/SELECTED_SOMA.json'
    if selection_path.exists():
        checks = gates(read('pass3_nonlinear_confirmation.jsonl'), json.loads(selection_path.read_text()))
        lines.extend(['## Registered confirmation checks', ''] + [f'- {key}: {value}' for key, value in checks.items()] + [''])
    lines.extend(['## Raw manifest', '', '| File | Records | SHA-256 |', '|---|---:|---|'])
    for path in sorted(results.glob('pass3_*.jsonl')):
        lines.append(f'| {path.name} | {len(path.read_text().splitlines())} | {hashlib.sha256(path.read_bytes()).hexdigest()} |')
    (ROOT / 'experiments/report/PASS3.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[:20]))


if __name__ == '__main__':
    main()
