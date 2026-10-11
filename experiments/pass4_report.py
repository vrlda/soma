"""Recompute pass-4 tables and two linked PNGs from immutable raw records."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'experiments/results'
REPORT = ROOT / 'experiments/report'


def load(name, expected=None):
    rows = [json.loads(x) for x in (RESULTS / name).read_text().splitlines()]
    if expected is not None:
        assert len(rows) == expected, (name, len(rows), expected)
        assert all(r.get('complete', True) for r in rows), name
    return rows


def pct(value):
    return f'{100 * value:.2f}%'


def mean(rows, field, metric='accuracy'):
    return statistics.mean(r[field][metric] for r in rows)


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def capacity(rows, variant):
    if variant in ('mlp', 'linear'):
        return 'reference'
    branches = (f"{statistics.mean(r['brain']['branches'] for r in rows):.1f}"
                if all('branches' in r['brain'] for r in rows) else '—')
    return branches + ' / ' + ' / '.join(
        f"{statistics.mean(r['brain'][key] for r in rows):.1f}"
        for key in ('hidden', 'synapses'))


def figures(confirmation, noise_old, noise_new, unified):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout='constrained')
    for variant, color, label in [('candidate', '#1b6d9c', 'Selected shared brain'),
                                  ('prediction_off', '#d37826', 'Prediction off'),
                                  ('shuffled', '#9e657e', 'Shuffled teacher'),
                                  ('pass1', '#777777', 'Raw organism'),
                                  ('mlp', '#36895b', 'MLP reference')]:
        rows = [r for r in confirmation if r['variant'] == variant]
        values = [[r['learning_curve'][i]['accuracy'] for r in rows] for i in range(6)]
        x = [100 + 200 * i for i in range(6)]
        axes[0].plot(x, [statistics.mean(v) for v in values], label=label, color=color)
        if variant == 'candidate':
            axes[0].fill_between(x, [min(v) for v in values], [max(v) for v in values], color=color, alpha=.15)
    axes[0].axhline(.95, color='#555555', linestyle=':', label='Registered final-window gate')
    axes[0].set(xlabel='Episode (200-episode windows)', ylabel='Prequential accuracy', ylim=(0, 1.05), title='Leak-free acquisition, ten confirmation seeds')
    axes[0].legend(fontsize=8, loc='lower right')
    for rows, color, label in [(noise_old, '#d37826', 'Two matches, no raw gate'),
                                (noise_new, '#1b6d9c', 'Selected raw-source gate')]:
        records = [r for r in rows if r['condition'] == 'independent']
        values = [[r['trajectory'][i]['births'] for r in records] for i in range(12)]
        x = [1000 * (i + 1) for i in range(12)]
        axes[1].plot(x, [statistics.mean(v) for v in values], label=label, color=color)
        axes[1].fill_between(x, [min(v) for v in values], [max(v) for v in values], color=color, alpha=.15)
    axes[1].set(xlabel='Independent-noise ticks', ylabel='Cumulative new branches', title='Finite noise probe, five seeds')
    axes[1].legend(fontsize=9)
    fig.savefig(REPORT / 'pass4_acquisition.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout='constrained')
    tasks = ('linear', 'grammar8', 'context')
    labels = ('Linear transfer', 'Grammar gap 8', 'Initial-context acquisition')
    for variant, shift, color, label in [('candidate', -.18, '#1b6d9c', 'Unchanged shared selection'),
                                        ('specialist', .18, '#d37826', 'Selected older specialist')]:
        values = []
        for task in tasks:
            rows = [r for r in unified if r['task'] == task and r['variant'] == variant]
            scores = [r['phases'][0]['last_400']['accuracy'] if task == 'context' else r['metrics']['accuracy'] for r in rows]
            values.append(statistics.mean(scores))
        axes[0].bar([i + shift for i in range(3)], values, width=.34, color=color, label=label)
    axes[0].set(xticks=range(3), xticklabels=labels, ylim=(0, 1.08), ylabel='Accuracy', title='One rule set still loses the older tasks')
    axes[0].tick_params(axis='x', labelsize=8)
    axes[0].legend(fontsize=8)
    # Actual seed-0 branch collision, not a diagram invented from the score.
    diagnosis = load('pass4_failure_diagnosis.jsonl', 6)[-1]
    keys = [k for k in diagnosis['winner_class_counts'] if k != 'abstention']
    for target, color in ((0, '#1b6d9c'), (1, '#d37826')):
        heights = [diagnosis['winner_class_counts'][k].get(str(target), 0) for k in keys]
        bottoms = [diagnosis['winner_class_counts'][k].get('0', 0) for k in keys] if target else None
        axes[1].bar(range(len(keys)), heights, bottom=bottoms, color=color, label=f'Outcome {target}')
    axes[1].set(xticks=range(len(keys)), xticklabels=['Branch ' + k for k in keys], ylabel='Last-400 query wins', title='Opposite answers share each query branch')
    axes[1].text(.02, .98, 'Trace oracle: 100%\nBest winner-only assignment: 56.25%', transform=axes[1].transAxes, va='top', fontsize=9)
    axes[1].legend(fontsize=9, loc='upper right')
    fig.savefig(REPORT / 'pass4_shared_limits.png', dpi=170)
    plt.close(fig)


def main():
    calibration = load('pass4_calibration_complete.jsonl', 90)
    hypothesis1 = load('pass4_shared_hypothesis1.jsonl', 20)
    hypothesis2 = load('pass4_shared_hypothesis2.jsonl', 25)
    hypothesis3 = load('pass4_shared_hypothesis3.jsonl', 20)
    noise_old = load('pass4_noise_hypothesis2.jsonl', 15)
    noise_new = load('pass4_noise_hypothesis3.jsonl', 15)
    controls = load('pass4_exploratory_controls.jsonl', 10)
    confirmation = load('pass4_acquisition_confirmation.jsonl', 80)
    raw_explore = load('pass4_raw_organism_exploration.jsonl', 5)
    raw_confirm = load('pass4_raw_organism_confirmation.jsonl', 10)
    confirmation += raw_confirm
    unified = load('pass4_unified.jsonl', 30)
    generalization = load('pass4_generalization.jsonl', 40)
    longer = load('pass4_longer_time.jsonl', 10)
    diagnosis = load('pass4_failure_diagnosis.jsonl', 6)
    replay = load('pass4_exact_replay.jsonl', 13)
    raw_replay = load('pass4_raw_exact_replay.jsonl', 1)
    legacy = load('pass4_pass3_replay.jsonl', 10) + load('pass4_pass3_hard_replay.jsonl', 2)
    assert all(r['exact_match'] for r in replay + raw_replay + legacy)
    candidate = [r for r in confirmation if r['variant'] == 'candidate']
    assert {r['seed'] for r in candidate} == set(range(10, 20))
    primary_count = sum(r['final_window']['accuracy'] >= .95 for r in candidate)
    frozen_count = sum(r['frozen_probe']['accuracy'] >= .95 for r in candidate)
    a_table = table(['Model', 'Final 400', 'End probe', 'Final-window seeds ≥ 95%', 'Mean branches / somas / weighted edges'], [
        [v, pct(mean([r for r in confirmation if r['variant'] == v], 'final_window')),
         pct(mean([r for r in confirmation if r['variant'] == v], 'frozen_probe')),
         str(sum(r['final_window']['accuracy'] >= .95 for r in confirmation if r['variant'] == v)) + '/10',
         capacity([r for r in confirmation if r['variant'] == v], v)]
        for v in ('candidate', 'pass1', 'pass3', 'recruitment_off', 'prediction_off', 'fixed_final', 'shuffled', 'linear', 'mlp')])
    seed_table = table(['Seed', 'Final 400', 'Frozen probe', 'Branches', 'Births', 'Deaths'], [
        [r['seed'], pct(r['final_window']['accuracy']), pct(r['frozen_probe']['accuracy']), r['brain']['branches'], r['brain']['births'], r['brain']['deaths']]
        for r in sorted(candidate, key=lambda r: r['seed'])])
    grid = defaultdict(list)
    for r in calibration:
        config = r['reference_config']
        key = (r['task'], r['variant'], config['size'] if config else 0, config['rate'] if config else 0)
        grid[key].append(r)
    grid_table = table(['Task', 'Reference', 'Size', 'Rate', 'Final 400', 'Post-train test', 'Seen probe'], [
        [t, v, n or '—', rate or '—', pct(mean(rows, 'final_window')), pct(mean(rows, 'metrics')), pct(mean(rows, 'seen_probe'))]
        for (t, v, n, rate), rows in sorted(grid.items())])
    b_table = table(['Older task, five seeds', 'Shared selection', 'Specialist'], [
        [t, pct(mean([r for r in unified if r['task'] == t and r['variant'] == 'candidate'], 'metrics')),
         pct(mean([r for r in unified if r['task'] == t and r['variant'] == 'specialist'], 'metrics'))]
        for t in ('linear', 'grammar8')])
    contexts = [r for r in unified if r['task'] == 'context']
    context_table = table(['Model / seed', 'Initial A final 400', 'Acquired ≥ 90%', 'B final 400', 'A return first 50', 'Retention eligible'], [
        [r['variant'] + ' / ' + str(r['seed']), pct(r['phases'][0]['last_400']['accuracy']), str(r['acquired']),
         pct(r['phases'][1]['last_400']['accuracy']), pct(r['phases'][2]['first_50']['accuracy']), str(r['retention_eligible'])]
        for r in contexts])
    g_table = table(['Original calibrated but leaky task', 'Held out', 'Seen', 'Frozen held out'], [
        [v, pct(mean([r for r in generalization if r['variant'] == v], 'metrics')),
         pct(mean([r for r in generalization if r['variant'] == v], 'seen_probe')),
         pct(mean([r for r in generalization if r['variant'] == v], 'frozen_probe')) if v in ('candidate', 'prediction_off') else '—']
        for v in ('candidate', 'prediction_off', 'nearest', 'mlp')])
    c_table = table(['Gap 16, five seeds', 'Final 400'], [[v, pct(mean([r for r in longer if r['variant'] == v], 'metrics'))] for v in ('candidate', 'trace_off')])
    noise_table = table(['Seed', 'Ungated noise births', 'Selected noise births', 'Selected noise deaths', 'Random-label births', 'Last-third random-label births', 'Random/ordinary hidden exact'], [
        [seed, next(r['brain']['births'] for r in noise_old if r['seed'] == seed and r['condition'] == 'independent'),
         next(r['brain']['births'] for r in noise_new if r['seed'] == seed and r['condition'] == 'independent'),
         next(r['brain']['deaths'] for r in noise_new if r['seed'] == seed and r['condition'] == 'independent'),
         next(r['brain']['births'] for r in noise_new if r['seed'] == seed and r['condition'] == 'random_labels'),
         next(r['trajectory'][-1]['births'] - r['trajectory'][7]['births'] for r in noise_new if r['seed'] == seed and r['condition'] == 'random_labels'),
         next(r['hidden_hash'] for r in noise_new if r['seed'] == seed and r['condition'] == 'random_labels') == next(r['hidden_hash'] for r in noise_new if r['seed'] == seed and r['condition'] == 'ordinary_labels')]
        for seed in range(5)])
    figures(confirmation, noise_old, noise_new, unified)
    detailed = f'''# Pass 4 measured results — 2026-10-11

Registered acquisition gate: {primary_count}/10 pass; frozen-probe gate:
{frozen_count}/10 pass. Both require at least 9/10 at 95%, unchanged.
CPython 3.12.14; selected configuration frozen before confirmation.

## Complete conventional calibration grid, seeds 0–4

{grid_table}

For acquisition, post-train test and seen probe both use fresh seen combinations.
For the old tasks, post-train test is held out. Best means were selected by task,
with size/rate tie breaks, BEFORE any SOMA confirmation. No second reference
grid on confirmation. The stronger benchmark fails the conventional held-out
calibration criterion; this is an inductive-bias target, not an impossibility
proof. Nearest confidence .99 is arbitrary; interpret its accuracy only.

## Acquisition confirmation, every required control

{a_table}

Chance is 50%. Neural/MLP end probes freeze weights and statistics; the linear
reference's incidental source-scale statistics continue without labels.
Competition and live structural schedules differ in ablations; report their
capacities, rather than claiming exact capacity yoking. Fixed-final starts with
the candidate's final branch-parent/channel topology, RANDOM strengths, and no
growth/pruning. Its initial weighted edge count exactly matches final candidate
capacity; it receives full exposure. Under vigilance those random fields may
never match, so this comparison does not isolate growth from activity-based
initialization. Targeted placement and branch-to-soma prediction are implemented;
their separate advantage over other placements is not established here.

{seed_table}

For SharedSoma, births/deaths count branches, not new somas; legacy controls
use their original counting conventions (— means unreported). Eight predictive somas remain;
one provisional dendrite and its replacements are separately observable.
No population cap applies to new branches. Output-edge counts include pruning
and regrowth. This is finite conjunction acquisition with learned denoising,
not a discovered factorized algorithm or emergent hierarchy.

## Noise regulation and random labels

{noise_table}

Independent-noise probes last 12000 ticks. Random/ordinary-label lifetimes last
2400 episodes on identical sensory streams. Evidence supports finite rejection
of independent noise, not asymptotic boundedness or all forms of noise. Hidden
hashes exclude receiving outcome weights/evidence and outcome edge counters,
which may legitimately differ; all hidden source histories, forecast/matching
strengths and structural events remain identical.

## One unchanged shared configuration

{b_table}

{context_table}

All five shared initial acquisitions fail: no shared return is retention or
forgetting evidence. Four specialist acquisitions pass; their eligible return
is 100%. Specialist seed 3 (71.5% initial A) remains ineligible despite its 96%
raw return. The acquisition task above uses the same selection; no per-task
tuning occurred.

## Observer-only diagnosis

Normalized labelled readout over actual predictive axons: mean
{pct(statistics.mean(r['observer']['accuracy'] for r in diagnosis[:-1]))},
versus native 80%; some native seeds are better. Native training-state hashes
and probabilities replay exactly. This observer does not alter SOMA, nor
replace its score. At seed 0 initial A, trace oracle 100%, native 53.5%, winner-
only best constant assignment 56.25%. Actual winners group by the second symbol
and merge opposite first-symbol answers. The oracle never supplies an answer
or teaching signal. One small failing initial-A boundary checkpoint is retained.

## Transfer after acquisition passes

{g_table}

Only the conventionally calibrated original task is tested. It still emits a
product-bearing view AFTER query. Intrinsic primary testing can learn from
that answer-bearing sensory structure; the fully frozen result is essential.
Seen acquisition reaches 100%, but held transfer fails and loses to the perfect
nearest baseline. No beyond-memorization or nonlinear composition claim.
These are fixed-selection measurements after acquisition; seeds/directions
were used earlier, so they are not a new blinded selection experiment.

## C: longer time only, after all A/B controls finish

{c_table}

No sleep/replay mechanism or benefit is claimed.

## Exact replay and historical integrity

Eight representative shared runs replay exactly (two exploratory versions
pinned to their source revisions, three confirmation variants, three older
interfaces); one complete raw-organism reference also replays exactly. Four
unchanged calibration restart cases match; immutability, stream pairing,
frozen selection and executed-source hashes pass. All twelve pass-3 cases
replay with CURRENT Soma code. No numerical tolerance is used. VISION,
organism.py, predictive.py and all historical raw records match reviewed base;
legacy Soma's original source prefix is byte-identical.

The first calibration dispatch failed on a generator adapter after four rows;
keep that partial batch and the corrected complete grid. One newborn-copy test
had the wrong expectation, and its shell committed despite failure; both the
failed commit and the explicit assertion/commit-guard correction remain in
history. Subsequent commits were test-gated. Earlier shared schema snapshots
need their recorded source revision for full-state replay, which the verifier
provides. These verification failures are not scientific successes.

## Raw-file manifest

'''
    for path in sorted(RESULTS.glob('pass4_*.jsonl')):
        count = len(path.read_text().splitlines())
        detailed += f'- `{path.name}`: {count} rows, SHA-256 `{hashlib.sha256(path.read_bytes()).hexdigest()}`.\n'
    (REPORT / 'PASS4.md').write_text(detailed)
    current = f'''# SOMA current research state — 2026-10-11

**Pass 4 establishes reliable finite nonlinear acquisition by local prediction
plus recruited conjunctive branches. It does not establish nonlinear
composition or a reliable unified brain.** On a leak-free raw-factor task,
all ten confirmation seeds pass: final-window mean {pct(mean(candidate, 'final_window'))},
frozen mean {pct(mean(candidate, 'frozen_probe'))}. Independent-noise recruitment
falls from 395 to zero births per 12000 ticks across five seeds.

The same configuration still loses the older tasks: linear transfer 80% versus
97%, grammar gap 8 49.05% versus 100%, and initial-context acquisition 51.05%
with zero of five eligible for retention. Generalization to calibrated held-out
pairs is only 47.69%, versus nearest-exemplar 100%. Real text, emergent hierarchy,
self-knowledge and endogenous sleep remain unearned.

## Pass 4: calibrate before building

The experiment-only online backprop MLP uses one tanh hidden layer, sizes
16/64 and rates .03/.15, one query update per episode, 1200 episodes. Full grid
and linear/nearest controls: 90 complete exploratory rows. Best MLP held means:
original task 95.25%, stronger task 49.75%; best seen stronger 79.83%. The
stronger held benchmark is not conventionally calibrated by this small grid
and does not gate SOMA. This is not a proof that another learner cannot solve
it. Acquisition reference (16, .15) reaches 100% on all exploratory final
windows/probes. Selected references were frozen before confirmation; the full
grid and partial failed dispatch remain in [PASS4.md](experiments/report/PASS4.md).

## Pass 4 A: one substrate with reliable acquisition

`SharedSoma` in [soma.py](brain/soma.py) instantiates one graph directly, without
specialist routing. Local linear next-input prediction learns from each soma's
own sensory observation and prior eligibility. Sensory/predictive event traces
and homeostasis feed competitive receptive branches. Three consistent local
matches were too restrictive; two matches and wider vigilance acquire reliably
but grow on noise. The final rule gates birth with RAW presynaptic lag covariance,
before traces can manufacture persistence. New branches copy provisional local
afferents at a failing predictor and also learn a branch-to-own-soma predictive
current. Only outcome synapses receive labels. [brain/README.md](brain/README.md)
states local inputs, competition, counting and remaining approximations.

| Exploratory hypothesis, five seeds | Final window | Frozen end probe | Independent-noise births |
|---|---:|---:|---:|
| Three matches, vigilance .20 | 81.70% | 89.80% | not measured |
| Two matches, vigilance .50 | 99.45% | 100% | 395 per 12000 ticks |
| Same, raw-source gate .12 | 99.45% | 100% | zero in every seed |

The new task exposes two binary factors in twelve randomly oriented raw
coordinates with orthogonal fresh nuisance .3. Eight blanks, two raw views,
a third raw view/query, and a FOURTH RAW VIEW after feedback. No sensory view
contains a product/parity/property target. All four combinations occur; this
is seen-combination acquisition. Predictions precede feedback. Final 400 of
1200 episodes and 200 fresh frozen probes are registered separately. A linear
population logistic optimum is chance by symmetry. Block-balanced ordering
can put adaptive weak readouts below chance; frozen probes expose that limit.

SELECTED_SHARED freezes every non-interface parameter before seeds 10–19.
All 90 confirmation rows (including the slow raw-organism cohort) complete.

{a_table}

Both registered reliability gates pass: {primary_count}/10 final windows and
{frozen_count}/10 frozen probes are at least 95%. All individual seeds:

{seed_table}

Prediction-off and shuffled controls show useful learned features; the
recruitment-off ablation shows needed nonlinear receptive capacity for this
configuration. The MLP's frozen probe is 100%, so this is not general
superiority to backprop. The fixed-size random-wiring comparison is limited
by vigilance and initialization, and targeted placement's independent value
is not isolated. Branches are neural compartments, not new neurons: eight
predictive somas stay fixed while branch and connection counts change.
Growth/pruning events, protected matching evidence and exact state/RNG resume
are observable. No hierarchy or compositional algorithm is claimed.

![Acquisition and finite noise regulation](experiments/report/pass4_acquisition.png)

Independent noise: all five 12000-tick runs have zero births. Random labels
on 2400 episodes cause exactly the same hidden development as ordinary labels;
last-third births are zero in each. Pruning can remove an unused initial branch;
all predictive somas survive. These finite results do not establish asymptotic
stability or suppression of every temporally structured noise process.

## Pass 4 B: losses and their measured causes

{b_table}

Initial-context acquisition: shared mean 51.05%, zero of five pass .90; B mean
48.8%. Its 50.4% raw first-50 return is NOT retention evidence. Specialists
acquire 94.3%, four of five pass, eligible return 100%; the ineligible seed 3
remains explicitly reported. Per-seed eligibility is in PASS4.md.

The seed-0 oracle decodes both retained traces at 100%, while actual query
branches merge opposite answers: one groups both first symbols with second=0,
the other both first symbols with second=1. Their best winner-only class
assignment is 56.25%, versus native 53.5%. This is acquisition aliasing before
B, not erased old weights. One small boundary brain is saved for diagnosis.

On the linear task, an external normalized labelled readout over actual
predictive axons scores 93.63% on average, versus native 80%. Native training
states and test probabilities remain exact. Some native seeds outperform the
observer; this is accessible-feature evidence and a conditioning hypothesis,
not a per-task repaired SOMA or proof that scale is the sole cause.

![Shared-task losses and actual branch collisions](experiments/report/pass4_shared_limits.png)

## Transfer and longer time

After the acquisition gate passes, the same frozen selection fails the old
conventionally calibrated task's unseen combinations:

{g_table}

Its future view still carries product structure after query, so both adaptive
and fully frozen results are essential. Seen 100% is insufficient: nearest
memory 100% and candidate held 47.69% rule out a composition claim here.
The stronger uncalibrated held task remains a later target, not a moved gate.

C chooses longer time only, after all A/B controls finish:

{c_table}

No endogenous replay or sleep benefit was tested.

## Verification and next research

All 60 final behavioral tests pass; exact replay covers nine representative new
runs, twelve current-code pass-3 cases, strict complete-state/probability checks,
source snapshots, unchanged historical files and frozen configurations. Two
early shared schema snapshots replay at their recorded source revisions. The
calibration adapter failure and one failed test/commit guard are preserved and
corrected explicitly in NOTEBOOK. No raw results, VISION or old kernels were
edited. CPython 3.12.14; brain uses only the standard library. Details and all
raw hashes: [PASS4.md](experiments/report/PASS4.md).

Next: preserve weak early traces in local branch matching without sacrificing
nuisance tolerance; condition each receiving outcome synapse's learning on its
source scale. Test both mechanisms in one configuration across all four tasks,
with acquisition gates first. Then pursue factor-level abstraction beyond
prototype memorization on a conventionally calibrated anti-template benchmark.
Do not scale to real text or replace these missing mechanisms with an LLM.

## Earlier results, condensed and still valid

| Pass | Earned result | Limit retained |
|---|---|---|
| 1: delayed cue | 100% through 64 blank steps | elementary associations, not hierarchy |
| 1: gap-8 grammar | 100% with local variance normalization/recruitment; about 50% without | fresh gap 16 remains near chance |
| 1: context lifetime gap 2 | 100% new/old; 82.4% return without pruning protection | finite labelled lifetime; sleep replay has no advantage over matched wake replay |
| 1: lifetime gap 8 | 98% return vs 78.2% unprotected | seed 11 failed initial acquisition; return is not clean retention for it |
| 2: linear withheld combinations | 95.5% prediction; frozen 66.8%, shuffled 67.9%, Oja 50.1%, linear memory 91.3% | denoising, not nonlinear composition; early sources need pinned replay |
| 2: uncertainty | output entropy error AUC about .95 familiar; sensory surprise about .5 | overconfident under shift; no learned ignorance detector |
| 3: original nonlinear task | 77.6% held, 76.1% frozen; fixed-final 80.6% | nearest raw memory 100%; required gates fail |
| 3: stronger nonlinear task | 26.9% held vs frozen 45.3%; seen 60.8% | failed acquisition/transfer; answer-bearing future view and no conventional calibration then |
| 3: structural hypotheses | noise births 33 → .4 per 4000 ticks | signed rule extinguishes brains; only 5/10 selected lifetimes have both births/deaths |
| 3: shared tasks | linear 75%, grammar 65.8%, initial A 56%; 0/5 acquired | specialists 97%/100%/94.3%; retention undefined for candidate |
| 3: longer time | 50.3% gap 16, trace-off 50.2% | unsolved; no sleep mechanism |

Earlier detail remains in [SUMMARY.md](experiments/report/SUMMARY.md),
[PASS2.md](experiments/report/PASS2.md), [PASS3.md](experiments/report/PASS3.md)
and the unchanged historical raw records. VISION remains the north star.
'''
    (ROOT / 'RESEARCH_REPORT.md').write_text(current)
    print('Recomputed PASS4.md, whole-state report, and two PNG figures from complete raw records.')


if __name__ == '__main__':
    main()
