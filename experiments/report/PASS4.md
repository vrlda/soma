# Pass 4 measured results — 2026-10-11

Registered acquisition gate: 10/10 pass; frozen-probe gate:
10/10 pass. Both require at least 9/10 at 95%, unchanged.
CPython 3.12.14; selected configuration frozen before confirmation.

## Complete conventional calibration grid, seeds 0–4

| Task | Reference | Size | Rate | Final 400 | Post-train test | Seen probe |
| --- | --- | --- | --- | --- | --- | --- |
| acquisition | linear | — | — | 42.05% | 49.60% | 49.60% |
| acquisition | mlp | 16 | 0.03 | 69.80% | 83.50% | 83.30% |
| acquisition | mlp | 16 | 0.15 | 100.00% | 100.00% | 100.00% |
| acquisition | mlp | 64 | 0.03 | 89.35% | 99.60% | 99.80% |
| acquisition | mlp | 64 | 0.15 | 100.00% | 100.00% | 100.00% |
| acquisition | nearest | — | — | 100.00% | 100.00% | 100.00% |
| original | linear | — | — | 52.80% | 51.12% | 52.17% |
| original | mlp | 16 | 0.03 | 97.60% | 91.62% | 99.83% |
| original | mlp | 16 | 0.15 | 100.00% | 85.12% | 100.00% |
| original | mlp | 64 | 0.03 | 99.25% | 95.25% | 100.00% |
| original | mlp | 64 | 0.15 | 100.00% | 90.00% | 100.00% |
| original | nearest | — | — | 100.00% | 100.00% | 100.00% |
| stronger | linear | — | — | 49.55% | 50.62% | 49.00% |
| stronger | mlp | 16 | 0.03 | 55.40% | 44.88% | 56.67% |
| stronger | mlp | 16 | 0.15 | 74.25% | 43.12% | 75.83% |
| stronger | mlp | 64 | 0.03 | 57.80% | 37.50% | 60.67% |
| stronger | mlp | 64 | 0.15 | 79.85% | 49.75% | 79.83% |
| stronger | nearest | — | — | 100.00% | 0.00% | 100.00% |

For acquisition, post-train test and seen probe both use fresh seen combinations.
For the old tasks, post-train test is held out. Best means were selected by task,
with size/rate tie breaks, BEFORE any SOMA confirmation. No second reference
grid on confirmation. The stronger benchmark fails the conventional held-out
calibration criterion; this is an inductive-bias target, not an impossibility
proof. Nearest confidence .99 is arbitrary; interpret its accuracy only.

## Acquisition confirmation, every required control

| Model | Final 400 | End probe | Final-window seeds ≥ 95% | Mean branches / somas / weighted edges |
| --- | --- | --- | --- | --- |
| candidate | 99.80% | 99.95% | 10/10 | 11.0 / 8.0 / 516.2 |
| pass1 | 32.75% | 50.50% | 0/10 | — / 789.8 / 7020.4 |
| pass3 | 46.88% | 54.25% | 0/10 | — / 1.4 / 99.4 |
| recruitment_off | 29.00% | 53.35% | 0/10 | 0.9 / 8.0 / 274.6 |
| prediction_off | 80.95% | 86.60% | 0/10 | 53.7 / 8.0 / 1497.8 |
| fixed_final | 28.98% | 52.65% | 0/10 | 11.0 / 8.0 / 516.4 |
| shuffled | 61.62% | 79.55% | 0/10 | 270.9 / 8.0 / 6485.8 |
| linear | 41.65% | 48.30% | 0/10 | reference |
| mlp | 97.58% | 100.00% | 9/10 | reference |

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

| Seed | Final 400 | Frozen probe | Branches | Births | Deaths |
| --- | --- | --- | --- | --- | --- |
| 10 | 100.00% | 100.00% | 8 | 7 | 0 |
| 11 | 99.75% | 100.00% | 13 | 13 | 1 |
| 12 | 99.75% | 100.00% | 16 | 15 | 0 |
| 13 | 100.00% | 100.00% | 7 | 6 | 0 |
| 14 | 100.00% | 100.00% | 6 | 5 | 0 |
| 15 | 99.75% | 100.00% | 10 | 9 | 0 |
| 16 | 99.50% | 100.00% | 11 | 11 | 1 |
| 17 | 100.00% | 100.00% | 9 | 8 | 0 |
| 18 | 99.75% | 99.50% | 21 | 20 | 0 |
| 19 | 99.50% | 100.00% | 9 | 9 | 1 |

For SharedSoma, births/deaths count branches, not new somas; legacy controls
use their original counting conventions (— means unreported). Eight predictive somas remain;
one provisional dendrite and its replacements are separately observable.
No population cap applies to new branches. Output-edge counts include pruning
and regrowth. This is finite conjunction acquisition with learned denoising,
not a discovered factorized algorithm or emergent hierarchy.

## Uncapped raw pass-1 reference, all fifteen streams

| Raw pass-1 seed / cohort | Final 400 | Frozen probe | Hidden units | Births | Deaths |
| --- | --- | --- | --- | --- | --- |
| 0 / explore | 35.50% | 52.00% | 818 | 9145 | 8328 |
| 1 / explore | 28.75% | 55.00% | 815 | 9094 | 8280 |
| 2 / explore | 32.00% | 51.00% | 766 | 8704 | 7939 |
| 3 / explore | 31.25% | 55.50% | 779 | 8747 | 7969 |
| 4 / explore | 38.00% | 51.50% | 799 | 9104 | 8306 |
| 10 / confirm | 31.50% | 54.50% | 770 | 8798 | 8029 |
| 11 / confirm | 32.25% | 49.00% | 807 | 9097 | 8291 |
| 12 / confirm | 37.25% | 50.00% | 788 | 8798 | 8011 |
| 13 / confirm | 33.75% | 49.50% | 779 | 8756 | 7978 |
| 14 / confirm | 36.50% | 50.00% | 790 | 8924 | 8135 |
| 15 / confirm | 32.50% | 49.00% | 764 | 8602 | 7839 |
| 16 / confirm | 30.50% | 53.00% | 817 | 9087 | 8271 |
| 17 / confirm | 34.75% | 49.00% | 775 | 8674 | 7900 |
| 18 / confirm | 24.25% | 50.50% | 792 | 8990 | 8199 |
| 19 / confirm | 34.25% | 50.50% | 816 | 9063 | 8248 |

This uses the actual immutable organism kernel and selected recruitment
settings on raw coordinates, with full exposure and no imposed population cap.
Births/deaths here count its recruited neurons, not SharedSoma branches.

## Noise regulation and random labels

| Seed | Ungated noise births | Selected noise births | Selected noise deaths | Random-label births | Last-third random-label births | Random/ordinary hidden exact |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 411 | 0 | 0 | 10 | 0 | True |
| 1 | 359 | 0 | 0 | 13 | 0 | True |
| 2 | 403 | 0 | 1 | 7 | 0 | True |
| 3 | 431 | 0 | 0 | 6 | 0 | True |
| 4 | 371 | 0 | 0 | 19 | 1 | True |

Independent-noise probes last 12000 ticks. Random/ordinary-label lifetimes last
2400 episodes on identical sensory streams. Evidence supports finite rejection
of independent noise, not asymptotic boundedness or all forms of noise. Hidden
hashes exclude receiving outcome weights/evidence and outcome edge counters,
which may legitimately differ; all hidden source histories, forecast/matching
strengths and structural events remain identical.

## One unchanged shared configuration

| Older task, five seeds | Shared selection | Specialist |
| --- | --- | --- |
| linear | 80.00% | 97.00% |
| grammar8 | 49.05% | 100.00% |

| Model / seed | Initial A final 400 | Acquired ≥ 90% | B final 400 | A return first 50 | Retention eligible |
| --- | --- | --- | --- | --- | --- |
| candidate / 0 | 53.50% | False | 49.75% | 36.00% | False |
| specialist / 0 | 100.00% | True | 100.00% | 100.00% | True |
| candidate / 1 | 49.00% | False | 49.25% | 44.00% | False |
| specialist / 1 | 100.00% | True | 100.00% | 100.00% | True |
| candidate / 2 | 57.75% | False | 46.75% | 66.00% | False |
| specialist / 2 | 100.00% | True | 100.00% | 100.00% | True |
| candidate / 3 | 46.75% | False | 49.75% | 56.00% | False |
| specialist / 3 | 71.50% | False | 100.00% | 96.00% | False |
| candidate / 4 | 48.25% | False | 48.50% | 50.00% | False |
| specialist / 4 | 100.00% | True | 100.00% | 100.00% | True |

All five shared initial acquisitions fail: no shared return is retention or
forgetting evidence. Four specialist acquisitions pass; their eligible return
is 100%. Specialist seed 3 (71.5% initial A) remains ineligible despite its 96%
raw return. The acquisition task above uses the same selection; no per-task
tuning occurred.

## Observer-only diagnosis

Normalized labelled readout over actual predictive axons: mean
93.62%,
versus native 80%; some native seeds are better. Native training-state hashes
and probabilities replay exactly. This observer does not alter SOMA, nor
replace its score. At seed 0 initial A, trace oracle 100%, native 53.5%, winner-
only best constant assignment 56.25%. Actual winners group by the second symbol
and merge opposite first-symbol answers. The oracle never supplies an answer
or teaching signal. One small failing initial-A boundary checkpoint is retained.

## Transfer after acquisition passes

| Original calibrated but leaky task | Held out | Seen | Frozen held out |
| --- | --- | --- | --- |
| candidate | 47.69% | 100.00% | 52.50% |
| prediction_off | 55.44% | 92.25% | 50.19% |
| nearest | 100.00% | 100.00% | — |
| mlp | 99.75% | 99.92% | — |

Only the conventionally calibrated original task is tested. It still emits a
product-bearing view AFTER query. Intrinsic primary testing can learn from
that answer-bearing sensory structure; the fully frozen result is essential.
Seen acquisition reaches 100%, but held transfer fails and loses to the perfect
nearest baseline. No beyond-memorization or nonlinear composition claim.
These are fixed-selection measurements after acquisition; seeds/directions
were used earlier, so they are not a new blinded selection experiment.

## C: longer time only, after all A/B controls finish

| Gap 16, five seeds | Final 400 |
| --- | --- |
| candidate | 53.15% |
| trace_off | 49.65% |

| Seed | Gap 16 candidate | Trace off |
| --- | --- | --- |
| 0 | 48.50% | 51.25% |
| 1 | 46.75% | 48.25% |
| 2 | 73.00% | 51.25% |
| 3 | 48.75% | 48.75% |
| 4 | 48.75% | 48.75% |

Only seed 2 reaches 73%; the other four score 46.75–48.75%.
The shared scores remain far below reliable acquisition; this small mean
difference does not establish a useful longer-time mechanism.
No sleep/replay mechanism or benefit is claimed.

## Exact replay and historical integrity

Eight representative shared runs replay exactly (two exploratory versions
pinned to their source revisions, three confirmation variants, three older
interfaces); one complete raw-organism reference also replays exactly. Four
unchanged calibration restart cases match; immutability, stream pairing,
frozen selection and all 90 distinct executed-source snapshots pass.
The report also checks raw-control pairing and exact initial fixed-size capacity. All twelve pass-3 cases
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

- `pass4_acquisition_confirmation.jsonl`: 80 rows, SHA-256 `e451f3b6f48e35ce56abc9e5908438124750892121318760437305e2357907ec`.
- `pass4_calibration.jsonl`: 4 rows, SHA-256 `17f122cfec278baa78e04226c908cbbff3bc686dee34c1728ad95ac9ec36fbdd`.
- `pass4_calibration_complete.jsonl`: 90 rows, SHA-256 `c936a99a240f61d0cf8f5ea8b5b4465d8b670a4877724f8a9b25e0e1ca214def`.
- `pass4_exact_replay.jsonl`: 13 rows, SHA-256 `70b74996c9370477de41dd9d393b891cd31452ccbf2821fff8153a386ee69b53`.
- `pass4_exploratory_controls.jsonl`: 10 rows, SHA-256 `71cc19622a03d0924e5df020201bad56fe8cf34ffe1f71cf7b337d000a2df64e`.
- `pass4_failure_diagnosis.jsonl`: 6 rows, SHA-256 `bd9a435fae2101ea930387233e94e8c0fda1b267664eccef0cfc8f6d660f8ae3`.
- `pass4_generalization.jsonl`: 40 rows, SHA-256 `6f60b15958985adddd06f261101db9b0f6318bbe7e719e7312456d7749e9ca10`.
- `pass4_longer_time.jsonl`: 10 rows, SHA-256 `caf2585a794f16bf6405bbc462689945ef77b64cae254f05eb0577e27d12ba1d`.
- `pass4_noise_hypothesis2.jsonl`: 15 rows, SHA-256 `14bf58ec7340f4d76ec295a9186ec10f04b5e9e2edf8886bbba5218351ae5718`.
- `pass4_noise_hypothesis3.jsonl`: 15 rows, SHA-256 `6bc0d6201c31afbd2e1582a3bb17d3b5fa4d8dbe4f576170fad96d2f5f10e8ab`.
- `pass4_pass3_hard_replay.jsonl`: 2 rows, SHA-256 `c3a9f2af06493a9b5cec696527de170b730d68d801989740c86408885e7f48c8`.
- `pass4_pass3_replay.jsonl`: 10 rows, SHA-256 `5c8441bdd8a997e5b99c98959c72e44ef617a4bf8a6ef7baeb26e273fc6b6e49`.
- `pass4_raw_exact_replay.jsonl`: 1 rows, SHA-256 `f99ad7d08101089404181cddd851680c8ae6a6f9129c38052c3fd73c68d23667`.
- `pass4_raw_organism_confirmation.jsonl`: 10 rows, SHA-256 `e8edc58ef9d04f2742e278aacb8c80f8faf5227a4f95516a0c1accde6bc7b153`.
- `pass4_raw_organism_exploration.jsonl`: 5 rows, SHA-256 `6de3518c4c015caa0ab44ea51d43a741200462ebe5444ba80b1b1ad8dd7e4f1d`.
- `pass4_shared_hypothesis1.jsonl`: 20 rows, SHA-256 `36db08f01a196a545d82e5f912b2133d675d9242fdb5cebe05be99220886d6aa`.
- `pass4_shared_hypothesis2.jsonl`: 25 rows, SHA-256 `a27329fa63f0496add12bc03c3f1ea9fd65a7f40ad5602f9dd997935ddb74917`.
- `pass4_shared_hypothesis3.jsonl`: 20 rows, SHA-256 `46056d165edb0d98dc90a4b110ea42ff05a83587bfa5d3bb4988f9139e3eb250`.
- `pass4_unified.jsonl`: 30 rows, SHA-256 `65005561317b77fff4843e4f2d8e57462bb85e90d506c2fe63b8186d5f0670de`.
