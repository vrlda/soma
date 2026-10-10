# Small, controlled SOMA experiments

Start with `PROTOCOL.md` (questions and gates registered before execution),
`../NOTEBOOK.md` (including failures), and the generated `report/SUMMARY.md`.
Raw per-seed configurations, prequential scores, and neuron/synapse trajectories
are preserved in `results/`. Initial seeds are 0–4; confirmation streams use
10–19 after a configuration is selected. Loss is measured in natural-log nats.

Run the validity checks from the repository root:

```sh
python -B -m unittest experiments.test_brain -v
```

Reproduce the first repeating-sequence experiment:

```sh
python -B -m experiments.run --suite sequence --output /tmp/soma-sequence.jsonl
```

Run a selected configuration without copying its settings by hand:

```sh
python -B -m experiments.reproduce --selection experiments/SELECTED_GRAMMAR.json \
  --suite grammar --gaps 0,2,8 --output /tmp/soma-grammar.jsonl
```

`SELECTED.json` selects the event-trace delay candidate;
`SELECTED_GRAMMAR.json` selects temporal representation recruitment;
`SELECTED_LIFETIME.json` adds consolidation/protected pruning for the longer
contextual lifetime. `SELECTED_HOMEOSTASIS.json` adds local afferent scaling;
`SELECTED_COMBINED.json` freezes their combination for the gap-8 lifetime.
Selected settings are experiment hypotheses, not a
universal production configuration. Defaults retain the initial failed rules.

The runner refuses to overwrite result files or checkpoints. Optional
`--checkpoint-dir` saves complete final brains, including RNG and traces.
`--budget-seconds` stops an expensive grammar/lifetime experiment without
capping the brain's population; incomplete runs are labelled and do not pass
accuracy gates. An empty `grammar_recruitment.jsonl` records the interrupted
query-only growth attempt; its separately reproduced partial diagnostic is
`results/recruitment_failure_diagnostic.json`.

Build the all-results table and standalone figures:

```sh
python -B -m experiments.report
```

The brain/runner need only Python's standard library. Figures use Matplotlib.
`trace_information.py` is an explicitly hand-designed oracle that measures
whether afferent traces retain information; it never changes or answers for
SOMA. The contextual replay buffer is raw-experience scaffolding, never an
inference path. Sleep/wake replay use the same presentations so a shared gain
is not misrepresented as a sleep-specific effect.

`confidence.py` probes checkpoint clones without supplying labels. It measures
confidence/evidence and transfer to novel inputs; it never implements a rule
that tells SOMA to say "unknown". `check_reproducibility.py` reruns representative
historical configurations and checks exact scores and recorded neural counters.

## Pass 2: future prediction and held-out factors

`composition.py` compares the new distributed predictor with frozen, shuffled,
exact-capacity, Oja, original-pool and labelled linear-memory controls. Twelve
factor combinations occur in learning; four never occur before the test.
Test outcomes are never supplied. All factor values occur in learning.

The one new candidate is frozen in `SELECTED_PREDICTIVE.json` and used unchanged
for the composition study and uncertainty probes. Old selections/defaults
remain available for exact historical reproduction. This is not yet evidence
that the new candidate replaces the older nonlinear grammar brain.

```sh
python -B -m experiments.composition \
  --selection experiments/SELECTED_PREDICTIVE.json --seeds 0,1,2,3,4 \
  --variants predictive,frozen,shuffled,matched_capacity,hebbian_capacity,linear_memory,competitive \
  --output /tmp/soma-pass2-exploration.jsonl
python -B -m experiments.confirm_composition \
  --selection experiments/SELECTED_PREDICTIVE.json --workers 2 \
  --output /tmp/soma-pass2-confirmation.jsonl
python -B -m experiments.predictive_uncertainty \
  --selection experiments/SELECTED_PREDICTIVE.json \
  --output /tmp/soma-pass2-uncertainty.jsonl
python -B -m experiments.pass2_report
```

Run from the repository root. The scientific reference runtime is CPython
3.12.14. PyPy passed within-interpreter tests but changed numerical state;
it was rejected for the paired study. Original controls are CPU-expensive
because analog nuisance triggers many prototype births. Parallel workers
hold separate brains; no state crosses seeds. Recorded budgets never impose
population caps and incomplete trials cannot pass.

The original four-probe transfer gate remains unchanged. Supplementary
160-probe frozen-hidden tests each start from the same learned checkpoint;
no basal/axon-statistic/structural update or outcome feedback occurs. Main
unlabelled tests continue intrinsic learning. Source hashes are captured at
import for confirmation, and identical stream and structural-schedule hashes
verify pairing. Initial completion-time source hashes have a provenance caveat
explained in the protocol; configurations/seeds remain reproducible.

`diagnose_retention.py` reconstructs exact earlier lifetimes and checks their
final saved states. `predictive_geometry.py` uses evaluator-only knowledge of
the environment to analyze weight energy and signal rank. Neither diagnostic
teaches the brain. `predictive_uncertainty.py` probes checkpoint clones and
reports within-seed, within-condition AUC; it contains no unknown-response
rule. Generated machine summaries and SVGs are excluded from commits.

Exact numerical replay of early pass-2 iterations needs the source revision:
adding Oja regrouped floating arithmetic, and full-test yoking corrected an
exploratory capacity difference. All original organism defaults/results remain
unchanged. The source-specific verifier compares latest and recorded revisions:

```sh
python -B -m experiments.check_pass2_reproducibility \
  --output /tmp/soma-pass2-source-replay.jsonl
```

It requires this repository's Git history and CPython 3.12.14. Current-code
mismatches are recorded, while a pinned-source mismatch fails the verifier.
