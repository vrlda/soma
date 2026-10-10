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
contextual lifetime. Selected settings are experiment hypotheses, not a
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
