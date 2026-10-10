# SOMA research notebook

Newest entries first. Each entry: date, question, setup, result, meaning.
Failures are recorded as carefully as successes.

---

## 2026-10-10: consolidated contextual lifetime, confirmed

**Question.** Can one growing/pruning brain keep an earlier temporal mapping
while learning its reversal in an observable new context?

**Setup.** `PROTOCOL.md` lifetime and consolidation entries. Two binary symbols
separated by two neutral events; seven primitive sensory channels include
context, two outcomes. 1200 original-context episodes, 1200 reversed-context
episodes, 400 original-context return episodes, no resets/replay. Score before
teaching. Select `SELECTED_LIFETIME.json` after five exploratory seeds, then
confirm ten independent streams (seeds 10–19). Protect pruning with synapse-local
match/use evidence decaying at .99999 per update; retain active representations
and permit missing prediction connections to regrow locally. Controls match
other settings and remove pruning protection or pruning itself.

**Result.** Confirmation: protected candidate 100% new-context final 400 and
100% old-context first 50; removing protection 100% new and 82.4% old return;
no pruning 100% on both. The first old response also precedes relearning;
its per-seed scores are recorded. Original pruning baseline returned at 86.4%
across five seeds, below the preregistered 90% gate. Full records/checkpoints:
`results/lifetime_confirmation.jsonl`, `checkpoints/lifetime_confirmation/`.

**Meaning.** Consolidation evidence earns its place in this small lifetime by
preventing structural forgetting. The test supports rare-path preservation,
not unlimited lifelong knowledge, general reasoning, or biological sleep.
A longer-gap combination with afferent normalization is being tested separately.

---

## 2026-10-10: temporal recruitment and its limits, confirmed

**Question.** Can local primitives form a representation that combines two
symbols across time, rather than just remembering one symbol?

**Setup.** Finite parity-terminal grammar; task rule exists only in experiments.
Sensory drive peak traces (.8 decay) feed competitive neurons. Start at one
cell, grow where persistent AND current input mismatch remains unmet, on every
experience rather than only labelled outcomes. Newborn afferents copy local
traces. Generic mismatch vigilance .03; smooth size cost, no size ceiling.
Selected settings frozen in `SELECTED_GRAMMAR.json`; ten confirmation seeds.

**Result.** 100% final-window accuracy at gaps 0 and 2, approximately 50% at
8. Removing recruitment: 51.3%, 49.4%, 49.4% at gaps 0/2/8. Removing input
traces: 49.8% at each gap. Freezing afferent weights ALSO scored 100% at 0/2:
recruitment and local outcome learning suffice here; afferent adaptation is
not established as necessary. End populations averaged 15.6 and 16.6 cells at
0/2, from one. Earlier fixed-16-cell candidate: 94.4%, 60.1%, 50.1% on initial
seeds. See `results/grammar_confirmation.jsonl` and earlier preserved results.

**Failure mechanism.** Query-only recruitment kept growing on mismatch from
blank/prefix experiences. The full run was interrupted for excessive cost;
a separate 120-episode reproduction reached 121 hidden cells and only 63.75%
accuracy. `results/recruitment_failure_diagnostic.json` is explicitly partial.
Aligning births with the currently unmet experience removed that failure.
Removing vigilance cost about 45 cells versus about 16 and scored only 89.4%
at gap 2 on initial seeds. Pruning preserved short-task accuracy but repeatedly
removed unlabelled prefix cells, producing substantial birth/death churn.

**Meaning.** A small population can acquire finite temporal conjunctions with
local rules and structural recruitment. It is not an emergent hierarchy or
compositional generalization. The surviving gap-8 failure remains a failure;
real text is deferred.

---

## 2026-10-10: what survives the longer delay (oracle diagnostic)

**Question.** Does grammar failure at longer gaps mean that the sensory trace
has lost the early symbol?

**Setup.** A pre-outcome observer copies traces; an explicitly hand-designed
ORACLE reads each sensory alphabet's larger trace and applies the environment's
parity rule. It does not teach or answer for SOMA. 1200 episodes, final 400,
five seeds per gap, selected grammar unchanged.

**Result.** Oracle 100% at gaps 8, 16, 64, while SOMA stayed near chance.
`results/trace_information.jsonl` includes pre-teaching trace samples.

**Meaning.** The information is still present. Ordinary Euclidean matching
can ignore the weaker older signal. A separately preregistered local variance
normalization experiment investigates representation resolution. The oracle
is an information upper bound, not an architectural shortcut inside SOMA.

---

## 2026-10-10: elementary memory and the mechanisms that did not earn a claim

**Question.** Which primitive rules explain elementary sequence, delay, context,
replay, and structural results?

**Setup.** Initial candidate and follow-ups in `PROTOCOL.md`, all seeds/configs
preserved in `results/`. Prediction always precedes outcome exposure. Five
initial seeds; selected event-trace and competitive settings then ten
confirmation streams.

**Results.** Initial sequence: 100% candidate, 25% disabled learning. Initial
averaging-trace delay candidate: 82.9%, 56.0%, 52.6%, 50.8% at gaps 0/4/16/64.
A full averaging-trace bank did not solve intervening distractors. Peak/event
traces did: selected confirmation 100% at 0/4/16/64; trace-only neurons also
scored 100% at 0/4/16. Removing all temporal paths scored 49.55%, 50.0%,
49.85% at 0/4/16. Conflicting-cue confirmation: selected candidate and trace-only
control both 100%. These are synaptic associations, not deep credit assignment.

Competitive context separation retained both meanings at 100%, whereas the
linear-only control's initial old-context probe was 19.5% in confirmation.
Frozen competitive afferents ALSO passed: do not call this necessary learned
organization. Replay labelled sleep and equal wake replay were numerically
identical; no sleep-specific consolidation was shown. Some initial replay
runs interfered with the new context rather than helping it.

**Matched structural control.** On random labels, prediction-error growth from
one cell made 150 births and 86 deaths in each seed, ending at 65 cells with
51.28% accuracy. Representation-novelty growth with the same initial size,
learning rule, interval, cost, pruning, and stream made 4–11 births; accuracy
remained chance. Novelty avoids interpreting irreducible label noise as an
endless need for neurons, but it does not solve unpredictable labels.

**Meaning.** Keep the evidence for local temporal state and representation
recruitment. Hidden Oja learning, outcome metaplasticity on elementary tasks,
and sleep-labelled replay have not independently established the larger
claims in the vision. Every failed candidate remains reproducible.

---

## 2026-10-10: initial local neural candidate (historical partial snapshot)

**Question.** Can local outcome learning plus leaky recurrent neurons and
averaging synaptic traces learn a sequence and retain a cue through distractors?

**Setup.** Registered `experiments/PROTOCOL.md` before execution. Standard-library
reference in `brain/organism.py`; predictions precede outcome exposure. Sparse
recurrent hidden neurons use local Oja updates. Prediction synapses use source
trace times their own target neuron's outcome error; no error propagates into
hidden cells. Five seeds, prequential metrics, controlled ablations. Six checks
passed, including exact checkpoint/RNG continuation and outcome causality.

**Partial results, not final.** All five candidate seeds scored 100% on the final
windows of the two four-symbol cycles; no-learning scored 25%. Removing hidden
plasticity also scored 100%: this task does not establish useful hidden learning.
The completed candidate runs at gaps 0, 4, 16 scored 82.9%, 56.0%, 52.6%.
The delayed-memory gate failed. Initial conflicting-cue runs also failed.
Full results and later confirmation will be recorded separately.

**Interpretation.** This is a functioning local readout, not deep credit
assignment, a learned hierarchy, or a working lifelong brain. A randomly
assigned averaging trace scales a one-off event by `(1 - decay)`: long time
constants preserve time but attenuate a brief cue. Randomly omitting particular
trace timescales also makes the primitive interface seed-dependent. Test that
mechanism explicitly next; do not conceal this failure by changing its gate.

---

## 2026-10-02: fresh start

**What happened.** The repository was reset around [`VISION.md`](VISION.md).
Everything earlier moved to `archive/` (last runnable commit `b6606cd`).
The earlier work drifted toward count tables, hand-built features, and
product engineering. Its lessons are in `VISION.md` section 8.

**Starting point for the new work:**

- Build a small, clean brain: neurons and synapses that hold numbers, with
  each kind of change as a separate, swappable rule (local learning, global
  modulators, growth and death, sleep).
- Test the first rules on tiny observable tasks: predict a repeating
  pattern, remember across a delay, learn a toy grammar, recall after
  sleep. Compare against trivial controls and ablations.
- Move to real text (the modern corpus in `data/modern/`) only once the
  small tasks are mastered.

**Baselines to beat eventually** (from the archive, Simple English
Wikipedia validation, bits per byte; lower is better): about 1.99 for
byte-context count tables at a 4M-entry budget, and about 1.88 for the same
plus copying from earlier in the document.
