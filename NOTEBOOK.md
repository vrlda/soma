# SOMA research notebook

Newest entries first. Each entry: date, question, setup, result, meaning.
Failures are recorded as carefully as successes.

---

## 2026-10-10: initial local neural candidate (in progress)

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
