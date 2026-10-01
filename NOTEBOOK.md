# SOMA research notebook

Newest entries first. Each entry: date, question, setup, result, meaning.
Failures are recorded as carefully as successes.

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
