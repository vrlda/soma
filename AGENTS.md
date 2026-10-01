# Instructions for agents working on SOMA

1. **Read [`VISION.md`](VISION.md) completely before doing anything.** It is
   the north star: a synthetic brain of neurons and synapses that learns by
   local rules, continuously, grows and prunes itself, and sleeps to
   consolidate. It is a research dream aiming at a breakthrough, not a
   product.
2. **Read the latest entries in [`NOTEBOOK.md`](NOTEBOOK.md)** to learn where
   the research stands.
3. **Check every idea against `VISION.md` section 5 (what SOMA is NOT).** No
   Transformers or LLM hybrids, no backpropagation inside SOMA, no
   retrieval-and-reader engineering, no count tables posing as neurons, no
   product race. Baselines are allowed only if clearly labelled.
4. **Experiment small and measured.** Use a tiny observable task first,
   always with a control, and run ablations for each mechanism.
5. **Record every experiment in `NOTEBOOK.md`** (date, question, setup,
   result, meaning), including failures. Never move goalposts after seeing
   results.
6. **Be honest with the owner.** Report what works and what doesn't.

## Layout

| Path | Contents |
|---|---|
| `VISION.md` | the vision: read first, re-read often |
| `NOTEBOOK.md` | the research log, newest first |
| `brain/` | the SOMA brain: neurons, synapses, rules (starts empty; built from here on) |
| `experiments/` | small experiments, each recorded in the notebook |
| `scripts/build_modern_corpus.py` | builds the modern text corpus into `data/modern/` (Wikipedia, SQuAD, dialogue; sealed test splits) |
| `data/modern/MANIFEST.json` | hashes and counts of that corpus (the text is git-ignored) |
| `archive/` | everything before 2026-10-02, reference only (see `archive/README.md`) |

Code style: Python standard library first, clear and inspectable. Add a faster
implementation (Rust, NumPy) only when an experiment needs the speed, and keep
the readable version as the reference.
