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
| `RESEARCH_REPORT.md` | current measured results, limits, and next steps (keep it up to date) |
| `brain/organism.py` | the SOMA brain: neurons, synapses, local rules, growth, pruning, save/resume |
| `experiments/` | protocol, runner, frozen selections (`SELECTED_*.json`), tests, raw results, saved brains |
| `scripts/build_modern_corpus.py` | builds the modern text corpus into `data/modern/` (Wikipedia, SQuAD, dialogue; sealed test splits) |
| `data/modern/MANIFEST.json` | hashes and counts of that corpus (the text is git-ignored) |
| `archive/` | everything before 2026-10-02, reference only (see `archive/README.md`) |

Run the tests before every commit: `python -B -m unittest experiments.test_brain`.
Never overwrite or edit files in `experiments/results/`: they are the record.

Every mechanism in `brain/` is currently an on/off switch in `Config`, so that
ablations stay reproducible. Once a mechanism has earned its place in confirmed
experiments, aim to make it part of the one shared brain rather than adding more
per-task switches: SOMA should end up as one brain with one set of rules.

Code style: Python standard library first, clear and inspectable. Add a faster
implementation (Rust, NumPy) only when an experiment needs the speed, and keep
the readable version as the reference.
