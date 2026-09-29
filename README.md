# SOMA — Self-Organizing Morphogenic Architecture

> **Project status:** research program past the synthetic-kernel era and mid
> language track. The organism core (v8–v15, frozen, 221-test baseline) now
> sits under a universal event system, calibrated evidence routing,
> sequence memory with E0–E2 English scaling, generation, dialogue,
> instruction behavior, tools, continual learning, red-team hardening, and a
> prosumer service. The active roadmap is
> [`SOMA_REAL_MODEL_MASTER_PLAN.md`](SOMA_REAL_MODEL_MASTER_PLAN.md);
> milestones R0–R4, R3A–R3D are complete (see table). This is not a
> consumer model and not AGI: capabilities below are exactly measured,
> with boundaries stated.

## Milestone board

| Milestone | Status | Evidence |
|---|---|---|
| R0 preserve kernel | ✅ | 221 frozen tests, hashed baseline |
| R1 evidence router | ✅ | default router; 10/10 suites green |
| R2 event organism | ✅ | 2 encodings learn, controls fail |
| R3A sequences | ✅ | lag-copy + periodic gates, both adapters |
| R3B English E0–E2 | ✅ | E2 held-out 0.3995 vs 0.559 bar; compositional transfer carried |
| R3C generation | ✅ | 7/7 + 28/28 valid UTF-8, lesion, silence |
| R3D dialogue | ✅ | 6/6 gates: uptake, correction, retention, release |
| R3 systematicity | ✅ | novel combos recombine, precision held |
| R4 continual | ✅ | lifelong 4/4 uptake, conflicts, quarantine |
| R7 instruction | partial | skills, uncertainty, refusal, tools; human preference floors open |
| R8 service | partial | CLI/chat/API/doctor work; installer, signatures, clean-machine open |
| R9 red-team | partial | 7/7 attack gates; telemetry, runbooks, long alpha open |
| R5 engine | partial | binary format, journal, profiler, Rust spike (~170x) |
| R6 useful scale | partial | E2 11 MB; E3+ needs engine |
| R10 release | open | needs R5/R6 |
| R11/R12 multimodal | future | after Text v1 |

## Try it (no coding)

Clone with Git LFS installed to download the large benchmark reports:

```bash
git lfs install
git clone https://github.com/vrlda/soma.git
cd soma
```

```bash
export SOMA_BRAINS=/tmp/soma-demo
python3 -m soma.service.main brain-create demo
python3 -m soma.service.main brain-teach demo data/e0/alice.txt
python3 -m soma.service.main brain-correct demo "USER the code is " "kite"
python3 -m soma.service.main chat demo
python3 -m soma.service.main serve --port 8765
```

Chat understands `repeat after me:`, `spell`, taught facts, corrections,
and says "I don't know" for out-of-training bytes. `doctor` checks health.

## Repository map

- `soma/` — organism core (frozen paths) + `events/`, `routing/`,
  `transducers/`, `memory/`, `evaluation/`, `persistence/`, `service/`
- `engine/soma-engine` — Rust sparse-graph spike with differential tapes
  (`bash engine/tests.sh` builds it and runs every Python/Rust parity gate)
- `tests/` — 482 unit tests (`python -m unittest discover -s tests`)
- `r*_benchmark.py` — frozen milestone gates with JSON reports in `reports/`;
  `r6_*_pilot.py` / `*_study.py` / `*_calibration.py` are exploratory R6
  experiments, and `r3b_reference_baselines.py` is a non-gating comparison.
  They stay at the root because tests import them and
  `reports/r0-baseline/SHA256SUMS` pins their paths.
- `docs/` — per-milestone records, honest boundaries included
- `data/` — licensed public-domain corpora with manifests

## Honesty section

- Language ability is statistical bit prediction plus exact episodic rules;
  E2 English (0.3995 bits/bit) sits between a frozen order-1 and order-2
  byte n-gram and behind gzip; a frozen order-5 n-gram reaches 0.258
  ([reference baselines](docs/r3b-reference-baselines.md)).
  there is no comprehension, and fluent nonsense is answered from marginals.
- Compositional transfer works for taught parts in novel arrangements;
  open-ended semantic generalization is unproven.
- Stochastic grammar, E3+ scale, GPU engine, and multimodal input are open.
- Every claim maps to a frozen script, a report hash, and a test.
