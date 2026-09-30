# SOMA — Self-Organizing Morphogenic Architecture

> **Project status (2026-09-30):** research program, mid language track.
> The frozen v8–v15 organism core sits under a universal event system,
> calibrated evidence routing, English sequence memory, generation,
> dialogue, instruction behavior, tools, continual learning, red-team
> hardening, and a local service. Milestones R0–R4 are complete; R5–R9 are
> partial. This is not a consumer model and not AGI. Every capability
> below is measured, and its boundaries are stated.

## Where we are and what's next

**Goal:** SOMA Text v1 (master plan §21). That is one persistent brain a
non-developer can install, chat with, teach, and trust to retain and
recover what it learned, within stated resource and safety bounds.

**Latest result:** the R6 circuit-mixing memory brings English prediction
to 1.93 bits/byte (was 3.20). It beats a frozen order-5 n-gram, and its
plastic arbitration is causal ([details](docs/r6-circuit-mixing.md)). The
result held on a pre-registered, never-seen test book: 1.90 bits/byte
against 2.07 for the n-gram ([untouched test](docs/r6-untouched-test.md)).

**Next, in order** (authoritative list with done-criteria:
[master plan §22](SOMA_REAL_MODEL_MASTER_PLAN.md#22-exact-next-work-on-resume)):

1. ~~**Trust the result:**~~ done. Hashes re-baselined, the untouched test
   held (all 3 hypotheses), and every mechanism now earns its place
   (ADR 0007).
2. **Answer the SOMA question:** measure forgetting across books, then add
   a consolidation mechanism that removes recency interference.
3. **Make it usable:** port forgetting, quarantine, and persistence to the
   new memory, and serve chat from the Rust engine.
4. **Scale and ship:** E3 scaling, Micro tier on the engine, human
   preference ratings, signatures, alpha, then Text v1.

## Milestone board

| Milestone | Status | Evidence |
|---|---|---|
| R0 preserve kernel | ✅ | 221 frozen tests, hashed baseline |
| R1 evidence router | ✅ | default router; 10/10 suites green |
| R2 event organism | ✅ | 2 encodings learn, controls fail |
| R3A sequences | ✅ | lag-copy + periodic gates, both adapters |
| R3B English E0–E2 | ✅ | E2 held-out 0.3995 vs 0.559 bar; E3-lite 18 MB |
| R3C generation | ✅ | 7/7 + 28/28 valid UTF-8, lesion, silence |
| R3D dialogue | ✅ | 6/6 gates: uptake, correction, retention, release |
| R3 systematicity | ✅ | novel combos recombine, precision held |
| R4 continual | ✅ | lifelong 4/4 uptake, conflicts, quarantine |
| R5 engine | partial | `.soma` v1, journal, profiler, 2M soak; Rust parity for graph, forward, learning, evidence, detector, circuit mixing. Open: growth on engine, serving, GPU, 72 h soak |
| R6 useful scale | partial | circuit mixing 0.2411 bits/bit, 7/7 gates, untouched book 0.2381 (beats order-5 n-gram); Small floor, factual 8/8, reclamation 4/4. Open: consolidation, E3, signed candidate |
| R7 instruction | partial | skills, uncertainty, refusal, tools. Open: human preference floors |
| R8 service | partial | CLI, chat, HTTP API, doctor, installer, backup/restore. Open: signatures, downloader, recorded clean-machine run |
| R9 red-team | partial | 7/7 attack gates, telemetry, runbooks. Open: closed alpha |
| R10 release | open | needs R5–R9 |
| R11 vision | first step | 6×6 glyphs through the unchanged core, 7/7 |
| R12 embodied | first step | simulated cart with a safety controller, 5/5 |

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
- `engine/soma-engine` — Rust engine: graph, kernels, and `soma-mixer`
  (`bash engine/tests.sh` builds it and runs every Python/Rust parity gate)
- `tests/` — 496 unit tests (about 7 min) (`python -m unittest discover -s tests`)
- `r*_benchmark.py` — frozen milestone gates with JSON reports in `reports/`;
  `r6_*_pilot.py` / `*_study.py` / `*_calibration.py` are exploratory R6
  experiments, and `r3b_reference_baselines.py` is a non-gating comparison.
  They stay at the root because tests import them and
  `reports/r0-baseline/SHA256SUMS` pins their paths.
- `docs/` — per-milestone records, honest boundaries included
- `data/` — licensed public-domain corpora with manifests

## Documentation map

| Read this | For |
|---|---|
| [`PROJECT.MD`](PROJECT.MD) | Founding vision and design principles (why SOMA exists) |
| [`SOMA_REAL_MODEL_MASTER_PLAN.md`](SOMA_REAL_MODEL_MASTER_PLAN.md) | **The plan:** architecture contract, milestone status (§16), Text v1 definition of done (§21), ordered next work (§22) |
| [`SOMA_ENGLISH_DEMO_RUNBOOK.md`](SOMA_ENGLISH_DEMO_RUNBOOK.md) | Procedure and gates for training the English demonstration brain |
| [`docs/README.md`](docs/README.md) | Index of per-milestone records, ADRs, and operations docs |
| [`SOMA_SYNTHETIC_KERNEL_HISTORY.md`](SOMA_SYNTHETIC_KERNEL_HISTORY.md) | Historical v8–v15 kernel record (not a plan) |
| [`CHANGELOG.md`](CHANGELOG.md) | What changed, by release |

## Honesty section

- Language ability is statistical bit prediction plus exact episodic rules;
  E2 English (0.3995 bits/bit) sits between a frozen order-1 and order-2
  byte n-gram and behind gzip; a frozen order-5 n-gram reaches 0.258
  ([reference baselines](docs/r3b-reference-baselines.md)). The R6
  circuit-mixing memory reaches 0.2411 on the same protocol
  ([docs](docs/r6-circuit-mixing.md)) and 0.2381 on a pre-registered
  untouched book; chat does not use it yet. There is no comprehension, and
  fluent nonsense is answered from marginals.
- Compositional transfer works for taught parts in novel arrangements;
  open-ended semantic generalization is unproven.
- Stochastic grammar, E3+ scale, GPU engine, and multimodal input are open.
- Every claim maps to a frozen script, a report hash, and a test.
