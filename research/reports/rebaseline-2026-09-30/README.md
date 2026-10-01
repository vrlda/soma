# Integrity re-baseline, 2026-09-30

Master plan §22 step 1. The pinned manifest had 28 mismatches: 21 code,
test, and report files (18 from commit 3532e39 plus three later edits) and
7 prose docs. Two of the 21 were LFS pointer files that matched once Git
LFS was installed. Every frozen benchmark covering the changed code was
re-run on Linux (Python 3.11) against the committed reports.
Committed reports are left untouched as historical evidence; re-run outputs
whose files differ are stored here.

| Benchmark | Re-run vs committed report | Gates |
|---|---|---|
| `final_lifetime_benchmark.py` (v15) | SOMA, structure-disabled, fixed-topology, and replay arms equal up to float noise (~1e-16). The learning-disabled control arm differs, see below | FINAL_LIFETIME_PASS |
| `r3b_e1_benchmark.py` E1 | test 0.3976183 vs 0.3976185; differences come from the v1→v2 protocol (acquisition-fitted bars, per-book document boundaries) | all pass |
| `r3b_e1_benchmark.py` E2 | test 0.3994975 vs 0.3994979; same protocol explanation | all pass |
| `r3b_e1_benchmark.py` E3 (v2 report) | **identical**, 859/859 values | all pass |
| `r3b_sequence_memory_benchmark.py` | model metrics identical; only the v2 unigram bar moved | all pass |
| `r3b_fusion_benchmark.py` | model metrics identical (lesion differs at 1e-16); only the v2 bar moved | all pass |
| R3C generation, R3D dialogue, R3D systematicity, R4 lifelong, R6 factual, R7 instruction, R7 tools, R9 red-team | **identical** | all pass |
| `r6_reclamation_benchmark.py` | **byte-identical** report; its in-harness determinism gate passes. It exercises `SequenceCircuitMemory` under constant forced eviction, which confirms the 2026-09-30 reclamation-scan fix preserves behavior exactly | 4/4 |
| Full unit suite | 496 tests OK in 424 s | — |

## Commit 3532e39 is behavior-preserving

Running `final_lifetime_benchmark.py` at 3532e39^ and at HEAD on the same
machine gives 2,614,729 equal values and no value differences. The only
change is 144 new state fields from the opt-in persistent scout.

## The v15 archive's learning-disabled arm is not reproducible here

`reports/r0-baseline/final-lifetime-v15.json` reproduces to float noise in
every arm except the learning-disabled control (85,375 differing values;
for example, phase A tail skill −0.210 archived vs −0.142 now). Its absence
of R1/R3 state fields shows it was produced by code older than 76f4dc5, the
commit that added it. Running 76f4dc5 here reproduces the *current*
numbers, not the archived ones. So the difference predates that commit,
and it may be platform-dependent (the archive was recorded on macOS with
Python 3.14). The gate result is unaffected. The archive stays pinned as
historical R0 evidence; a macOS re-run could settle the cause.

## Files

- `r3b-e1-v2.json`, `r3b-e2-v2.json`, `r3b-sequence-memory-v2.json`,
  `r3b-fusion-v2.json`: re-run reports under the current v2 protocols.
- `final-lifetime-v15-summary.json`: gate summary of the v15 re-run (the
  full 80 MB report is not duplicated).
