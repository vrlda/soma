# R5 Scalable Runtime (Core + Kernels Shipped; GPU + Full Port Future)

## Rust kernels with differential parity

Beyond graph mutation: forward propagation (all cell types, dormant
freeze, adaptation/observe dynamics) agrees at ~1e-16, the Hebbian +
actor update kernels agree bitwise (gap 0.0), the R1 evidence router
pure functions agree at ~5e-14 (libm), and the CUSUM detector update
(predictor, variance, cusum, freeze, confidence, streak, global
surprise) agrees at ~1e-15 — all across seeds. Toolchain:
`engine/differential_{forward,learn,evidence,detector}.py`.
Fingerprint/probe state machines and structural growth stay Python-side
by design (documented in code); porting them is the defined next engine
milestone, not assumed done. Long soak: 500k steps bounded
(`reports/r5-soak-500k.json`).

## Binary `.soma` format v1 (`soma/persistence/soma_v1.py`)

Single-file container: magic, chunk table (manifest/brain/dialogue/episodic
JSON), per-chunk SHA-256, footer hash. Atomic writes (temp + fsync +
rename). Reads reject bad magic, truncation, hash mismatch, duplicates.
Store `export_soma` / `import_soma` round-trips byte-identically (tested).

## Crash safety (`soma/persistence/journal.py`, wired into BrainStore)

Dirty marker plus previous-generation rotation on every save; loads
validate and fall back to the previous generation. Fault-injection tested
(simulated mid-save crash recovers). Clone/export exclude journal and
previous generations.

## Profiler, budgets (`soma/service/profile.py`, store budgets)

Per-step microseconds (organism + memory), active-cell counts, state bytes,
RSS. State budgets enforced on save (tested). 20k-step soak proxy green
with hard bounds held (`reports/r5-soak.json`).

## Tier ceilings (`configs/tiers.json`)

Micro/Tiny/Small/Prosumer cell/synapse budgets frozen; the 2M-step soak
(`reports/r5-soak-2m.json`, 4x500k chunks with save/load resume between
chunks) holds hard bounds with validation throughout.

## Timing (in-process compute, `reports/r5-forward-timing.json`)

Graph mutation ~170x Rust over Python; forward propagation 3.3x on a
35-cell network (small-graph overhead dominates both sides; larger nets
unmeasured, no claim). Binary-IPC figures are reported separately where
measured and never conflated with compute.

## Host-language decision: Rust (measured)

`engine/soma-engine`: sparse graph with stable ids + generational slots,
JSON snapshot, shared LCG op tapes. `cargo test` green (4 tests).
Differential `engine/differential_tape.py`: identical 50k/200k op tapes in
Rust and Python agree bitwise (worst gap 0.0); Rust is ~170x faster at
graph mutation. Full neural semantics, Metal/CUDA backends, and packaging
remain future work; the spike de-risks the choice, not the port.
