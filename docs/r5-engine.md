# R5 Scalable Runtime (Core Shipped; GPU + Full Port Future)

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

## Host-language decision: Rust (measured)

`engine/soma-engine`: sparse graph with stable ids + generational slots,
JSON snapshot, shared LCG op tapes. `cargo test` green (4 tests).
Differential `engine/differential_tape.py`: identical 50k/200k op tapes in
Rust and Python agree bitwise (worst gap 0.0); Rust is ~170x faster at
graph mutation. Full neural semantics, Metal/CUDA backends, and packaging
remain future work; the spike de-risks the choice, not the port.
