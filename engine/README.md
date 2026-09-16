# SOMA Production Engine (R5 Spike)

Rust host selected by measurement, not taste: ~170x faster than Python on
sparse graph mutation with bitwise-identical structure/strengths on shared
operation tapes (`differential_tape.py`), and 1e-16 forward agreement
(`differential_forward.py`) plus bitwise learning-kernel parity
(`differential_learn.py`).

Ported: sparse graph (stable ids, generational slots), forward propagation
(all cell types, dormant freeze, adaptation, observe dynamics), Hebbian and
actor update kernels. Explicitly NOT ported: routing/detectors,
fingerprints, structural growth, probes, GPU backends, packaging beyond
`cargo build --release`.

Build: `cargo build --release --manifest-path engine/soma-engine/Cargo.toml`
Test: `cargo test --manifest-path engine/soma-engine/Cargo.toml`

GPU is deferred until dense-batch workloads exist that justify it; sparse
small-graph inference is memory-latency bound, where this host already wins.
