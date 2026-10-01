# SOMA Production Engine (R5)

The Rust host was selected by measurement: about 170× faster than Python on
sparse graph mutation, with identical results on shared operation tapes.
Every ported kernel has a Python reference and a differential harness.

| Kernel | Rust | Harness | Parity |
|---|---|---|---|
| Sparse graph (stable ids, generational slots) | `graph.rs` | `differential_tape.py` | bitwise |
| Forward propagation (all cell types, dormant freeze, adaptation) | `neuron.rs` | `differential_forward.py` | ~1e-16 |
| Hebbian and actor learning updates | `neuron.rs` | `differential_learn.py` | bitwise |
| R1 evidence router functions | `evidence.rs` | `differential_evidence.py` | ~5e-14 (libm) |
| CUSUM detector update | `detector.rs` | `differential_detector.py` | ~1e-15 |
| Circuit-mixing language memory (`soma/memory/mixing.py`), including trust-weighted learning and `SOMAMIX1` state | `mixer.rs`, `soma-mixer` | `differential_mixer.py` | bitwise; saved state byte-identical |

`soma-mixer-serve` hosts one circuit-mixing memory for the chat service
(line-delimited JSON on stdin/stdout; client `soma.memory.engine`). It is
checked against the Python reference by `differential_serve.py`; see
`docs/r8-engine-serving.md`.

Not ported yet: fingerprints, probes, and organism structural growth
(Python-side by design until the next engine milestone), serving from the
engine, and GPU backends.

- Build: `cargo build --release --manifest-path engine/soma-engine/Cargo.toml`
- All engine gates (unit tests plus every harness): `bash engine/tests.sh`

GPU is deferred until a measured dense-batch workload justifies it. Sparse
inference is memory-latency bound, and that is where this host already wins.
