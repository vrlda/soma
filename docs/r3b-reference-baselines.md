# R3B reference baselines (context, not a gate)

`r3b_reference_baselines.py` places the E2 English score next to the
standard references. It changes no frozen gate. Report:
`reports/r3b-e2-reference-baselines.json`.

Same manifest (`reports/e2-manifest.json`), same 11.5 MB acquisition, same
validation (Jekyll) and sealed test (Time Machine) bytes. Unit: bits/bit
(byte-level bits/byte ÷ 8, equal to SOMA's bit cross-entropy by the chain
rule).

| Model | Protocol | Validation | Test |
|---|---|---|---|
| byte unigram (gate bar) | frozen | 0.565 | 0.559 |
| Witten-Bell byte n-gram, order 1 | frozen | 0.443 | 0.441 |
| **SOMA E2, order-16 bit circuits** | frozen | **0.398** | **0.3995** |
| gzip -9 | adaptive, eval bytes only | 0.392 | 0.386 |
| Witten-Bell byte n-gram, order 2 | frozen | 0.359 | 0.363 |
| xz -9e | adaptive, eval bytes only | 0.356 | 0.345 |
| bz2 -9 | adaptive, eval bytes only | 0.319 | 0.310 |
| Witten-Bell byte n-gram, order 3 | frozen | 0.297 | 0.302 |
| xz -9e primed on acquisition | adaptive, conditional | 0.289 | 0.288 |
| Witten-Bell byte n-gram, order 5 | frozen | **0.258** | **0.258** |

## Reading

- The frozen n-gram uses exactly SOMA's evaluation protocol: fit on
  acquisition, score validation/test with no updates. It needs no tuning
  and fits in about 36 s of pure Python.
- SOMA E2 falls between the order-1 and order-2 byte n-gram. That is
  expected: `max_order=16` over MSB-first bits is at most two bytes of
  context. Byte-unigram beating is real but a weak bar.
- Order-5 byte n-gram (≈2.06 bits/byte) is the like-for-like target the
  structural mechanisms should eventually beat. gzip, which never sees the
  acquisition books, already beats the current E2 score.

## What the E2 score depends on

`r3b_e1_benchmark.py` trains a bare `soma.memory.SequenceCircuitMemory`; no
`Organism` is constructed. Service chat reaches the same class through
`Organism.sequence_memory`, a pass-through. The E2 number therefore measures
the suffix-circuit memory (whose circuits do grow and are reclaimed under a
budget), not the organism's cell/synapse plasticity, homeostasis, or
consolidation. The organism participates only in the fused/motor runs
(`run_english_fused`, E0 parity 0.37509).

## Suggested next gates (proposal)

*Item 2 is done; item 3 is now master plan §22 step 6.*

1. Report these references in every future E-series result.
2. Add a gate "beats frozen order-2 byte n-gram", then order 3. (Done for
   the R6 circuit-mixing memory: `r6_circuit_mixing_benchmark.py` gates on
   order 2 and order 5.)
3. A language result in which organism plasticity is on the scoring path,
   with an ablation (plasticity off, circuits kept) showing it helps.

Reproduce: `python3 r3b_reference_baselines.py` (≈2 min; the n-gram peaks
at ≈0.7 GB RSS and the xz -9e encoder adds ≈0.7 GB; `--skip-primed` drops
the slow xz conditional pass).
