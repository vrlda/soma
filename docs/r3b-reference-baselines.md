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

## Suggested next gates (proposal)

1. Report these references in every future E-series result.
2. Add a gate "beats frozen order-2 byte n-gram", then order 3.
3. An ablation that disables organism structural plasticity (growth,
   pruning, consolidation) while keeping sequence circuits, to measure how
   much of the language score depends on the SOMA-specific mechanisms.

Reproduce: `python3 r3b_reference_baselines.py` (≈2 min; the n-gram peaks
at ≈0.7 GB RSS and the xz -9e encoder adds ≈0.7 GB; `--skip-primed` drops
the slow xz conditional pass).
