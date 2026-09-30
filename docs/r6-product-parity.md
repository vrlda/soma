# Circuit mixing: product feature parity

Master plan §22 step 7. The circuit-mixing memory can now replace
`SequenceCircuitMemory` wherever the product uses a background language
memory. This page records what was ported, how it is verified, and what is
deliberately out of scope.

## What the product actually needs

A survey of every caller (service, instruction router, generation,
dialogue, R4, R9) shows the background memory is used only through:
`observe(bit, learn, weight)`, `distribution()`, `reset_history()`, and
`symbol_index`, plus the organism's quarantine and checkpoints.
`SequenceCircuitMemory.unobserve` and `consolidate` have no product caller.
Forgetting in the service removes episodic rules (`forget_fact`), so exact
forgetting stays at the episodic layer.

## Ported (Python reference and Rust port, bit-exact)

| Feature | Behavior |
|---|---|
| `distribution()` / `probability()` | returns `({0: p0, 1: p1}, order)`. `order` is the bit length of the longest live context, so the router's byte-scale evidence check keeps its meaning |
| trust-weighted `observe(..., weight=w)` | counts rise by `w`, capped; arbitration updates once. Used by quarantine approval |
| binary state `SOMAMIX1` | `dumps()` / `loads()`: canonical header, sorted circuits, fixed float block, FNV-1a checksum. **Python and Rust write byte-identical files** and load each other's |
| organism ownership | `Organism.enable_sequence_memory((0, 1), kind="mixing")`. Checkpoints embed the state; quarantine and approval work unchanged |

## Verification

- `engine/differential_mixer.py`, six configurations (trust weight 3 on
  the second document, tight budgets, hashed orders, every mechanism
  toggled): identical probabilities and weights, **byte-identical saved
  state**, Rust scoring from Python's state equal, Python reload equal.
- **At scale:** the compact E2 brain (3.8M circuits) saves to 107.5 MB. A
  fresh process loads it and scores Jekyll validation in 1.8 s (369 MB RSS),
  reproducing 0.24576 exactly.
- **Gates on the new memory:** R4 lifelong **6/6**
  (`reports/r4-lifelong-mixing.json`; background E0 held-out 0.2648 vs
  0.3751 for the old memory) and R9 red-team **7/7**
  (`reports/r9-redteam-mixing.json`). The default runs still reproduce the
  frozen `r4-lifelong.json` and `r9-redteam.json` byte-for-byte.
- **Unit tests:** the distribution contract, weight capping, round trip and
  corruption detection, and organism checkpoint plus quarantine.

## Out of scope, with reasons

- **Exact statistical unforgetting.** Counts are capped and arbitration
  weights are plastic, so unlearning a document from the statistical memory
  cannot be exact. The product forgets at the episodic layer; that is where
  exactness is kept.
- **Chunk promotion.** It measured zero gain in ADR 0006 and has no
  product caller.
- **Serving from the engine.** That is step 8. The Python reference runs
  at about 19 µs per bit, which suits tests and small brains but not
  E2-scale brains (millions of circuits). Those must be served by the Rust
  engine.
