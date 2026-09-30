# ADR 0009: The circuit memory is the organism's language substrate

- Status: accepted 2026-09-30. Master plan §22 step 6. This decision
  records the current evidence; it is open to revision by the re-entry
  criterion below.
- Question: SOMA's language score comes from `CircuitMixingMemory`, while
  the `Organism` cell/synapse network is not on the scoring path. Should
  organism-level plasticity be wired into the language path, or should the
  circuit memory be recognized as the organism's language substrate?
- Evidence:
  - **The organism path does not model English.** With the memory
    lesioned, the fused E0 run scores 0.994 bits/bit held-out, worse than
    the byte-unigram bar of 0.593 (`reports/r3b-fusion.json`).
  - **Fusion adds nothing.** Organism plus memory scores 0.3750937 vs
    0.3750949 for the memory alone, a difference of about 1e-6 (same
    report).
  - **There is a recorded stop condition.** The organism's temporal
    composition through the real text transducer passed 4/16 cases against
    a pre-declared 75% floor. English organism pilots are blocked until a
    new architecture experiment is approved
    (`docs/r6-text-temporal-transport-2048.md`).
  - **The memory carries SOMA's mechanisms, each shown causal:** evidence-
    gated growth (0.0033), plastic arbitration (0.0206), and budgeted
    reclamation (bounded, 7/7 gates) (ADR 0007), plus metaplastic
    consolidation (ADR 0008).
  - **It is domain-neutral.** It contains no text rules. Contexts are
    preceding 8-bit symbols, the unit every transducer already emits. It
    learns a seeded non-text Markov byte process to 0.233 bits/bit against
    a 0.198 entropy floor and a 0.5 unigram
    (`tests/test_r6_circuit_mixing.py`).
- Decision:
  1. `CircuitMixingMemory` is the organism's **sequence-prediction memory
     subsystem**, part of the memory hierarchy in master plan §5.5. It is
     the language substrate for Text v1.
  2. For §21's "SOMA core has causal language contribution", *core* means
     the organism's event core plus its memory subsystems. The causal
     evidence is the memory's ablations. This interpretation is explicit
     here so it cannot drift silently.
  3. The organism cell/synapse network stays off the language scoring
     path. Its role remains control, routing, and the non-text domains
     (R11 glyphs, R12 cart).
- Re-entry criterion: organism-level plasticity rejoins the language path
  only through a pre-declared experiment that improves a frozen language
  metric over the memory alone, with a causal ablation. The designated
  first candidate targets the open step-5 residue. The organism's
  calibrated context detector (R1) would route the memory's arbitration
  among per-context weight banks, to cut the remaining weight drift of
  0.0009–0.0014.
- Consequences: step 7 (feature parity for the product) proceeds on the
  circuit memory. The R6 English organism pilots stay as development
  evidence and do not gate anything.
