# ADR 0005: Blank vs trained distribution

- Status: accepted for R0.
- Context: Master plan 2.1, 3.1, 3.2. Blank universal brain vs acquired trained brain, same architecture.
- Decision: blank = genesis cells, generic plasticity/structure rules, channels, limits, persistence, safety. No vocabulary, categories, skills, policy. Trained (e.g. Text v1) = descendant with acquired state.
- Constraints: sharing defaults to sanitized base snapshot; episodes, secrets, provenance, private deltas excluded unless selected. See `docs/genesis-state.md`.
- Test: genesis doc + `SOMA.create` baseline guard in `tests/test_r0_baseline.py`.
