# ADR 0001: Runtime, Brain, Transducer separation

- Status: accepted for R0.
- Context: Master plan 2.1, 3.4, 12. Code is not the brain.
- Decision: three independent artifacts. Runtime = executable engine + CLI/API. Brain = mutable topology, numeric state, memories, evidence, lineage. Transducer = signed adapter mapping external medium to event protocol.
- Constraints: runtime update never overwrites brain; clone creates distinct identity/lineage; brain identity + ancestor recorded.
- Test: `tests/test_r0_contracts.py` asserts separation doc exists; future `.soma` manifest enforces hashes.
