# ADR 0004: Transducer boundary

- Status: accepted for R0.
- Context: Master plan 5.2. Adapters translate interfaces; they hold no task intelligence.
- Decision: adapters declare channel schemas, sampling, units, clocks, reversible encoding, hardware needs, safety bounds. Must never inject task identity, labels, answers, future info, pretrained solutions, semantic categories.
- Constraints: boundary test fails any envelope carrying `task_id`, `phase`, `target`, `schedule` into organism state. Minimally interpreted path required for universality claim.
- Test: `tests/test_r0_contracts.py` boundary rejection case.
