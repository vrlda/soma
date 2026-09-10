# ADR 0003: Active workspace vs persistent memory

- Status: accepted for R0.
- Context: Master plan 2.1, 5.5. Brain is persistent state, not context window.
- Decision: bounded disposable workspace (current events, activations, goals, retrieved candidates) separate from persistent brain (topology, evidence, episodic/semantic stores).
- Constraints: old info affects behavior only via consolidation, retrieval, or re-supply. Finite limits published: buffer size, active cells/event, retrieval bandwidth, storage. See `docs/workspace-contract.md`.
- Test: workspace doc + bounds assertion in `tests/test_r0_contracts.py`.
