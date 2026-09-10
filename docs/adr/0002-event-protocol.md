# ADR 0002: Universal event protocol

- Status: accepted for R0.
- Context: Master plan 5.1. Native input is time-ordered event envelope, not token or fixed vector.
- Decision: `formats/event-protocol/v1.json` defines Event / ActionEvent / OutcomeEvent + channel, clock, correlation. Validator at `soma/r0/event_schema.py` (stdlib only, no organism import).
- Constraints: core learns compositions of events; never assumes words, pixels, motor commands. Rates synced by clocks/buffers.
- Test: `tests/test_r0_contracts.py` valid/invalid envelopes + action/outcome correlation.
