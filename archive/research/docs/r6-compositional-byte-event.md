# R6 development discriminator: byte/event product task

`r6_compositional_byte_event_benchmark.py` is a development-only mechanism
experiment (protocol v2). It is not an R6 tier gate and does not modify,
regenerate, or interpret the locked `research/reports/r6-tier.json` acceptance result.

The fixture emits one universal `Event` envelope per transition. Its only
domain payload is an opaque integer in `[0, 7]`; the transducer decodes the
three bits to simultaneous `(-1, +1)` sensor values. The evaluator computes
the product target outside the event boundary and supplies only a delayed
scalar outcome. Event validation rejects `target`, `label`, `answer`, and
other task metadata.

The fixture rotates leave-one-combination-out over all eight symbols. For each
case, the other seven symbols are balanced and deterministically shuffled for
each frozen seed, so both product signs are covered. The model is the existing
target-blind `enable_variable_order_learning(max_order=3)` path. The matched
control is explicitly named `nonlinear_additive_context`: it has the same
hidden-afferent graph and actor learning but no order-N product substrate. A
shuffled-reward control, constant positive/negative/zero controls, and a table
diagnostic are reported without entering the organism.

Held-out scoring is read-only and independent per frame. Each symbol is scored
by cloning the acquisition checkpoint, emitting exactly one byte event, and
discarding the clone. No held-out `OutcomeEvent` is sent, and the source
checkpoint digest must remain unchanged. The intact and lesion branches use
the same frozen evaluation procedure. The causal check zeros only the
installed `k3:input-000|input-001|input-002` feature-to-owner edge in a clone.
The report includes skill, MAE, and sign accuracy. A positive diagnostic
requires at least two fixed seeds, balanced all-symbol/positive-negative
coverage, a predeclared macro-average 0.10 margin over the matched additive
control, and a predeclared macro-average 0.05 lesion drop. These are
diagnostic criteria, not qualification gates; the report always contains
`development_only: true`, `acceptance_gating: false`, and
`r6_qualification: not_evaluated`.

Run a local smoke report with:

```text
python3 research/benchmarks/r6_compositional_byte_event_benchmark.py --report /tmp/r6-product-dev.json
```

The report records the frozen manifest digest, event/transducer schema, source
hash, hashes of the relevant Organism/EventBridge/transducer dependencies,
repository revision/dirty status, environment metadata, stream hashes,
matched starting structure, all controls, and lesion metrics. It intentionally
does not claim semantic factuality or R6 qualification.
