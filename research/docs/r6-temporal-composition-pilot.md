# R6 temporal-composition pilot

`r6_temporal_composition_pilot.py` (protocol v3) is a development-only, non-gating
mechanism pilot. It does not modify or reinterpret the locked R6 tier reports.

The fixture uses the existing `HistoryBitsTransducer` channel schema and
`EventBridge`. Raw iid 3-bit symbols are exposed as an eight-deep lag
workspace. For the signed core representation, the existing bit adapter is
subclassed only to center its declared bit channels to `{-1,+1}`; no task
feature or target is added to an event. The evaluator target is the product of
the nonadjacent taps `lag0.bit0 × lag2.bit1 × lag5.bit2`, represented by the
declared core feature key `k3:input-000|input-007|input-017`.

Every case is leave-one-combination-out over all eight combinations. Acquisition
uses iid raw symbols while retaining seven combinations; held-out examples
retain only the withheld combination. The model is compared with identical
legacy variable-order, explicitly persistent-scout variable-order,
nonlinear-additive, and shuffled-reward Organisms plus acquisition-fitted
order-3 and order-5 suffix-to-action controls. Persistent mode carries only
candidate sum/sumsq/count statistics across fingerprint episodes, uses
prediction-error-centered credit, and has a frozen calibration-selected
declaration of 256 action outcomes, 32 observations per candidate, and a conservative
multiplicity-adjusted winner/runner heuristic over the full declared
pair/triple family (not a calibrated FWER test). It runs consecutive 16-step
fingerprints until the first budget-complete terminal decision. Constant plus,
constant minus, wrong-coordinate, and
deterministic random-k3 lesion controls are reported diagnostically. Controls
never fit validation or held-out examples.

Held-out scoring is one-shot and read-only: each workspace is scored from an
independent acquisition-checkpoint clone with no target-derived outcome. The
source checkpoint digest must remain unchanged. The report includes exact
checkpoint/resume digest checks, resource/state metrics, target-feature
installation and targeted lesion results, dependency/source hashes, and
environment metadata. A lesion is counted only when the exact declared k3
target feature is installed; unrelated k3 features or bias/current-bit
features do not qualify. Legacy and persistent modes are matched on streams and
action tapes; these extra controls do not gate R6.

Run a small smoke report with:

```text
python3 research/benchmarks/r6_temporal_composition_pilot.py --seed 0 --acquisition-examples 64 --evaluation-examples 4 --report /tmp/r6-temporal.json
```

The report always states `development_only: true`, `acceptance_gating: false`,
and `r6_qualification: not_evaluated`. Partial target-k3 installation or a
positive mean lesion is not a success claim: a credible mechanism result would
also require broad target-k3 coverage, consistent targeted lesion degradation,
and consistent held-out advantage over the matched controls. Any observed
feature installation or lesion effect is exploratory evidence and is not an R6
qualification claim.
