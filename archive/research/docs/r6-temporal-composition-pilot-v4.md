# Temporal composition pilot v4

`r6_temporal_composition_pilot_v4.py` is a development-only continuation. It
reuses the frozen v3 leave-one-combination-out task, streams, controls,
thresholds, lesions, read-only scoring, and resume checks. It does not alter
R6 tier acceptance.

The v3 budget 256 failed transportability in the frozen leave-one-out pilot
(5/8 persistent target-feature installs, inconsistent causal lesion). V4 uses
the already-predeclared 512 evidence budget as an exploratory continuation;
it is pending matched calibration and is not an R6 gate.

V4 reports, per held-out case, target rank, held-out sign, persistent skill,
margins versus the nonlinear-additive and acquisition-fitted suffix order-3/
order-5 controls, target-feature installation, and targeted lesion effect. It
also retains all v3 telemetry, checkpoint/resume, frozen read-only, wrong-
coordinate, random-k3, shuffled-reward, and constant controls.

Run the single frozen eight-case leave-one-out seed:

```text
python3 research/benchmarks/r6_temporal_composition_pilot_v4.py \
  --seed 0 --acquisition-examples 512 --evaluation-examples 16 \
  --report /tmp/r6-temporal-composition-pilot-v4.json
```
