# Persistent scout calibration

`r6_persistent_scout_calibration.py` is a frozen, development-only calibration
protocol. It does not modify R6 acceptance or the locked tier reports.

The protocol was declared before execution with seeds `0,1,2,3`, budgets
`128,256,512`, 512 acquisition examples per run, minimum candidate count 32,
and the complete 2,600 pair/triple family. Target-present streams and null
streams use independent seeds. Nulls are both shuffled-reward and independent
random-reward streams.

The predeclared selection rule was:

- target-present accept rate at least 0.75 (3/4 seeds);
- maximum null accept rate 0.0 and maximum null false-install rate 0.0;
- select the smallest budget satisfying all criteria.

The empirical null target is a fixed 5% family-wise false-positive objective;
with four seeds this calibration requires zero observed null accepts. The
scout's multiplicity adjustment remains a conservative heuristic, not a
formal FWER-calibrated test.

Observed calibration selected budget 256:

| budget | target accept | shuffled null | random null | selected |
| --- | ---: | ---: | ---: | --- |
| 128 | 0.25 | 0.00 | 0.00 | no |
| 256 | 0.75 | 0.00 | 0.00 | yes |
| 512 | 1.00 | 0.00 | 0.00 | no (larger) |

The selected value is frozen in temporal pilot protocol v3 as
`PERSISTENT_EVIDENCE_BUDGET = 256`. Calibration reports include per-run target
rank, winner z, runner score, winner/runner margin, maximum statistic, terminal
decision, install status, state bytes, and stream/checkpoint digests.

Run with:

```text
python3 research/benchmarks/r6_persistent_scout_calibration.py --report /tmp/r6-persistent-scout-calibration-v1.json
```
