# TextBytes transport-2048 development protocol

`r6_text_temporal_transport_2048.py` is a separately versioned,
development-only continuation of the TextBytes transport pilot. It reuses the
v1 event path and frozen task exactly, with only these predeclared changes:

- persistent-scout evidence budget: `2048`;
- independent seeds: `2137` and `2271`;
- `2048` eligible acquisition examples per seed/held-out case;
- `64` continuous frozen evaluation examples per case;
- all eight leave-one-combination-out target cases for each seed (`16` cases);
- the existing 680-member candidate family (pairs and triples over the
  bias-excluded lag inputs), genuine no-credit for withheld acquisition frames,
  paired shuffled-reward null using the identical persistent scout and target
  inspection path, targeted lesion, and checkpoint/resume digest.

The protocol is frozen before execution. A persistent case passes only when
the evaluator-only target feature is installed and its targeted lesion reduces
held-out skill by more than `0.05`. The predeclared matrix criterion is at
least `75%` passing persistent cases. The paired null criterion is zero
target-feature installs in shuffled-reward cases. “Null install” means the
evaluator target feature specifically; unrelated variable-order features are
reported separately and are not relabeled as target discovery.

Every mode/case is subject to a 180-second case guard and a 2,000,000-byte
checkpoint-state guard. Execution, frozen-read-only, resume, no-credit,
balanced-coverage, and candidate-family checks are required alongside the
quality criteria. The resulting report is exploratory evidence only: it does
not modify `reports/r6-tier.json`, R6 thresholds, or acceptance status. English
pilots remain blocked pending review of this transport result.

The initial v2-2048 report was produced before the shuffled control was made
architecturally symmetric. Its zero null-target-install metric is invalid and
is retained only as a superseded historical artifact. The target-plus-positive-
lesion result was 4/16 (25%) against the predeclared 75% floor. This is a stop
condition: do not run the English pilot until a new predeclared architecture
experiment is approved; no threshold or R6 acceptance changes follow from the
invalid null metric.
