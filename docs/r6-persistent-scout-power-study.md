# Persistent-scout power study

`r6_persistent_scout_power_study.py` is a development-only, predeclared
power study. It does not modify scout thresholds or R6 acceptance reports.
Budgets are fixed at 512, 1024, and 2048. Each target-present case is a
leave-one-combination-out stream over the existing temporal fixture; a paired
shuffled-reward row is the null. Seeds, held-out combinations, evidence
minimum, and the lesion margin are recorded in the report.

A target case passes only when the target is the first persistent winner, the
target feature is actually installed, and an independent held-out score drops
by more than 0.05 after the target path lesion. A budget qualifies only when
at least 75% of target cases pass and both shuffled-null accept and install
rates are zero. Incomplete staged matrices never qualify a budget.

The command defaults to two staged cases (`--max-cases 2`) to keep local
development bounded. Use `--max-cases` explicitly for a larger predeclared
matrix; no result is silently promoted to acceptance.

The completed 16-case matrix (seeds `137,271`, all eight held-out
combinations, 32 held-out examples per case) took 1064.3 seconds. The
predeclared result was:

| budget | target pass | target-first | install | positive lesion | shuffled null accept/install | selected |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 512 | 56.25% | 75% | 56.25% | 56.25% | 0% / 0% | no |
| 1024 | 56.25% | 75% | 56.25% | 56.25% | 0% / 0% | no |
| 2048 | 75% | 75% | 75% | 75% | 0% / 0% | smallest development qualifier |

The earlier staged report made before the lesion-sign correction is retained
as an invalid historical artifact and is not used for interpretation.
