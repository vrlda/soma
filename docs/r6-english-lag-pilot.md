# Exploratory English lag-workspace pilot

`r6_english_lag_pilot.py` is development evidence, not an R6 acceptance run.
It compares three otherwise matched event paths:

* `lag_persistent`: `TextBytesTransducer` plus the generic depth-8 bounded
  lag workspace and the opt-in persistent scout (the predeclared 512-example
  evidence budget);
* `raw_text`: the raw `TextBytesTransducer` with variable-order learning; and
* `lag_additive`: the same lag workspace and organism topology without
  variable-order learning.

Acquisition bytes are the only bytes that receive next-bit outcomes. Each
validation/test partition is scored continuously by one independent clone,
without resetting at byte boundaries. The bridge uses explicit no-credit
drains during scoring; target-derived evaluation feedback is never sent.
Checkpoint state includes both the organism and lag workspace, and a JSON
round-trip digest is recorded.

The acquisition-fitted byte-unigram and suffix controls are conventional
baselines. Constant controls, if added for diagnostics, are explicitly oracle
diagnostics and cannot gate a result. Source/dependency hashes and corpus
hashes are written to the report. The pilot does not update `reports/r6-tier.json`
and does not qualify R6.
