# R6 English event-path pilot

`r6_english_event_pilot.py` is a small, development-only mechanism pilot. It
is not an R6 gate and does not modify or reinterpret `r6-tier-v2` evidence.

The fixture uses the existing `TextBytesTransducer` and `EventBridge`: each
UTF-8 bit and byte-boundary flag crosses the universal event boundary, while
the next-bit target is evaluator-owned and never appears in an event payload.
The generic Organism is run with and without
`enable_variable_order_learning(max_order=3)` using identical starting
topology and acquisition stream. Controls are an acquisition-fitted variable-
order suffix table and an acquisition-fitted byte unigram.

Acquisition, validation, and test byte partitions are frozen in the script and
hashed in every report. Controls and the Organism train only on acquisition.
Held-out validation/test scoring is read-only: each byte is evaluated from an
independent acquisition-checkpoint clone. Zero outcomes are used only to
satisfy EventBridge's delayed-action contract inside the disposable clone;
no target-derived held-out outcome is sent, and the source checkpoint digest
must remain unchanged. Checkpoint round-trip digest equality, resource/state
metrics, dependency hashes, repository revision, and environment metadata are
recorded.

If a product feature is installed, the pilot clones the checkpoint again and
zeros only that feature's motor edge. The lesion reports validation/test NLL
delta in bits. A k3 path is reported separately when present; a smaller
product path is still identified honestly rather than relabeled as k3.

Run the four-seed pilot with:

```text
python3 r6_english_event_pilot.py --report /tmp/r6-english-event-pilot.json
```

The report always contains `development_only: true`,
`acceptance_gating: false`, and `r6_qualification: not_evaluated`. Any NLL
margin or lesion effect is exploratory evidence only. In particular, a small
aggregate lesion delta is not a causal success claim unless it is consistent
across seeds and also clears the conventional suffix/unigram comparisons;
this pilot must not be promoted to R6 qualification without a separately
frozen protocol and review.
