# Exploratory TextBytes lag-workspace transport pilot

`r6_text_temporal_transport_pilot.py` carries the previously exercised sparse
temporal composition fixture through the real `TextBytesTransducer` event
path and the domain-neutral `BoundedLagWorkspaceTransducer`. The depth-8
workspace only retains declared signals, signed-centers binary values, pads
the transient with zero, and serializes its wrapped adapter state.

The held-out cases are leave-one-combination-out cases over all eight signs of
the three lag taps. Target labels are evaluator-side; event payloads contain
only raw bit and byte-boundary signals. Acquisition controls are fitted only
on eligible acquisition frames. Withheld acquisition frames remain in the
event/lag stream but use the bridge's explicit no-credit drain; they are not
represented as numeric zero rewards. No-credit suppresses reward-derived
credit only; the current observation still propagates through the organism to
preserve event timing and workspace state. The persistent and shuffled-reward
modes use the same persistent scout, candidate family, target inspection, and
lesion path; only reward-to-event pairing differs. Additive and raw-text modes
are compared with continuous frozen evaluation, checkpoint
round-trip digests, source/dependency hashes, and a winner-path lesion when a
target feature is actually installed. Byte-unigram and suffix controls are
ordinary baselines; constant controls would be oracle diagnostics only.

This pilot is explicitly development-only and does not alter or qualify the
locked R6 tier report.

The first v2-2048 execution predates this control correction: its shuffled
mode did not enable the persistent scout, so its reported zero null-target
installs are invalid and must not be used as a null result. Its independent
target quality result remains 4/16 target-plus-positive-lesion cases versus
the frozen 75% floor. The protocol therefore stops here; no English pilot is
authorized until a newly predeclared architecture experiment is reviewed.
