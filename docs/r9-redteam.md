# R9 Red Team (`r9_redteam_benchmark.py`, all gates pass)

Threat model (adversarial local user / untrusted text):

- Correction hijack: lower-trust supersede of a higher-trust rule is
  blocked and logged (`blocked-lower-trust`); equal trust keeps
  latest-wins so legitimate corrections still work.
- Flood: per-provenance episodic quotas bound blast radius; quota breach
  raises; legitimate rules survive.
- Bulk poisoning: untrusted `brain-teach --trust untrusted` stages into
  quarantine with zero learning (byte-identical predictions); explicit
  `brain-approve` crosses the boundary.
- Prompt injection: known patterns refused; benign neighbors pass (R7).

Ranks: system 3, correction 2, user/approved 1, untrusted 0; unknown
labels default to user level (compat, documented). Conflict and blocked
logs are bounded (1024) and persisted.

Operations (`tests/test_r9_ops.py`): consent-gated telemetry (metadata
only by default, content needs a second flag), content-free support
bundles, runbook set (`docs/runbooks.md` with taxonomy), and the R7
preference harness (blind pairs, agreement, win rates; synthetic demo
rater included, human judgments pending).
