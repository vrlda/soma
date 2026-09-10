"""Pure calibrated circuit-evidence calculations (R1, Section 6).

No organism imports. No randomness. No behavior control.
All functions deterministic given inputs.
"""

import math

SCHEMA_VERSION = 1
STUDENT_T_DF = 5.0
LLR_CLIP = 5.0
MIN_SCALE = 1e-3
MAX_EFF_N = 1000000.0


class CircuitEvidence(object):
    """Versioned per-circuit outcome model + uncertainty + calibration bins."""

    VERSION = 1

    def __init__(self, circuit_id, outcome_mean=0.0, outcome_scale=1.0, eff_n=0.0):
        self.circuit_id = str(circuit_id)
        self.model_version = self.VERSION
        self.outcome_mean = float(outcome_mean)
        self.outcome_scale = max(float(outcome_scale), MIN_SCALE)
        self.eff_n = float(eff_n)
        self.log_evidence = 0.0
        self.posterior = 0.0
        self.normal_outcomes = 0
        self.probe_outcomes = 0
        self.scored_outcomes = 0
        self.probe_log_evidence = 0.0
        self.probe_scored = 0
        self.last_used_step = -1
        self.reuse_count = 0
        self.stability = 0.0
        self.protected = False
        self.pending_ids = []
        self.calibration_bins = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.calibration_hits = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    def to_dict(self):
        return {
            "version": self.VERSION,
            "circuit_id": self.circuit_id,
            "model_version": self.model_version,
            "outcome_mean": self.outcome_mean,
            "outcome_scale": self.outcome_scale,
            "eff_n": self.eff_n,
            "log_evidence": self.log_evidence,
            "posterior": self.posterior,
            "normal_outcomes": self.normal_outcomes,
            "probe_outcomes": self.probe_outcomes,
            "scored_outcomes": self.scored_outcomes,
            "probe_log_evidence": self.probe_log_evidence,
            "probe_scored": self.probe_scored,
            "last_used_step": self.last_used_step,
            "reuse_count": self.reuse_count,
            "stability": self.stability,
            "protected": self.protected,
            "pending_ids": list(self.pending_ids),
            "calibration_bins": list(self.calibration_bins),
            "calibration_hits": list(self.calibration_hits),
        }

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, dict):
            raise ValueError("CircuitEvidence payload must be a mapping")
        if payload.get("version") != cls.VERSION:
            raise ValueError("unsupported CircuitEvidence version: %r" % (payload.get("version"),))
        item = cls(payload.get("circuit_id", "?"))
        for key in ("outcome_mean", "outcome_scale", "eff_n", "log_evidence", "posterior", "stability",
                    "probe_log_evidence"):
            value = float(payload.get(key, 0.0))
            if not math.isfinite(value):
                raise ValueError("CircuitEvidence field %s must be finite" % key)
            setattr(item, key, value)
        item.outcome_scale = max(item.outcome_scale, MIN_SCALE)
        for key in ("normal_outcomes", "probe_outcomes", "scored_outcomes", "probe_scored",
                    "last_used_step", "reuse_count"):
            value = int(payload.get(key, 0))
            if value < -1:
                raise ValueError("CircuitEvidence field %s must be >= -1" % key)
            setattr(item, key, value)
        item.model_version = int(payload.get("model_version", cls.VERSION))
        item.protected = bool(payload.get("protected", False))
        item.pending_ids = list(payload.get("pending_ids", []))
        bins = list(payload.get("calibration_bins", [0] * 10))
        hits = list(payload.get("calibration_hits", [0] * 10))
        if len(bins) != 10 or len(hits) != 10:
            raise ValueError("calibration bins must have length 10")
        item.calibration_bins = [int(v) for v in bins]
        item.calibration_hits = [int(v) for v in hits]
        for value in item.calibration_bins + item.calibration_hits:
            if value < 0:
                raise ValueError("calibration counts must be nonnegative")
        return item


def student_t_logpdf(x, mean, scale, df=STUDENT_T_DF):
    scale = max(float(scale), MIN_SCALE)
    z = (float(x) - float(mean)) / scale
    half = (float(df) + 1.0) / 2.0
    return (
        math.lgamma(half)
        - math.lgamma(float(df) / 2.0)
        - 0.5 * math.log(float(df) * math.pi)
        - math.log(scale)
        - half * math.log(1.0 + z * z / float(df))
    )


def comparative_log_likelihood(evidence, outcome):
    """Clipped log-likelihood of one outcome under a circuit's model. No mutation."""
    if not math.isfinite(float(outcome)):
        raise ValueError("outcome must be finite")
    return max(-LLR_CLIP, min(LLR_CLIP, student_t_logpdf(
        float(outcome), evidence.outcome_mean, evidence.outcome_scale)))


def update_evidence(evidence, outcome, from_probe=False, decay=0.999, step=-1, accumulate=True):
    """Bounded local update of sufficient stats. Returns clipped log-likelihood."""
    if not math.isfinite(float(outcome)):
        raise ValueError("outcome must be finite")
    ll = comparative_log_likelihood(evidence, outcome)
    # Online mean/scale update with bounded decay on sufficient stats.
    eff = min(MAX_EFF_N, evidence.eff_n * float(decay) + 1.0)
    delta = float(outcome) - evidence.outcome_mean
    mean = evidence.outcome_mean + delta / eff
    var = evidence.outcome_scale * evidence.outcome_scale
    var = ((eff - 1.0) / eff) * var + (delta * delta) / (eff * eff) if eff > 1.0 else var
    evidence.outcome_mean = mean
    evidence.outcome_scale = max(math.sqrt(max(var, MIN_SCALE * MIN_SCALE)), MIN_SCALE)
    evidence.eff_n = eff
    if accumulate:
        evidence.log_evidence += max(-LLR_CLIP, min(LLR_CLIP, ll))
    if from_probe:
        evidence.probe_outcomes += 1
    else:
        evidence.normal_outcomes += 1
    if step >= 0:
        evidence.last_used_step = int(step)
    # Calibration bin by posterior confidence decile.
    bin_index = min(9, max(0, int(evidence.posterior * 10.0)))
    evidence.calibration_bins[bin_index] += 1
    return max(-LLR_CLIP, min(LLR_CLIP, ll))


def normalize_posterior(log_scores, novelty_mass=0.05):
    """Normalize log evidence scores with explicit reserved novelty mass."""
    if not 0.0 < float(novelty_mass) < 1.0:
        raise ValueError("novelty_mass must be in (0, 1)")
    if not log_scores:
        return {"__novelty__": 1.0}
    maximum = max(float(v) for v in log_scores.values())
    weights = {k: math.exp(float(v) - maximum) for k, v in log_scores.items()}
    total = sum(weights.values())
    scale = (1.0 - float(novelty_mass)) / total if total > 0.0 else 0.0
    posterior = {k: weights[k] * scale for k in weights}
    posterior["__novelty__"] = float(novelty_mass)
    return posterior


def should_switch(challenger_post, incumbent_post, challenger_n, incumbent_n,
                  min_evidence=16, enter_margin=0.15, exit_margin=0.05):
    """Hysteresis rule. Enter bar stricter than exit bar. One noisy reward can't switch."""
    if int(challenger_n) < int(min_evidence) or int(incumbent_n) < int(min_evidence):
        return False
    return (float(challenger_post) - float(incumbent_post)) >= float(enter_margin)


def should_return_to_incumbent(challenger_post, incumbent_post, exit_margin=0.05):
    return (float(incumbent_post) - float(challenger_post)) >= float(exit_margin)


def probe_tally_posterior(candidate_ll, incumbent_ll, novelty_mass=0.05):
    """Posterior over {candidate, incumbent, novelty} from probe tallies."""
    return normalize_posterior(
        {"candidate": float(candidate_ll), "incumbent": float(incumbent_ll)},
        novelty_mass=novelty_mass,
    )


def probe_decision(candidate_ll, candidate_n, incumbent_ll, incumbent_n,
                   min_probe=2, enter_margin=0.15, exit_margin=0.05,
                   novelty_mass=0.05):
    """Bounded probe verdict: accept / reject / continue. Pure."""
    if int(candidate_n) < int(min_probe) or int(incumbent_n) < int(min_probe):
        return "continue"
    posterior = probe_tally_posterior(candidate_ll, incumbent_ll, novelty_mass=novelty_mass)
    advantage = posterior["candidate"] - posterior["incumbent"]
    if advantage >= float(enter_margin):
        return "accept"
    if -advantage >= float(exit_margin):
        return "reject"
    return "continue"


def calibration_metrics(posteriors, outcomes):
    """NLL, Brier, ECE over locked binary outcome stream. Pure, deterministic."""
    if len(posteriors) != len(outcomes):
        raise ValueError("posteriors and outcomes must align")
    n = len(posteriors)
    if n == 0:
        return {"n": 0, "nll": 0.0, "brier": 0.0, "ece": 0.0}
    nll = 0.0
    brier = 0.0
    bins = [[0, 0.0] for _ in range(10)]
    for prob, outcome in zip(posteriors, outcomes):
        prob = min(1.0 - 1e-9, max(1e-9, float(prob)))
        target = 1.0 if outcome else 0.0
        nll += -(target * math.log(prob) + (1.0 - target) * math.log(1.0 - prob))
        brier += (prob - target) ** 2
        idx = min(9, max(0, int(prob * 10.0)))
        bins[idx][0] += 1
        bins[idx][1] += target
    ece = 0.0
    for idx, (count, hits) in enumerate(bins):
        if count:
            center = (idx + 0.5) / 10.0
            ece += abs(hits / count - center) * (count / n)
    return {"n": n, "nll": nll / n, "brier": brier / n, "ece": ece}
