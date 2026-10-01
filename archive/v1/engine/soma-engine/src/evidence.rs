//! Calibrated circuit-evidence calculations, mirroring soma/routing/evidence.py.
//!
//! Pure, deterministic, organism-free. Student-t likelihoods use libm so
//! special functions match reference behavior within test tolerances.

use serde::{Deserialize, Serialize};

pub const STUDENT_T_DF: f64 = 5.0;
pub const LLR_CLIP: f64 = 5.0;
pub const MIN_SCALE: f64 = 1e-3;
pub const MAX_EFF_N: f64 = 1000000.0;

fn clip_ll(value: f64) -> f64 {
    value.max(-LLR_CLIP).min(LLR_CLIP)
}

pub fn student_t_logpdf(x: f64, mean: f64, scale: f64, df: f64) -> f64 {
    let scale = scale.max(MIN_SCALE);
    let z = (x - mean) / scale;
    let half = (df + 1.0) / 2.0;
    libm::lgamma(half) - libm::lgamma(df / 2.0)
        - 0.5 * libm::log(df * std::f64::consts::PI)
        - libm::log(scale)
        - half * libm::log(1.0 + z * z / df)
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CircuitEvidence {
    pub outcome_mean: f64,
    pub outcome_scale: f64,
    pub eff_n: f64,
    pub log_evidence: f64,
    pub posterior: f64,
    pub normal_outcomes: i64,
    pub probe_outcomes: i64,
    pub last_used_step: i64,
    pub calibration_bins: [i64; 10],
}

impl Default for CircuitEvidence {
    fn default() -> Self {
        CircuitEvidence {
            outcome_mean: 0.0,
            outcome_scale: 1.0,
            eff_n: 0.0,
            log_evidence: 0.0,
            posterior: 0.0,
            normal_outcomes: 0,
            probe_outcomes: 0,
            last_used_step: -1,
            calibration_bins: [0; 10],
        }
    }
}

pub fn comparative_log_likelihood(evidence: &CircuitEvidence, outcome: f64) -> f64 {
    clip_ll(student_t_logpdf(outcome, evidence.outcome_mean, evidence.outcome_scale, STUDENT_T_DF))
}

pub fn update_evidence(
    evidence: &mut CircuitEvidence,
    outcome: f64,
    from_probe: bool,
    decay: f64,
    step: i64,
    accumulate: bool,
) -> f64 {
    let ll = comparative_log_likelihood(evidence, outcome);
    let eff = MAX_EFF_N.min(evidence.eff_n * decay + 1.0);
    let delta = outcome - evidence.outcome_mean;
    let mean = evidence.outcome_mean + delta / eff;
    let mut var = evidence.outcome_scale * evidence.outcome_scale;
    if eff > 1.0 {
        var = ((eff - 1.0) / eff) * var + (delta * delta) / (eff * eff);
    }
    evidence.outcome_mean = mean;
    evidence.outcome_scale = var.max(MIN_SCALE * MIN_SCALE).sqrt().max(MIN_SCALE);
    evidence.eff_n = eff;
    if accumulate {
        evidence.log_evidence += clip_ll(ll);
    }
    if from_probe {
        evidence.probe_outcomes += 1;
    } else {
        evidence.normal_outcomes += 1;
    }
    if step >= 0 {
        evidence.last_used_step = step;
    }
    let bin = ((evidence.posterior * 10.0) as i64).max(0).min(9) as usize;
    evidence.calibration_bins[bin] += 1;
    clip_ll(ll)
}

pub fn normalize_posterior(scores: &[(String, f64)], novelty_mass: f64) -> Vec<(String, f64)> {
    if scores.is_empty() {
        return vec![("__novelty__".to_string(), 1.0)];
    }
    let maximum = scores.iter().map(|(_, v)| *v).fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<(String, f64)> = scores.iter().map(|(k, v)| (k.clone(), (*v - maximum).exp())).collect();
    let total: f64 = weights.iter().map(|(_, w)| *w).sum();
    let scale = if total > 0.0 { (1.0 - novelty_mass) / total } else { 0.0 };
    let mut posterior: Vec<(String, f64)> = weights.iter().map(|(k, w)| (k.clone(), w * scale)).collect();
    posterior.push(("__novelty__".to_string(), novelty_mass));
    posterior
}

pub fn should_switch(
    challenger_post: f64,
    incumbent_post: f64,
    challenger_n: i64,
    incumbent_n: i64,
    min_evidence: i64,
    enter_margin: f64,
) -> bool {
    if challenger_n < min_evidence || incumbent_n < min_evidence {
        return false;
    }
    challenger_post - incumbent_post >= enter_margin
}

pub fn should_return_to_incumbent(challenger_post: f64, incumbent_post: f64, exit_margin: f64) -> bool {
    incumbent_post - challenger_post >= exit_margin
}

pub fn probe_decision(
    candidate_ll: f64,
    candidate_n: i64,
    incumbent_ll: f64,
    incumbent_n: i64,
    min_probe: i64,
    enter_margin: f64,
    exit_margin: f64,
    novelty_mass: f64,
) -> &'static str {
    if candidate_n < min_probe || incumbent_n < min_probe {
        return "continue";
    }
    let posterior = normalize_posterior(
        &[("candidate".to_string(), candidate_ll), ("incumbent".to_string(), incumbent_ll)],
        novelty_mass,
    );
    let candidate = posterior.iter().find(|(k, _)| k == "candidate").map(|(_, v)| *v).unwrap_or(0.0);
    let incumbent = posterior.iter().find(|(k, _)| k == "incumbent").map(|(_, v)| *v).unwrap_or(0.0);
    let advantage = candidate - incumbent;
    if advantage >= enter_margin {
        "accept"
    } else if -advantage >= exit_margin {
        "reject"
    } else {
        "continue"
    }
}

pub fn calibration_metrics(posteriors: &[f64], outcomes: &[bool]) -> (f64, f64, f64) {
    assert_eq!(posteriors.len(), outcomes.len());
    let n = posteriors.len();
    if n == 0 {
        return (0.0, 0.0, 0.0);
    }
    let mut nll = 0.0;
    let mut brier = 0.0;
    let mut bins = [(0usize, 0.0f64); 10];
    for (prob, outcome) in posteriors.iter().zip(outcomes.iter()) {
        let prob = prob.max(1e-9).min(1.0 - 1e-9);
        let target = if *outcome { 1.0 } else { 0.0 };
        nll += -(target * libm::log(prob) + (1.0 - target) * libm::log(1.0 - prob));
        brier += (prob - target).powi(2);
        let idx = ((prob * 10.0) as usize).min(9);
        bins[idx].0 += 1;
        bins[idx].1 += target;
    }
    let mut ece = 0.0;
    for (idx, (count, hits)) in bins.iter().enumerate() {
        if *count > 0 {
            let center = (idx as f64 + 0.5) / 10.0;
            ece += ((hits / *count as f64) - center).abs() * (*count as f64 / n as f64);
        }
    }
    let n = n as f64;
    (nll / n, brier / n, ece)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_noise_cannot_switch() {
        assert!(!should_switch(0.9, 0.1, 1, 100, 16, 0.15));
        assert!(should_switch(0.8, 0.4, 32, 64, 16, 0.15));
    }

    #[test]
    fn empty_scores_yield_full_novelty() {
        let posterior = normalize_posterior(&[], 0.05);
        assert_eq!(posterior, vec![("__novelty__".to_string(), 1.0)]);
    }

    #[test]
    fn probe_floor_holds() {
        assert_eq!(probe_decision(0.0, 1, -10.0, 1, 2, 0.15, 0.05, 0.05), "continue");
        assert_eq!(probe_decision(0.0, 8, -10.0, 8, 2, 0.15, 0.05, 0.05), "accept");
    }
}
