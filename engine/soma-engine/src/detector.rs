//! Page-Hinkley/CUSUM detector update, mirroring the normal-policy branch
//! of Organism._apply_context_reward exactly (predictor, variance, CUSUM,
//! freeze, confidence, streak, and global surprise statistics).

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DetectorParams {
    pub predictor_rate: f64,
    pub surprise_leak: f64,
    pub surprise_drift: f64,
    pub detector_drift: f64,
    pub detector_recovery: f64,
    pub freeze_threshold: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DetectorState {
    pub predictor: Vec<f64>,
    pub mean: f64,
    pub variance: f64,
    pub count: i64,
    pub cusum: f64,
    pub frozen: bool,
    pub confidence: f64,
    pub negative_streak: i64,
    pub baseline: f64,
    pub surprise_mean: f64,
    pub surprise_variance: f64,
    pub surprise_count: i64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DetectorOutcome {
    pub state: DetectorState,
    pub standardized_surprise: f64,
}

pub fn context_features(inputs: &[f64], action: f64) -> Vec<f64> {
    let mut features = Vec::with_capacity(3 + 3 * inputs.len());
    features.push(1.0);
    features.extend_from_slice(inputs);
    features.push(action);
    for value in inputs {
        features.push(value * action);
    }
    for value in inputs {
        features.push(value * value);
    }
    features.push(action * action);
    features
}

pub fn update(
    params: &DetectorParams,
    mut state: DetectorState,
    features: &[f64],
    reward_value: f64,
    pending_prediction: f64,
    has_pending_features: bool,
    action_mode_normal: bool,
    pending_scale: f64,
) -> DetectorOutcome {
    if state.count == 0 {
        state.predictor = vec![0.0; features.len()];
        state.predictor[0] = reward_value;
        state.mean = reward_value;
        state.variance = 0.01;
    }
    let prediction = if has_pending_features {
        pending_prediction
    } else {
        state.predictor.iter().zip(features.iter()).map(|(w, v)| w * v).sum()
    };
    let detector_delta = reward_value - prediction;
    if !state.frozen && action_mode_normal {
        let norm: f64 = 1.0 + features.iter().map(|v| v * v).sum::<f64>();
        let rate = params.predictor_rate / norm;
        state.predictor = state
            .predictor
            .iter()
            .zip(features.iter())
            .map(|(w, v)| w + rate * detector_delta * v)
            .collect();
        state.mean = state.predictor.iter().zip(features.iter()).map(|(w, v)| w * v).sum();
        state.variance = (params.surprise_leak * state.variance
            + (1.0 - params.surprise_leak) * detector_delta * detector_delta)
            .max(1e-5);
    } else {
        state.mean = prediction;
    }
    state.count += 1;
    let scale = if pending_scale > 0.0 { pending_scale } else { state.variance.sqrt() };
    let standardized_surprise = detector_delta / scale;
    let negative_deviation = (0.0f64).max(-standardized_surprise - params.detector_drift);
    let positive_recovery = (0.0f64).max(standardized_surprise);
    state.cusum = (params.surprise_leak * state.cusum + negative_deviation
        - params.detector_recovery * positive_recovery)
        .max(0.0);
    if state.cusum >= params.freeze_threshold {
        state.frozen = true;
    }
    let module_rpe = reward_value - state.baseline;
    state.confidence = (0.98 * state.confidence + 0.02 * if module_rpe > 0.0 { 1.0 } else { 0.0 })
        .max(0.0)
        .min(1.0);
    if standardized_surprise < -0.5 {
        state.negative_streak += 1;
    } else {
        state.negative_streak = (state.negative_streak - 1).max(0);
    }
    state.surprise_count += 1;
    let global_delta = reward_value - state.surprise_mean;
    state.surprise_mean += params.surprise_drift * global_delta;
    state.surprise_variance = (params.surprise_leak * state.surprise_variance
        + (1.0 - params.surprise_leak) * global_delta * global_delta)
        .max(1e-5);
    DetectorOutcome { state, standardized_surprise }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> DetectorParams {
        DetectorParams {
            predictor_rate: 0.5,
            surprise_leak: 0.995,
            surprise_drift: 0.05,
            detector_drift: 0.35,
            detector_recovery: 0.20,
            freeze_threshold: 2.5,
        }
    }

    fn state() -> DetectorState {
        DetectorState {
            predictor: vec![0.0; 4],
            mean: 0.0,
            variance: 0.01,
            count: 0,
            cusum: 0.0,
            frozen: false,
            confidence: 0.0,
            negative_streak: 0,
            baseline: 0.0,
            surprise_mean: 0.0,
            surprise_variance: 0.01,
            surprise_count: 0,
        }
    }

    #[test]
    fn first_outcome_seeds_predictor() {
        let outcome = update(&params(), state(), &[1.0, 0.5, 0.0, 0.0], 0.4, 0.0, false, true, 0.0);
        assert_eq!(outcome.state.predictor[0], 0.4);
        assert_eq!(outcome.state.count, 1);
    }

    #[test]
    fn sustained_negative_surprise_accumulates() {
        let mut state = state();
        for _ in 0..10 {
            let outcome = update(&params(), state, &[1.0, 0.0, 0.0, 0.0], 0.5, 0.0, false, true, 0.1);
            state = outcome.state;
        }
        // Warning-state freeze: the predictor stops tracking, so a regime
        // shift accumulates CUSUM evidence instead of being absorbed.
        state.frozen = true;
        for _ in 0..40 {
            let outcome = update(&params(), state, &[1.0, 0.0, 0.0, 0.0], -2.0, 0.0, false, true, 0.1);
            state = outcome.state;
        }
        assert!(state.cusum > 0.0);
        assert!(state.frozen);
    }
}
