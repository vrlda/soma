//! Differential detector runner: JSON tape in, detector states out.

use soma_engine::detector::{update, DetectorParams, DetectorState};
use std::io::Read;

fn main() {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).expect("stdin");
    let tape: serde_json::Value = serde_json::from_str(&input).expect("tape");
    let params_value = tape.get("params").cloned().unwrap_or(serde_json::Value::Null);
    let params: DetectorParams = serde_json::from_value(params_value).expect("params");
    let mut out = String::from("{\"trials\":[");
    let trials = tape.get("trials").and_then(|v| v.as_array()).cloned().unwrap_or_default();
    for (index, trial) in trials.iter().enumerate() {
        let state_value = trial.get("state").cloned().unwrap_or(serde_json::Value::Null);
        let state: DetectorState = serde_json::from_value(state_value).expect("state");
        let has_pending = trial.get("has_pending").and_then(|v| v.as_bool()).unwrap_or(false);
        let features: Vec<f64> = if has_pending {
            trial
                .get("features")
                .and_then(|v| v.as_array())
                .map(|list| list.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect())
                .unwrap_or_default()
        } else {
            let inputs: Vec<f64> = trial
                .get("inputs")
                .and_then(|v| v.as_array())
                .map(|list| list.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect())
                .unwrap_or_default();
            let action = trial.get("pending_action").and_then(|v| v.as_f64()).unwrap_or(0.0);
            soma_engine::detector::context_features(&inputs, action)
        };
        let outcome = update(
            &params,
            state,
            &features,
            trial.get("reward").and_then(|v| v.as_f64()).unwrap_or(0.0),
            trial.get("pending_prediction").and_then(|v| v.as_f64()).unwrap_or(0.0),
            trial.get("has_pending").and_then(|v| v.as_bool()).unwrap_or(false),
            trial.get("mode_normal").and_then(|v| v.as_bool()).unwrap_or(true),
            trial.get("pending_scale").and_then(|v| v.as_f64()).unwrap_or(0.0),
        );
        if index > 0 {
            out.push(',');
        }
        out.push_str(&serde_json::to_string(&outcome).expect("encode"));
    }
    out.push_str("]}");
    println!("{}", out);
}
