//! Differential evidence runner: JSON op tape in, state plus returns out.
//!
//! Ops: {"op":"update","circuit":name,"outcome":x,"probe":b,"decay":d,
//!        "step":s,"accumulate":b}, {"op":"posterior","scores":{k:v},
//!        "novelty":n}, {"op":"switch","c":x,"i":y,"cn":n,"in":m},
//! {"op":"probe","cll":x,"cn":n,"ill":y,"in":m},
//! {"op":"calibration","probs":[...],"outcomes":[...]}.

use soma_engine::evidence::{
    calibration_metrics, normalize_posterior, probe_decision, should_switch, update_evidence,
    CircuitEvidence,
};
use std::collections::BTreeMap;
use std::io::Read;

fn get<'a>(value: &'a serde_json::Value, key: &str) -> &'a serde_json::Value {
    value.get(key).unwrap_or(&serde_json::Value::Null)
}

fn as_f64(value: &serde_json::Value) -> f64 {
    value.as_f64().unwrap_or(0.0)
}

fn as_i64(value: &serde_json::Value) -> i64 {
    value.as_i64().unwrap_or(0)
}

fn as_bool(value: &serde_json::Value) -> bool {
    value.as_bool().unwrap_or(false)
}

fn main() {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).expect("stdin");
    let tape: serde_json::Value = serde_json::from_str(&input).expect("tape");
    let mut circuits: BTreeMap<String, CircuitEvidence> = BTreeMap::new();
    let mut returns: Vec<String> = Vec::new();
    let ops = tape.get("ops").and_then(|v| v.as_array()).cloned().unwrap_or_default();
    for op in &ops {
        let kind = op.get("op").and_then(|v| v.as_str()).unwrap_or("");
        match kind {
            "update" => {
                let name = op.get("circuit").and_then(|v| v.as_str()).unwrap_or("c").to_string();
                let evidence = circuits.entry(name).or_default();
                let value = update_evidence(
                    evidence,
                    as_f64(get(op, "outcome")),
                    as_bool(get(op, "probe")),
                    as_f64(get(op, "decay")),
                    as_i64(get(op, "step")),
                    as_bool(get(op, "accumulate")),
                );
                returns.push(format!("{:?}", value));
            }
            "posterior" => {
                let scores: Vec<(String, f64)> = op
                    .get("scores")
                    .and_then(|v| v.as_object())
                    .map(|map| map.iter().map(|(k, v)| (k.clone(), as_f64(v))).collect())
                    .unwrap_or_default();
                let novelty = as_f64(get(op, "novelty"));
                let posterior = normalize_posterior(&scores, novelty);
                let mut parts: Vec<String> = posterior.iter().map(|(k, v)| format!("{:?}:{:?}", k, v)).collect();
                parts.sort();
                returns.push(parts.join(","));
            }
            "switch" => {
                let value = should_switch(
                    as_f64(get(op, "c")), as_f64(get(op, "i")),
                    as_i64(get(op, "cn")), as_i64(get(op, "in")), 16, 0.15,
                );
                returns.push(value.to_string());
            }
            "probe" => {
                let value = probe_decision(
                    as_f64(get(op, "cll")), as_i64(get(op, "cn")),
                    as_f64(get(op, "ill")), as_i64(get(op, "in")),
                    2, 0.15, 0.05, 0.05,
                );
                returns.push(value.to_string());
            }
            "calibration" => {
                let probs: Vec<f64> = op.get("probs").and_then(|v| v.as_array())
                    .map(|list| list.iter().map(as_f64).collect()).unwrap_or_default();
                let outcomes: Vec<bool> = op.get("outcomes").and_then(|v| v.as_array())
                    .map(|list| list.iter().map(|v| v.as_i64().unwrap_or(0) != 0).collect())
                    .unwrap_or_default();
                let (nll, brier, ece) = calibration_metrics(&probs, &outcomes);
                returns.push(format!("{:?},{:?},{:?}", nll, brier, ece));
            }
            _ => {}
        }
    }
    let mut state_parts = Vec::new();
    for (name, evidence) in &circuits {
        state_parts.push(format!(
            "{:?}:{:?},{:?},{:?},{:?},{:?},{:?}",
            name,
            evidence.outcome_mean,
            evidence.outcome_scale,
            evidence.eff_n,
            evidence.log_evidence,
            evidence.normal_outcomes,
            evidence.probe_outcomes,
        ));
    }
    let mut out = String::from("{\"returns\":[");
    for (index, value) in returns.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str(&format!("{:?}", value));
    }
    out.push_str("],\"state\":[");
    for (index, part) in state_parts.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str(&format!("{:?}", part));
    }
    out.push_str("]}");
    println!("{}", out);
}
