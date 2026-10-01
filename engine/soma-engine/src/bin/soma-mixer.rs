//! CircuitMixingMemory runner: JSON job on stdin, JSON report on stdout.
//!
//! Job: {"config": {...MixerConfig fields...}, "train": [path, ...],
//!       "eval": {"name": path, ...}, "eval_every_file": bool,
//!       "trace": {"name": n}, "train_weights": [w, ...],
//!       "load_state": path, "save_state": path}. Each file is one document: history resets
//! at its start. Evaluation is frozen (learn=false) and leaves circuits and
//! weights untouched.

use soma_engine::mixer::{CircuitMixingMemory, MixerConfig};
use std::collections::BTreeMap;
use std::io::Read;
use std::time::Instant;

fn peak_rss_mb() -> f64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kb: f64 = rest.trim().trim_end_matches("kB").trim().parse().unwrap_or(-1.0);
            return kb / 1024.0;
        }
    }
    -1.0
}

fn config_from(value: &serde_json::Value) -> MixerConfig {
    let mut config = MixerConfig::default();
    if let Some(orders) = value.get("orders").and_then(|v| v.as_array()) {
        config.orders = orders.iter().map(|o| o.as_u64().expect("order") as u32).collect();
    }
    if let Some(v) = value.get("max_circuits").and_then(|v| v.as_u64()) {
        config.max_circuits = v as usize;
    }
    if let Some(v) = value.get("learning_rate").and_then(|v| v.as_f64()) {
        config.learning_rate = v;
    }
    if let Some(v) = value.get("count_limit").and_then(|v| v.as_u64()) {
        config.count_limit = v as u32;
    }
    if let Some(v) = value.get("initial_weight").and_then(|v| v.as_f64()) {
        config.initial_weight = v;
    }
    if let Some(v) = value.get("reclaim_fraction").and_then(|v| v.as_f64()) {
        config.reclaim_fraction = v;
    }
    if let Some(v) = value.get("arbitration").and_then(|v| v.as_bool()) {
        config.arbitration = v;
    }
    if let Some(v) = value.get("halve_above").and_then(|v| v.as_u64()) {
        config.halve_above = v as u32;
    }
    if let Some(v) = value.get("calibration").and_then(|v| v.as_bool()) {
        config.calibration = v;
    }
    if let Some(v) = value.get("calibration_limit").and_then(|v| v.as_u64()) {
        config.calibration_limit = v as u32;
    }
    if let Some(v) = value.get("gate_bit_position").and_then(|v| v.as_bool()) {
        config.gate_bit_position = v;
    }
    if let Some(v) = value.get("correction").and_then(|v| v.as_bool()) {
        config.correction = v;
    }
    if let Some(v) = value.get("growth_threshold").and_then(|v| v.as_u64()) {
        config.growth_threshold = v as u32;
    }
    if let Some(v) = value.get("gate_partial").and_then(|v| v.as_bool()) {
        config.gate_partial = v;
    }
    if let Some(v) = value.get("plasticity_tau").and_then(|v| v.as_f64()) {
        config.plasticity_tau = v;
    }
    if let Some(v) = value.get("freeze_arbitration_after").and_then(|v| v.as_u64()) {
        config.freeze_arbitration_after = v;
    }
    if let Some(v) = value.get("growth_pressure").and_then(|v| v.as_u64()) {
        config.growth_pressure = v as u32;
    }
    if let Some(v) = value.get("correction_rate").and_then(|v| v.as_f64()) {
        config.correction_rate = v;
    }
    config
}

fn evaluate(
    memory: &mut CircuitMixingMemory,
    evals: &BTreeMap<String, Vec<u8>>,
    traces: &BTreeMap<String, usize>,
    trace_out: &mut BTreeMap<String, Vec<f64>>,
) -> BTreeMap<String, f64> {
    let mut scores = BTreeMap::new();
    for (name, data) in evals {
        memory.reset_history();
        let mut trace = Vec::new();
        let score = memory.score_bytes(data, false, Some(&mut trace));
        if let Some(&limit) = traces.get(name) {
            trace.truncate(limit);
            trace_out.insert(name.clone(), trace);
        }
        scores.insert(name.clone(), score);
    }
    memory.reset_history();
    scores
}

fn main() {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).expect("stdin");
    let job: serde_json::Value = serde_json::from_str(&input).expect("job json");
    let config = config_from(job.get("config").unwrap_or(&serde_json::Value::Null));
    let train: Vec<String> = job
        .get("train")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().map(|p| p.as_str().expect("path").to_string()).collect())
        .unwrap_or_default();
    let mut evals = BTreeMap::new();
    if let Some(map) = job.get("eval").and_then(|v| v.as_object()) {
        for (name, path) in map {
            let data = std::fs::read(path.as_str().expect("path")).expect("eval file");
            evals.insert(name.clone(), data);
        }
    }
    let mut traces = BTreeMap::new();
    if let Some(map) = job.get("trace").and_then(|v| v.as_object()) {
        for (name, limit) in map {
            traces.insert(name.clone(), limit.as_u64().unwrap_or(0) as usize);
        }
    }
    let every = job.get("eval_every_file").and_then(|v| v.as_bool()).unwrap_or(false);
    // Evaluate after every `eval_interval` files (and after the last one).
    let interval = job.get("eval_interval").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    // Diagnostic: snapshot_evals[i] names the eval to score at the end with
    // the final circuits but the arbitration weights saved after train[i].
    let snapshot_evals: Vec<String> = job
        .get("snapshot_evals")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().map(|p| p.as_str().unwrap_or("").to_string()).collect())
        .unwrap_or_default();
    let mut snapshots: Vec<Vec<Vec<f64>>> = Vec::new();

    let started = Instant::now();
    let mut memory = match job.get("load_state").and_then(|v| v.as_str()) {
        Some(path) => {
            let data = std::fs::read(path).expect("load_state file");
            CircuitMixingMemory::loads(&data).expect("valid SOMAMIX1 state")
        }
        None => CircuitMixingMemory::new(config),
    };
    let mut curve = Vec::new();
    let mut cumulative: u64 = 0;
    let mut trace_out = BTreeMap::new();
    for (index, path) in train.iter().enumerate() {
        let data = std::fs::read(path).expect("train file");
        cumulative += data.len() as u64;
        memory.reset_history();
        let weight = job
            .get("train_weights")
            .and_then(|v| v.as_array())
            .and_then(|a| a.get(index))
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as u32;
        memory.observe_bytes_weighted(&data, true, weight);
        // Save the post-training state before any evaluation resets history.
        if index + 1 == train.len() {
            if let Some(path) = job.get("save_state").and_then(|v| v.as_str()) {
                std::fs::write(path, memory.dumps()).expect("save_state file");
            }
        }
        if !snapshot_evals.is_empty() {
            snapshots.push(memory.weights.clone());
        }
        if every || index + 1 == train.len() || (interval > 0 && (index + 1) % interval == 0) {
            let mut scratch = BTreeMap::new();
            let scores = evaluate(&mut memory, &evals, &BTreeMap::new(), &mut scratch);
            curve.push(serde_json::json!({
                "file": path,
                "cumulative_bytes": cumulative,
                "circuits": memory.circuits.len(),
                "circuits_created": memory.circuits_created,
                "circuits_reclaimed": memory.circuits_reclaimed,
                "elapsed_seconds": started.elapsed().as_secs_f64(),
                "bits_per_bit": scores,
            }));
        }
    }
    let final_scores = evaluate(&mut memory, &evals, &traces, &mut trace_out);
    let mut snapshot_scores = BTreeMap::new();
    if !snapshot_evals.is_empty() {
        let final_weights = memory.weights.clone();
        for (index, name) in snapshot_evals.iter().enumerate() {
            if let (Some(weights), Some(data)) = (snapshots.get(index), evals.get(name)) {
                memory.weights = weights.clone();
                memory.reset_history();
                let score = memory.score_bytes(data, false, None);
                snapshot_scores.insert(name.clone(), score);
            }
        }
        memory.weights = final_weights;
        memory.reset_history();
    }
    let report = serde_json::json!({
        "engine": "soma-mixer",
        "orders": memory.config.orders,
        "max_circuits": memory.config.max_circuits,
        "learning_rate": memory.config.learning_rate,
        "count_limit": memory.config.count_limit,
        "arbitration": memory.config.arbitration,
        "config": format!("{:?}", memory.config),
        "events_seen": memory.events_seen,
        "circuits": memory.circuits.len(),
        "circuits_created": memory.circuits_created,
        "circuits_reclaimed": memory.circuits_reclaimed,
        "weights": memory.weights,
        "bits_per_bit": final_scores,
        "snapshot_scores": snapshot_scores,
        "curve": curve,
        "trace": trace_out,
        "total_seconds": started.elapsed().as_secs_f64(),
        "peak_rss_mb": peak_rss_mb(),
    });
    println!("{}", serde_json::to_string(&report).unwrap());
}
