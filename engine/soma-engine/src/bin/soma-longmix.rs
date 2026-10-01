//! LongMixer runner (ADR 0010): JSON job on stdin, JSON report on stdout.
//!
//! Job: {"config": {...LongConfig fields...},
//!       "train": [path, ...], "eval": {"name": path, ...},
//!       "qa": {"prompts": path, "out": path, "max_bytes": 64}}
//! Every file holds documents separated by the byte 0x1E; history resets at
//! each document. Training learns; evaluation is frozen and reports
//! bits/byte; QA greedily continues each prompt document until a newline
//! and writes one JSON string per line to "out".

use soma_engine::longmix::{LongConfig, LongMixer};
use std::collections::BTreeMap;
use std::io::{Read, Write};
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

fn config_from(value: &serde_json::Value) -> LongConfig {
    let mut c = LongConfig::default();
    let get = |name: &str| value.get(name);
    if let Some(orders) = get("byte_orders").and_then(|v| v.as_array()) {
        c.byte_orders = orders.iter().map(|o| o.as_u64().expect("order") as u32).collect();
    }
    if let Some(v) = get("word_contexts").and_then(|v| v.as_bool()) { c.word_contexts = v; }
    if let Some(v) = get("match_model").and_then(|v| v.as_bool()) { c.match_model = v; }
    if let Some(v) = get("max_circuits").and_then(|v| v.as_u64()) { c.max_circuits = v as usize; }
    if let Some(v) = get("learning_rate").and_then(|v| v.as_f64()) { c.learning_rate = v; }
    if let Some(v) = get("count_limit").and_then(|v| v.as_u64()) { c.count_limit = v as u32; }
    if let Some(v) = get("halve_above").and_then(|v| v.as_u64()) { c.halve_above = v as u32; }
    if let Some(v) = get("initial_weight").and_then(|v| v.as_f64()) { c.initial_weight = v; }
    if let Some(v) = get("reclaim_fraction").and_then(|v| v.as_f64()) { c.reclaim_fraction = v; }
    if let Some(v) = get("growth_threshold").and_then(|v| v.as_u64()) { c.growth_threshold = v as u32; }
    if let Some(v) = get("growth_pressure").and_then(|v| v.as_u64()) { c.growth_pressure = v as u32; }
    if let Some(v) = get("plasticity_tau").and_then(|v| v.as_f64()) { c.plasticity_tau = v; }
    if let Some(v) = get("byte_gate").and_then(|v| v.as_bool()) { c.byte_gate = v; }
    c
}

fn documents(path: &str) -> Vec<Vec<u8>> {
    let data = std::fs::read(path).expect("input file");
    data.split(|&b| b == 0x1e).filter(|d| !d.is_empty()).map(|d| d.to_vec()).collect()
}

fn main() {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).expect("stdin");
    let job: serde_json::Value = serde_json::from_str(&input).expect("job json");
    let config = config_from(job.get("config").unwrap_or(&serde_json::Value::Null));
    let mut memory = LongMixer::new(config);
    let started = Instant::now();
    let mut trained_bytes: u64 = 0;
    let mut trained_docs: u64 = 0;
    if let Some(files) = job.get("train").and_then(|v| v.as_array()) {
        for file in files {
            for doc in documents(file.as_str().expect("path")) {
                memory.reset_history();
                memory.observe_bytes(&doc, true);
                trained_bytes += doc.len() as u64;
                trained_docs += 1;
            }
        }
    }
    let train_seconds = started.elapsed().as_secs_f64();
    let mut scores = BTreeMap::new();
    if let Some(map) = job.get("eval").and_then(|v| v.as_object()) {
        for (name, path) in map {
            let (mut total, mut bits) = (0.0, 0u64);
            for doc in documents(path.as_str().expect("path")) {
                memory.reset_history();
                let (t, c) = memory.score_bytes(&doc, false);
                total += t;
                bits += c;
            }
            scores.insert(name.clone(), serde_json::json!({
                "bits_per_byte": 8.0 * total / bits.max(1) as f64, "bytes": bits / 8}));
        }
    }
    let mut qa_count = 0;
    if let Some(qa) = job.get("qa") {
        let prompts = documents(qa.get("prompts").and_then(|v| v.as_str()).expect("prompts"));
        let max_bytes = qa.get("max_bytes").and_then(|v| v.as_u64()).unwrap_or(64) as usize;
        let mut out = std::fs::File::create(qa.get("out").and_then(|v| v.as_str()).expect("out"))
            .expect("qa out");
        for prompt in prompts {
            memory.reset_history();
            memory.observe_bytes(&prompt, false);
            let answer = memory.generate(max_bytes, b'\n');
            let text = String::from_utf8_lossy(&answer).to_string();
            writeln!(out, "{}", serde_json::to_string(&text).unwrap()).unwrap();
            qa_count += 1;
        }
    }
    let report = serde_json::json!({
        "engine": "soma-longmix",
        "config": format!("{:?}", memory.config),
        "trained_bytes": trained_bytes,
        "trained_documents": trained_docs,
        "train_seconds": train_seconds,
        "circuits": memory.circuits.len(),
        "circuits_created": memory.circuits_created,
        "circuits_reclaimed": memory.circuits_reclaimed,
        "scores": scores,
        "qa_answers": qa_count,
        "total_seconds": started.elapsed().as_secs_f64(),
        "peak_rss_mb": peak_rss_mb(),
    });
    println!("{}", serde_json::to_string(&report).unwrap());
}
