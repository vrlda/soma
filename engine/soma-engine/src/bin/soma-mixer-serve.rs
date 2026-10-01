//! Long-lived CircuitMixingMemory host: one JSON request per line on stdin,
//! one JSON reply per line on stdout. The Python client is
//! `soma.memory.engine.EngineMixingMemory`.
//!
//! Requests:
//!   {"op":"new","config":{...}}          fresh memory (MixerConfig fields)
//!   {"op":"load","path":p}               load a SOMAMIX1 state
//!   {"op":"save","path":p}               write a SOMAMIX1 state (atomic rename)
//!   {"op":"reset"}                       start a new stream (history only)
//!   {"op":"observe","bits":"0110","learn":b,"weight":w,"distribution":b}
//!   {"op":"distribution"}                -> {"p1":x,"order":n}
//!   {"op":"summary"}                     -> counters
//!   {"op":"quit"}
//! Every reply has "ok"; failures reply {"ok":false,"error":...}.

use soma_engine::mixer::{CircuitMixingMemory, MixerConfig};
use std::io::{BufRead, Write};

fn config_from(value: &serde_json::Value) -> MixerConfig {
    let mut config = MixerConfig::default();
    let get = |name: &str| value.get(name);
    if let Some(orders) = get("orders").and_then(|v| v.as_array()) {
        config.orders = orders.iter().map(|o| o.as_u64().unwrap_or(0) as u32).collect();
    }
    if let Some(v) = get("max_circuits").and_then(|v| v.as_u64()) { config.max_circuits = v as usize; }
    if let Some(v) = get("learning_rate").and_then(|v| v.as_f64()) { config.learning_rate = v; }
    if let Some(v) = get("count_limit").and_then(|v| v.as_u64()) { config.count_limit = v as u32; }
    if let Some(v) = get("initial_weight").and_then(|v| v.as_f64()) { config.initial_weight = v; }
    if let Some(v) = get("reclaim_fraction").and_then(|v| v.as_f64()) { config.reclaim_fraction = v; }
    if let Some(v) = get("arbitration").and_then(|v| v.as_bool()) { config.arbitration = v; }
    if let Some(v) = get("halve_above").and_then(|v| v.as_u64()) { config.halve_above = v as u32; }
    if let Some(v) = get("calibration").and_then(|v| v.as_bool()) { config.calibration = v; }
    if let Some(v) = get("calibration_limit").and_then(|v| v.as_u64()) { config.calibration_limit = v as u32; }
    if let Some(v) = get("gate_bit_position").and_then(|v| v.as_bool()) { config.gate_bit_position = v; }
    if let Some(v) = get("gate_partial").and_then(|v| v.as_bool()) { config.gate_partial = v; }
    if let Some(v) = get("correction").and_then(|v| v.as_bool()) { config.correction = v; }
    if let Some(v) = get("correction_rate").and_then(|v| v.as_f64()) { config.correction_rate = v; }
    if let Some(v) = get("growth_threshold").and_then(|v| v.as_u64()) { config.growth_threshold = v as u32; }
    if let Some(v) = get("plasticity_tau").and_then(|v| v.as_f64()) { config.plasticity_tau = v; }
    if let Some(v) = get("freeze_arbitration_after").and_then(|v| v.as_u64()) { config.freeze_arbitration_after = v; }
    config
}

fn distribution_reply(memory: &mut CircuitMixingMemory) -> serde_json::Value {
    let (p1, order) = memory.distribution();
    serde_json::json!({"ok": true, "p1": p1, "order": order})
}

fn handle(memory: &mut Option<CircuitMixingMemory>, request: &serde_json::Value) -> Result<serde_json::Value, String> {
    let op = request.get("op").and_then(|v| v.as_str()).unwrap_or("");
    match op {
        "new" => {
            let config = config_from(request.get("config").unwrap_or(&serde_json::Value::Null));
                *memory = Some(CircuitMixingMemory::new(config));
            return Ok(serde_json::json!({"ok": true}));
        }
        "load" => {
            let path = request.get("path").and_then(|v| v.as_str()).ok_or("missing path")?;
            let data = std::fs::read(path).map_err(|e| e.to_string())?;
            *memory = Some(CircuitMixingMemory::loads(&data)?);
            return Ok(serde_json::json!({"ok": true}));
        }
        "quit" => return Ok(serde_json::json!({"ok": true})),
        _ => {}
    }
    let memory = memory.as_mut().ok_or("no memory loaded: send new or load first")?;
    match op {
        "save" => {
            let path = request.get("path").and_then(|v| v.as_str()).ok_or("missing path")?;
            let temporary = format!("{}.tmp", path);
            std::fs::write(&temporary, memory.dumps()).map_err(|e| e.to_string())?;
            std::fs::rename(&temporary, path).map_err(|e| e.to_string())?;
            Ok(serde_json::json!({"ok": true}))
        }
        "reset" => {
            memory.reset_history();
            Ok(serde_json::json!({"ok": true}))
        }
        "observe" => {
            let bits = request.get("bits").and_then(|v| v.as_str()).unwrap_or("");
            let learn = request.get("learn").and_then(|v| v.as_bool()).unwrap_or(true);
            let weight = request.get("weight").and_then(|v| v.as_u64()).unwrap_or(1);
            if weight < 1 || weight > u32::MAX as u64 {
                return Err("weight must be a positive integer".to_string());
            }
            for character in bits.bytes() {
                let bit = match character {
                    b'0' => 0,
                    b'1' => 1,
                    _ => return Err("bits must be a string of 0 and 1".to_string()),
                };
                memory.observe_weighted(bit, learn, weight as u32);
            }
            if request.get("distribution").and_then(|v| v.as_bool()).unwrap_or(false) {
                Ok(distribution_reply(memory))
            } else {
                Ok(serde_json::json!({"ok": true}))
            }
        }
        "distribution" => Ok(distribution_reply(memory)),
        "summary" => Ok(serde_json::json!({
            "ok": true,
            "circuits": memory.circuits.len(),
            "circuits_created": memory.circuits_created,
            "circuits_reclaimed": memory.circuits_reclaimed,
            "events_seen": memory.events_seen,
        })),
        _ => Err(format!("unknown op: {}", op)),
    }
}

fn main() {
    std::panic::set_hook(Box::new(|_| {}));
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let mut memory: Option<CircuitMixingMemory> = None;
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(line) => line,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }
        let reply = match serde_json::from_str::<serde_json::Value>(&line) {
            Ok(request) => {
                let quit = request.get("op").and_then(|v| v.as_str()) == Some("quit");
                // A panic (for example an assertion on an invalid configuration)
                // becomes an error reply; the loaded memory is dropped because
                // it may be half-updated.
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    handle(&mut memory, &request)
                }));
                let reply = match outcome {
                    Ok(Ok(reply)) => reply,
                    Ok(Err(error)) => serde_json::json!({"ok": false, "error": error}),
                    Err(_) => {
                        memory = None;
                        serde_json::json!({"ok": false, "error": "engine panic; memory unloaded"})
                    }
                };
                if quit {
                    let _ = writeln!(out, "{}", reply);
                    let _ = out.flush();
                    break;
                }
                reply
            }
            Err(error) => serde_json::json!({"ok": false, "error": error.to_string()}),
        };
        if writeln!(out, "{}", reply).is_err() || out.flush().is_err() {
            break;
        }
    }
}
