//! Forward differential runner: JSON fixture in, outputs JSON out.
//!
//! Reads a ForwardFixture (see neuron.rs), runs every step, and prints
//! per-step output activations plus final cell activations.

use soma_engine::neuron::{ForwardFixture, Network};
use std::io::Read;

fn main() {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).expect("stdin");
    let fixture: ForwardFixture = serde_json::from_str(&input).expect("fixture");
    let rounds: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(1);
    let started = std::time::Instant::now();
    let mut outputs = Vec::with_capacity(fixture.steps.len());
    let mut network = Network::from_fixture(&fixture);
    for _ in 0..rounds {
        network = Network::from_fixture(&fixture);
        outputs.clear();
        for values in &fixture.steps {
            outputs.push(network.step(values));
        }
    }
    let elapsed = started.elapsed();
    let mut final_cells: Vec<(&String, &f64)> = network
        .cells
        .iter()
        .map(|(id, cell)| (id, &cell.activation))
        .collect();
    final_cells.sort_by(|a, b| a.0.cmp(b.0));
    let mut out = String::from("{\"outputs\":[");
    for (index, row) in outputs.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push('[');
        for (position, value) in row.iter().enumerate() {
            if position > 0 {
                out.push(',');
            }
            out.push_str(&format!("{:?}", value));
        }
        out.push(']');
    }
    out.push_str("],\"final\":{");
    for (index, (id, activation)) in final_cells.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str(&format!("{:?}:{:?}", id, activation));
    }
    out.push_str(&format!("}},\"microseconds\":{}}}", elapsed.as_micros()));
    println!("{}", out);
}
