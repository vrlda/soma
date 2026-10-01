//! Forward propagation semantics, mirroring Organism._propagate.
//!
//! Scope: frozen weights, no representation noise, no adaptive proposals,
//! no homeostasis pass. Dormant motor freeze, product cells, threshold,
//! bias, adaptation, and observe-dynamics are exact. Tanh differences
//! across libm implementations are covered by test tolerances, never by
//! Silva-specific rounding.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CellInit {
    pub id: String,
    pub kind: String,
    pub threshold: f64,
    pub adaptation: f64,
    pub bias: f64,
    pub activation: f64,
    pub activation_type: String,
    pub dendritic_sources: Vec<String>,
    pub dendritic_normalizer: f64,
    pub dormant: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SynapseInit {
    pub source: String,
    pub destination: String,
    pub strength: f64,
    #[serde(default)]
    pub plasticity: f64,
    #[serde(default)]
    pub trace: f64,
    #[serde(default)]
    pub actor_trace: f64,
}

#[derive(Clone, Debug)]
pub struct Edge {
    pub source: String,
    pub strength: f64,
    pub plasticity: f64,
    pub trace: f64,
    pub actor_trace: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ForwardFixture {
    pub cells: Vec<CellInit>,
    pub synapses: Vec<SynapseInit>,
    pub input_ids: Vec<String>,
    pub output_ids: Vec<String>,
    pub steps: Vec<Vec<f64>>,
    pub passes: usize,
}

pub struct Cell {
    pub kind: String,
    pub threshold: f64,
    pub adaptation: f64,
    pub bias: f64,
    pub activation: f64,
    pub activation_type: String,
    pub dendritic_sources: Vec<String>,
    pub dendritic_normalizer: f64,
    pub dormant: bool,
    pub signed_ema: f64,
    pub activity_ema: f64,
    pub utility: f64,
    pub age: u64,
}

pub struct Network {
    pub cells: BTreeMap<String, Cell>,
    pub incoming: BTreeMap<String, Vec<Edge>>,
    pub input_ids: Vec<String>,
    pub output_ids: Vec<String>,
    pub passes: usize,
}

fn edge_strength(edges: &[Edge], source: &str) -> f64 {
    edges.iter().find(|edge| edge.source == source).map(|edge| edge.strength).unwrap_or(0.0)
}

fn clamp(value: f64) -> f64 {
    value.max(-1.0).min(1.0)
}

impl Network {
    pub fn from_fixture(fixture: &ForwardFixture) -> Self {
        let mut cells = BTreeMap::new();
        for init in &fixture.cells {
            cells.insert(init.id.clone(), Cell {
                kind: init.kind.clone(),
                threshold: init.threshold,
                adaptation: init.adaptation,
                bias: init.bias,
                activation: init.activation,
                activation_type: init.activation_type.clone(),
                dendritic_sources: init.dendritic_sources.clone(),
                dendritic_normalizer: init.dendritic_normalizer,
                dormant: init.dormant,
                signed_ema: 0.0,
                activity_ema: 0.0,
                utility: 0.0,
                age: 0,
            });
        }
        let mut incoming: BTreeMap<String, Vec<Edge>> = BTreeMap::new();
        for synapse in &fixture.synapses {
            incoming.entry(synapse.destination.clone()).or_default().push(Edge {
                source: synapse.source.clone(),
                strength: synapse.strength,
                plasticity: synapse.plasticity,
                trace: synapse.trace,
                actor_trace: synapse.actor_trace,
            });
        }
        for edges in incoming.values_mut() {
            edges.sort_by(|a, b| a.source.cmp(&b.source));
        }
        Network {
            cells,
            incoming,
            input_ids: fixture.input_ids.clone(),
            output_ids: fixture.output_ids.clone(),
            passes: fixture.passes.max(1),
        }
    }

    fn drive(&self, id: &str, previous: &BTreeMap<String, f64>) -> f64 {
        let cell = &self.cells[id];
        match cell.activation_type.as_str() {
            "dendritic_product" => {
                let a = &cell.dendritic_sources[0];
                let b = &cell.dendritic_sources[1];
                let edges = self.incoming.get(id);
                let strength = |source: &str| {
                    edges.and_then(|list| list.iter().find(|edge| edge.source == source))
                        .map(|edge| edge.strength).unwrap_or(0.0)
                };
                previous.get(a).copied().unwrap_or(0.0) * strength(a)
                    * previous.get(b).copied().unwrap_or(0.0) * strength(b)
            }
            "dendritic_product_n" => {
                let mut drive = 1.0;
                for source in &cell.dendritic_sources {
                    drive *= previous.get(source).copied().unwrap_or(0.0)
                        * edge_strength(self.incoming.get(id).map(Vec::as_slice).unwrap_or(&[]), source);
                }
                drive
            }
            "dendritic_product_composed" => {
                let mut drive = cell.dendritic_normalizer;
                for source in &cell.dendritic_sources {
                    drive *= previous.get(source).copied().unwrap_or(0.0)
                        * edge_strength(self.incoming.get(id).map(Vec::as_slice).unwrap_or(&[]), source);
                }
                drive
            }
            _ => {
                self.incoming.get(id).map(|edges| {
                    edges.iter().map(|edge| {
                        previous.get(&edge.source).copied().unwrap_or(0.0) * edge.strength
                    }).sum()
                }).unwrap_or(0.0)
            }
        }
    }

    fn observe(&mut self, id: &str, value: f64) {
        let cell = self.cells.get_mut(id).expect("cell");
        cell.activation = clamp(value);
        cell.activity_ema = 0.95 * cell.activity_ema + 0.05 * cell.activation.abs();
        cell.signed_ema = 0.95 * cell.signed_ema + 0.05 * cell.activation;
        cell.utility = 0.995 * cell.utility + 0.005 * cell.activation.abs();
        cell.age += 1;
    }

    /// Three-factor Hebbian update over correlation traces.
    /// Mirrors Organism._apply_reward exactly (plasticity-gated, clamped).
    pub fn apply_reward(&mut self, learning_rate: f64, prediction_error: f64) {
        for edges in self.incoming.values_mut() {
            for edge in edges.iter_mut() {
                edge.strength += learning_rate * edge.plasticity * edge.trace * prediction_error;
                edge.strength = edge.strength.max(-1.0).min(1.0);
            }
        }
    }

    /// Actor update over motor perturbation traces.
    /// Mirrors Organism._apply_actor_reward exactly, including trace reset.
    pub fn apply_actor_reward(&mut self, learning_rate: f64, prediction_error: f64) {
        for destination in self.output_ids.clone() {
            if let Some(edges) = self.incoming.get_mut(&destination) {
                for edge in edges.iter_mut() {
                    edge.strength += learning_rate * edge.actor_trace * prediction_error;
                    edge.strength = edge.strength.max(-1.0).min(1.0);
                    edge.actor_trace = 0.0;
                }
            }
        }
    }

    /// Canonical strength snapshot for differential comparison.
    pub fn strengths(&self) -> Vec<(String, String, f64)> {
        let mut rows = Vec::new();
        for (destination, edges) in &self.incoming {
            for edge in edges {
                rows.push((edge.source.clone(), destination.clone(), edge.strength));
            }
        }
        rows.sort_by(|a, b| (a.0.clone(), a.1.clone()).cmp(&(b.0.clone(), b.1.clone())));
        rows
    }

    pub fn step(&mut self, values: &[f64]) -> Vec<f64> {
        for (identifier, value) in self.input_ids.clone().iter().zip(values.iter()) {
            self.observe(identifier, *value);
        }
        let mut previous: BTreeMap<String, f64> = self.cells.iter().map(|(id, cell)| (id.clone(), cell.activation)).collect();
        for _ in 0..self.passes {
            let mut current = previous.clone();
            let ids: Vec<String> = {
                let mut keys: Vec<String> = self.cells.keys().cloned().collect();
                keys.sort();
                keys
            };
            for identifier in &ids {
                let kind = self.cells[identifier].kind.clone();
                if kind == "input" {
                    continue;
                }
                let dormant = self.cells[identifier].dormant;
                if kind == "motor_module" && dormant {
                    current.insert(identifier.clone(), previous[identifier]);
                    continue;
                }
                let drive = self.drive(identifier, &previous);
                let (threshold, bias, adaptation) = {
                    let cell = &self.cells[identifier];
                    (cell.threshold, cell.bias, cell.adaptation)
                };
                let value = ((drive + bias - threshold) / (1.0 + adaptation)).tanh();
                current.insert(identifier.clone(), value);
                self.observe(identifier, value);
                if let Some(cell) = self.cells.get_mut(identifier) {
                    cell.adaptation = 0.96 * cell.adaptation + 0.04 * value.abs();
                }
            }
            previous = current;
        }
        self.output_ids.iter().map(|id| previous[id]).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> Network {
        Network::from_fixture(&ForwardFixture {
            cells: vec![
                CellInit { id: "in".into(), kind: "input".into(), threshold: 0.0, adaptation: 0.0, bias: 0.0, activation: 0.0, activation_type: "additive".into(), dendritic_sources: vec![], dendritic_normalizer: 1.0, dormant: false },
                CellInit { id: "out".into(), kind: "output".into(), threshold: 0.0, adaptation: 0.0, bias: 0.0, activation: 0.0, activation_type: "additive".into(), dendritic_sources: vec![], dendritic_normalizer: 1.0, dormant: false },
            ],
            synapses: vec![
                SynapseInit { source: "in".into(), destination: "out".into(), strength: 0.5, plasticity: 0.5, trace: 0.0, actor_trace: 0.0 },
            ],
            input_ids: vec!["in".into()],
            output_ids: vec!["out".into()],
            steps: vec![],
            passes: 1,
        })
    }

    #[test]
    fn affine_tanh_matches_closed_form() {
        let mut network = tiny();
        let outputs = network.step(&[0.8]);
        let expected = (0.8 * 0.5f64).tanh();
        assert!((outputs[0] - expected).abs() < 1e-12, "got {}", outputs[0]);
    }

    #[test]
    fn adaptation_damps_second_pass() {
        // Multi-pass dynamics (adaptation gain) are covered against Python
        // by the differential tape; here we pin the direction only.
        let mut network = tiny();
        let first = network.step(&[0.8])[0];
        let second = network.step(&[0.8])[0];
        assert!(second < first, "adaptation should damp repeat drive");
        assert!(second > 0.0, "output must stay positive");
    }
}
