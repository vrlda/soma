//! Sparse directed graph with stable logical IDs and generational slots.
//!
//! Production-engine spike for the R5 host-language decision: measures
//! dynamic sparse graph mutation (add/remove with slot reuse) and emits a
//! JSON-compatible state snapshot for differential testing against the
//! Python reference on a shared operation tape.

use std::collections::{BTreeMap, HashMap};

/// Stable logical synapse identity plus a generation counter that
/// invalidates stale handles when a physical slot is reused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SynapseId {
    pub index: u64,
    pub generation: u64,
}

#[derive(Clone, Debug)]
pub struct Synapse {
    pub id: SynapseId,
    pub source: u64,
    pub destination: u64,
    pub strength: f64,
    pub plasticity: f64,
}

#[derive(Default)]
pub struct SparseGraph {
    cells: BTreeMap<u64, ()>,
    slots: HashMap<(u64, u64), usize>,
    storage: Vec<Option<Synapse>>,
    free: Vec<usize>,
    next_cell: u64,
    next_index: u64,
    pair_ids: HashMap<(u64, u64), u64>,
    retired_generations: HashMap<u64, u64>,
}

impl SparseGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_cell(&mut self) -> u64 {
        let id = self.next_cell;
        self.next_cell += 1;
        self.cells.insert(id, ());
        id
    }

    pub fn add_synapse(&mut self, source: u64, destination: u64, strength: f64) -> SynapseId {
        if let Some(&slot) = self.slots.get(&(source, destination)) {
            return self.storage[slot].as_ref().expect("slot mapped").id;
        }
        let pair = (source, destination);
        let index = *self.pair_ids.entry(pair).or_insert_with(|| {
            let assigned = self.next_index;
            self.next_index += 1;
            assigned
        });
        let generation = self.retired_generations.get(&index).copied().unwrap_or(0);
        let id = SynapseId { index, generation };
        let synapse = Synapse { id, source, destination, strength, plasticity: 0.5 };
        if let Some(slot) = self.free.pop() {
            self.storage[slot] = Some(synapse);
            self.slots.insert((source, destination), slot);
        } else {
            self.slots.insert((source, destination), self.storage.len());
            self.storage.push(Some(synapse));
        }
        id
    }

    pub fn remove_synapse(&mut self, source: u64, destination: u64) -> bool {
        let Some(slot) = self.slots.remove(&(source, destination)) else {
            return false;
        };
        if let Some(synapse) = self.storage[slot].take() {
            self.retired_generations.insert(synapse.id.index, synapse.id.generation + 1);
            self.free.push(slot);
            return true;
        }
        false
    }

    pub fn get(&self, source: u64, destination: u64) -> Option<&Synapse> {
        self.slots.get(&(source, destination)).and_then(|slot| self.storage[*slot].as_ref())
    }

    pub fn synapse_count(&self) -> usize {
        self.slots.len()
    }

    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    /// Canonical snapshot: sorted edge list with stable ids and generations.
    pub fn snapshot(&self) -> Vec<(u64, u64, u64, u64, f64)> {
        let mut edges: Vec<(u64, u64, u64, u64, f64)> = self
            .slots
            .iter()
            .filter_map(|((source, destination), slot)| {
                self.storage[*slot].as_ref().map(|synapse| {
                    (*source, *destination, synapse.id.index, synapse.id.generation, synapse.strength)
                })
            })
            .collect();
        edges.sort_by(|a, b| {
            (a.0, a.1, a.2, a.3)
                .partial_cmp(&(b.0, b.1, b.2, b.3))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        edges
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_reuse_bumps_generation() {
        let mut graph = SparseGraph::new();
        let first = graph.add_synapse(0, 1, 0.5);
        assert!(graph.remove_synapse(0, 1));
        let second = graph.add_synapse(0, 1, -0.25);
        assert_ne!(first, second);
        assert!(second.generation > first.generation);
        assert_eq!(graph.synapse_count(), 1);
    }

    #[test]
    fn duplicate_add_is_idempotent() {
        let mut graph = SparseGraph::new();
        let first = graph.add_synapse(2, 3, 0.1);
        let second = graph.add_synapse(2, 3, 0.9);
        assert_eq!(first, second);
        assert_eq!(graph.get(2, 3).expect("edge").strength, 0.1);
    }

    #[test]
    fn snapshot_is_canonical() {
        let mut graph = SparseGraph::new();
        graph.add_synapse(5, 1, 0.2);
        graph.add_synapse(1, 5, 0.3);
        let snapshot = graph.snapshot();
        assert_eq!(snapshot.len(), 2);
        assert!(snapshot[0] < snapshot[1]);
    }
}
