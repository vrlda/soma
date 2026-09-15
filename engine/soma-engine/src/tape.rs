//! Shared operation tape: JSON op list executed by both the Rust spike
//! and the Python reference for differential agreement testing.
//!
//! Tape format: [{"op": "add", "source": s, "destination": d, "strength": w},
//!               {"op": "remove", "source": s, "destination": d}, ...]
//! Report format: {"edges": [[source, destination, index, generation,
//! strength], ...], "synapse_count": n}

use crate::graph::SparseGraph;

#[derive(Debug)]
pub enum TapeOp {
    Add { source: u64, destination: u64, strength: f64 },
    Remove { source: u64, destination: u64 },
}

pub fn run_tape(ops: &[TapeOp]) -> SparseGraph {
    let mut graph = SparseGraph::new();
    for op in ops {
        match *op {
            TapeOp::Add { source, destination, strength } => {
                graph.add_synapse(source, destination, strength);
            }
            TapeOp::Remove { source, destination } => {
                graph.remove_synapse(source, destination);
            }
        }
    }
    graph
}

pub fn report_inner(graph: &SparseGraph) -> String {
    let edges = graph.snapshot();
    let mut out = String::from("\"edges\":[");
    for (index, (source, destination, edge_index, generation, strength)) in edges.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "[{},{},{},{},{}]",
            source,
            destination,
            edge_index,
            generation,
            print_float(*strength)
        ));
    }
    out.push_str(&format!("],\"synapse_count\":{}", graph.synapse_count()));
    out
}

pub fn report(graph: &SparseGraph) -> String {
    format!("{{{}}}", report_inner(graph))
}

fn print_float(value: f64) -> String {
    if value == value.trunc() && value.abs() < 1e15 {
        format!("{}.0", value.trunc() as i64)
    } else {
        format!("{:?}", value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_tape_reports_empty_graph() {
        let report = report(&run_tape(&[]));
        assert!(report.contains("\"synapse_count\":0"));
    }
}
