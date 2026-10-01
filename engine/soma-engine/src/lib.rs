//! SOMA production-engine spike (R5 host-language decision prototype).
//!
//! Scope: dynamic sparse graph mutation with stable logical ids plus a
//! JSON-compatible snapshot for differential testing. Full neural semantics,
//! GPU backends, and packaging follow only if this spike validates.

pub mod detector;
pub mod evidence;
pub mod graph;
pub mod longmix;
pub mod mixer;
pub mod neuron;
pub mod tape;

pub use detector::DetectorState;
pub use evidence::CircuitEvidence;
pub use graph::{SparseGraph, SynapseId};
pub use neuron::{ForwardFixture, Network};
