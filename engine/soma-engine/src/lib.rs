//! SOMA production-engine spike (R5 host-language decision prototype).
//!
//! Scope: dynamic sparse graph mutation with stable logical ids plus a
//! JSON-compatible snapshot for differential testing. Full neural semantics,
//! GPU backends, and packaging follow only if this spike validates.

pub mod graph;
pub mod tape;

pub use graph::{SparseGraph, SynapseId};
