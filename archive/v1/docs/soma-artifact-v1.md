# .soma artifact v1 (target)

- Layers (one lineage): base snapshot + lifetime delta + write-ahead journal. Compaction creates new immutable generation, clears only incorporated journal entries.
- Manifest: brain id, ancestor id, format/runtime compatibility versions, transducer schemas + hashes, chunk checksums / Merkle hashes, migration history, safety policy version, model card + license.
- Storage: cells table + logical-to-physical map, compressed sparse blocks, strengths + plasticity, circuit registry, episodic/semantic stores + provenance, router calibration + pending credit, workspace clocks, budgets, WAL.
- Rules: journal, fsync, checksum, atomic generation switch; restore latest valid on torn write; imports resource-limited, never execute embedded code.
- Ops: create / clone / branch / inspect / verify / compact / export / import / backup / restore / retire.
