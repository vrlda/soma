# Checkpoint schema v12

Source: `Organism.state_dict` / `from_state_dict` in `soma/organism.py`. Version `12`, loader accepts 5-11 via migration, rejects others.

- `version`: must equal 12 after migration.
- `seed`, `step_count`, `input_ids`, `output_ids`: identity and lineage.
- `learning_rate`, `trace_decay`, `homeostasis_rate`, `actor_learning_rate`, flags (`legacy_learning_enabled`, `structural_plasticity_enabled`): learning semantics.
- `reward_baseline`, `reward_baseline_rate`, `pending_outcome`: credit state.
- `cells`: `{id: Cell-as-dict}`. `synapses`: `synapse_to_dict` list. `next_synapse_index`, `retired_synapse_ids`: stable IDs.
- `resources`: `ResourceBudget.to_dict()` with `max_cells`, `max_synapses`, counters, energy.
- `events`, `coactivity`, `metrics_history`: bounded logs.
- RNG: `rng_state`, `exploration_rng_state`, `representation_rng_state`, `variable_order_rng_state`, `composition_rng_state`, tapes + cursors: exact resume.
- Delayed credit: `delayed_credit_enabled/delay/queue`: pending outcomes survive restore.
- Routing/structure: `variable_order_*`, `general_*`, `composition_*`, motor modules, fingerprints, probes, audits, install/route counts.
- Atomicity: tempfile + fsync + rename; corrupt payload rejected; failed save preserves prior bytes byte-for-byte.
- Compound: `save_checkpoint` / `load_checkpoint` bundles organism + environment at exact decision boundary.
