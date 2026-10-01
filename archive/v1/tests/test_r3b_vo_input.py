import unittest

from soma.organism import Organism


def _input_aff_organism(seed=101):
    organism = Organism.create_default(input_size=4, hidden_size=4, output_size=1, seed=seed)
    organism.enable_context_modules(max_modules=3)
    organism.enable_variable_order_learning(seed + 7919, max_order=3)
    return organism


def _install(organism, sources, module_id, credit):
    """Mirror _variable_order_route: transition flag plus reverse mapping."""
    key = organism._dendritic_feature_key(tuple(sorted(sources)))
    organism._variable_order_internal_transition = True
    try:
        cell_id = organism.install_dendritic_feature(sources, module_id, credit)
    finally:
        organism._variable_order_internal_transition = False
    organism.variable_order_module_features[module_id] = key
    organism.validate()
    return cell_id


class VOInputAffTests(unittest.TestCase):
    def test_install_on_input_aff_owner(self):
        organism = _input_aff_organism()
        module_id = organism.active_motor_module
        self.assertEqual(organism.motor_modules[module_id].afferent_kind, "input")
        cell_id = _install(organism, ("input-000", "input-001"), module_id, 0.5)
        self.assertIn(module_id, organism.variable_order_extra_sources)
        self.assertIn(cell_id, organism.variable_order_extra_sources[module_id])
        cell = organism.cells[cell_id]
        self.assertEqual(cell.activation_type, "dendritic_product")
        self.assertIsNotNone(organism.graph.get(cell_id, organism.motor_modules[module_id].cell_id))
        module = organism.motor_modules[module_id]
        self.assertIn(cell_id, [s.source for s in organism._module_synapses(module)])
        organism.validate()
        mean = organism._module_mean_from_current_inputs(module)
        self.assertTrue(-1.0 <= mean <= 1.0)

    def test_hidden_install_leaves_extra_map_empty(self):
        organism = Organism.create_default(input_size=3, hidden_size=6, output_size=1, seed=102)
        organism.enable_context_modules(max_modules=3, afferent_kind="hidden")
        organism.enable_variable_order_learning(102 + 7919, max_order=3)
        module_id = organism.active_motor_module
        _install(organism, ("input-000", "input-001"), module_id, 0.5)
        self.assertEqual(organism.variable_order_extra_sources, {})
        organism.validate()

    def test_state_round_trip_with_extras(self):
        organism = _input_aff_organism()
        module_id = organism.active_motor_module
        _install(organism, ("input-002", "input-003"), module_id, -0.25)
        restored = Organism.from_state_dict(organism.state_dict())
        self.assertEqual(restored.state_dict(), organism.state_dict())
        self.assertEqual(restored.variable_order_extra_sources,
                         organism.variable_order_extra_sources)

    def test_legacy_checkpoint_migrates_empty(self):
        organism = _input_aff_organism()
        payload = organism.state_dict()
        del payload["variable_order_extra_sources"]
        restored = Organism.from_state_dict(payload)
        self.assertEqual(restored.variable_order_extra_sources, {})
        restored.validate()

    def test_duplicate_feature_rejected(self):
        organism = _input_aff_organism()
        module_id = organism.active_motor_module
        _install(organism, ("input-000", "input-001"), module_id, 0.5)
        organism._variable_order_internal_transition = True
        try:
            with self.assertRaises(ValueError):
                organism.install_dendritic_feature(("input-000", "input-001"), module_id, 0.5)
        finally:
            organism._variable_order_internal_transition = False

    def test_unknown_owner_rejected(self):
        organism = _input_aff_organism()
        with self.assertRaises(ValueError):
            organism.install_dendritic_feature(("input-000", "input-001"), "motor-module-999", 0.5)


if __name__ == "__main__":
    unittest.main()
