import unittest

from r6_compositional_byte_event_benchmark import (
    ByteTripleTransducer,
    ProductExperimentConfig,
    _event,
    frozen_manifest,
    run_product_experiment,
    symbol_to_values,
)
from soma.events.envelope import Event


class R6ProductEventDevelopmentTests(unittest.TestCase):
    def test_byte_transducer_round_trip_covers_complete_alphabet(self):
        transducer = ByteTripleTransducer()
        for symbol in range(8):
            event = _event(symbol, symbol).to_dict()
            frame = {"byte": event}
            self.assertEqual(transducer.pack_frame(frame), symbol_to_values(symbol))

    def test_event_boundary_rejects_target_metadata(self):
        with self.assertRaises(ValueError):
            # Event validation is intentionally exercised directly so a task
            # answer cannot be smuggled into the byte channel.
            Event("byte", "test", 0, 0.0, {"target": 1.0}, {"trust": "test"})

    def test_manifest_holds_out_exactly_one_combination(self):
        manifest = frozen_manifest(0, ProductExperimentConfig(acquisition_repetitions=8), heldout_symbol=7)
        self.assertEqual(set(manifest["acquisition_symbols"]), set(range(7)))
        self.assertEqual(manifest["heldout_symbol"], 7)
        self.assertNotIn(manifest["heldout_symbol"], manifest["acquisition_symbols"])

    def test_development_run_is_non_qualifying_and_product_path_is_causal(self):
        report = run_product_experiment((0,), ProductExperimentConfig(
            acquisition_repetitions=192,
        ))
        self.assertTrue(report["development_only"])
        self.assertFalse(report["acceptance_gating"])
        self.assertEqual(report["r6_qualification"], "not_evaluated")
        self.assertTrue(report["matched_streams"])
        self.assertTrue(report["matched_initial_structure"])
        self.assertTrue(report["diagnostics"]["balanced_leave_one_out_coverage"])
        self.assertTrue(report["diagnostics"]["both_product_signs_covered"])
        self.assertTrue(report["diagnostics"]["frozen_read_only_all_cases"])
        self.assertTrue(report["diagnostics"]["intact_lesion_protocol_match"])
        variable = report["runs"]["variable_order"][0]
        additive = report["runs"]["nonlinear_additive_context"][0]
        self.assertTrue(variable["k3_product_feature_installed_all_cases"])
        self.assertTrue(variable["product_path_lesion_applied_all_cases"])
        self.assertTrue(all(case["heldout_checkpoint_unchanged"] for case in variable["cases"]))
        lesion_values = [case["product_path_lesion_drop"] for case in variable["cases"]]
        self.assertGreater(sum(lesion_values) / len(lesion_values), 0.05)
        margins = [
            variable["cases"][i]["frozen_heldout_metrics"]["skill"]
            - additive["cases"][i]["frozen_heldout_metrics"]["skill"]
            for i in range(8)
        ]
        self.assertGreater(sum(margins) / len(margins), 0.10)


if __name__ == "__main__":
    unittest.main()
