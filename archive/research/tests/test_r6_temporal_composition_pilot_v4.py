import unittest

import r6_temporal_composition_pilot_v4 as pilot


class R6TemporalCompositionPilotV4Tests(unittest.TestCase):
    def test_v4_is_non_gating_and_uses_predeclared_512_budget(self):
        self.assertEqual(pilot.PROTOCOL, "r6-temporal-composition-pilot-v4")
        self.assertEqual(pilot.PERSISTENT_EVIDENCE_BUDGET, 512)
        organism = pilot._new_organism(0, "persistent_variable_order")
        self.assertEqual(organism.variable_order_persistent_evidence_budget, 512)

    def test_v4_smoke_exposes_heldout_control_margins(self):
        report = pilot.run_temporal_pilot((0,), acquisition_examples=32, evaluation_examples=4)
        self.assertTrue(report["development_only"])
        self.assertFalse(report["acceptance_gating"])
        self.assertEqual(report["protocol"], "r6-temporal-composition-pilot-v4")
        self.assertEqual(len(report["v4_heldout_performance"]["cases"]), 8)
        self.assertTrue(all("persistent_vs_additive_skill" in case for case in report["v4_heldout_performance"]["cases"]))
        self.assertTrue(report["diagnostics"]["frozen_read_only_all"])
        self.assertTrue(report["diagnostics"]["resume_digest_all"])


if __name__ == "__main__":
    unittest.main()
