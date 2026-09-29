import unittest

import r6_text_temporal_transport_pilot as pilot
import r6_text_temporal_transport_2048 as pilot_2048


class R6TextTemporalTransportTests(unittest.TestCase):
    def test_smoke_declares_transport_and_read_only_resume_invariants(self):
        report = pilot.run_transport_pilot((0,), acquisition_examples=16, evaluation_examples=3)
        self.assertTrue(report["development_only"])
        self.assertFalse(report["acceptance_gating"])
        self.assertEqual(report["transport"]["depth"], 8)
        self.assertEqual(report["diagnostics"]["balanced_leave_one_out"], True)
        self.assertTrue(report["diagnostics"]["frozen_read_only_all"])
        self.assertTrue(report["diagnostics"]["resume_digest_all"])
        self.assertTrue(report["diagnostics"]["withheld_frames_no_credit_all"])
        self.assertTrue(report["diagnostics"]["candidate_family_excludes_fixed_bias"])
        self.assertIn("dependencies", report)
        self.assertEqual(len(report["comparison"]["persistent_minus_byte_unigram_skill"]), 8)
        persistent = report["runs"]["lag_persistent"][0]["cases"][0]
        self.assertIn("withheld_no_credit_examples", persistent["resources"])
        self.assertIn("target_winner", persistent)
        self.assertIn("target_rank_available", persistent)

    def test_event_payload_does_not_contain_evaluator_target(self):
        events = pilot._frame(1, 0, 0)
        self.assertTrue(all("target" not in event["payload"] for event in events))
        self.assertTrue(all("combo" not in event["payload"] for event in events))

    def test_v2_protocol_is_frozen_and_budget_is_parameterized(self):
        organism, _ = pilot._new_organism(7, "lag_persistent", persistent_budget=2048)
        state = organism.state_dict()
        shuffled, _ = pilot._new_organism(7, "lag_shuffled", persistent_budget=2048)
        shuffled_state = shuffled.state_dict()
        self.assertEqual(pilot_2048.PROTOCOL, "r6-text-temporal-transport-pilot-v2-2048")
        self.assertEqual(pilot_2048.PREDECLARED_CRITERIA["matrix"]["candidate_family_size"], 680)
        self.assertEqual(state["variable_order_persistent_evidence_budget"], 2048)
        self.assertTrue(shuffled_state["variable_order_persistent_scout_enabled"])
        self.assertEqual(shuffled_state["variable_order_persistent_evidence_budget"], 2048)
        self.assertEqual(pilot_2048.EVALUATION_EXAMPLES, 64)
        self.assertEqual(pilot_2048.TARGET_CASE_FLOOR, 0.75)

    def test_v2_null_target_install_is_counted_symmetrically(self):
        report = pilot.run_transport_pilot((2137,), acquisition_examples=16, evaluation_examples=3,
                                           persistent_budget=32)
        report["runs"]["lag_shuffled"][0]["cases"][0]["target_k3_installed"] = True
        summarized = pilot_2048._summarize(report)
        self.assertEqual(summarized["qualification_summary"]["null_target_install_count"], 1)
        self.assertFalse(summarized["qualification_summary"]["quality_gates"]["paired_shuffled_zero_target_installs"])


if __name__ == "__main__":
    unittest.main()
