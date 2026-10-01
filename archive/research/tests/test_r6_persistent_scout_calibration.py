import unittest

import r6_persistent_scout_calibration as calibration
import r6_temporal_composition_pilot as pilot


class PersistentScoutCalibrationTests(unittest.TestCase):
    def test_protocol_declares_full_family_and_fixed_null_objective(self):
        self.assertEqual(calibration.FAMILY_SIZE, 2600)
        self.assertEqual(calibration.BUDGETS, (128, 256, 512))
        self.assertEqual(calibration.POWER_FLOOR, 0.75)
        self.assertEqual(calibration.NULL_FWER_TARGET, 0.05)

    def test_smoke_reports_target_and_both_null_streams(self):
        report = calibration.run_calibration(seeds=(0,), budgets=(128,))
        self.assertTrue(report["development_only"])
        self.assertFalse(report["acceptance_gating"])
        self.assertEqual(report["family_size"], 2600)
        self.assertEqual({row["reward_mode"] for row in report["rows"]}, {"target", "shuffled", "random"})
        self.assertTrue(all(row["family_size"] == 2600 for row in report["rows"]))

    def test_temporal_pilot_freezes_calibrated_budget(self):
        self.assertEqual(pilot.PERSISTENT_EVIDENCE_BUDGET, 256)


if __name__ == "__main__":
    unittest.main()
