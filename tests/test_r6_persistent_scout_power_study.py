import unittest

import r6_persistent_scout_power_study as study


class R6PersistentScoutPowerStudyTests(unittest.TestCase):
    def test_lesion_drop_sign_is_intact_minus_lesioned(self):
        self.assertAlmostEqual(study._lesion_drop(0.90857, -0.01632), 0.92489, places=5)

    def test_staged_smoke_is_incomplete_and_cannot_qualify(self):
        report = study.run_power_study(
            seeds=(0,), budgets=(32,), heldout_combos=(0, 1),
            max_cases=1, evaluation_examples=2)
        self.assertTrue(report["development_only"])
        self.assertFalse(report["acceptance_gating"])
        self.assertEqual(report["cases_run"], 1)
        self.assertIsNone(report["selected_budget"])
        self.assertFalse(report["by_budget"]["32"]["complete"])
        self.assertFalse(report["by_budget"]["32"]["qualifies"])
        self.assertEqual(len(report["rows"]), 2)


if __name__ == "__main__":
    unittest.main()
