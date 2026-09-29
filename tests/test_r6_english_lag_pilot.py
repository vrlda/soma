import unittest

import r6_english_lag_pilot as pilot


class R6EnglishLagPilotTests(unittest.TestCase):
    def test_continuous_frozen_scoring_and_resume(self):
        report = pilot.run_english_lag_pilot((0,), acquisition=b"River road. " * 4,
                                             validation=b"River.", test=b"road!")
        self.assertTrue(report["development_only"])
        self.assertFalse(report["acceptance_gating"])
        self.assertEqual(report["r6_qualification"], "not_evaluated")
        self.assertTrue(report["diagnostics"]["continuous_frozen_scoring"])
        self.assertTrue(report["diagnostics"]["frozen_read_only_all"])
        self.assertTrue(report["diagnostics"]["single_clone_all"])
        self.assertTrue(report["diagnostics"]["resume_digest_all"])
        self.assertEqual(report["controls"]["fit_partition"], "acquisition_only")
        self.assertEqual(report["controls"]["constant_controls"], "oracle_diagnostics")


if __name__ == "__main__":
    unittest.main()
