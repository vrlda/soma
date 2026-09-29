import hashlib
import json
import unittest

import r6_english_event_pilot as pilot


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


class R6EnglishEventPilotTests(unittest.TestCase):
    def test_partitions_are_frozen_and_utf8(self):
        acquisition = b"".join(pilot.ACQUISITION_DOCUMENTS)
        pilot.validate_utf8(acquisition)
        pilot.validate_utf8(pilot.VALIDATION_BYTES)
        pilot.validate_utf8(pilot.TEST_BYTES)
        self.assertNotEqual(hashlib.sha256(acquisition).hexdigest(), hashlib.sha256(pilot.TEST_BYTES).hexdigest())
        self.assertNotEqual(hashlib.sha256(pilot.VALIDATION_BYTES).hexdigest(), hashlib.sha256(pilot.TEST_BYTES).hexdigest())

    def test_frozen_score_does_not_mutate_checkpoint(self):
        acquisition = b"The river moved quietly. The keeper marked each door.\n" * 4
        organism = pilot._new_organism(0, "variable_order")
        checkpoint, _ = pilot._train(organism, acquisition, 0)
        before = _digest(checkpoint)
        _, metrics = pilot._frozen_score(checkpoint, b"The keeper returned.\n", 0)
        self.assertEqual(before, _digest(checkpoint))
        self.assertTrue(metrics["frozen_read_only"])
        self.assertTrue(metrics["checkpoint_unchanged"])
        self.assertGreater(metrics["independent_clone_count"], 1)

    def test_pilot_is_explicitly_non_gating_and_controls_fit_acquisition(self):
        report = pilot.run_english_pilot((0,))
        self.assertTrue(report["development_only"])
        self.assertFalse(report["acceptance_gating"])
        self.assertEqual(report["r6_qualification"], "not_evaluated")
        self.assertFalse(report["event_path"]["target_in_payload"])
        self.assertFalse(report["event_path"]["heldout_target_derived_outcomes_sent"])
        self.assertTrue(report["diagnostics"]["frozen_read_only_all"])
        self.assertTrue(report["diagnostics"]["checkpoint_resume_digest_all"])
        self.assertTrue(report["diagnostics"]["product_path_installed_all_variable_order"])
        self.assertEqual(report["controls"]["fit_partition"], "acquisition_only")


if __name__ == "__main__":
    unittest.main()
