import hashlib
import json
import unittest

import r6_temporal_composition_pilot as pilot
from soma.organism import Organism


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


class R6TemporalCompositionPilotTests(unittest.TestCase):
    def test_iid_workspace_excludes_only_declared_combination(self):
        acquisition = pilot._examples(0, 7, 64)
        evaluation = pilot._examples(0, 7, 16, require_heldout=True)
        self.assertTrue(all(example["combo"] != 7 for example in acquisition))
        self.assertTrue(all(example["combo"] == 7 for example in evaluation))
        self.assertEqual({example["combo"] for example in acquisition}, set(range(7)))
        self.assertEqual({example["combo"] for example in evaluation}, {7})

    def test_declared_target_k3_is_installed_and_lesionable(self):
        examples = pilot._examples(0, 1, 512)
        checkpoint, _ = pilot._train(pilot._new_organism(0, "variable_order"), examples, 0)
        self.assertIn(pilot.TARGET_KEY, checkpoint["variable_order_feature_owners"])
        lesioned, changed = pilot._lesion_target_k3(checkpoint)
        self.assertTrue(changed)
        self.assertNotEqual(_digest(checkpoint), _digest(lesioned))

    def test_frozen_score_preserves_checkpoint(self):
        examples = pilot._examples(0, 1, 64)
        checkpoint, _ = pilot._train(pilot._new_organism(0, "variable_order"), examples, 0)
        before = _digest(checkpoint)
        _, metrics = pilot._frozen_score(checkpoint, pilot._examples(1, 1, 8, require_heldout=True), 0)
        self.assertEqual(before, _digest(checkpoint))
        self.assertTrue(metrics["frozen_read_only"])
        self.assertTrue(metrics["checkpoint_unchanged"])

    def test_pilot_is_explicitly_non_gating(self):
        report = pilot.run_temporal_pilot((0,), acquisition_examples=32, evaluation_examples=4)
        self.assertTrue(report["development_only"])
        self.assertFalse(report["acceptance_gating"])
        self.assertEqual(report["r6_qualification"], "not_evaluated")
        self.assertTrue(report["diagnostics"]["balanced_leave_one_out"])
        self.assertTrue(report["diagnostics"]["both_target_signs_covered"])
        self.assertTrue(report["diagnostics"]["frozen_read_only_all"])
        self.assertTrue(report["diagnostics"]["resume_digest_all"])
        self.assertIn("persistent_variable_order", pilot.MODES)
        self.assertIn("constant_plus", report["controls"]["cases"][0])
        self.assertEqual(report["controls"]["fit_partition"], "acquisition_only")

    def test_persistent_mode_roundtrips_evidence_configuration(self):
        organism = pilot._new_organism(0, "persistent_variable_order")
        state = organism.state_dict()
        self.assertTrue(state["variable_order_persistent_scout_enabled"])
        self.assertEqual(state["variable_order_persistent_evidence_budget"], 256)
        self.assertEqual(state["variable_order_persistent_min_count"], 32)
        self.assertEqual(state["variable_order_persistent_terminal_state"], "active")
        restored = Organism.from_state_dict(json.loads(json.dumps(state)))
        self.assertEqual(restored.state_dict(), state)


if __name__ == "__main__":
    unittest.main()
