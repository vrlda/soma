import unittest

from soma import SOMA
from soma.organism import Organism


class R0BaselineTests(unittest.TestCase):
    def test_organism_version_frozen_at_12(self):
        self.assertEqual(Organism.VERSION, 12)

    def test_public_profile_unchanged(self):
        model = SOMA.create(input_size=2, seed=7)
        status = model.inspect(0)
        self.assertEqual(status["profile"], "continuous-v1")
        self.assertEqual(status["checkpoint_version"], 12)

    def test_manifests_importable(self):
        from soma import FINAL_LIFETIME_MANIFEST, CONTINUOUS_LIFETIME_MANIFEST
        self.assertTrue(len(FINAL_LIFETIME_MANIFEST) > 0)
        self.assertTrue(len(CONTINUOUS_LIFETIME_MANIFEST) > 0)


if __name__ == "__main__":
    unittest.main()
