import unittest

from soma.evaluation.dialogue import (
    bits_to_bytes,
    fresh_dialogue,
    respond,
    sanitize_export,
    teach_fact,
    text_to_bits,
)
from soma.memory import EpisodicBuffer, SequenceCircuitMemory


def _background():
    memory = SequenceCircuitMemory((0, 1), max_order=8, max_circuits=512)
    for symbol in ([0, 0, 1, 1, 0, 1] * 20):
        memory.observe(symbol)
    return memory


class R3DDialogueTests(unittest.TestCase):
    def test_text_bits_round_trip(self):
        self.assertEqual(bits_to_bytes(text_to_bits("Hi!")), b"Hi!")

    def test_teach_and_recall(self):
        dialogue = fresh_dialogue()
        episodic = EpisodicBuffer()
        teach_fact(dialogue, episodic, "USER code Q AGENT A apple ", "USER code Q ", "apple")
        response = respond(_background(), dialogue, episodic, "USER code Q ", 6, seed=1)
        self.assertTrue(response.startswith(b"apple"))

    def test_supersede_replaces_completion(self):
        dialogue = fresh_dialogue()
        episodic = EpisodicBuffer()
        first = teach_fact(dialogue, episodic, "USER k QR AGENT A one ", "USER k QR ", "one")
        second = teach_fact(dialogue, episodic, "USER k QR AGENT A two ", "USER k QR ", "two")
        self.assertEqual(first, second)
        response = respond(_background(), dialogue, episodic, "USER k QR ", 4, seed=1)
        self.assertTrue(response.startswith(b"two"))

    def test_longest_trigger_wins(self):
        dialogue = fresh_dialogue()
        episodic = EpisodicBuffer()
        teach_fact(dialogue, episodic, "USER AB AGENT A wide ", "USER AB ", "wide")
        teach_fact(dialogue, episodic, "USER B AGENT A narrow ", "USER B ", "narrow")
        response = respond(_background(), dialogue, episodic, "USER AB ", 5, seed=1)
        self.assertTrue(response.startswith(b"wide"))

    def test_remove_forgets(self):
        dialogue = fresh_dialogue()
        episodic = EpisodicBuffer()
        entry = teach_fact(dialogue, episodic, "USER z9 AGENT A yes ", "USER z9 ", "yes")
        self.assertTrue(respond(_background(), dialogue, episodic, "USER z9 ", 4, seed=1).startswith(b"yes"))
        episodic.remove(entry)
        self.assertFalse(respond(_background(), dialogue, episodic, "USER z9 ", 4, seed=1).startswith(b"yes"))
        with self.assertRaises(ValueError):
            episodic.remove(entry)

    def test_sanitize_filters_blocked_provenance(self):
        dialogue = fresh_dialogue()
        episodic = EpisodicBuffer()
        teach_fact(dialogue, episodic, "USER pub AGENT A one ", "USER pub ", "one", provenance="public")
        teach_fact(dialogue, episodic, "USER sec AGENT A two ", "USER sec ", "two", provenance="private")
        _, kept = sanitize_export(dialogue, episodic)
        restored = EpisodicBuffer.from_state_dict(kept)
        self.assertEqual(len(restored.entries), 1)
        self.assertTrue(respond(_background(), dialogue, restored, "USER pub ", 4, seed=1).startswith(b"one"))
        self.assertFalse(respond(_background(), dialogue, restored, "USER sec ", 4, seed=1).startswith(b"two"))

    def test_emissions_never_train(self):
        background = _background()
        dialogue = fresh_dialogue()
        episodic = EpisodicBuffer()
        before = (len(background.circuits), len(dialogue.circuits))
        respond(background, dialogue, episodic, "USER hello ", 8, seed=3)
        after = (len(background.circuits), len(dialogue.circuits))
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
