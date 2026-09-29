import unittest

from r6_factual_benchmark import starts_with


class R6FactualBenchmarkTests(unittest.TestCase):
    def test_prefix_accepts_incomplete_utf8_tail(self):
        # A fixed-length response may stop in the middle of a later codepoint;
        # that must not invalidate an already-complete factual answer.
        self.assertTrue(starts_with(b"fifthI$\xc9", "fifth"))

    def test_prefix_rejects_wrong_answer(self):
        self.assertFalse(starts_with(b"sixth", "fifth"))


if __name__ == "__main__":
    unittest.main()
