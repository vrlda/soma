import unittest

from r6_retention_benchmark import orders_for, retention_metrics, split_probes


class RetentionMetricTests(unittest.TestCase):
    def test_probes_are_held_out_tails(self):
        split = split_probes([("a", b"x" * 100 + b"TAIL")], probe_bytes=4)
        self.assertEqual(split, [("a", b"x" * 100, b"TAIL")])
        with self.assertRaises(ValueError):
            split_probes([("short", b"abc")], probe_bytes=4)

    def test_orders_are_deterministic_permutations(self):
        names = ["a", "b", "c", "d", "e"]
        first, second = orders_for(names), orders_for(names)
        self.assertEqual(first, second)
        self.assertEqual(first["reversed"], names[::-1])
        for order in first.values():
            self.assertEqual(sorted(order), names)

    def test_forgetting_is_final_minus_best_after_learning(self):
        order = ["a", "b", "c"]
        rows = [(0.50, 0.90, 0.90), (0.55, 0.40, 0.80), (0.60, 0.45, 0.30)]
        curve = [{"bits_per_bit": {"probe:a": r[0], "probe:b": r[1], "probe:c": r[2],
                                   "validation": 0.5}} for r in rows]
        metrics = retention_metrics(order, curve)
        self.assertAlmostEqual(metrics["forgetting"]["a"], 0.10)
        self.assertAlmostEqual(metrics["forgetting"]["b"], 0.05)
        self.assertNotIn("c", metrics["forgetting"])
        self.assertAlmostEqual(metrics["mean_forgetting"], 0.075)
        self.assertEqual(metrics["matrix"][2], [0.60, 0.45, 0.30])


if __name__ == "__main__":
    unittest.main()
