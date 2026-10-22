"""Run with: python -m unittest discover -s /workspace/outputs -v"""

import unittest

import numpy as np

from experiment_011_protocol import (
    get_split, orbit_representatives, paired_seed_sign_test,
    per_seed_metrics, query_pools, routing_metrics,
)


class ProtocolTests(unittest.TestCase):
    def test_key_partition_is_complete_and_orbit_disjoint(self):
        partitions = get_split()
        self.assertEqual([len(keys) for keys in partitions], [61488, 2022, 2026])
        self.assertEqual(len(np.unique(np.concatenate(partitions))), 65536)
        groups = [set(orbit_representatives(keys)) for keys in partitions]
        for i in range(3):
            for j in range(i + 1, 3):
                self.assertFalse(groups[i] & groups[j])

    def test_original_overlap_and_corrected_queries(self):
        _, val, test = get_split()
        original = np.random.default_rng(42).permutation(test)
        overlap = set(orbit_representatives(original[:256])) & set(
            orbit_representatives(original[256:512]))
        self.assertEqual(len(overlap), 63)
        train_q, eval_q = query_pools(val, test)
        self.assertFalse(set(orbit_representatives(train_q)) & set(orbit_representatives(eval_q)))
        self.assertTrue(set(train_q) <= set(val))
        self.assertTrue(set(eval_q) <= set(test))

    def test_query_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            query_pools(np.arange(256), np.arange(256))

    def test_per_seed_predictions_stay_with_their_rows(self):
        # Deliberately interleave seeds; one always wants head, the other teacher.
        rows = [
            {'seed': s, 't_vec': float(s == 16), 's_vec': float(s == 15)}
            for _ in range(3) for s in (15, 16)
        ]
        predictions = [0, 1, 0, 1, 0, 1]
        for result in per_seed_metrics(rows, predictions, 'test'):
            self.assertEqual(result['correct'], 3)
            self.assertEqual(result['routed_mean'], 1.)

    def test_ties_and_regret_have_separate_meaning(self):
        rows = [
            {'seed': 0, 't_vec': .5, 's_vec': .5},
            {'seed': 0, 't_vec': 1., 's_vec': 0.},
        ]
        result = routing_metrics(rows, [1, 0], 'test')
        self.assertEqual(result['ties'], 1)
        self.assertEqual(result['n_untied'], 1)
        self.assertEqual(result['routing_acc'], 0.)
        self.assertEqual(result['regret'], .5)
        self.assertIsNone(routing_metrics(rows[:1], [0], 'test')['routing_acc'])

    def test_sign_test_counts_seeds_not_task_rows(self):
        rows = [{'seed': s, 't_vec': 1., 's_vec': 0.} for s in range(5) for _ in range(8)]
        result = paired_seed_sign_test(rows, [1] * 40, [0] * 40)
        self.assertEqual(result['wins'], 5)
        self.assertEqual(result['one_sided_sign_p'], 1 / 32)
        tied = paired_seed_sign_test(rows, [1] * 40, [1] * 40)
        self.assertEqual(tied['ties'], 5)
        self.assertEqual(tied['one_sided_sign_p'], 1.)

    def test_invalid_routes_and_lengths_raise(self):
        rows = [{'seed': 0, 't_vec': 1., 's_vec': 0.}]
        for predictions in ([], [0, 1], [2], [[1]]):
            with self.assertRaises(ValueError):
                routing_metrics(rows, predictions, 'test')


if __name__ == '__main__':
    unittest.main()
