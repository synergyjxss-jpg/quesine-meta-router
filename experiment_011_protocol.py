"""Data partitions and reporting for Experiment 011; no ML dependencies."""

from math import comb

import numpy as np


def orbit_representatives(keys):
    keys = np.asarray(keys, dtype=np.int32)
    swapped = (keys % 256) * 256 + keys // 256
    return np.minimum.reduce([keys, swapped, keys ^ 65535, swapped ^ 65535])


def get_split(split_seed=20261024):
    keys = np.arange(65536, dtype=np.int32)
    representatives = orbit_representatives(keys)
    unique = np.random.default_rng(split_seed).permutation(np.unique(representatives))
    return tuple(
        keys[np.isin(representatives, selected)]
        for selected in (unique[1024:], unique[:512], unique[512:1024])
    )


def query_pools(val_keys, test_keys, size=256):
    """Reserve the original validation partition for router-training labels."""
    train_queries = np.random.default_rng(42).choice(val_keys, size, replace=False)
    eval_queries = np.random.default_rng(43).choice(test_keys, size, replace=False)
    overlap = np.intersect1d(
        orbit_representatives(train_queries), orbit_representatives(eval_queries)
    )
    if overlap.size:
        raise ValueError("Router training and evaluation share symmetry groups")
    return train_queries, eval_queries


def aligned_predictions(rows, predictions):
    predictions = np.asarray(predictions)
    if predictions.ndim != 1 or len(predictions) != len(rows):
        raise ValueError("Exactly one prediction per row is required")
    if not np.isin(predictions, [0, 1]).all():
        raise ValueError("Route predictions must be 0 (head) or 1 (teacher)")
    if not rows:
        raise ValueError("At least one evaluation row is required")
    return predictions


def routed_scores(rows, predictions):
    predictions = aligned_predictions(rows, predictions)
    return np.asarray([
        row['t_vec'] if route else row['s_vec']
        for row, route in zip(rows, predictions)
    ])


def routing_metrics(rows, predictions, name):
    predictions = aligned_predictions(rows, predictions)
    teacher = np.asarray([row['t_vec'] for row in rows])
    head = np.asarray([row['s_vec'] for row in rows])
    routed = routed_scores(rows, predictions)
    oracle = np.maximum(teacher, head)
    # Accuracies are computed on the same query count; equality is a real tie.
    untied = teacher != head
    correct = int(((predictions == (teacher > head)) & untied).sum())
    n_untied = int(untied.sum())
    return {
        'name': name,
        'n': len(rows),
        'n_untied': n_untied,
        'ties': len(rows) - n_untied,
        'correct': correct,
        'routing_acc': correct / n_untied if n_untied else None,
        'routed_mean': float(routed.mean()),
        'oracle_mean': float(oracle.mean()),
        'regret': float((oracle - routed).mean()),
    }


def per_seed_metrics(rows, predictions, name):
    predictions = aligned_predictions(rows, predictions)
    result = []
    for seed in sorted({row['seed'] for row in rows}):
        indices = np.asarray([i for i, row in enumerate(rows) if row['seed'] == seed])
        group = [rows[i] for i in indices]
        result.append({
            'seed': seed,
            **routing_metrics(group, predictions[indices], name),
        })
    return result


def paired_seed_sign_test(rows, candidate, baseline):
    """One-sided sign test on mean routed-accuracy differences by seed.

    Tests the probability of a positive difference among nonzero seed
    differences against 1/2, conditional on the fixed tasks, queries and
    fitted router. It does NOT test episode-level accuracy or a mean effect.
    Independent seed runs are the inferential units. Exact ties are omitted.
    """
    differences = routed_scores(rows, candidate) - routed_scores(rows, baseline)
    deltas = []
    for seed in sorted({row['seed'] for row in rows}):
        mask = np.asarray([row['seed'] == seed for row in rows])
        deltas.append({'seed': seed, 'delta': float(differences[mask].mean())})
    wins = sum(row['delta'] > 0 for row in deltas)
    losses = sum(row['delta'] < 0 for row in deltas)
    n = wins + losses
    pvalue = sum(comb(n, k) for k in range(wins, n + 1)) / 2**n if n else 1.0
    return {
        'seed_differences': deltas,
        'mean_seed_delta': float(np.mean([row['delta'] for row in deltas])),
        'wins': wins,
        'losses': losses,
        'ties': len(deltas) - n,
        'one_sided_sign_p': pvalue,
        'inference': 'Conditional on the fixed task suite, queries, and fitted router',
    }
