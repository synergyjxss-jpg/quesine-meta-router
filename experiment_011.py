"""QUESINE Experiment 011: meta-router evaluation on held-out backbone seeds.

Run from this directory: python experiment_011.py
Requires NumPy, PyTorch, and scikit-learn. See experiment_011_review.md.
"""

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.ensemble import RandomForestClassifier

from experiment_011_protocol import (
    get_split, per_seed_metrics, paired_seed_sign_test, query_pools, routing_metrics,
)

N_TASK = 5
GATES = ['XOR', 'AND', 'OR', 'NAND', 'XNOR']
OPERANDS = ('A', 'B', 'A_left', 'A_right', 'B_left', 'B_right')
BITMASK = (1 << np.arange(7, -1, -1)).astype(np.int32)
TASK_NAMES = ['known_rule30', 'unseen_roll2', 'unseen_majority'] + [f'random_{i}' for i in range(5)]
FEATURE_NAMES = ['teacher_best_bit', 'teacher_second_bit', 'teacher_margin',
                 'head_train_bit', 'head_train_vector']


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def encode(keys, tid):
    keys = np.asarray(keys, dtype=np.int32)
    a = ((keys // 256)[:, None] & BITMASK[None, :]) > 0
    b = ((keys % 256)[:, None] & BITMASK[None, :]) > 0
    x = np.zeros((len(keys), 21), dtype=np.float32)
    x[:, :8], x[:, 8:16], x[:, 16 + tid] = a, b, 1.
    if tid == 0:
        y = np.logical_xor(a, b)
    elif tid == 1:
        y = np.logical_and(a, b)
    elif tid == 2:
        y = np.logical_or(a, b)
    elif tid == 3:
        y = np.logical_not(np.logical_and(a, b))
    else:
        y = np.logical_not(np.logical_xor(a, b))
    return x, y.astype(np.float32)


def net(depth=1, width=64):
    layers, size = [], 21
    for _ in range(depth):
        layers.extend([nn.Linear(size, width), nn.ReLU()])
        size = width
    layers.append(nn.Linear(size, 8))
    return nn.Sequential(*layers)


def make_training(seed, train_keys):
    rng = np.random.default_rng(8100 + seed)
    pool = [rng.choice(train_keys, 1536, replace=False) for _ in range(N_TASK)]
    mem = [rng.choice(p, 192, replace=False) for p in pool]
    xtr, memory = [], []
    for tid in range(N_TASK):
        xtr.append(tuple(torch.from_numpy(v) for v in encode(pool[tid], tid)))
        memory.append(tuple(torch.from_numpy(v) for v in encode(mem[tid], tid)))
    return xtr, memory


def train_initial_four(seed, train_keys, steps=250):
    # The supplied implementation is uniform replay, regardless of method name.
    seed_all(20290 + seed)
    rng = np.random.default_rng(4124 + seed)
    xtr, memory = make_training(seed, train_keys)
    model = net()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    for tid in range(4):
        xnew, ynew = xtr[tid]
        for _ in range(steps):
            idx = rng.choice(len(xnew), size=64, replace=False)
            xb, yb = xnew[idx], ynew[idx]
            if tid > 0:
                m_idx = rng.choice(tid, size=32, replace=True)
                mx, my = [], []
                for m_tid in m_idx:
                    mem_x, mem_y = memory[m_tid]
                    k = rng.choice(len(mem_x), size=1)[0]
                    mx.append(mem_x[k])
                    my.append(mem_y[k])
                xb = torch.cat((xb, torch.stack(mx)))
                yb = torch.cat((yb, torch.stack(my)))
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            loss.backward()
            optimizer.step()
    return model


def task_target(x, condition, task_seed=0):
    a, b = x[:, :8] > 0.5, x[:, 8:16] > 0.5
    if condition == 'known_rule30':
        return (a ^ (torch.roll(b, 1, 1) | b)).float()
    if condition == 'unseen_roll2':
        return (~(a ^ torch.roll(b, -2, 1))).float()
    if condition == 'unseen_majority':
        bl = torch.roll(b, 1, 1)
        return ((a & b) | (a & bl) | (b & bl)).float()
    if condition not in TASK_NAMES[3:]:
        raise ValueError(f'Unknown task: {condition}')
    rng = np.random.default_rng(task_seed)
    gate = ['AND', 'OR', 'XOR', 'NAND'][rng.integers(0, 4)]
    shift = int(rng.integers(-2, 3))
    shifted = torch.roll(b, shift, 1)
    if gate == 'AND':
        out = a & shifted
    elif gate == 'OR':
        out = a | shifted
    elif gate == 'XOR':
        out = a ^ shifted
    else:
        out = ~(a & shifted)
    return (~out if rng.integers(0, 2) else out).float()


def hidden(model, x, context=4):
    # Preserve the original context choice. Context 4 was not pretrained.
    xx = x.clone()
    xx[:, 16:] = 0
    xx[:, 16 + context] = 1
    return model[:2](xx)


def head_predict(head, model, x):
    with torch.no_grad():
        return head(hidden(model, x)) > 0


def predict_gate(model, first, second, gate):
    inp = torch.zeros((first.shape[0], 21), dtype=torch.float32)
    inp[:, :8], inp[:, 8:16], inp[:, 16 + gate] = first.float(), second.float(), 1
    with torch.no_grad():
        return model(inp) > 0


def generate_grammar(model, x):
    a, b = x[:, :8] > 0.5, x[:, 8:16] > 0.5
    vals = {'A': a, 'B': b, 'A_left': torch.roll(a, 1, 1),
            'A_right': torch.roll(a, -1, 1), 'B_left': torch.roll(b, 1, 1),
            'B_right': torch.roll(b, -1, 1)}
    expressions = []
    for g in range(4):
        for i, u in enumerate(OPERANDS):
            for v in OPERANDS[i + 1:]:
                inner = predict_gate(model, vals[u], vals[v], g)
                desc = f'{GATES[g]}({u},{v})'
                for invert in (False, True):
                    expressions.append((('NOT ' if invert else '') + desc,
                                        ~inner if invert else inner))
                for root in ('A', 'B'):
                    for outer in range(4):
                        combined = predict_gate(model, vals[root], inner, outer)
                        desc2 = f'{GATES[outer]}({root},{desc})'
                        for invert in (False, True):
                            expressions.append((('NOT ' if invert else '') + desc2,
                                                ~combined if invert else combined))
    return expressions


def train_scratch_head(model, xtrain, ytrain, seed, steps=250):
    features = hidden(model, xtrain).detach()
    # Honor the seed independently of task order; restore the caller's RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        head = nn.Linear(features.shape[1], 8)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.01)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(head(features), ytrain)
        loss.backward()
        optimizer.step()
    return head


def vec_acc(pred, target):
    return float((pred.bool() == (target > 0.5)).all(dim=1).float().mean())


def extract_features(model, expr_support, xs, ys, head):
    ranked = sorted(
        [(name, float((pred == (ys > 0.5)).float().mean()))
         for name, pred in expr_support],
        key=lambda item: item[1], reverse=True,
    )
    best, second = ranked[0][1], ranked[1][1]
    head.eval()
    correct = head_predict(head, model, xs) == (ys > 0.5)
    features = np.array([best, second, best - second,
                         float(correct.float().mean()),
                         float(correct.all(dim=1).float().mean())])
    return features, ranked[0][0]


def run(smoke=False):
    torch.set_num_threads(1)
    train_keys, val_keys, test_keys = get_split()
    train_queries, eval_queries = query_pools(val_keys, test_keys)
    train_seeds = [0, 1] if smoke else list(range(15))
    test_seeds = [15] if smoke else list(range(15, 20))
    backbone_steps, head_steps = (5, 5) if smoke else (250, 250)
    features_train, labels_train, eval_rows = [], [], []
    train_ties = 0
    print('QUESINE Experiment 011: held-out-seed meta-router evaluation', flush=True)
    if smoke:
        print('SMOKE TEST: reduced training; no scientific conclusions.', flush=True)
    for seed in train_seeds + test_seeds:
        is_train = seed in train_seeds
        print(f"Seed {seed} ({'train' if is_train else 'TEST'})", flush=True)
        model = train_initial_four(seed, train_keys, backbone_steps)
        model.eval().requires_grad_(False)
        support = np.random.default_rng(77000 + seed).choice(train_keys, 96, replace=False)
        xs = torch.from_numpy(encode(support, 0)[0])
        xquery = torch.from_numpy(encode(train_queries if is_train else eval_queries, 0)[0])
        expr_support = generate_grammar(model, xs)
        # Predictions are task independent and contain no query-label information.
        expr_query = dict(generate_grammar(model, xquery))
        for task_idx, task in enumerate(TASK_NAMES):
            task_seed = 5000 + task_idx
            ys = task_target(xs, task, task_seed)
            head = train_scratch_head(model, xs, ys, 90000 + 100 * seed + task_idx, head_steps)
            features, best_expression = extract_features(model, expr_support, xs, ys, head)
            # Query labels only define training labels or measure final outcomes.
            yquery = task_target(xquery, task, task_seed)
            teacher = vec_acc(expr_query[best_expression], yquery)
            scratch = vec_acc(head_predict(head, model, xquery), yquery)
            winner = int(teacher > scratch)
            if is_train:
                # Keep the original tie-to-head rule as a deterministic label.
                train_ties += int(teacher == scratch)
                features_train.append(features)
                labels_train.append(winner)
            else:
                eval_rows.append({
                    'seed': seed, 'method': 'UniformReplay', 'task': task,
                    'feats': features.tolist(), 'best_expression': best_expression,
                    't_vec': teacher, 's_vec': scratch, 'winner': winner,
                })
    rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    rf.fit(features_train, labels_train)
    predictions = {
        'Learned RF': rf.predict(np.asarray([row['feats'] for row in eval_rows])),
        'Hardcoded 5pp (Support)': np.asarray([
            int(row['feats'][0] > row['feats'][3] + 0.05) for row in eval_rows
        ]),
        'Always-Teacher': np.ones(len(eval_rows), dtype=int),
        'Always-Head': np.zeros(len(eval_rows), dtype=int),
    }
    metrics = [routing_metrics(eval_rows, pred, name) for name, pred in predictions.items()]
    per_seed = per_seed_metrics(eval_rows, predictions['Learned RF'], 'Learned RF')
    # One designated comparison; other baselines remain descriptive.
    comparison = paired_seed_sign_test(
        eval_rows, predictions['Learned RF'], predictions['Hardcoded 5pp (Support)']
    )
    print(f'\n{len(eval_rows)} episodes, {len(test_seeds)} held-out seeds, one replay method')
    print(f"{'Router':26s} {'Acc*':>8s} {'Correct*':>10s} {'Routed':>9s} {'Regret':>9s}")
    for row in metrics:
        accuracy = 'n/a' if row['routing_acc'] is None else f"{100 * row['routing_acc']:.2f}%"
        correct = f"{row['correct']}/{row['n_untied']}"
        print(f"{row['name']:26s} {accuracy:>8s} {correct:>10s} "
              f"{row['routed_mean']:9.4f} {row['regret']:9.4f}")
    print('* Routing accuracy excludes query-performance ties; routed score includes all rows.')
    print('\nPer-seed RF results:')
    for row in per_seed:
        print(f"Seed {row['seed']}: routed_mean={row['routed_mean']:.4f}, "
              f"correct={row['correct']}/{row['n_untied']}, ties={row['ties']}")
    print('\nPaired seed comparison, RF minus Hardcoded 5pp:')
    print(json.dumps(comparison, indent=2))
    importances = dict(zip(FEATURE_NAMES, rf.feature_importances_.tolist()))
    print('\nFeature importances:', importances)
    print('Same task definitions appear in meta-training and evaluation; this tests new seeds.')
    print(f'Evaluation uses {len(test_seeds)} seeds; significance is not proof.')
    for name, prediction in predictions.items():
        for row, route in zip(eval_rows, prediction):
            row.setdefault('routes', {})[name] = int(route)
    result = {
        'smoke_test': smoke, 'train_seeds': train_seeds, 'test_seeds': test_seeds,
        'train_ties_labelled_head': train_ties,
        'meta_train_labels': {str(v): labels_train.count(v) for v in (0, 1)},
        'meta_train_queries': train_queries.tolist(), 'meta_eval_queries': eval_queries.tolist(),
        'metrics': metrics, 'per_seed': per_seed, 'primary_comparison': comparison,
        'feature_importances': importances, 'evaluation_rows': eval_rows,
    }
    destination = Path('/tmp/experiment_011_smoke.json' if smoke
                       else '/workspace/outputs/experiment_011_results.json')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Results saved to {destination}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true', help='Check execution with minimal training')
    run(smoke=parser.parse_args().smoke)
