# Experiment 011: review and corrections

The supplied experiment cannot establish a "statistical proof" of the router. Two implementation bugs affect its reported evidence, and its evaluation supports a narrower claim than its task names suggest. The revision preserves the backbone architecture, replay schedule, task definitions, grammar, five router features, and random-forest hyperparameters.

## Findings in the supplied code

1. **The two methods implement the same procedure.** `train_initial_four(method, ...)` never reads `method`. It resets all random generators from the same seed each time. Thus `HardExample` and `UniformScored` are duplicate runs of uniform replay under deterministic execution. The 80 evaluation rows represent 40 distinct seed/task settings, with only five held-out backbone seeds. Training rows are duplicated too. The revision runs one method, named `UniformReplay`. It does not invent a missing hard-example algorithm.

2. **The per-seed report misaligns predictions.** Filtering `eval_rows` and then zipping with all of `preds_rf` always reuses the first 16 predictions. The report for seed 15 is aligned; later reports generally are not. The overall metrics, which zip the full rows and predictions, do not have this indexing bug. The revision retains original row indices when selecting predictions for each seed.

3. **The original binomial p-values do not establish improvement over a baseline.** A success null of 0.5 describes a fair random routing decision on non-tied episodes. It is not the performance of the majority route or the support-threshold router. Duplicate observations and dependence within a backbone seed also defeat an independent episode-level interpretation. SciPy describes exactly the probability hypothesis tested by [`binomtest`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html); that hypothesis is not a paired router comparison.

4. **Meta-training and evaluation query keys are disjoint, but their symmetry groups are not.** Executing the original split with NumPy gives 61,488 training, 2,022 validation, and 2,026 test keys. The two 256-key query subsets share no exact keys, but they share **63 swap/complement orbits**. This violates the group-separation rationale of the outer partition. It does not itself prove a particular amount of accuracy inflation. The revision samples meta-training queries from the original validation partition and evaluation queries from the test partition. Those partitions are already orbit-disjoint. Group-aware separation is consistent with [scikit-learn's guidance for dependent samples](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data).

5. **The router does not face held-out task definitions.** All eight tasks appear in meta-training and evaluation, with the same `5000 + task_idx` seeds. `unseen_roll2` and `unseen_majority` may be unseen during primitive backbone training, but they are seen by the meta-router through training labels. The evaluation measures transfer to new backbone/support seeds on a fixed task suite. A claim about new tasks requires a separate task-level holdout, preferably deduplicating equivalent generated Boolean functions.

6. **Ties are labeled as head victories.** This is a permissible deterministic training convention, but it makes reported routing accuracy penalize a teacher selection that has exactly the same utility. The revision retains the training tie convention, reports the number of ties, excludes ties from routing-accuracy denominators, and includes every episode in routed accuracy and regret.

7. **The head's `seed` argument is unused.** The original loop is still reproducible because earlier code seeds PyTorch; however, calling this function with a seed does not independently control its initialization. The revision isolates head initialization with `torch.random.fork_rng` and derives a seed from the backbone seed and task index. This deliberate correction means revised numerical results need not reproduce the original values.

8. **"Scratch" means a new linear head over a pretrained, frozen backbone.** It is not a neural network trained from scratch. Its support accuracy is measured on its own fitting data; the teacher's maximum support score is also selected on that support. Both are legal support-only features, but neither is an unbiased estimate of query accuracy. Context 4 was not used during primitive training. The revision keeps this original context choice and labels the baseline `Always-Head`; a context ablation or cross-fitted support features would be a new experiment.

## What the revised analysis tests

The primary utility is exact eight-bit vector accuracy on query examples. Regret is the mean difference between the better of the two measured query scores and the routed score. That oracle is an empirical upper bound over the two candidates on this query set, not knowledge of their population accuracies.

The designated paired comparison is RF minus the 5-percentage-point support rule. Each held-out seed contributes one difference after averaging its eight tasks. A one-sided exact sign test counts positive versus negative seed differences, dropping exact zero differences. This tests the chance that RF wins among nonzero seed differences, not whether its mean improvement is positive. The reported mean difference and the individual seed differences describe magnitude separately.

The sign test assumes independent seed runs conditional on the fixed query set, task suite, and fitted router. It does not account for resampling task definitions, query datasets, or meta-training cohorts. Five held-out seeds provide little inferential resolution: even five wins give a one-sided p-value of 1/32 = 0.03125, or 0.0625 for the corresponding two-sided sign test. A small p-value would still not constitute proof. Other baselines are reported descriptively; this revision does not run multiple unadjusted significance comparisons.

For broader evidence, specify the task-generalization target and primary comparison before examining results, use more independent evaluation seeds, and repeat complete meta-training/evaluation splits if uncertainty over the fitted router matters. Changes made after reviewing previous results should be followed by a fresh confirmatory evaluation.

## Files and execution

- `experiment_011.py`: revised neural experiment; default 15 training seeds and 5 evaluation seeds.
- `experiment_011_protocol.py`: orbit partitioning, aligned reporting, tie handling, and paired seed comparison.
- `test_experiment_011.py`: seven regression tests using NumPy and the Python standard library.

Place the three Python files together. Install NumPy, PyTorch, and scikit-learn in your chosen environment. SciPy is no longer imported directly; the small exact seed sign-test calculation uses Python's standard library.

```bash
python -m unittest discover -s /workspace/outputs -v
python /workspace/outputs/experiment_011.py --smoke
python /workspace/outputs/experiment_011.py
```

The default run saves detailed evaluation rows, route choices, split keys, feature importances, and summaries to `/workspace/outputs/experiment_011_results.json`. The smoke run deliberately uses minimal training and writes only `/tmp/experiment_011_smoke.json`; its scores must not be interpreted as experiment results.

## Verification status

- Seven protocol regression tests passed.
- All three Python files passed syntax parsing.
- Verified the original query overlap: 63 shared orbits and zero exact-key overlap.
- Reproduced the original indexing error with synthetic rows: a seed with 16/16 correct predictions was incorrectly reported as 0/16; the aligned calculation recovered 16/16.
- A reduced neural smoke run passed in an isolated CPU environment: two meta-training seeds, one evaluation seed, eight tasks, five backbone steps per task, and five head-training steps. Grammar generation, head fitting, random-forest fitting, reporting, and JSON export completed successfully. Export row counts and route assignments were also checked.
- The smoke scores are not scientific performance results. A subsequent full run completed all 20 seeds (15 meta-training and 5 evaluation), with 250 backbone steps per primitive task and 250 head-training steps per task. The full results are summarized in `experiment_011_results.md` and exported in `experiment_011_results.json`.
- The full-run export was independently checked: 120 meta-training episodes, 40 evaluation episodes, three query-performance ties, zero cross-partition orbit overlap, and agreement of all aggregate scores, per-seed RF scores, and paired differences with the detailed rows. RF achieved 37/37 correct non-tied routes and 57.17% mean exact-vector accuracy, versus 41.44% for the support rule. The designated one-sided seed sign-test p-value is 0.03125; the interpretation remains conditional on the fixed task suite, queries, and fitted router.
- `experiment_011_validation.json` records the checks and exact installed library versions. `requirements-experiment_011.txt` lists direct dependencies; the validation record supplies tested versions without requiring a particular CPU/GPU wheel source.
