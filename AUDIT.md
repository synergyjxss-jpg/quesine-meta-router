# Experiment 011: full run results

The revised experiment completed with 15 meta-training seeds and 5 held-out evaluation seeds. It used 120 meta-training episodes, 40 evaluation episodes, and 256 query examples per episode. The replay procedure was UniformReplay.

| Router | Routing accuracy (non-ties) | Mean exact-vector accuracy | Regret (percentage points) |
|---|---:|---:|---:|
| Learned RF | 37/37 (100.00%) | 57.17% | 0.00 |
| Hardcoded 5pp (Support) | 23/37 (62.16%) | 41.44% | 15.73 |
| Always-Teacher | 15/37 (40.54%) | 40.61% | 16.56 |
| Always-Head | 22/37 (59.46%) | 39.23% | 17.94 |

There were three ties in measured candidate query performance. They are excluded from routing-accuracy denominators and included in vector accuracy and regret. Perfect routing here means selecting the better of the two candidates on the measured query set; task accuracy is 57.17%, not 100%.

The RF improved mean exact-vector accuracy by 15.73 percentage points over the 5pp support rule. It improved in all five held-out seeds. The designated one-sided seed sign test gives p = 0.03125. The corresponding two-sided value is 0.0625.

| Held-out seed | RF exact-vector accuracy | RF minus support rule (percentage points) |
|---|---:|---:|
| 15 | 61.52% | +19.58 |
| 16 | 58.50% | +21.68 |
| 17 | 52.05% | +17.63 |
| 18 | 53.42% | +10.79 |
| 19 | 60.35% | +8.98 |

The evidence is conditional on this fixed eight-task suite, query set, and fitted router. The task definitions also appeared in meta-training. These results support improved routing across these held-out seeds; they do not establish generalization to new task families or statistical proof.

All saved aggregate scores, per-seed RF scores, paired differences, row counts, and symmetry-group separation were independently checked against the exported rows. The neural training procedure was not changed after viewing these results.

Artifacts: `experiment_011_results.json` contains the detailed rows and metrics; `experiment_011_run.log` contains captured console output; `experiment_011_validation.json` records environment versions, validation status, and source/result hashes.
