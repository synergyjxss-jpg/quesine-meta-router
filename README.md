# Quesine: Support-Only Meta-Routing for Continual Learning

**Status:** Exploratory reference implementation. PyTorch, NumPy, Scikit-Learn.

## Overview
This repository contains the code for **Quesine**, an experimental architecture that explores how a frozen continual learner can route between reusing a compositional DSL skill and training a scratch adapter. 

The system uses a learned meta-router (Random Forest) that evaluates support-set features to predict which candidate model will generalize better to unseen queries.

## Key Results
*   **Learned Routing:** The Random Forest meta-router achieved **100% routing accuracy** (37/37 non-tied episodes) on 5 held-out seeds, with **zero regret** vs the test oracle. It improved mean exact-vector accuracy by 15.73 pp over a fixed 5pp heuristic.
*   **Statistical Significance:** A paired seed sign test yields \(p = 0.03125\).
*   **Critical Boundary (LOFO):** A Leave-One-Family-Out diagnostic revealed that the router fails when asked to route a structurally novel task family (e.g., three-input majority gates). It learns statistical associations, not semantic task geometry. 

## Repository Structure
*   `experiment_011.py`: The corrected, rigorous experiment execution.
*   `experiment_011_protocol.py`: Data splitting, orbit isolation, and paired seed sign tests.
*   `test_experiment_011.py`: Regression tests for the protocol and metrics.
*   `REPORT.md`: Full results, detailed metrics, and the LOFO diagnostic.
*   `AUDIT.md`: Independent code and statistical review of the protocol.
*   `experiment_011_results.md`: Final run results table.
