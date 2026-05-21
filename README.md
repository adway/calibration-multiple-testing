# calibration-multiple-testing

Replication code for "Calibration without labels in multiple testing."

The repository contains:

- simulation code for generating two-group mixture p-values,
- calibrators based on raw p-values, q-values, and local-fdr estimators,
- evaluation code for calibration error under a known data-generating model,
- notebooks to replicate figures and tables in paper.

## Simulation studies

Simulations use a two-group mixture model:

- null p-values are sampled from `Uniform(0, 1)`,
- non-null p-values are sampled from `Beta(alpha, beta)`,
- the null proportion is `pi0`.

Given a p-value `p`, the target calibration function is the conditional null probability
`P(H = 0 | p)`, computed in the code via `conditional_mean(...)`.

## Implemented calibrators

The main estimators live in `src/calibration_helpers/calibrators.py`:

- `p-value`: identity map,
- `q-value`: Storey-style q-values,
- `g-lfdr`: local fdr from a Grenander density estimate,
- `l-lfdr`: local fdr from Lindsey's method with spline-Poisson density estimation,
- `s-mle-lfdr`: spline-based pseudo-MLE local fdr estimator.

Calibration regret is computed in `src/calibration_helpers/metrics.py` as mean squared error between the target conditional null probability and the calibrator output. For smooth estimators, the simulation script uses Monte Carlo evaluation.

## Repository layout

- `src/calibration_sims.py`: main simulation launcher; builds tasks and submits chunked jobs with `submitit`.
- `src/combine_calibration_results.py`: combines per-job CSVs into a summary table.
- `src/calibration_helpers/`: mixture model, calibrators, and evaluation helpers.
- `npeb/`: local package dependency providing Grenander and related empirical Bayes utilities.
- `slurm/`: cluster submission scripts.
- `notebooks/`: exploratory and figure-generation notebooks.
- `data/`: cached summaries and external data used in analysis.
- `figures/`: exported plots.
