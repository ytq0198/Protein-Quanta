# Low-Capacity Uncertainty Gate LOOCV Design

## Purpose

Test whether inference-time, SE(3)-invariant state features can identify when a stronger Static-residual anchor is safe and useful. This validation-only experiment follows the no-go of the scenario-only `T1/T2/T3=8/8/1` policy. It does not alter NeuralMD weights and does not access the internal test split.

## Data boundary

- Source: the 30 archived validation records formed by 10 MISATO-100 complexes and T1/T2/T3.
- Group: complex ID. Every held-out fold contains all three scenarios of one complex.
- Inputs may use only information available at inference: scenario name, ligand atom count, the final two observed frames, and the unanchored NeuralMD prediction.
- Future ground truth is used only to construct validation labels and to evaluate out-of-fold predictions. It is never an input feature.
- No internal-test trajectory, metric, label, or result is read during model fitting, feature selection, threshold selection, or OOF evaluation.

## Fixed candidate actions and labels

The gate chooses between the current frozen `beta=1` action and one stronger action `beta=8`, both with `decay_scale_frames=98`.

For each validation record, `beta=8` receives a positive label only if, relative to `beta=1`:

1. coordinate RMSE increases by at most 2%;
2. RMSF MAE increases by at most 5%;
3. Matching decreases strictly; and
4. Stability increases strictly.

Otherwise the label is negative and the safe fallback is `beta=1`. The observed label distribution before fitting is T1 `5/10`, T2 `3/10`, T3 `2/10`, for `10/30` positives overall.

## Fixed features

Continuous features are transformed with `log1p(max(value, 0))` where appropriate and standardized using the training groups in each fold only:

1. `log_atom_count`: ligand heavy-atom count;
2. `log_observed_speed`: RMS displacement between the last two observed frames;
3. `log_prediction_pair_drift`: RMS change of intramolecular pair distances across the unanchored predicted trajectory;
4. `log_prediction_rg_shift`: absolute radius-of-gyration change from the last observed frame to the final predicted frame.

Two fixed scenario indicators, `is_T2` and `is_T3`, are appended; T1 is the reference. Coordinates, orientations, translations, sample IDs, future errors, future metrics, and truth-derived statistics are excluded.

## Model and training

- Model: binary logistic regression implemented in NumPy.
- Capacity: one intercept and six feature coefficients.
- Regularization: fixed L2 coefficient `10.0` on coefficients, not the intercept.
- Optimizer: deterministic Newton/IRLS updates, at most 100 iterations, probability clipping `1e-8`, convergence tolerance `1e-10`, and a small `1e-8` diagonal numerical stabilizer.
- Class weighting: balanced inverse-frequency weights computed from the training fold only.
- Decision threshold: fixed probability `>=0.5` selects `beta=8`; otherwise select `beta=1`.
- Randomness: none.

No hyperparameter, feature subset, action beta, or probability threshold is tuned from OOF results. If the fixed model fails, the experiment is recorded as no-go rather than modified in place.

## Grouped leave-one-complex-out protocol

For each of the 10 folds:

1. hold out all T1/T2/T3 records of one complex;
2. fit feature normalization and logistic coefficients on the remaining 27 records;
3. predict the three held-out probabilities;
4. choose `beta=1` or `beta=8` independently for each held-out scenario; and
5. aggregate all 30 predictions only after every record has been held out exactly once.

The report stores fold membership, training-label counts, normalization statistics, coefficients, held-out probabilities, chosen betas, true labels, and per-record metrics.

## OOF success and stop rules

Compare the OOF gated trajectories with global `beta=1`, using unweighted means over the same 10 complexes per scenario.

The gate is considered validation-feasible only if all conditions pass:

1. T1 and T2 each improve aggregate Matching and Stability;
2. no scenario aggregate coordinate RMSE worsens by more than 2%;
3. no scenario aggregate RMSF MAE worsens by more than 5%;
4. the gate selects `beta=8` for at least 20% and at most 80% of the 30 OOF records;
5. OOF balanced accuracy for the safe/useful label is at least 0.60; and
6. every record is predicted exactly once by a model that did not train on that complex.

Failure of any condition is a no-go. Passing all conditions supports further validation research but does not promote the initial candidate and does not authorize reading the internal test split in this phase.

## Artifacts and claims

- Code: reusable feature extraction, label construction, deterministic logistic fitting, grouped OOF evaluation, and a CLI for archived trajectories.
- Evidence: one machine-readable OOF report, one compact comparison figure, and updates to the experiment-progress and research-design documents.
- Claims are limited to internal validation feasibility. No official score, general protein-ligand validity, Phys improvement, or hidden-test performance is claimed.

