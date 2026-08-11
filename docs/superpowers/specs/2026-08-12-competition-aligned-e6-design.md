# Competition-Aligned T1/T2/T3 Evaluation and E6 Design

## Status and decision authority

The team has given standing authorization to continue without interactive approval while they are away. This design records the decision that would otherwise be reviewed interactively. It does not author the competition submission: the supplied preliminary template explicitly says AI-generated answers are prohibited, so repository outputs are limited to reproducible code, experiment evidence, figures, and internal research notes that the team must independently interpret and write up.

## Problem statement

The competition guide weights technical performance by scenario and scientific property:

- `Final = 50% T1 + 30% T2 + 20% T3`;
- each scenario uses `40% Geo + 25% Phys + 25% Dyn + 10% Stab` after official baseline normalization;
- T1 is next-frame/short-range prediction, T2 observes the first 80% and predicts the last 20%, and T3 observes the first 20% and predicts the last 80%.

Our current first-two-frame 100-frame evaluator is a useful stress test, but it is not any official scenario. The corrected official NeuralMD training now reproduces the published checkpoint almost exactly, while its final weights have normal 20-frame training loss and catastrophic 100-frame rollout. The next experiment must therefore align validation with T1/T2/T3 before changing the loss.

## Approaches considered

### A. Add pairwise-distance loss immediately

This is the shortest path to another training run, but the existing evaluator cannot say whether a gain belongs to T1, T2, or T3. It risks optimizing an internal proxy while missing the 50/30/20 competition structure. Rejected as the immediate next step.

### B. Build scenario-aligned validation, then add one geometry loss

First construct a single evaluator that can initialize NeuralMD at an arbitrary observed frame and report all three scenarios. Then add only a ligand intramolecular pair-distance auxiliary loss, calibrate its gradient contribution, and compare against the corrected seed-42 baseline. This gives attribution, respects the competition's Geo/Phys/Dyn/Stab emphasis, and is the recommended approach.

### C. Reproduce the complete official score locally

This would include bond angles, stereochemistry, energy validity, hydrogen bonds, free-energy distributions, exact normalization, and hidden split logic. The guide does not publish the complete executable definitions or normalization constants, and the current MISATO preprocessing lacks a reliable bond table. Implementing a guessed score would create false confidence. Deferred until official evaluation code or complete field definitions are available.

## Scenario proxy protocol

All protocols use the official 100-frame MISATO trajectories and semi-flexible protein setting. The two most recent observed ligand frames initialize position and velocity. Metrics are computed only on unobserved target frames.

| Scenario | Observed frames | Initializer | Evaluated frames | Rationale |
|---|---:|---:|---:|---|
| T1 proxy | 0–1 | 0–1 | 2–19 | “next k” is unspecified; 18 target frames make a 20-frame local window consistent with official training |
| T2 proxy | 0–79 | 78–79 | 80–99 | exact first-80% / last-20% split |
| T3 proxy | 0–19 | 18–19 | 20–99 | exact first-20% / last-80% split |

The T1 proxy must always be labeled as a project proxy because the guide does not specify `k`. T2 and T3 match the stated fractions, but all locally reconstructed metrics remain proxy diagnostics until the organizer releases executable scoring.

## Metric mapping

The evaluator reports raw metrics and relative improvement against both the official NeuralMD checkpoint and Static. It must not fabricate a 0–100 composite score without official normalization.

| Competition module | Available proxy evidence | Explicit limitation |
|---|---|---|
| Geo | coordinate MAE/RMSE; aligned ligand RMSD | no pocket-coordinate prediction in semi-flexible setting |
| Phys | NeuralMD covalent-radius ligand collision and binding collision rates | no verified bond table, bond angles, stereochemistry, or energy |
| Dyn | pair-distance Matching; Rg error; RMSF error; intraligand contact agreement | no hydrogen-bond typing or free-energy surface in the first version |
| Stab | distance Stability; short/mid/long error-window summaries; rollout growth slope | locally chosen summaries are not official normalization |

Every report includes per-complex rows and unweighted means. Validation chooses methods and checkpoints; no new test evaluation is allowed until a method passes its preregistered validation gate.

## Components and data flow

1. `protein_quanta/scenarios.py` owns immutable scenario definitions and validates frame ranges.
2. `scripts/evaluate_neuralmd_scenarios.py` loads each complex once, initializes at the scenario-specific last two observed frames, runs NeuralMD only over the required interval, and evaluates NeuralMD/Static/Linear on the target frames.
3. `protein_quanta/metrics.py` gains windowed error-growth summaries. Collision proxies call the pinned NeuralMD reference implementation rather than duplicating covalent radii.
4. JSON reports store the scenario definitions, raw metrics, checkpoint hash, upstream commit, sample IDs, runtime, and the “not official score” status.
5. A compact figure compares T1/T2/T3 relative to Static and highlights anti-collapse metrics.

The published checkpoint is evaluated first. The corrected seed-42 best checkpoint must reproduce the same scenario metrics within numerical tolerance; otherwise E6 training does not begin.

## E6 loss design

The first E6 model changes one learning signal only:

\[
L = L_{pos} + \lambda_d L_{pair}
\]

For each complex, frame, and unordered ligand heavy-atom pair, `L_pair` applies Smooth-L1 with `beta=0.5 Å` to the predicted-minus-true distance error. Pair losses are averaged within each complex and then across the batch so large ligands do not dominate. Diagonal and duplicate pairs are excluded. The distance construction is translation/rotation invariant and does not alter NeuralMD's SE(3)-equivariant architecture.

`lambda_d` is not hand-picked from test performance. On the first ten deterministic training batches, compute separate gradient norms for `L_pos` and `L_pair`; set

\[
\lambda_d = 0.1\,\mathrm{median}\left(\|\nabla L_{pos}\|_2 / \max(\|\nabla L_{pair}\|_2,10^{-12})\right)
\]

and freeze it. This targets an initial auxiliary-gradient contribution of roughly 10%. Record the ten ratios and selected value. The first experiment uses the corrected official 20-frame window, seed 42, no clipping, and otherwise identical hyperparameters. Gradient monitoring and non-finite protection remain instrumentation, not an experimental variable.

## Checkpoint and experiment policy

- Save a checkpoint every 5 epochs outside Git, plus best coordinate and best preregistered scenario checkpoint.
- A 20-epoch feasibility run is sufficient for the first decision because the reproduced official best occurs at epoch 15.
- Extend to 100 epochs only if the 20-epoch result passes the validation gate.
- Run seeds 0 and 123 only after the 100-epoch seed-42 result passes.
- Store code, hashes, summaries, figures, and small JSON reports in Git; never store trajectories, MISATO HDF5, or checkpoints.

## Validation gate

E6 advances from 20 to 100 epochs only if one saved checkpoint satisfies all conditions against the corrected seed-42 coordinate-best baseline:

1. T1 coordinate RMSE is no worse than 2%.
2. T2 Stability does not decrease by more than 1 percentage point and T2 Matching does not worsen by more than 2%.
3. T3 improves Stability by at least 2 percentage points or Matching by at least 5%.
4. T3 RMSF error is no more than 105% of baseline, preventing Static collapse.
5. No non-finite update occurs and ligand/binding collision proxies do not worsen materially.

The 100-epoch version is a candidate for multi-seed validation only if it preserves the same directional result. Exact official score claims are prohibited without organizer code.

## Error handling and auditability

- Reject a scenario if fewer than two observed frames or no target frames remain.
- Reject prediction/truth shape mismatches and non-finite coordinates before aggregation.
- Record failures per sample; do not silently drop a complex from the mean.
- Validate every checkpoint with strict state-dict loading and SHA256.
- Explicitly record the official hyperparameter false flags in logs and reports.
- Treat the 1-epoch upstream `IndexError` as a documented preflight boundary failure; use 5 epochs for checkpoint-saving preflight.

## Testing

- Unit tests for exact T1/T2/T3 frame indices and invalid boundaries.
- Unit tests showing arbitrary-frame initialization preserves the two observed frames and evaluates only targets.
- Metric tests with hand-derived windowed error growth.
- Pair-loss tests for rigid-motion invariance, zero loss on truth, per-complex balancing, exclusion of duplicate/diagonal pairs, and finite gradients.
- Integration preflight on one validation complex and one training epoch-equivalent batch before any multi-epoch run.

## Expected competition evidence

The initial submission evidence should emphasize the verified baseline reproduction, C1's frozen positive result, the corrected failure analysis, and an early E6 validation result only if it passes. The technical narrative maps each result to Geo/Phys/Dyn/Stab and T1/T2/T3, while scientific significance explains why preserving dynamic fluctuations and physical validity matters for binding dynamics. Dependencies, upstream provenance, data boundaries, and licenses remain explicit for the 5% open-source dimension.
