# E16: distributional dynamics validation

Date: 2026-08-13 (Asia/Shanghai)  
Split: MISATO-100 validation, 10 complexes  
Status: project Dyn diagnostics; not an official Dyn score

## Question

Does unanchored seed-42 epoch 5 improve dynamics distributions relative to the published NeuralMD checkpoint, rather than only reducing mean trajectory errors? This experiment was run after the Cartesian anchor failed E15. The Static trajectory and ground truth are mandatory controls.

## Metrics

The evaluator uses only saved trajectories and reports unweighted means over complexes:

- RMSF-profile MAE, Pearson correlation and Spearman correlation;
- 1-D Wasserstein distance for radius of gyration distributions;
- mean per-atom-pair Wasserstein distance for intraligand distances;
- mean per-atom Wasserstein distance for one-frame displacement;
- mean step-amplitude ratio relative to truth, whose ideal is one;
- velocity-autocorrelation MAE for lags 1--10.

These metrics are project diagnostics. They do not reconstruct the organizer's normalization and are not combined into an invented competition score.

## Epoch-5 versus published NeuralMD

| Scenario | RMSF Pearson (epoch5 / published) | RMSF Spearman | Rg W1 change | pair-distance W1 change | step amplitude (epoch5 / published / truth) |
|---|---:|---:|---:|---:|---:|
| T1 | 0.259 / 0.238 | 0.159 / 0.137 | +0.16% | +0.22% | 0.00901 / 0.00908 / 1.0 |
| T2 | 0.618 / 0.574 | 0.529 / 0.468 | +0.93% | +0.37% | 0.00960 / 0.00969 / 1.0 |
| T3 | 0.271 / 0.259 | 0.263 / 0.246 | **-8.48%** | **-11.93%** | 0.01317 / 0.01400 / 1.0 |

Lower W1 is better. Epoch 5 improves RMSF profile correlation in all three scenarios and materially improves the T3 Rg and intraligand-distance distributions. It is slightly worse on most T1/T2 distribution distances, RMSF MAE, step displacement and velocity autocorrelation. Therefore it does **not** satisfy the preregistered E16 rule requiring an overall distributional improvement in at least two scenarios.

## Direct bond-aware comparison

The same validation trajectories were evaluated on the 9/10 topology-covered subset.

| Scenario | bond-length MAE epoch5 | published | relative change | extreme events epoch5 / published |
|---|---:|---:|---:|---:|
| T1 | 0.040269 A | 0.040960 A | -1.69% | 0 / 0% |
| T2 | 0.042463 A | 0.043427 A | -2.22% | 0 / 0% |
| T3 | 0.044818 A | 0.100947 A | -55.60% | 0.2666 / 0.1082% |

No graph-distance->2 nonbonded collision was observed in either model. The T3 mean and >20% violation rates are substantially better at epoch 5, but its rare extreme-event rate is higher. This supports keeping epoch 5 as the safer working baseline while withholding a comprehensive Phys-improvement claim.

## Scientific finding and decision

Both NeuralMD checkpoints reproduce only about 0.9--1.4% of the truth's mean one-frame displacement amplitude. Static has zero amplitude, so the models are not exactly static, but they remain strongly under-dynamic at this temporal resolution. This gives a sharper mechanism than generic "long-rollout drift": the official short-window position objective permits an overly smooth local vector field, while closed-loop rollout still accumulates structural error.

Decision:

1. E16 distribution gate is **no-go** for claiming epoch 5 as an overall Dyn improvement.
2. Epoch 5 remains the active safe baseline because its T3 structural distributions and most bond diagnostics are better, with explicit rare extreme-event caveats.
3. E17 must target local displacement distribution/velocity dynamics during training, with bond-aware validity as a safety gate. Further Cartesian anchoring or validation tuning is prohibited.

## Reproducibility

- Implementation: `protein_quanta/metrics.py`
- Evaluator: `scripts/evaluate_dynamics_distributions.py`
- Raw Dyn JSON: `reports/reproduction/dynamics_epoch005_vs_published_val.json`
- Raw Phys JSONs: `reports/reproduction/bond_aware_epoch005_val.json` and `reports/reproduction/bond_aware_published_val.json`
- Published rollout trajectories and large arrays remain outside Git under `/mnt/localDisk3/weizian/runs/protein-quanta/`.
- The published checkpoint trajectory regeneration used validation only; no new internal-test distribution diagnostics were run.
