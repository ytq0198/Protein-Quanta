# RNN/LSTM/GRU/Transformer temporal architecture screen

Date: 2026-08-13 (Asia/Shanghai)  
Decision split: MISATO-100 validation, 10 complexes  
Training split: MISATO-100 train, 80 complexes  
Status: invariant architecture evidence; not competition Geo/Phys/Dyn/Stab scores

## Motivation and design

Direction two is not a single fixed-history forecasting task. T1 observes 2 frames and predicts 18; T2 observes 80 and predicts 20; T3 observes 20 and predicts 80. To test long-memory architectures without confounding the result with rotation, translation, ligand size, or a different spatial backbone, the first-stage screen predicts a 12-dimensional rigid-motion-invariant frame state: Rg, pair-distance statistics and one-frame displacement-magnitude statistics.

The design was committed before training (`7c1ad03`): same train/validation split, seed 42, Adam 1e-3, 200 epochs, one-step teacher forcing, validation every 10 epochs, no internal-test access, and fewer than 10k trainable parameters per model. The comparison includes last-observation Static and an order-1 MLP control.

## Results

| Architecture | Parameters | selected epoch | T1 RMSE | T2 RMSE | T3 RMSE | weighted RMSE | Static scenarios beaten | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| MLP | 5,772 | 110 | **0.3514** | 0.3351 | 0.5392 | **0.38409** | 2 | control |
| RNN | 5,916 | 150 | 0.4483 | 0.3798 | 0.7057 | 0.47923 | 0 | no-go |
| LSTM | 5,412 | 180 | 0.5145 | 0.3694 | 0.5187 | 0.47182 | 1 | no-go |
| GRU | 5,584 | 200 | 0.3894 | 0.3224 | 0.7014 | 0.43170 | 0 | no-go |
| Transformer | 4,860 | 190 | 0.3916 | **0.3065** | **0.4838** | 0.38453 | 2 | no-go |
| Static | -- | -- | 0.3648 | 0.3088 | 0.5643 | -- | -- | control |

The Transformer is only 0.11% worse than MLP on the competition-weighted proxy while using fewer parameters. Relative to MLP, it improves T2 by 8.5% and T3 by 10.3%, but worsens T1 by 11.4%. Since T1 contributes 50%, it fails the pre-registered rule requiring weighted improvement over MLP. LSTM shows a small T3 signal but is much worse on T1/T2; RNN and GRU provide no robust overall benefit.

## Decision

No architecture passes the complete Level-A promotion gate, so none is promoted as a full 3D replacement and no internal-test architecture comparison is allowed.

There is nevertheless a useful new hypothesis: short-history T1 favors a local/no-memory core while T2/T3 favor causal attention. A scenario-adaptive MLP/Transformer combination cannot be reported from this validation result because it was proposed after seeing the split. It requires a new grouped cross-validation or held-out validation design before any 3D experiment.

## Scientific interpretation

1. "Long sequence" is scenario-specific: attention helps when history or horizon is long, not when only two frames are observed.
2. More expressive recurrence is not automatically better on high-dimensional molecular dynamics summaries; state representation and temporal resolution remain central.
3. The next relevant architecture is not an absolute-coordinate Transformer, but a shared E(3)-equivariant spatial encoder with a short-memory local path and a causal temporal-attention path.
4. Because the screen uses invariant summaries, it cannot establish atomic Geo, bond-aware Phys or official Dyn improvement.

## Evidence

- Preregistration: `configs/temporal_architecture_screen.json`
- Implementation: `protein_quanta/temporal_features.py`, `protein_quanta/temporal_models.py`
- Training/evaluation: `scripts/train_temporal_architecture_screen.py`
- Raw report: `reports/reproduction/temporal_architecture_screen_val.json`
- Figure: `reports/figures/temporal_architecture_screen.png`
- Research plan and literature boundary: `docs/temporal-architecture-research-plan.md`

No internal-test sample or official competition evaluator was used.
