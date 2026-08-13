# Dense equivariant 64/16 effect gate — live audit

> Status: training in progress. This file records process and artifact checks only; it contains no holdout target result and is not an official competition score.

## Sealed execution

- Actual training/evaluation source commit at pipeline launch: `d078920`.
- Pipeline root: server-only `dense-effect-gate/sealed-v1`; checkpoints and trajectories are not tracked in Git.
- Seeds run sequentially as `0, 42, 123`; each seed trains local-ODE control and multiscale candidate for 50 final-only epochs.
- Evaluation has a program barrier: all three `training_complete_holdout_unread` manifests and all six checkpoint SHA-256 values must verify before holdout IDs, topology report, or MISATO target trajectories are read.
- Run identity was sealed at `2026-08-13T18:07:49Z`, before the first seed finished and before any holdout target evaluation. It records hashes for both scripts, config, split files, topology report, and MISATO-100 HDF5.

## Seed 0 training audit

Seed 0 finished at `2026-08-13T18:08:09Z`.

| Item | Control | Multiscale candidate |
|---|---:|---:|
| Epochs | 50 | 50 |
| Non-finite updates | 0 | 0 |
| Clipped updates | 0 | 0 |
| Parameter L2 from common initialization | 1.23924494 | 1.26480377 |
| Checkpoint SHA-256 | `29aeadb0…36f0` | `463977e7…c92a` |

Candidate-to-control parameter L2 is `0.04874384`, which proves the extra objective had a non-negligible optimization effect. This does **not** establish predictive improvement; that decision remains sealed until all seeds finish and the single holdout evaluation runs.

## Seed 42 training audit

Seed 42 finished at `2026-08-13T18:19:35Z` and the pipeline immediately advanced to seed 123.

| Item | Control | Multiscale candidate |
|---|---:|---:|
| Epochs | 50 | 50 |
| Non-finite updates | 0 | 0 |
| Clipped updates | 0 | 0 |
| Parameter L2 from common initialization | 1.16674054 | 1.18863535 |
| Checkpoint SHA-256 | `6c3e74f4…3b22` | `a2c73ecc…3760` |

Candidate-to-control parameter L2 is `0.06489188`. Together with seed 0, this rules out the preflight failure mode in which the extra loss was numerically inert, while still leaving predictive effect sealed.

## Restart reproducibility observation

An obsolete combined train/evaluate script had previously been terminated after six control epochs because its holdout-access order violated the pre-registered policy. No checkpoint or evaluation output was produced. After splitting training from evaluation and restarting from scratch, seed 42 control epochs 1 and 2 reproduced the earlier logged values (`8.52745507`, `17.85635709`). This is a narrow same-server deterministic-restart check, not a claim of general cross-platform determinism.

## Next audit points

1. Verify seed 42 and seed 123 manifests, epoch counts, non-finite/clipping counts, parameter displacement, and checkpoint hashes.
2. Confirm evaluation begins only after all six hashes pass.
3. Independently recompute the gate with the pure aggregation module and require exact agreement with the frozen server evaluator.
4. Archive the raw report, decision table, paired-seed/scenario figure, and a Chinese technical report with limitations adjacent to every claim.
