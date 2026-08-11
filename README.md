# Protein-Quanta

Reproducible baselines and physics-aware trajectory learning for the GOAI
protein–ligand binding trajectory prediction task.

## Current scope

This bootstrap branch contains:

- a validated trajectory record;
- static and constant-velocity linear rollout baselines;
- coordinate MAE and RMSE with atom masking;
- NeuralMD-compatible Matching and Stability metrics;
- MISATO HDF5 schema auditing and ligand preprocessing;
- server-side Static and Linear baseline evaluation;
- standard-library unit tests (plus `h5py` for MISATO tests).

The official MISATO-100 file has passed schema and finite-coordinate auditing
for all 100 complexes. Static, Linear, and the official NeuralMD checkpoint
have been evaluated on the official 10-complex test split under one protocol.
See
[`reports/reproduction/2026-08-11-neuralmd-misato100-baseline.md`](reports/reproduction/2026-08-11-neuralmd-misato100-baseline.md).

The project also reports Kabsch-aligned RMSD, radius-of-gyration error, RMSF
error, and intramolecular contact-map agreement. These are explicitly labeled
as project-defined diagnostic proxies, not official competition scores.

## Run the tests

```bash
python -m unittest discover -s tests -v
```

Evaluate the reproducible Static and Linear baselines on the official tiny
MISATO file:

```bash
python -m scripts.evaluate_naive_baselines \
  /path/to/tiny_md.hdf5 \
  reports/reproduction/misato_tiny_proxy_diagnostics.json \
  --observed-frames 2 \
  --contact-cutoff 4.5
```

Run the NeuralMD loader/model smoke test with the official checkpoint after
installing the isolated NeuralMD environment:

```bash
python -m scripts.smoke_neuralmd \
  --upstream /path/to/NeuralMD \
  --h5 /path/to/tiny_md.hdf5 \
  --checkpoint /path/to/NeuralMD_ODE/MISATO_100_seed_42/model.pth \
  --sample-id 10GS \
  --device cuda:0 \
  --output reports/reproduction/neuralmd_10GS_checkpoint_smoke.json
```

Evaluate a NeuralMD checkpoint on a named MISATO split:

```bash
python -m scripts.evaluate_neuralmd_checkpoint \
  --upstream /path/to/NeuralMD \
  --h5 /path/to/MISATO_100/raw/MD.hdf5 \
  --split /path/to/MISATO_100/raw/test_MD.txt \
  --checkpoint /path/to/model.pth \
  --rollout-frames 100 \
  --ode-step-size 5 \
  --output reports/reproduction/neuralmd_misato100_test_rollout100.json
```

The evaluator strictly loads the checkpoint, records its SHA256, checks
preprocessing agreement per complex, and computes unweighted per-complex
means. Its diagnostics are explicitly not labeled as official scores.

## Data policy

Do not commit MISATO data, competition test trajectories, checkpoints, secrets,
or generated runs. On the school server the intended locations are:

```text
/mnt/localDisk3/weizian/datasets/misato
/mnt/localDisk3/weizian/checkpoints/protein-quanta
/mnt/localDisk3/weizian/runs/protein-quanta
```

The local `Data set.zip` supplied for the virtual-cell direction is not an
input to this project.

## Reproduction policy

Every reported experiment must record its upstream commit, data version and
split, project commit, environment, GPU, configuration, seed, command, runtime,
metrics, deviations from the paper, and failures. See
[`reports/reproduction/TEMPLATE.md`](reports/reproduction/TEMPLATE.md).

## Research roadmap and living report

- Research design: [`docs/experiment-design-and-research-roadmap.md`](docs/experiment-design-and-research-roadmap.md)
- Living experiment log: [`reports/experiment-progress-report.md`](reports/experiment-progress-report.md)
- Anchor-residual plan: [`docs/superpowers/plans/2026-08-11-anchor-residual-feasibility.md`](docs/superpowers/plans/2026-08-11-anchor-residual-feasibility.md)

The first innovation feasibility test applies a validation-selected,
time-decayed Static anchor to frozen NeuralMD trajectories. Beta 4 improved
held-out test Matching by about 7.0%, Stability by 3.78 percentage points,
radius-of-gyration error by about 28.6%, and contact agreement by about 0.9%.
Coordinate RMSE and RMSF error traded off by about 1.2% and 2.8%. The test
split was evaluated once with the validation-frozen beta and was not rescanned.
See
[`reports/figures/anchor_residual_tradeoff.png`](reports/figures/anchor_residual_tradeoff.png).
