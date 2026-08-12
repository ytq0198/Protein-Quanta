# Protein-Quanta

Reproducible baselines and physics-aware trajectory learning for the GOAI
protein–ligand binding trajectory prediction task.

## Current scope

This research branch contains:

- a validated trajectory record;
- static and constant-velocity linear rollout baselines;
- coordinate MAE and RMSE with atom masking;
- NeuralMD-compatible Matching and Stability metrics;
- MISATO HDF5 schema auditing and ligand preprocessing;
- server-side Static and Linear baseline evaluation;
- T1/T2/T3 competition-aligned NeuralMD evaluation and error-growth diagnostics;
- invariant per-complex pair-distance loss and gradient calibration;
- scenario-aware checkpoint selection and time-decayed Static anchoring;
- project-defined intraligand and ligand-protein collision diagnostics;
- privacy-minimized server environment capture for reproducibility;
- standard-library unit tests (plus `h5py` for MISATO tests).

The official MISATO-100 file has passed schema and finite-coordinate auditing
for all 100 complexes. Static, Linear, and the official NeuralMD checkpoint
have been evaluated on the official 10-complex test split under one protocol.
See
[`reports/reproduction/2026-08-11-neuralmd-misato100-baseline.md`](reports/reproduction/2026-08-11-neuralmd-misato100-baseline.md).

The project also reports Kabsch-aligned RMSD, radius-of-gyration error, RMSF
error, and intramolecular contact-map agreement. These are explicitly labeled
as project-defined diagnostic proxies, not official competition scores.

## Historical frozen candidate and active status

The historical candidate was NeuralMD seed 42 at epoch 5 followed by a frozen
time-decayed Static residual anchor (`beta=1`, decay scale 98 frames). The
checkpoint and anchor were selected on the 10-complex validation split with
explicit T1/T2/T3 guards. Relative to the published checkpoint, the frozen
internal test improves Matching by 2.48%/2.27%/4.99% and Stability by
0.51/0.42/1.61 percentage points on T1/T2/T3. Coordinate RMSE improves on
T1/T2 and trades off by 0.36% on T3; RMSF improves in all three scenarios.

These values remain local proxy diagnostics because organizer normalization
code is unavailable. The machine-readable source of truth is
[`configs/frozen_candidate.json`](configs/frozen_candidate.json); the full
causal ablation, including the no-go pair-loss experiments, is in
[`reports/reproduction/2026-08-12-neuralmd-pair-loss-and-earlystop.md`](reports/reproduction/2026-08-12-neuralmd-pair-loss-and-earlystop.md).

The candidate has also passed a preliminary overlap audit. Binding overlap is
very small and does not increase consistently, but the intraligand proxy rises
slightly in five of six validation/test scenario comparisons. Because the
available preprocessing lacks a bond graph, bonded neighbours cannot be
excluded and this is not a chemically valid clash rate. The result is recorded
as a Phys gap, not as evidence of an official-score improvement; see
[`reports/reproduction/2026-08-12-collision-proxy-audit.md`](reports/reproduction/2026-08-12-collision-proxy-audit.md).

This candidate was **demoted on 2026-08-13** after a stricter bond-aware
validation. Relative to unanchored epoch-5 NeuralMD, the anchor worsened
bond-length MAE by 8.99% on T1 and 13.69% on T2, exceeding the pre-registered
5% guard and adding small extreme-bond event rates. No new bond-aware test
evaluation was run and beta was not retuned. The active safe baseline is now
unanchored seed-42 epoch 5 pending a direct score-facing Phys/Dyn comparison
with published NeuralMD. See
[`reports/reproduction/2026-08-13-bond-aware-phys-validation.md`](reports/reproduction/2026-08-13-bond-aware-phys-validation.md).
The machine-readable current state is
[`configs/active_candidate.json`](configs/active_candidate.json); the old
anchor remains in [`configs/frozen_candidate.json`](configs/frozen_candidate.json)
only as a traceable historical no-go.

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

Evaluate all three competition-aligned scenarios and optionally retain local
trajectories outside Git:

```bash
python -m scripts.evaluate_neuralmd_scenarios \
  --upstream /path/to/NeuralMD \
  --h5 /path/to/MISATO_100/raw/MD.hdf5 \
  --split /path/to/MISATO_100/raw/val_MD.txt \
  --checkpoint /path/to/model_epoch_005.pth \
  --trajectory-dir /path/outside/git/trajectories \
  --device cuda:0 \
  --output /path/outside/git/epoch005_val.json
```

Apply the frozen anchor to those saved trajectories:

```bash
python -m scripts.evaluate_anchor_scenarios \
  --trajectory-dir /path/outside/git/trajectories \
  --reference-report /path/outside/git/epoch005_val.json \
  --beta 1 \
  --decay-scale-frames 98 \
  --output /path/outside/git/epoch005_anchor1_val.json
```

Audit project-defined collision proxies on the saved trajectories:

```bash
python -m scripts.evaluate_collision_scenarios \
  --upstream /path/to/NeuralMD \
  --h5 /path/to/MISATO_100/raw/MD.hdf5 \
  --trajectory-dir /path/outside/git/trajectories \
  --reference-report /path/outside/git/epoch005_val.json \
  --beta 1 \
  --decay-scale-frames 98 \
  --output /path/outside/git/epoch005_anchor1_collision_val.json
```

This diagnostic does not exclude bonded ligand pairs and is not the official
Phys metric.

The canonical corrected NeuralMD training command makes the 20-frame sampling
boolean explicit, disables test evaluation during development, and can retain
every fifth validation checkpoint:

```bash
python -m scripts.run_neuralmd_training /path/outside/git/run 20 0 \
  --seed 42 \
  --save-every-epoch 5
```

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

## Environment provenance

Capture the active runtime without writing usernames, hostnames, paths, or
environment variables:

```bash
python -m scripts.capture_environment \
  reports/reproduction/server-environment.json
```

The archived school-server snapshot records Python 3.10.20, PyTorch
2.6.0+cu124, CUDA 12.4, the whitelisted scientific package versions, and four
NVIDIA RTX A6000 devices. See
[`reports/reproduction/server-environment.json`](reports/reproduction/server-environment.json).

Run the read-only public-release audit before a milestone or submission:

```bash
python -m scripts.audit_release \
  --output reports/reproduction/release-audit.json
```

It rejects tracked checkpoints/trajectory data, files over 5 MiB, common token
prefixes, missing manifest evidence, and any drift toward claiming an official
competition score. A missing project-level license remains a warning because
license selection requires a team decision.

## Reproduction policy

Every reported experiment must record its upstream commit, data version and
split, project commit, environment, GPU, configuration, seed, command, runtime,
metrics, deviations from the paper, and failures. See
[`reports/reproduction/TEMPLATE.md`](reports/reproduction/TEMPLATE.md).

## Research roadmap and living report

- Research design: [`docs/experiment-design-and-research-roadmap.md`](docs/experiment-design-and-research-roadmap.md)
- Living experiment log: [`reports/experiment-progress-report.md`](reports/experiment-progress-report.md)
- Scenario/pair-loss plan: [`docs/superpowers/plans/2026-08-12-competition-scenarios-and-pair-loss.md`](docs/superpowers/plans/2026-08-12-competition-scenarios-and-pair-loss.md)
- Preliminary evidence index (not submission prose): [`docs/preliminary-evidence-index.md`](docs/preliminary-evidence-index.md)
- Preliminary submission compliance checklist (internal, not submission prose): [`docs/preliminary-submission-compliance-checklist.md`](docs/preliminary-submission-compliance-checklist.md)
- Official/public supplementary-material audit: [`reports/reproduction/2026-08-12-official-materials-audit.md`](reports/reproduction/2026-08-12-official-materials-audit.md)

The original full-rollout anchor feasibility experiment passed, while the
constant-weight pair-distance objective did not survive causal ablation. The
strongest current result combines scenario-aware epoch-5 selection with a
weaker beta-1 anchor. See
[`reports/figures/neuralmd_earlystop_anchor1_tradeoff.png`](reports/figures/neuralmd_earlystop_anchor1_tradeoff.png).
