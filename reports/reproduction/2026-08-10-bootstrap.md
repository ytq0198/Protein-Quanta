# Bootstrap and upstream audit — 2026-08-10 to 2026-08-11

## Objective

Create the smallest reproducible baseline layer and pin the two mandatory
upstream projects before downloading large data or training models.

## Protein-Quanta environment

- Local development branch: `codex/reproduction-bootstrap` (unborn branch;
  no commit or push yet)
- Server project: `/mnt/localDisk3/weizian/Protein-Quanta`
- Server data: `/mnt/localDisk3/weizian/datasets/misato`
- Server checkpoints: `/mnt/localDisk3/weizian/checkpoints/protein-quanta`
- Server runs: `/mnt/localDisk3/weizian/runs/protein-quanta`
- Lightweight test interpreter:
  `/mnt/localDisk3/weizian/conda_envs/protein-quanta/bin/python`
- Lightweight environment: Python 3.9.25, NumPy 2.0.2, h5py 3.14.0,
  pandas 2.3.3
- NeuralMD environment:
  `/mnt/localDisk3/weizian/conda_envs/protein-quanta-neuralmd`
- NeuralMD runtime: Python 3.10, PyTorch 2.6.0+cu124, PyG 2.6.1,
  torch-scatter 2.1.2+pt26cu124
- GPU snapshot: 4 × NVIDIA RTX A6000, each 49,140 MiB; all idle at preflight
- Disk snapshot: 2.6 TiB available on `/mnt/localDisk3`

## Verified upstream sources

### MISATO

- Source: https://github.com/t7morgen/misato-dataset
- Commit: `7b06d532e2ed0719411fcc1b3ac39743db4ca10d`
- Branch: `master`
- License evidence: repository `LICENSE` file (LGPL-2.1)
- Useful included smoke-test assets:
  - `data/MD/h5_files/tiny_md.hdf5` (about 92 MB)
  - `data/MD/h5_files/tiny_md_out.hdf5` (about 47 MB)
  - tiny train/validation/test split files

The included tiny MD file should be audited before downloading the 725 MB
NeuralMD MISATO-100 subset. This reduces the cost of discovering unit, schema,
or atom-mapping errors.

### NeuralMD

- Source: https://github.com/chao1224/NeuralMD
- Commit: `a2ae030838c6ea0251eb6a29bfe99dc9d8ee1cfe`
- Branch: `main`
- License evidence: `setup.py` declares MIT; no standalone license file was
  found in this commit
- Official multi-trajectory entry point:
  `examples/main_MISATO_multi_traj_NeuralMD.py`
- Official semi-flexible loader:
  `NeuralMD/datasets/MISATO/dataset_MISATO_semi_flexible.py`

## Important implementation observations

1. The NeuralMD loader expects 100 frames and transposes coordinates to
   `(atoms, frames, 3)` internally.
2. It removes hydrogen atoms from the ligand.
3. It centers coordinates using the mean over every atom and every frame, then
   freezes protein coordinates at frame 0 in the semi-flexible setting.
4. Its Matching metric is per-frame RMSE between all ligand pairwise-distance
   matrices.
5. Its Stability metric is the percentage of pairwise-distance errors no
   larger than 0.5 Å; the implementation includes diagonal and both symmetric
   matrix entries.
6. The main example defaults to MISATO-100, batch size 32, 32 epochs, seed 42,
   ten predicted frames, Euler ODE, and NeuralMD_Binding01.
7. Paper metrics and competition Geo/Phys/Dyn/Stab are not identical; the
   project must retain both rather than relabel paper metrics as official
   competition scores.

## Baseline layer implemented

- Static rollout
- Constant-velocity linear rollout
- Coordinate MAE and RMSE with atom mask
- NeuralMD-compatible Matching and Stability metrics
- Validated trajectory record with sample ID, unit, finite-coordinate check,
  and atom mask
- Chunked MISATO HDF5 auditor
- NeuralMD-compatible ligand extraction: global centering and ligand-hydrogen
  removal

## MISATO tiny audit and naive baseline results

The official `tiny_md.hdf5` was evaluated on the server. The JSON artifacts
are `misato_tiny_schema.json` and `misato_tiny_static_linear.json` in this
report directory.

- 20/20 complexes passed required-field, shape, ligand-index, finite-value,
  and per-frame energy checks.
- Every complex contains 100 frames.
- Total atom counts range from 1,767 to 6,600 before ligand extraction.
- Evaluation protocol: observe frames 0 and 1, predict frames 2 through 99,
  retain ligand heavy atoms, and use the NeuralMD centering convention.
- This is a T1 proxy and must not be labeled as an official competition score.

| Baseline | Coordinate RMSE (Å) | Matching (Å) | Stability (%) |
|---|---:|---:|---:|
| Static | 3.1184 | 0.6087 | 76.7683 |
| Linear | 50.4211 | 78.3472 | 6.8904 |

The constant-velocity baseline diverges badly over a 98-frame rollout. This
supports prioritizing error-control and geometric stability in the learned
baseline rather than relying on unconstrained velocity extrapolation.

## NeuralMD loader and forward-pass smoke test

The pinned upstream loader and `NeuralMD_Binding01` model were run on `10GS`
using GPU 0. The artifact is `neuralmd_10GS_smoke.json`.

- Ligand heavy atoms: 33
- Trajectory frames: 100
- Protein backbone atoms: 1,245
- Maximum absolute difference between our preprocessing and the official
  loader: `4.31e-7 Å`
- Model parameters: 2,198,924
- Output acceleration and velocity shapes: `(33, 3)`
- Outputs finite: yes
- Peak allocated GPU memory: 48.90 MiB
- Checkpoint loaded: no

PyG reported that the upstream use of `MessagePassing.jittable()` is deprecated
and PyTorch reported a future change to the default `torch.cross` dimension.
Neither warning blocked the forward pass, but both should be recorded before
any dependency upgrade.

## Network incident and resolution

Direct GitHub object transfer from the server was very slow and an interrupted
client left an older clone process running. That process raced with a retry and
cleaned the shared target path. The old processes were terminated, incomplete
directories were retained with dated names, and the verified shallow clones
were transferred from the local workstation over the campus link. No research
data or unrelated server process was removed.

The official NeuralMD model repository was also verified in the browser. The
target checkpoint is `NeuralMD_ODE/MISATO_100_seed_42/model.pth`, size
4,813,458 bytes, SHA256
`364404a7ce61ec1180fa3800c0dcbddf377b672d22f915fc6b19d202710a4a5a`.
The direct official download route timed out, but the public Hugging Face
mirror completed on the local workstation and produced exactly the published
SHA256. The file remains in an ignored local cache; no checkpoint inference is
claimed yet.

## Manual boundary correction

The competition manual states that direction two is evaluated uniformly on
MISATO and compared with NeuralMD and other baselines. It does not publish a
`Geo/Phys/Dyn/Stab` weighted formula or `T1/T2/T3` definitions. Those labels
from the earlier planning document must therefore be treated as internal
diagnostic organization only, not official scoring rules.

## Local proxy-diagnostic rerun - 2026-08-11

The 20-complex official `tiny_md.hdf5` was rerun with observed frames 0 and 1
and forecast frames 2 through 99. The generated artifact is
`misato_tiny_proxy_diagnostics.json`. All additional quantities below are
project-defined diagnostic proxies, not official scores.

| Baseline | Aligned RMSD (A) | Rg MAE (A) | RMSF MAE (A) | Contact agreement |
|---|---:|---:|---:|---:|
| Static | 0.9332 | 0.1319 | 3.2692 | 0.9671 |
| Linear | 56.6338 | 54.8009 | 35.4890 | 0.5785 |

The Linear failure remains severe after rigid alignment and is accompanied by
large radius-of-gyration error. The dominant failure is therefore structural
expansion/distortion rather than only global translation. This supports
prioritizing multi-step error control and geometry-preserving constraints when
the learned baseline is available.

## Next action

1. Transfer the verified checkpoint to the ignored server checkpoint directory.
2. Run deterministic checkpoint inference on `10GS`, then on one MISATO-100
   validation/test complex when the subset is available.
3. Download MISATO-100 (725 MB) to the server only; do not place it in Git.
4. Compare Static, Linear, and checkpoint NeuralMD with the same evaluator.

## Checkpoint inference and rollout - 2026-08-11

The verified NeuralMD-ODE checkpoint was transferred to the server checkpoint
directory and its SHA256 was rechecked before loading. Strict state-dict
loading initially exposed a configuration mismatch: the upstream parser
defaults use 96 radial bases and velocity refinement coefficient 1, whereas
the published checkpoint folder's `hyperparameter.txt` specifies 100 and 0.
The smoke runner now infers those architecture switches from the state dict;
strict loading succeeds with zero missing and zero unexpected keys.

The official evaluation entry point also depends on the author's
condition-aware torchdiffeq fork rather than the PyPI package. The fork was
pinned at `3d7c7ec8c534a9b18b8b7c7d1fea0c235e6468d0` and installed editable in
the isolated NeuralMD environment.

On one A6000, the `10GS` checkpoint smoke test produced finite acceleration
and velocity arrays of shape `(33, 3)`, retained the preprocessing maximum
absolute difference of `4.31e-7 A`, and used about 45 MiB peak allocated GPU
memory. The loaded model has 1,193,090 parameters.

A 100-frame Euler rollout was then evaluated on frames 2-99 after using frames
0-1 for initialization. This is a single tiny-MISATO complex and must not be
presented as a test-set or aggregate result.

| Model | Coord RMSE (A) | Matching (A) | Stability (%) | Aligned RMSD (A) | Rg MAE (A) | RMSF MAE (A) | Contact agreement |
|---|---:|---:|---:|---:|---:|---:|---:|
| NeuralMD-ODE checkpoint | 2.7716 | 0.8059 | 61.8317 | 1.2625 | 0.0671 | 1.4414 | 0.9579 |
| Static | 2.7817 | 0.7355 | 65.7746 | 1.3012 | 0.1487 | 1.7817 | 0.9697 |
| Linear | 47.0839 | 86.9971 | 4.9755 | 63.0644 | 60.3993 | 29.6383 | 0.7216 |

NeuralMD is slightly better than Static on coordinate RMSE, aligned RMSD,
radius of gyration, and RMSF for this complex, but worse on Matching,
Stability, and contact-map agreement. It strongly outperforms unconstrained
Linear. The mixed comparison reinforces the need for an official-split,
multi-complex evaluation and motivates geometry/contact-aware stabilization;
it is not evidence that NeuralMD globally beats or loses to Static.
