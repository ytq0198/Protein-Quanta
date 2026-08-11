# Third-party source register

External repositories are not copied into this Git repository by default.
Every local checkout must be pinned to an exact commit and reviewed for its
license before code is reused.

| Project | Source | Purpose | Pinned commit | License review |
|---|---|---|---|---|
| MISATO | https://github.com/t7morgen/misato-dataset | Dataset audit and loader reference | `7b06d532e2ed0719411fcc1b3ac39743db4ca10d` | Repository contains an LGPL-2.1 license file; review obligations before copying code |
| NeuralMD | https://github.com/chao1224/NeuralMD | Main trajectory baseline | `a2ae030838c6ea0251eb6a29bfe99dc9d8ee1cfe` | `setup.py` declares MIT, but this commit has no standalone license file; treat reuse as pending clarification |
| torchdiffeq fork | https://github.com/chao1224/torchdiffeq | Condition-aware ODE integration required by NeuralMD | `3d7c7ec8c534a9b18b8b7c7d1fea0c235e6468d0` | MIT license file present |
| DynamicBind | https://github.com/luwei0917/DynamicBind | Conditional pocket-flexibility study | not scheduled | pending |

The verified checkouts live outside the project repository at
`/mnt/localDisk3/weizian/external/`. They are references until license review
and code provenance are complete.

## Official NeuralMD checkpoint target

- Repository: https://huggingface.co/chao1224/NeuralMD/tree/main
- File: `NeuralMD_ODE/MISATO_100_seed_42/model.pth`
- Published size: 4.81 MB
- Published SHA256:
  `364404a7ce61ec1180fa3800c0dcbddf377b672d22f915fc6b19d202710a4a5a`
- Status: downloaded on the local workstation through a public Hugging Face
  mirror and verified against the SHA256 published by the official repository.
  The same verified file is now present in the server checkpoint directory;
  both copies remain outside Git.

The checkpoint folder also publishes `hyperparameter.txt`. Architecture and
rollout settings relevant to reproduction are: 100 radial bases, velocity
refinement coefficient 0, Euler ODE, step size 5, scaling 100, 20 training
frames, batch size 8, 100 epochs, seed 42, and critically
`--no_NeuralMD_Binding_start_with_first_frame`. Without that explicit boolean,
the upstream parser starts every sample at frame 0 and silently bypasses the
20-frame window. These differ from several defaults in the current upstream
script, so checkpoint configuration must not be reconstructed from parser
defaults. The canonical command lives in `protein_quanta/neuralmd_training.py`.

## PyTorch 2.6 dataset-cache compatibility

Apply `patches/neuralmd-pytorch26.patch` to the pinned NeuralMD checkout when
using PyTorch 2.6+. It only opts the locally generated, trusted PyG dataset
cache out of PyTorch's weights-only loader. Model checkpoints remain loaded
with strict weights-only handling in this project.

## Stable-training instrumentation

Apply `patches/neuralmd-stable-training.patch` after the PyTorch 2.6 patch with
`git apply --unidiff-zero`; the zero-context form keeps the patch artifact free
of whitespace-only lines inherited from the upstream script.
The patch makes the pinned MISATO training entry point import
`protein_quanta.training_stability`, adds `--max_grad_norm` (default `0`,
disabled), and reports mean/maximum pre-clip gradient norm plus clipped and
non-finite batch counts for every epoch. Run the upstream script with this
repository on `PYTHONPATH`; on the school server this is:

```bash
export PYTHONPATH=/mnt/localDisk3/weizian/Protein-Quanta
```

E3 uses `--max_grad_norm 1`. Finite loss spikes are still optimized; only
non-finite loss or gradient updates are skipped and counted.
