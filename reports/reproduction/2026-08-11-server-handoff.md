# 2026-08-11 NeuralMD server handoff

## Objective

Complete the first deterministic NeuralMD checkpoint inference without
changing system CUDA, committing data, or starting long training.

## Inputs already verified locally

- Checkpoint: `NeuralMD_ODE/MISATO_100_seed_42/model.pth`
- Size: 4,813,458 bytes
- SHA256: `364404a7ce61ec1180fa3800c0dcbddf377b672d22f915fc6b19d202710a4a5a`
- Upstream NeuralMD commit: `a2ae030838c6ea0251eb6a29bfe99dc9d8ee1cfe`
- Upstream MISATO commit: `7b06d532e2ed0719411fcc1b3ac39743db4ca10d`

The checkpoint must stay under
`/mnt/localDisk3/weizian/checkpoints/protein-quanta/` and must not enter Git.

## Required server sequence

1. Record `nvidia-smi`, free disk, current project commit/status, Python path,
   and package versions.
2. Copy the verified checkpoint from the workstation to:
   `/mnt/localDisk3/weizian/checkpoints/protein-quanta/NeuralMD_ODE/MISATO_100_seed_42/model.pth`.
3. Recompute SHA256 on the server and stop immediately if it differs.
4. Run the complete project tests in the lightweight environment.
5. Activate `/mnt/localDisk3/weizian/conda_envs/protein-quanta-neuralmd`.
6. Run one `10GS` checkpoint smoke test on an available GPU:

```bash
cd /mnt/localDisk3/weizian/Protein-Quanta
python -m scripts.smoke_neuralmd \
  --upstream /mnt/localDisk3/weizian/external/NeuralMD \
  --h5 /mnt/localDisk3/weizian/external/misato-dataset/data/MD/h5_files/tiny_md.hdf5 \
  --checkpoint /mnt/localDisk3/weizian/checkpoints/protein-quanta/NeuralMD_ODE/MISATO_100_seed_42/model.pth \
  --sample-id 10GS \
  --device cuda:0 \
  --output reports/reproduction/neuralmd_10GS_checkpoint_smoke.json
```

7. Require `checkpoint_loaded: true`, the published SHA256, no missing or
   unexpected state-dict keys, finite outputs, and the previously observed
   preprocessing difference near `4.31e-7 A`.
8. Only after the single-sample check succeeds, obtain MISATO-100 outside Git
   and run the official split. Do not start full MISATO training yet.

## Stop conditions

- Checkpoint hash mismatch.
- Any missing/unexpected model key.
- Atom count, atom order, output shape, or coordinate unit mismatch.
- Non-finite output.
- Dependency repair would require changing system CUDA or the driver.
- Data download would place large files inside the Git working tree.

## Cursor prompt

```text
请在 /mnt/localDisk3/weizian/Protein-Quanta 完成 NeuralMD 官方 checkpoint 的单样本推理闭环。先完整阅读 reports/reproduction/2026-08-11-server-handoff.md，并严格执行其中顺序与停止条件。

约束：不得修改系统 CUDA/驱动，不得删除或覆盖现有数据，不得下载完整 MISATO，不得 commit 或 push。先记录 GPU、磁盘、git status、Python 与关键依赖。确认 checkpoint 位于 /mnt/localDisk3/weizian/checkpoints/protein-quanta/NeuralMD_ODE/MISATO_100_seed_42/model.pth，核对 SHA256 必须为 364404a7ce61ec1180fa3800c0dcbddf377b672d22f915fc6b19d202710a4a5a。随后运行项目完整测试，再在既有 protein-quanta-neuralmd 环境中对 10GS 执行文档给出的 smoke 命令。

完成后只报告：环境、GPU、命令、耗时、显存峰值、checkpoint 哈希、missing/unexpected keys、预处理差异、输出形状、是否有限、产物路径、git status 和未解决问题。任何停止条件出现时保留完整错误日志并停止，不要通过随意降级依赖伪造成功。
```
