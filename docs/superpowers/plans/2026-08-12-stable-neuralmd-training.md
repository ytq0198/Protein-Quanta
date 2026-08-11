# Stable NeuralMD Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Isolate whether gradient clipping prevents the observed epoch-48 NeuralMD training collapse without changing model architecture, data, ODE settings, loss, or seed.

**Architecture:** Add a repository-owned, unit-tested `stable_backward_step` helper and a small upstream patch that calls it. Preserve official training behavior except for norm logging, clipping at 1.0, and skipping only non-finite loss/gradient updates. Run the same 100-epoch seed-42 configuration, evaluate best/final checkpoints with the unified validator, and compare against the unclipped reproduction.

**Tech Stack:** PyTorch 2.6, NeuralMD, MISATO-100, condition-aware torchdiffeq, unittest, JSON/Markdown reports, NVIDIA A6000.

## Global Constraints

- Clip threshold is fixed at global L2 norm 1.0 for E3.
- Finite large losses are logged and backpropagated; they are not silently skipped.
- Only non-finite loss or gradient updates are skipped and counted.
- Validation chooses checkpoints; test is not used for hyperparameter selection.
- Server checkpoints/logs remain outside Git; Git stores code, patch, summaries, figures and hashes.
- E3 must use the exact seed-42 baseline configuration documented in the experiment report.

---

### Task 1: Stable optimizer step

**Files:**
- Create: `protein_quanta/training_stability.py`
- Create: `tests/test_training_stability.py`

**Interfaces:**
- Produces: `stable_backward_step(loss, model, optimizer, max_grad_norm) -> dict`.
- Result keys: `applied`, `reason`, `loss`, `grad_norm_before_clip`, `max_grad_norm`, `clipped`.

- [x] Write tests proving a large finite gradient is clipped and updates parameters finitely, while NaN loss skips the update without parameter change.
- [x] Run `python -m unittest tests.test_training_stability -v` and confirm failure before implementation.
- [x] Implement zero-grad, finite-loss check, backward, `clip_grad_norm_`, finite-gradient check and optimizer step.
- [x] Run focused and full tests; all must pass.

### Task 2: Minimal upstream integration patch

**Files:**
- Create: `third_party/patches/neuralmd-stable-training.patch`
- Modify external checkout only: `.external/NeuralMD/examples/main_MISATO_multi_traj_NeuralMD.py`
- Modify: `third_party/README.md`

**Interfaces:**
- New upstream CLI: `--max_grad_norm`, default 0 (disabled).
- Training epoch output includes mean/max pre-clip norm, clipped batch count and skipped non-finite batch count.

- [x] Patch the training loop to call `stable_backward_step` and aggregate returned statistics.
- [x] Add `--max_grad_norm` and keep default behavior unchanged at 0.
- [x] Record the exact patch and invocation requirements in `third_party/README.md`.
- [x] Run a 1-epoch server preflight with threshold 1.0; require finite weights and complete checkpoint save.

### Task 3: Full E3 run and unified evaluation

**Files:**
- Create server artifacts under `runs/protein-quanta/stability/neuralmd-seed42-clip1/`.
- Create: `reports/reproduction/neuralmd_misato100_clip1_best_val.json`
- Create: `reports/reproduction/neuralmd_misato100_clip1_best_test.json`
- Create: `reports/reproduction/neuralmd_misato100_clip1_final_val.json`

- [x] Run 100 epochs with seed 42 and persistent stdout log.
- [x] Verify no non-finite update, record maximum gradient norm and every loss spike.
- [x] Evaluate best and final checkpoints using the same 100-frame unified validator. The planned test evaluation was intentionally not unlocked after validation no-go.
- [x] Apply the go criterion: final validation Stability improves by at least 10 points over unclipped final while coordinate RMSE worsens no more than 2%; also compare best checkpoints.

### Task 4: Report and milestone

**Files:**
- Modify: `reports/experiment-progress-report.md`
- Modify: `docs/experiment-design-and-research-roadmap.md`
- Create: `reports/figures/neuralmd_clip1_training_comparison.png` if log contains enough checkpoints.

- [x] Record configuration, runtime, gradient statistics, checkpoint hashes, unified metrics, failures and go/no-go decision.
- [x] Run local/server test suites, JSON validation, secret scan and `git diff --check`.
- [ ] Commit as `experiment: stabilize NeuralMD training with gradient clipping` and push `codex/reproduction-bootstrap` without force.
