# Competition Scenarios and Pair-Distance Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add T1/T2/T3-aligned NeuralMD validation and test whether a gradient-calibrated pair-distance auxiliary loss improves long-horizon geometry without collapsing dynamics.

**Architecture:** A small scenario module owns all observed/target frame boundaries. The evaluator reuses the existing strict checkpoint loader but initializes NeuralMD at each scenario's last two observed frames, aggregates raw proxy metrics, and never invents organizer normalization. A separate invariant loss module supplies per-complex Smooth-L1 pair-distance loss; the pinned upstream training entry point receives the smallest patch needed to calibrate its weight and save periodic checkpoints.

**Tech Stack:** Python, NumPy, PyTorch 2.6, NeuralMD, condition-aware torchdiffeq, MISATO-100, unittest, JSON/Markdown, NVIDIA A6000.

## Global Constraints

- Use official window flags: `--no_NeuralMD_Binding_start_with_first_frame --NeuralMD_Binding_frame_num 20`.
- T1 proxy observes frames 0–1 and evaluates 2–19; T2 observes 0–79 and evaluates 80–99; T3 observes 0–19 and evaluates 20–99.
- Label every local scenario result as a proxy, not an official competition score.
- Do not construct a 0–100 composite without organizer normalization code.
- Validation alone selects checkpoints and method settings; do not unlock a new test run before the preregistered gate passes.
- E6 changes only `L_pair`; all official model/data/optimizer/ODE/seed settings remain fixed.
- `L_pair` uses unordered non-diagonal heavy-atom pairs, Smooth-L1 beta 0.5 Å, per-complex then per-batch averaging.
- Calibrate `lambda_d` from ten deterministic batches to target a 10% initial auxiliary gradient contribution, then freeze it.
- Checkpoints, trajectories and logs remain outside Git; Git stores code, small JSON, hashes, figures and reports.
- The supplied preliminary submission template says AI-generated answers are prohibited; produce evidence and internal notes only, not final submission prose.

---

### Task 1: Immutable scenario definitions

**Files:**
- Create: `protein_quanta/scenarios.py`
- Test: `tests/test_scenarios.py`

**Interfaces:**
- Produces: `TrajectoryScenario(name, observed_start, observed_end, target_start, target_end)`.
- Produces: `competition_scenarios(total_frames=100) -> tuple[TrajectoryScenario, ...]`.
- Produces: `scenario_time_grid(scenario, scaling) -> np.ndarray` including the two initializer frames and all target frames.

- [ ] Write failing tests with literal expected initializer/target indices for T1 `(0,1)->2:19`, T2 `(78,79)->80:99`, and T3 `(18,19)->20:99`; test invalid frame counts and nonpositive scaling.
- [ ] Run `python -m unittest tests.test_scenarios -v`; require missing-module failure.
- [ ] Implement a frozen dataclass with validation, `initializer_indices`, `target_indices`, and the exact three-scenario factory.
- [ ] Run focused and full tests; require all pass.

### Task 2: Arbitrary-frame rollout comparison and stability windows

**Files:**
- Modify: `scripts/smoke_neuralmd.py`
- Modify: `protein_quanta/metrics.py`
- Test: `tests/test_smoke_neuralmd.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces: `_scenario_rollout_comparison(prediction, truth, observed_local_frames, contact_cutoff)` where prediction/truth contain the two initializer frames followed by targets.
- Produces: `error_growth_summary(prediction, truth) -> dict` with early/middle/late coordinate RMSE and least-squares RMSE growth slope.

- [ ] Add failing tests proving arbitrary-frame comparison excludes both observed frames, Static repeats the last observed frame, and a handcrafted increasing error has exact window means/slope.
- [ ] Run focused tests and confirm behavior failures.
- [ ] Refactor the existing two-frame comparison into the generic function while keeping `_rollout_comparison` backward compatible.
- [ ] Implement three approximately equal contiguous target windows using `np.array_split`; compute per-frame coordinate RMSE and its linear slope against zero-based target time.
- [ ] Add the growth summary to `_evaluate` output without changing existing metric names.
- [ ] Run focused and full tests.

### Task 3: Scenario evaluator

**Files:**
- Create: `scripts/evaluate_neuralmd_scenarios.py`
- Test: `tests/test_evaluate_neuralmd_scenarios.py`

**Interfaces:**
- Consumes: strict checkpoint/data helpers from `scripts.evaluate_neuralmd_checkpoint` and scenarios from Task 1.
- Produces: one JSON report with `protocol.scenarios`, `summary[scenario][model]`, and `samples[*].scenarios`.

- [ ] Write failing helper tests for aggregation by scenario/model, duplicate sample rejection, and report status text `competition-aligned proxy; not official score`.
- [ ] Run focused tests and confirm missing behavior.
- [ ] Implement one model/data load per sample, scenario-specific position/velocity initialization, Euler integration over local time, and NeuralMD/Static/Linear comparisons.
- [ ] Record checkpoint SHA, exact frame indices, upstream path/commit, sample IDs, runtime, peak memory, and per-sample preprocessing agreement.
- [ ] Run a CPU helper test and local full suite.
- [ ] Run one-complex server smoke evaluation for all scenarios; require finite outputs and valid JSON.

### Task 4: Published and corrected baseline scenario table

**Files:**
- Create: `reports/reproduction/neuralmd_scenarios_published_val.json`
- Create: `reports/reproduction/neuralmd_scenarios_corrected_best_val.json`
- Create: `reports/figures/neuralmd_scenario_baselines.png`
- Modify: `reports/experiment-progress-report.md`
- Modify: `docs/experiment-design-and-research-roadmap.md`

**Interfaces:**
- Produces the frozen validation reference used by E6 gates.

- [ ] Evaluate the published checkpoint and corrected seed-42 best checkpoint on all 10 validation complexes.
- [ ] Require numerical agreement between their scenario summaries within `1e-3` for scalar metrics; investigate before E6 if this fails.
- [ ] Plot T1/T2/T3 NeuralMD-vs-Static coordinate RMSE, Matching, Stability, RMSF and error-growth slope; label as proxy.
- [ ] Record frame definitions, raw values, limitations, hashes and the no-official-normalization rule in both living documents.
- [ ] Commit and push the scenario-evaluation milestone after local/server tests, JSON validation, secret scan and `git diff --check`.

### Task 5: Invariant per-complex pair-distance loss

**Files:**
- Create: `protein_quanta/geometry_loss.py`
- Test: `tests/test_geometry_loss.py`

**Interfaces:**
- Produces: `pair_distance_smooth_l1(prediction, truth, atom_batch, beta=0.5) -> torch.Tensor`.
- Input shape: `(frames, total_atoms, 3)` for coordinates and `(total_atoms,)` for complex membership.

- [ ] Write failing tests for zero-on-truth, rigid translation/rotation invariance, duplicate/diagonal exclusion, unequal-complex balancing, invalid shapes, positive beta, and finite backward gradients.
- [ ] Run focused tests and confirm missing-module failure.
- [ ] Implement unique upper-triangle pairs separately for each complex, Smooth-L1 on distance residuals, mean within complex, then mean across nonempty complexes.
- [ ] Reject complexes with fewer than two atoms rather than silently changing the batch denominator.
- [ ] Run focused and full tests.

### Task 6: Gradient-scale calibration

**Files:**
- Modify: `protein_quanta/geometry_loss.py`
- Test: `tests/test_geometry_loss.py`

**Interfaces:**
- Produces: `calibrated_auxiliary_weight(position_grad_norms, pair_grad_norms, target_fraction=0.1, epsilon=1e-12) -> float`.

- [ ] Write failing literal tests for the median ratio, zero pair gradient handling, mismatched/empty sequences, non-finite values and invalid target fraction.
- [ ] Run focused tests and confirm failures.
- [ ] Implement the exact preregistered formula `target_fraction * median(pos / max(pair, epsilon))` and strict validation.
- [ ] Run focused and full tests.

### Task 7: Minimal upstream E6 patch and periodic checkpoints

**Files:**
- Create: `third_party/patches/neuralmd-pair-loss.patch`
- Modify external checkout: `.external/NeuralMD/examples/main_MISATO_multi_traj_NeuralMD.py`
- Modify: `third_party/README.md`
- Modify: `protein_quanta/neuralmd_training.py`
- Test: `tests/test_neuralmd_training_command.py`

**Interfaces:**
- New CLI: `--pair_loss_coefficient`, default `0`; `--pair_loss_beta`, default `0.5`; `--save_every_epoch`, default `0`.
- Epoch log adds `loss_pair` and the frozen coefficient.
- Checkpoints named `model_epoch_005.pth`, `model_epoch_010.pth`, and so on outside Git.

- [ ] Extend command tests first to require the three explicit E6 arguments while defaults preserve the corrected baseline.
- [ ] Patch the training loop to compute `L_pair` and add it only when coefficient is positive; preserve exact baseline behavior at zero.
- [ ] Save periodic checkpoints after validation epochs without changing upstream best-coordinate logic.
- [ ] Create a calibration mode that runs ten deterministic batches, records separate gradient norms, writes a small JSON, and exits before optimization; do not reuse those batches for model selection.
- [ ] Generate a clean provenance patch and verify it applies to pinned upstream commit with the PyTorch compatibility and stable-training patches in order.
- [ ] Run zero-coefficient 5-epoch equivalence preflight; require matching metrics/checkpoint tensors relative to the corrected preflight within numerical tolerance.

### Task 8: E6 20-epoch feasibility and decision

**Files:**
- Create server artifacts: `runs/protein-quanta/e6/neuralmd-pair-seed42-20ep/`
- Create: `reports/reproduction/neuralmd_pair20_scenarios_val.json`
- Modify: `reports/experiment-progress-report.md`
- Modify: `docs/experiment-design-and-research-roadmap.md`

**Interfaces:**
- Consumes the frozen baseline table from Task 4 and go gate from the design spec.

- [ ] Run ten-batch calibration and freeze `lambda_d`; record all ratios and the selected coefficient.
- [ ] Run seed 42 for 20 epochs with periodic checkpoints and persistent logs.
- [ ] Evaluate checkpoints 5/10/15/20 on validation scenarios; select only by the preregistered multi-condition gate.
- [ ] Advance to 100 epochs only if T1 RMSE ≤102%, T2 Stability ≥baseline−1 point, T2 Matching ≤102%, and T3 improves Stability ≥2 points or Matching ≥5%, with T3 RMSF ≤105% and no material collision regression.
- [ ] If no checkpoint passes, record no-go and retain C1 as the initial-submission candidate; do not evaluate test or run more seeds.
- [ ] If a checkpoint passes, run the 100-epoch extension, reapply the gate, then schedule seeds 0 and 123.
- [ ] Update living reports with config, hashes, per-scenario tables, failures, figures and scientific interpretation.
- [ ] Verify local/server tests, all JSON, secret patterns, tracked large artifacts and `git diff --check`; commit and push the E6 milestone.
