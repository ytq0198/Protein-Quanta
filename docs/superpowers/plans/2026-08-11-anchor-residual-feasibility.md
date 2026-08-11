# Static Anchor Residual Feasibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether a time-decayed residual between Static and NeuralMD improves long-horizon geometry without erasing learned ligand dynamics.

**Architecture:** Keep the official checkpoint and ODE rollout frozen. A focused NumPy module transforms each forecast as `anchor + exp(-beta * step / 98) * (prediction - anchor)`. A separate evaluator scans predeclared beta values on validation trajectories, applies an RMSF anti-collapse constraint, freezes one beta, and evaluates it once on the official test split.

**Tech Stack:** Python 3.10+, NumPy, existing `protein_quanta.metrics`, JSON/NPZ artifacts, unittest, MISATO-100, NeuralMD checkpoint inference.

## Global Constraints

- Never tune beta on the test split.
- Frames 0 and 1 are observations; metrics use frames 2 through 99.
- Use `decay_scale_frames=98` for every candidate so beta has one fixed meaning.
- Beta candidates are exactly `0, 0.25, 0.5, 1, 2, 4, 8`.
- A candidate is eligible only when validation RMSF MAE is no more than 105% of beta-0 NeuralMD and coordinate RMSE is no more than 102% of beta-0 NeuralMD.
- Among eligible candidates choose maximum Stability, breaking ties by lower Matching and then lower beta.
- All reported metrics are reproduction proxies, not official competition scores.
- Raw trajectories and checkpoints remain outside Git.

---

### Task 1: Anchored residual transform

**Files:**
- Create: `protein_quanta/anchoring.py`
- Test: `tests/test_anchoring.py`

**Interfaces:**
- Consumes: `prediction: np.ndarray` with shape `(frames, atoms, 3)`, `history: np.ndarray` with shape `(2, atoms, 3)`.
- Produces: `anchored_residual_rollout(prediction, history, beta, decay_scale_frames=98) -> np.ndarray` with the same shape as prediction; observed frames equal history and forecast residuals decay toward `history[-1]`.

- [ ] **Step 1: Write failing tests**

```python
def test_beta_zero_keeps_neural_forecast_and_observed_history():
    result = anchored_residual_rollout(prediction, history, beta=0)
    np.testing.assert_allclose(result[:2], history)
    np.testing.assert_allclose(result[2:], prediction[2:])

def test_positive_beta_decays_forecast_residual_toward_static_anchor():
    result = anchored_residual_rollout(prediction, history, beta=2, decay_scale_frames=2)
    assert np.linalg.norm(result[-1] - history[-1]) < np.linalg.norm(prediction[-1] - history[-1])

def test_negative_beta_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        anchored_residual_rollout(prediction, history, beta=-1)
```

- [ ] **Step 2: Run the focused test and confirm import/function failure**

Run: `python -m unittest tests.test_anchoring -v`
Expected: FAIL because `protein_quanta.anchoring` does not exist.

- [ ] **Step 3: Implement the transform**

```python
def anchored_residual_rollout(prediction, history, beta, decay_scale_frames=98):
    prediction = np.asarray(prediction)
    history = np.asarray(history)
    # validate ranks, matching atom/XYZ dimensions, finite values, beta and scale
    result = prediction.copy()
    result[:2] = history
    steps = np.arange(1, prediction.shape[0] - 1, dtype=float)
    weights = np.exp(-float(beta) * steps / float(decay_scale_frames))
    anchor = history[-1]
    result[2:] = anchor + weights[:, None, None] * (prediction[2:] - anchor)
    return result
```

- [ ] **Step 4: Run focused and full tests**

Run: `python -m unittest tests.test_anchoring -v`
Expected: PASS.
Run: `python -m unittest discover -s tests -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit the isolated transform**

```bash
git add protein_quanta/anchoring.py tests/test_anchoring.py
git commit -m "experiment: add static-anchor residual transform"
```

### Task 2: Validation beta scan and frozen selection

**Files:**
- Create: `scripts/evaluate_anchor_residual.py`
- Test: `tests/test_evaluate_anchor_residual.py`

**Interfaces:**
- Consumes: directory of `{sample_id}.npz` files containing `prediction` and `truth`; beta list; split label.
- Produces: JSON containing protocol, per-beta summary, per-sample metrics, eligibility, selection rule, and `selected_beta`.
- Produces helper `select_beta(candidates, rmsf_tolerance=1.05, coordinate_tolerance=1.02) -> float`.

- [ ] **Step 1: Write failing selection tests**

```python
def test_selection_rejects_static_collapse_even_if_stability_is_highest():
    candidates = make_candidates(beta0_rmsf=1.0, collapsed_rmsf=1.2)
    assert select_beta(candidates) != 8.0

def test_selection_maximizes_stability_then_matching_then_lower_beta():
    candidates = make_eligible_tie_candidates()
    assert select_beta(candidates) == 0.5
```

- [ ] **Step 2: Run test and confirm module failure**

Run: `python -m unittest tests.test_evaluate_anchor_residual -v`
Expected: FAIL because the evaluator module does not exist.

- [ ] **Step 3: Implement trajectory loading, evaluation, aggregation and frozen rule**

Use existing `scripts.evaluate_naive_baselines._evaluate` for every candidate. Validate that every NPZ has matching finite arrays and at least three frames. Aggregate scalar metrics as an unweighted mean across complexes. Mark beta 0 as the reference and apply the exact eligibility constraints from Global Constraints.

- [ ] **Step 4: Run focused and full tests**

Run: `python -m unittest tests.test_evaluate_anchor_residual -v`
Expected: PASS.
Run: `python -m unittest discover -s tests -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit evaluator**

```bash
git add scripts/evaluate_anchor_residual.py tests/test_evaluate_anchor_residual.py
git commit -m "experiment: add validation-only anchor beta scan"
```

### Task 3: Generate validation trajectories and run the scan

**Files:**
- Modify: `reports/experiment-progress-report.md`
- Create: `reports/reproduction/anchor_residual_val_scan.json`

**Interfaces:**
- Consumes: official NeuralMD checkpoint and official `val_MD.txt`.
- Produces: server-only NPZ trajectories in `/mnt/localDisk3/weizian/runs/protein-quanta/neuralmd/misato100-val-rollout100/` and Git-safe JSON summary.

- [ ] **Step 1: Re-run frozen checkpoint inference with trajectory output**

Run the existing evaluator with `--trajectory-dir /mnt/localDisk3/weizian/runs/protein-quanta/neuralmd/misato100-val-rollout100`, 100 frames, Euler step 5, scaling 100, and the official validation split.
Expected: 10 NPZ files and the same beta-0 summary as `neuralmd_misato100_val_euler5.json` within floating-point tolerance.

- [ ] **Step 2: Run the predeclared beta scan**

Run `python -m scripts.evaluate_anchor_residual` with the exact beta list from Global Constraints.
Expected: valid JSON and one selected beta; beta 0 may legitimately win.

- [ ] **Step 3: Check the feasibility gate**

Go if selected beta is greater than 0 and improves validation Stability by at least 2 percentage points or reduces Matching by at least 5%, while satisfying both anti-collapse constraints. Otherwise record C1 as negative and do not promote fixed anchoring.

- [ ] **Step 4: Update the progress report**

Record every beta, selected beta, eligibility, improvement percentages, and the go/no-go decision. Explicitly state that validation selected the hyperparameter.

- [ ] **Step 5: Commit validation evidence**

```bash
git add reports/reproduction/anchor_residual_val_scan.json reports/experiment-progress-report.md
git commit -m "report: record anchor residual validation scan"
```

### Task 4: One-shot frozen test and figure

**Files:**
- Create: `scripts/plot_anchor_residual.py`
- Test: `tests/test_plot_anchor_residual.py`
- Create: `reports/reproduction/anchor_residual_test_frozen.json`
- Create: `reports/figures/anchor_residual_tradeoff.png`
- Modify: `reports/experiment-progress-report.md`

**Interfaces:**
- Consumes: selected validation beta and existing test NPZ trajectories.
- Produces: one frozen test result and one validation/test trade-off figure.

- [ ] **Step 1: Write a failing rendering test**

Create a two-candidate fixture and assert `render_figure(...)` creates a non-empty PNG with validation and test series.

- [ ] **Step 2: Run the rendering test and confirm failure**

Run: `python -m unittest tests.test_plot_anchor_residual -v`
Expected: FAIL because the plotting module does not exist.

- [ ] **Step 3: Evaluate exactly beta 0 and the frozen selected beta on test**

Do not scan or reselect on test. Store `selection_source: validation` and the validation artifact SHA256 in the test JSON.

- [ ] **Step 4: Implement and render the trade-off figure**

Plot coordinate RMSE, Matching, Stability, Rg MAE and RMSF MAE as percent change from beta 0. Positive bars mean improvement after direction normalization. Label every panel as proxy diagnostics.

- [ ] **Step 5: Verify image and all tests**

Visually inspect the PNG. Run `python -m unittest discover -s tests -v`; expected all tests PASS.

- [ ] **Step 6: Update report and commit milestone**

```bash
git add scripts/plot_anchor_residual.py tests/test_plot_anchor_residual.py reports/reproduction/anchor_residual_test_frozen.json reports/figures/anchor_residual_tradeoff.png reports/experiment-progress-report.md
git commit -m "report: validate frozen static-anchor residual"
```

### Task 5: Milestone verification and publication

**Files:**
- Modify: `README.md`
- Modify: `reports/experiment-progress-report.md`

**Interfaces:**
- Consumes: all C1 artifacts and commits.
- Produces: a reproducible milestone suitable for pushing to `origin`.

- [ ] **Step 1: Verify local and server tests**

Run the full unittest suite in both environments. Validate every JSON with Python's JSON parser and scan tracked files for credentials or large binary artifacts.

- [ ] **Step 2: Verify repository scope**

Run `git status --short`, `git diff --check`, and confirm no HDF5, NPZ, PTH, credentials, or server paths containing secrets are staged.

- [ ] **Step 3: Update README milestone links**

Link the design, progress report, frozen JSON and trade-off figure; state the go/no-go decision without overstating official score impact.

- [ ] **Step 4: Commit and push the milestone**

```bash
git add README.md reports/experiment-progress-report.md
git commit -m "docs: publish anchor residual feasibility milestone"
git push -u origin codex/reproduction-bootstrap
```

If GitHub authentication prevents the push, give Cursor this exact instruction: “In `Protein-Quanta`, review the staged milestone for secrets and large files, run the full unittest suite, then push branch `codex/reproduction-bootstrap` to `origin` without rewriting history or changing existing files.”
