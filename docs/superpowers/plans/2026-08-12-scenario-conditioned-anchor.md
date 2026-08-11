# Scenario-Conditioned Anchor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate a preregistered `T1=8, T2=8, T3=1` Static-residual anchor policy without changing NeuralMD weights or using frozen-test results for parameter selection.

**Architecture:** Keep the numerical anchor transform unchanged. Add a strict policy validator in `protein_quanta.anchoring`, add CLI parsing and per-scenario dispatch to the existing evaluator, then run the frozen validation-selected policy on archived server trajectories. Reports retain per-complex metrics and explicitly distinguish validation selection from frozen internal testing.

**Tech Stack:** Python 3.10, NumPy, argparse, unittest/pytest, existing Protein-Quanta metric and scenario aggregation utilities.

## Global Constraints

- Candidate grid is exactly `{0, 0.5, 1, 2, 4, 8}` and the selected mapping is exactly `T1=8, T2=8, T3=1`.
- `decay_scale_frames` remains `98`.
- Selection uses MISATO-100 validation only; frozen internal test is run once and never changes the policy.
- No local composite competition score is invented because official normalization is unavailable.
- Existing global scalar-beta CLI behavior remains available.
- Checkpoints, trajectories, datasets, credentials, and absolute personal paths are not committed.

---

### Task 1: Strict scenario-policy validation

**Files:**
- Modify: `protein_quanta/anchoring.py`
- Modify: `tests/test_anchoring.py`

**Interfaces:**
- Produces: `validate_scenario_betas(beta_by_scenario, expected_scenarios) -> dict[str, float]`.
- The returned mapping contains every expected scenario exactly once, contains no unknown scenarios, and contains only finite non-negative floats.

- [ ] **Step 1: Write failing unit tests**

Add tests that expect `{"T1": 8, "T2": 8, "T3": 1}` to normalize to floats and that reject a missing key, an unknown key, a negative value, and `NaN`.

- [ ] **Step 2: Verify the tests fail for the missing interface**

Run: `python -m pytest -q tests/test_anchoring.py`

Expected: collection or test failure because `validate_scenario_betas` does not exist.

- [ ] **Step 3: Implement the minimal validator**

Implement exact set comparison against `expected_scenarios`, reject duplicate expected names, convert numeric values to floats, and validate with `numpy.isfinite(value) and value >= 0`.

- [ ] **Step 4: Verify Task 1**

Run: `python -m pytest -q tests/test_anchoring.py`

Expected: all anchoring tests pass.

- [ ] **Step 5: Commit Task 1**

Commit message: `feat: validate scenario-conditioned anchor policies`

### Task 2: CLI parsing, dispatch, and auditable protocol metadata

**Files:**
- Modify: `scripts/evaluate_anchor_scenarios.py`
- Modify: `tests/test_evaluate_anchor_scenarios.py`

**Interfaces:**
- Produces: `_parse_scenario_betas(values: list[str]) -> dict[str, float]` for `SCENARIO=BETA` tokens.
- CLI accepts mutually exclusive `--beta FLOAT` and `--scenario-betas T1=8 T2=8 T3=1`.
- CLI accepts `--split-label`, `--selection-allowed`, and `--policy-source` so JSON does not mislabel frozen-test evidence.
- The protocol records `anchor_policy.type`, `anchor_policy.beta_by_scenario`, `split_label`, `selection_allowed`, and `policy_source`.

- [ ] **Step 1: Write failing parser and dispatch tests**

Test successful parsing, duplicate token rejection, malformed token rejection, per-scenario beta dispatch, and mutual exclusion through argparse behavior or a small resolver function.

- [ ] **Step 2: Verify the new tests fail**

Run: `python -m pytest -q tests/test_evaluate_anchor_scenarios.py`

Expected: failure because parser/resolver interfaces do not exist.

- [ ] **Step 3: Implement the minimal CLI extension**

Parse only `name=value` tokens, reject empty names and duplicate keys, validate the completed mapping against scenario names from the reference report, and use the resolved beta inside the existing sample/scenario loop. Preserve a default scalar beta of `4.0` only when neither policy option is supplied.

- [ ] **Step 4: Verify Task 2 and full regression suite**

Run: `python -m pytest -q tests/test_evaluate_anchor_scenarios.py tests/test_anchoring.py`

Then run: `python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 5: Commit Task 2**

Commit message: `feat: evaluate scenario-conditioned anchor gates`

### Task 3: Validation replay, frozen test, and evidence decision

**Files:**
- Create: `reports/reproduction/neuralmd_scenario_anchor881_val.json`
- Create: `reports/reproduction/neuralmd_scenario_anchor881_test.json`
- Create: `reports/reproduction/2026-08-12-scenario-conditioned-anchor.md`
- Create: `reports/figures/neuralmd_scenario_anchor881_tradeoff.png`
- Modify: `reports/experiment-progress-report.md`
- Modify: `docs/experiment-design-and-research-roadmap.md`
- Modify: `docs/preliminary-evidence-index.md`
- Modify: `configs/frozen_candidate.json` only if every preregistered promotion condition passes.

**Interfaces:**
- Consumes archived `epoch005-val-trajectories` and `epoch005-test-trajectories` on the server.
- Produces JSON with the existing aggregate/per-complex schema plus the new policy metadata.
- Produces a promotion/no-go decision against the existing global `beta=1` reports.

- [ ] **Step 1: Run validation replay with the frozen mapping**

Use the pinned server Python environment, the validation trajectory directory, `--scenario-betas T1=8 T2=8 T3=1`, `--split-label validation`, `--selection-allowed`, and a policy source naming the committed preregistration spec.

- [ ] **Step 2: Check validation protocol and guards before touching test**

Confirm all ten sample IDs and three scenarios are present, the policy mapping is exact, all metrics are finite, and the T3 coordinate guard remains within the preregistered threshold.

- [ ] **Step 3: Run the frozen internal test exactly once**

Use the same mapping and scale, `--split-label frozen-internal-test`, no selection flag, and the same policy source. Do not rerun with changed betas.

- [ ] **Step 4: Apply the preregistered promotion rules**

Compare against `neuralmd_earlystop_anchor1_{val,test}.json`: T1/T2 Matching and Stability must improve; every coordinate RMSE change must be at most +2%; every RMSF MAE change must be at most +5%. Record each Boolean gate and the final decision.

- [ ] **Step 5: Create the comparison figure and research record**

Plot per-scenario relative coordinate RMSE, Matching, Stability-point, and RMSF changes versus global `beta=1`, labeling them internal proxy diagnostics. Record commands, hashes, environment, observed numbers, limitations, and whether learned feature gating remains justified.

- [ ] **Step 6: Update the living reports and frozen manifest conditionally**

If all gates pass, promote the mapping while retaining the exact previous candidate in the experiment report. If any gate fails, leave `configs/frozen_candidate.json` unchanged and record the experiment as no-go.

- [ ] **Step 7: Verify release and commit**

Run locally: `python -m pytest -q` and `python scripts/audit_release.py --output reports/reproduction/release-audit.json`.

Run the full test suite in the clean server clone. Confirm no `.npz`, checkpoint, HDF5, credential, or oversized artifact is tracked.

Commit message: `experiment: evaluate scenario-conditioned anchor gate`

