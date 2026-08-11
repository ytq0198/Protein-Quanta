# Scenario-Conditioned Anchor Design

## Purpose

Test whether the fixed global Static-residual anchor can be improved by conditioning its decay strength on the known competition scenario. This is a validation-selected, falsifiable bridge between the current global `beta=1` candidate and a future learned uncertainty gate. It must not alter NeuralMD weights or use the frozen internal test for selection.

## Evidence motivating the design

On the MISATO-100 validation split, increasing anchor strength from `beta=0` to `beta=8` monotonically improves coordinate RMSE, Matching, Stability, and RMSF for T1. T2 improves through the same range except that Matching is slightly better at `beta=4` than at `beta=8`. T3 behaves differently: `beta=1` improves Matching, Stability, and RMSF, while larger values exceed the preregistered two-percent coordinate guard and eventually reverse geometry gains.

This interaction rules out a single globally optimal anchor strength under the current proxy evaluation. The competition scenario is available before inference, so using it is not target leakage.

## Alternatives considered

1. **Scenario-conditioned deterministic gate — selected.** Map T1 and T2 to stronger anchoring and T3 to the current conservative strength. It is interpretable, has three auditable decisions, and requires no new training.
2. **Feature-conditioned learned gate — deferred.** Fit a small model from SE(3)-invariant rollout statistics such as displacement growth and radius-of-gyration drift. With only ten validation complexes, this has high overfitting risk and should follow only if the deterministic interaction survives frozen evaluation.
3. **Further global-beta tuning — rejected.** It cannot resolve the observed T1/T2 versus T3 interaction and adds little scientific novelty.

## Frozen selection protocol

- Candidate grid: `beta in {0, 0.5, 1, 2, 4, 8}` with `decay_scale_frames=98`.
- Selection data: MISATO-100 validation only, ten complexes, evaluated separately for T1/T2/T3.
- Per-scenario guard: coordinate RMSE no more than 102% of the unanchored epoch-5 checkpoint and RMSF MAE no more than 105% of that checkpoint.
- Within the guard, prefer lower Matching and higher Stability. Coordinate RMSE and RMSF are reported as protected metrics rather than collapsed into an invented score.
- The frozen policy selected from current validation evidence is `T1=8`, `T2=8`, `T3=1`.
- Frozen internal test is evaluated once after implementation. No beta changes are permitted in response to test results.

## Promotion and stop rules

The policy is promoted over global `beta=1` only if frozen evaluation satisfies all of the following:

1. T1 and T2 each improve Matching and Stability relative to global `beta=1`.
2. No scenario worsens coordinate RMSE by more than 2% relative to global `beta=1`.
3. No scenario worsens RMSF MAE by more than 5% relative to global `beta=1`.
4. Results are labeled as internal proxy metrics, not an official competition score.

Failure of any condition makes the experiment a documented no-go. The existing `seed=42, epoch=5, beta=1` candidate remains frozen unless every condition passes.

## Architecture and interfaces

- `protein_quanta/anchoring.py` retains the coordinate transform and gains a small policy validator that returns the beta for a named scenario.
- `scripts/evaluate_anchor_scenarios.py` accepts either the existing scalar `--beta` or a mutually exclusive `--scenario-betas T1=8 T2=8 T3=1` policy.
- The output protocol records the complete mapping, its validation-only source, and whether selection was allowed. The evaluator continues to emit per-complex and aggregate metrics through the existing scenario evaluation path.
- No checkpoint, trajectory, dataset, or credential is added to Git.

## Failure handling

Unknown scenarios, missing mappings, duplicate scenario keys, negative/non-finite beta values, or simultaneous scalar and mapped policies terminate before evaluation with a clear error. Saved trajectories must continue to pass the existing finite-value and shape checks.

## Verification

- Unit tests cover policy parsing, missing/duplicate keys, invalid beta values, and correct per-scenario dispatch.
- Existing scalar-beta behavior remains unchanged.
- The full local test suite and release audit must pass.
- The server run uses the archived epoch-5 validation and test trajectories and the pinned Python environment; resulting JSON is copied into `reports/reproduction/`.
- The experiment report records the exact command, policy, validation rationale, frozen result, and promotion/no-go decision.

