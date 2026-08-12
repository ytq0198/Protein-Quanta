# E17: local-displacement loss feasibility no-go

Date: 2026-08-13 (Asia/Shanghai)  
Decision split: MISATO-100 validation only  
Status: validation no-go; no new internal-test evaluation

## Hypothesis and frozen design

E16 found that epoch-5 and published NeuralMD reproduce only about 0.9--1.4% of the truth's mean one-frame displacement amplitude. E17 therefore tested whether an SE(3)-equivariant consecutive-displacement Smooth-L1 term could restore local dynamics without sacrificing T1 Geo, bond-aware Phys, or T2/T3 safety.

The experiment was committed before training in `configs/e17_displacement_preregistration.json` (commit `b393d38`). It changed one training variable:

- seed 42, official 20-frame random-window configuration;
- five epochs, Adam 1e-4, position MSE unchanged;
- displacement Smooth-L1 beta `0.5`, fixed coefficient `1.0`;
- no pair loss, no architecture/integrator/data-order change;
- validation only; coefficient sweep and new internal-test access prohibited after a failed gate.

## Training integrity

The five-epoch run completed without non-finite updates or gradient clipping. The final epoch logged displacement loss `0.56142` and coefficient `1.0`. Training-time test fields remained `nan`, confirming that `--no_eval_test_during_training` was active.

Candidate checkpoint SHA256: `4626ca99b322889ad168ef15c88a0a0970861d1a0af600e6c0afa5d05152240b`.

## Gate result

The machine-readable gate contains 20 checks. Nineteen safety checks passed; the single mandatory mechanism check failed:

| Check | Candidate / epoch-5 | Threshold | Decision |
|---|---:|---:|---|
| T1 coordinate RMSE | 0.999999996 | <=1.02 | pass |
| T1 Matching | 0.999999957 | <=1.00 | pass |
| T1 Stability change | 0.000 points | >=0 | pass |
| T1 bond-length MAE | 1.000000047 | <=1.05 | pass |
| T1 step-amplitude ideal-gap ratio | **0.9999999998** | **<=0.95** | **fail** |

T2/T3 Geo/Dyn safety ratios were also within numerical noise of one, and no scenario added extreme bond events. The candidate is effectively indistinguishable from the epoch-5 baseline despite having a different checkpoint hash.

## Decision and interpretation

E17 is **no-go**. It did not affect the mechanism it was designed to change, so passing safety checks cannot be counted as an innovation gain. Per preregistration:

- do not increase or sweep the coefficient on the same validation split;
- do not inspect a new internal-test result;
- do not promote the displacement term into the active candidate;
- retain unanchored seed-42 epoch 5 as the initial-round safe baseline.

The scientific inference is narrower: uncalibrated displacement-vector loss at coefficient one is gradient-dominated by the existing rollout position objective. A later round may use training-only gradient calibration or true short closed-loop exposure, but it requires a fresh validation design and larger independent data, not deadline-driven retuning.

## Evidence

- Preregistration: `configs/e17_displacement_preregistration.json`
- Gate: `reports/reproduction/e17_displacement_gate.json`
- Geo/Stab: `reports/reproduction/e17_displacement_val_scenarios.json`
- Dyn: `reports/reproduction/e17_displacement_val_dynamics.json`
- Phys: `reports/reproduction/e17_displacement_val_phys.json`
- Training log: `reports/reproduction/e17_displacement_training.log`
- Gate implementation: `scripts/evaluate_e17_gate.py`

All scores remain project diagnostics; no official normalized competition score is claimed.
