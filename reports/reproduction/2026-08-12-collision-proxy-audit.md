# Frozen-candidate collision proxy audit

## Question

The competition guide assigns 25% of each scenario score to physical
reasonableness (Phys).  Before treating the frozen `epoch 5 + beta=1` candidate
as submission-ready, we therefore tested whether its geometric gains coincide
with an obvious increase in atom overlap.

This is a project diagnostic, **not the organizer's Phys score**.  MISATO's
current NeuralMD preprocessing does not expose a covalent bond graph, so the
intramolecular calculation cannot exclude bonded pairs.  Its absolute value
therefore includes normal covalent neighbours and must not be interpreted as a
chemically valid clash rate.

## Protocol

- Model: NeuralMD seed 42, epoch 5, checkpoint SHA256
  `0e7d5150aa5f305499f17663d3a74b1063b0591733f534e676303ce11de50b8c`.
- Post-processing: time-decayed Static residual anchor, `beta=1`, decay scale
  98 frames.
- Data: the same 10-complex validation split used for selection, followed by
  the already frozen 10-complex internal test split without retuning.
- Scenarios: competition-aligned T1/T2/T3 windows.
- Pair rule: Euclidean distance below the sum of the two covalent radii.
- Ligand diagnostic: unique unordered intraligand pairs, excluding self-pairs.
- Binding diagnostic: every ligand-protein atom pair.
- Aggregation: mean frame percentage within each complex, then unweighted mean
  over complexes.

## Results

Values below are percentage points. `Delta` is anchored minus unanchored
epoch-5 NeuralMD; lower is nominally preferable, subject to the bond-graph
limitation above.

| Split | Scenario | Ligand anchored | Ligand delta | Binding anchored | Binding delta |
|---|---|---:|---:|---:|---:|
| Validation | T1 | 4.096643 | +0.187458 | 0.000000 | +0.000000 |
| Validation | T2 | 4.579677 | +0.229778 | 0.016659 | -0.000354 |
| Validation | T3 | 4.097439 | +0.060177 | 0.002724 | +0.001135 |
| Frozen internal test | T1 | 2.781360 | +0.170025 | 0.000000 | +0.000000 |
| Frozen internal test | T2 | 2.506842 | +0.090809 | 0.012592 | -0.000120 |
| Frozen internal test | T3 | 3.032785 | -0.044150 | 0.000588 | -0.000027 |

## Decision

The binding-overlap proxy remains extremely small and does not show a
consistent increase: four of six comparisons are unchanged or lower, while
validation T3 rises by only 0.001135 percentage points.  The ligand proxy,
however, rises in five of six comparisons by 0.060-0.230 points.  Because
bonded pairs are not excluded, this cannot establish that the method creates
steric clashes; it does establish that the geometric improvement is **not yet
sufficient evidence of a Phys improvement**.

The frozen candidate remains the best internal multi-metric candidate, but its
status is now explicitly conditional: it is strong on Geo/Dyn/Stab proxies and
has only a preliminary overlap screen for Phys.  We will not claim improved
physical reasonableness or construct a local total score.  A later Phys gate
must add bond-aware bond-length/angle/stereochemistry checks and, if the
organizer exposes it, the official energy/validity implementation.

## Reproducibility artifacts

- Validation report: `neuralmd_earlystop_anchor1_collision_val.json`
- Frozen test report: `neuralmd_earlystop_anchor1_collision_test.json`
- Metric implementation: `protein_quanta/collision.py`
- Evaluation entry point: `scripts/evaluate_collision_scenarios.py`
- Regression tests: `tests/test_collision_metrics.py`

