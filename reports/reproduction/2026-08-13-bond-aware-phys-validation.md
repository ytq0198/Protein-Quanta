# E13-E15: bond-aware Phys audit and validation no-go

## Question and pre-registered rule

The direction-two guide assigns 25% of every T1/T2/T3 scenario score to
physical validity. The current `epoch 5 + beta=1` candidate therefore cannot
be promoted from proxy status unless its geometric gains survive a traceable
bond-aware check.

Before the result was inspected, `docs/scoring-aligned-execution-plan.md`
fixed the minimum gate: no scenario may worsen bond-length error by more than
5% relative to the unanchored epoch-5 trajectory; new extreme bond events are
not allowed. Validation failure stops the experiment and does not authorize
inspection of a new Phys test result.

## E13 topology audit

The MISATO-100 HDF5 file contains atom types and trajectories but no explicit
bond graph. MISATO publishes a 54.3 GB AMBER restart/topology archive on
Zenodo, but the public endpoint was too slow for the preliminary deadline.
We therefore used the corresponding RCSB PDB records as a smaller, traceable
source and accepted a ligand topology only when all of the following held:

1. the PDB heavy-element sequence exactly matched MISATO atom order;
2. pairwise-distance MAE between PDB coordinates and MISATO frame 0 was at
   most 2.0 A, validating atom order without using distance to infer bonds;
3. every heavy atom was covered by explicit PDB `CONECT` bonds;
4. multiple crystallographic copies were accepted only when their indexed
   topologies were identical.

Coverage was 9/10 validation complexes and 8/10 frozen-internal-test
complexes. Validation mapping distance MAE ranged from 0.082 to 0.762 A. The
strictly rejected validation sample was 2FMB, an alternate-conformation
ligand; rejected test samples were 3FT2 and 3F5J, which require peptide-like
or multi-residue topology handling. They were not filled with guessed bonds.

## E14 metrics

On the topology-covered validation subset we measured same-frame bond-length
MAE, relative MAE, the percentage of bonds above 20% relative error, extreme
events below 0.5x or above 1.5x the reference length, and conservative
nonbonded clashes after excluding covalent graph distance one and two.

These are project Phys diagnostics, not the organizer's normalized Phys
score. They do not yet cover bond angles, stereochemistry, or energy.

## E15 validation decision

| Scenario | Epoch-5 bond MAE | Anchored bond MAE | Relative change | Gate |
|---|---:|---:|---:|---|
| T1 | 0.040269 A | 0.043889 A | +8.99% | fail |
| T2 | 0.042463 A | 0.048278 A | +13.69% | fail |
| T3 | 0.044818 A | 0.044070 A | -1.67% | pass |

The anchored candidate also introduced non-zero extreme-event rates in T1
and T2 where epoch-5 NeuralMD had zero, although the absolute rates were small
(0.0398% and 0.0895%). Conservative nonbonded clashes were zero in T1/T3 and
0.00279% in anchored T2 versus zero in NeuralMD.

**Decision: validation no-go.** `beta=1` violates the pre-registered 5%
bond-length guard in the two highest-weight scenarios. No new bond-aware
frozen-test evaluation is run, and beta is not retuned. The prior
`configs/frozen_candidate.json` remains an immutable historical result, but it
is demoted from the active direction-two candidate. The active safe baseline
returns to unanchored seed-42 epoch 5 pending a direct Phys/Dyn comparison
against published NeuralMD.

## Scientific interpretation

Anchoring is not merely a harmless variance reduction. It linearly blends
Cartesian coordinates from two conformations; even when both endpoints look
reasonable, the interpolation can distort local covalent geometry. The
opposite T3 direction shows that the effect depends on prediction horizon,
but the T1/T2 failures are enough to reject a global anchor.

The next innovation must preserve internal coordinates explicitly rather than
obtain stability by Cartesian contraction toward a static structure. This
connects the score and research question: long-rollout risk control is useful
only when local molecular validity is a hard constraint.

## Reproducibility and limitations

- Topology implementation: `protein_quanta/topology.py`
- Phys implementation: `protein_quanta/collision.py`
- Topology audit: `scripts/audit_ligand_topologies.py`
- Phys evaluation: `scripts/evaluate_bond_aware_phys.py`
- Machine-readable topology reports: `topology_val.json`, `topology_test.json`
- Server Phys output: kept outside Git under
  `/mnt/localDisk3/weizian/runs/protein-quanta/phys/bond_aware_val.json`
- Local regression: 107 tests passed.
- Server focused topology/Phys tests passed; the unrelated full server suite
  exposed a pre-existing missing `mpmath` dependency through SymPy/PyTorch.
- Coverage is incomplete and no official Phys evaluator is available. The
  no-go is valid for the covered validation subset and pre-registered guard;
  it is not an estimate of the official Phys score.
