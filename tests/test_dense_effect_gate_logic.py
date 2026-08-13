import copy

from protein_quanta.dense_effect_gate import evaluate_gate


CONFIG = {
    "pass_gate": {
        "paired_seed_wins_at_least": 2,
        "T1_coordinate_rmse_max_worsening_fraction": 0.02,
        "T3_coordinate_rmse_min_improvement_fraction": 0.03,
        "T3_rmse_slope_min_improvement_fraction": 0.05,
        "T3_dynamic_metrics_improved_at_least": 2,
        "bond_length_mae_max_worsening_fraction": 0.05,
        "extreme_bond_event_max_increase_percentage_points": 0.05,
    }
}


def metrics(coordinate, slope, rmsf, rg, contact, bond, extreme):
    return {
        "coordinate_rmse_angstrom": coordinate,
        "rmse_slope_angstrom_per_frame": slope,
        "rmsf_mae_angstrom": rmsf,
        "radius_of_gyration_mae_angstrom": rg,
        "contact_map_agreement_mean": contact,
        "nonfinite_frame_fraction": 0.0,
        "bond_length_mae_angstrom": bond,
        "extreme_bond_event_percent": extreme,
        "dynamics_distribution": {"velocity_autocorrelation_mae": 0.2},
    }


def passing_seed(seed):
    control = {
        "T1": metrics(1.0, 0.10, 1.0, 1.0, 0.90, 1.0, 0.0),
        "T2": metrics(2.0, 0.10, 1.0, 1.0, 0.90, 1.0, 0.0),
        "T3": metrics(3.0, 0.10, 1.0, 1.0, 0.90, 1.0, 0.0),
    }
    candidate = {
        "T1": metrics(1.01, 0.10, 1.0, 1.0, 0.90, 1.04, 0.04),
        "T2": metrics(1.80, 0.09, 0.9, 0.9, 0.91, 1.04, 0.04),
        "T3": metrics(2.70, 0.09, 0.9, 0.9, 0.91, 1.04, 0.04),
    }
    return {"seed": seed, "summary": {"control": control, "candidate": candidate}}


def test_all_preregistered_checks_pass_in_correct_directions():
    result = evaluate_gate(CONFIG, [passing_seed(seed) for seed in (0, 42, 123)])

    assert result["passed"]
    assert all(result["checks"].values())
    assert result["macro_coordinate_rmse"]["improvement_fraction"] > 0
    assert len([row for row in result["paired_seed_macro"] if row["candidate_improves"]]) == 3


def test_equal_candidate_fails_effect_checks_without_failing_safety():
    rows = [passing_seed(seed) for seed in (0, 42, 123)]
    for row in rows:
        row["summary"]["candidate"] = copy.deepcopy(row["summary"]["control"])

    result = evaluate_gate(CONFIG, rows)

    assert not result["passed"]
    assert not result["checks"]["macro_coordinate_rmse_improves"]
    assert not result["checks"]["paired_seed_wins_at_least"]
    assert not result["checks"]["T3_coordinate_rmse_min_improvement_fraction"]
    assert result["checks"]["bond_length_mae_max_worsening_fraction"]
    assert result["checks"]["all_rollouts_finite"]
