"""Preregistered validation gate for the E6 pair-distance experiment."""


def evaluate_e6_gate(
    baseline_summary,
    candidate_summary,
    collision_relative_regression,
    max_collision_relative_regression=0.02,
):
    baseline = {
        scenario: baseline_summary[scenario]["neuralmd"]
        for scenario in ("T1", "T2", "T3")
    }
    candidate = {
        scenario: candidate_summary[scenario]["neuralmd"]
        for scenario in ("T1", "T2", "T3")
    }
    deltas = {
        "t1_rmse_ratio": candidate["T1"]["coordinate_rmse_angstrom"]
        / baseline["T1"]["coordinate_rmse_angstrom"],
        "t2_stability_points": candidate["T2"]["stability_mean_percent"]
        - baseline["T2"]["stability_mean_percent"],
        "t2_matching_ratio": candidate["T2"]["matching_mean_angstrom"]
        / baseline["T2"]["matching_mean_angstrom"],
        "t3_stability_points": candidate["T3"]["stability_mean_percent"]
        - baseline["T3"]["stability_mean_percent"],
        "t3_matching_ratio": candidate["T3"]["matching_mean_angstrom"]
        / baseline["T3"]["matching_mean_angstrom"],
        "t3_rmsf_ratio": candidate["T3"]["rmsf_mae_angstrom"]
        / baseline["T3"]["rmsf_mae_angstrom"],
        "collision_relative_regression": float(collision_relative_regression),
    }
    tolerance = 1e-12
    checks = {
        "t1_coordinate_guard": deltas["t1_rmse_ratio"] <= 1.02 + tolerance,
        "t2_stability_guard": deltas["t2_stability_points"] >= -1.0 - tolerance,
        "t2_matching_guard": deltas["t2_matching_ratio"] <= 1.02 + tolerance,
        "t3_geometry_improvement": (
            deltas["t3_stability_points"] >= 2.0 - tolerance
            or deltas["t3_matching_ratio"] <= 0.95 + tolerance
        ),
        "t3_dynamics_guard": deltas["t3_rmsf_ratio"] <= 1.05 + tolerance,
        "collision_regression": (
            collision_relative_regression
            <= max_collision_relative_regression + tolerance
        ),
    }
    return {
        "status": "validation-only preregistered proxy gate",
        "passed": all(checks.values()),
        "thresholds": {
            "t1_rmse_max_ratio": 1.02,
            "t2_stability_min_point_change": -1.0,
            "t2_matching_max_ratio": 1.02,
            "t3_stability_min_point_improvement": 2.0,
            "t3_matching_max_ratio": 0.95,
            "t3_rmsf_max_ratio": 1.05,
            "max_collision_relative_regression": max_collision_relative_regression,
        },
        "deltas": deltas,
        "checks": checks,
    }
