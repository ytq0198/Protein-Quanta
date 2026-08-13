"""Pure aggregation and pass/fail logic for the dense equivariant effect gate."""

import numpy as np


def mean_summaries(seed_results, arm):
    result = {}
    for scenario in ("T1", "T2", "T3"):
        rows = [row["summary"][arm][scenario] for row in seed_results]
        result[scenario] = {}
        for key, value in rows[0].items():
            if key == "dynamics_distribution":
                result[scenario][key] = {
                    nested: float(np.mean([row[key][nested] for row in rows]))
                    for nested in value
                }
            else:
                result[scenario][key] = float(np.mean([row[key] for row in rows]))
    return result


def improvement_fraction(control, candidate):
    denominator = max(abs(control), np.finfo(float).eps)
    return float((control - candidate) / denominator)


def evaluate_gate(config, seed_results):
    control = mean_summaries(seed_results, "control")
    candidate = mean_summaries(seed_results, "candidate")
    scenarios = ("T1", "T2", "T3")
    macro_control = float(np.mean([
        control[name]["coordinate_rmse_angstrom"] for name in scenarios
    ]))
    macro_candidate = float(np.mean([
        candidate[name]["coordinate_rmse_angstrom"] for name in scenarios
    ]))
    seed_macro = []
    for row in seed_results:
        left = float(np.mean([
            row["summary"]["control"][name]["coordinate_rmse_angstrom"]
            for name in scenarios
        ]))
        right = float(np.mean([
            row["summary"]["candidate"][name]["coordinate_rmse_angstrom"]
            for name in scenarios
        ]))
        seed_macro.append({
            "seed": row["seed"],
            "control": left,
            "candidate": right,
            "candidate_minus_control": right - left,
            "candidate_improves": right < left,
        })

    dynamic_checks = {
        "rmsf_mae_angstrom": (
            candidate["T3"]["rmsf_mae_angstrom"]
            < control["T3"]["rmsf_mae_angstrom"]
        ),
        "radius_of_gyration_mae_angstrom": (
            candidate["T3"]["radius_of_gyration_mae_angstrom"]
            < control["T3"]["radius_of_gyration_mae_angstrom"]
        ),
        "contact_map_agreement_mean": (
            candidate["T3"]["contact_map_agreement_mean"]
            > control["T3"]["contact_map_agreement_mean"]
        ),
    }
    all_phys_control = np.mean([
        control[name]["bond_length_mae_angstrom"] for name in scenarios
    ])
    all_phys_candidate = np.mean([
        candidate[name]["bond_length_mae_angstrom"] for name in scenarios
    ])
    all_extreme_control = np.mean([
        control[name]["extreme_bond_event_percent"] for name in scenarios
    ])
    all_extreme_candidate = np.mean([
        candidate[name]["extreme_bond_event_percent"] for name in scenarios
    ])
    thresholds = config["pass_gate"]
    checks = {
        "macro_coordinate_rmse_improves": macro_candidate < macro_control,
        "paired_seed_wins_at_least": (
            sum(row["candidate_improves"] for row in seed_macro)
            >= thresholds["paired_seed_wins_at_least"]
        ),
        "T1_coordinate_rmse_max_worsening_fraction": (
            (candidate["T1"]["coordinate_rmse_angstrom"]
             - control["T1"]["coordinate_rmse_angstrom"])
            / max(control["T1"]["coordinate_rmse_angstrom"], np.finfo(float).eps)
            <= thresholds["T1_coordinate_rmse_max_worsening_fraction"]
        ),
        "T3_coordinate_rmse_min_improvement_fraction": (
            improvement_fraction(
                control["T3"]["coordinate_rmse_angstrom"],
                candidate["T3"]["coordinate_rmse_angstrom"],
            ) >= thresholds["T3_coordinate_rmse_min_improvement_fraction"]
        ),
        "T3_rmse_slope_min_improvement_fraction": (
            improvement_fraction(
                control["T3"]["rmse_slope_angstrom_per_frame"],
                candidate["T3"]["rmse_slope_angstrom_per_frame"],
            ) >= thresholds["T3_rmse_slope_min_improvement_fraction"]
        ),
        "T3_dynamic_metrics_improved_at_least": (
            sum(dynamic_checks.values())
            >= thresholds["T3_dynamic_metrics_improved_at_least"]
        ),
        "bond_length_mae_max_worsening_fraction": (
            (all_phys_candidate - all_phys_control)
            / max(all_phys_control, np.finfo(float).eps)
            <= thresholds["bond_length_mae_max_worsening_fraction"]
        ),
        "extreme_bond_event_max_increase_percentage_points": (
            all_extreme_candidate - all_extreme_control
            <= thresholds["extreme_bond_event_max_increase_percentage_points"]
        ),
        "all_rollouts_finite": all(
            candidate[name]["nonfinite_frame_fraction"] == 0.0
            and control[name]["nonfinite_frame_fraction"] == 0.0
            for name in scenarios
        ),
    }
    deltas = [row["candidate_minus_control"] for row in seed_macro]
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "dynamic_checks": dynamic_checks,
        "mean_summary": {"control": control, "candidate": candidate},
        "macro_coordinate_rmse": {
            "control": macro_control,
            "candidate": macro_candidate,
            "improvement_fraction": improvement_fraction(
                macro_control, macro_candidate
            ),
        },
        "paired_seed_macro": seed_macro,
        "paired_candidate_minus_control_mean": float(np.mean(deltas)),
        "paired_candidate_minus_control_sd": float(np.std(deltas, ddof=1)),
        "bond_length_mae_macro": {
            "control": float(all_phys_control),
            "candidate": float(all_phys_candidate),
        },
        "extreme_bond_event_macro": {
            "control": float(all_extreme_control),
            "candidate": float(all_extreme_candidate),
        },
    }
