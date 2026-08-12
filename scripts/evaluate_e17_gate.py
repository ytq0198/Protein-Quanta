"""Apply the frozen E17 validation-only promotion gate."""

import argparse
import json
from pathlib import Path


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _relative(candidate, baseline):
    if baseline <= 0:
        raise ValueError("relative comparison requires a positive baseline")
    return float(candidate / baseline)


def evaluate_gate(baseline_geo, candidate_geo, dynamics, baseline_phys, candidate_phys):
    checks = []

    def add(name, value, threshold, passed, scenario, module):
        checks.append(
            {
                "name": name,
                "scenario": scenario,
                "module": module,
                "value": float(value),
                "threshold": threshold,
                "passed": bool(passed),
            }
        )

    baseline_t1 = baseline_geo["summary"]["T1"]["neuralmd"]
    candidate_t1 = candidate_geo["summary"]["T1"]["neuralmd"]
    ratio = _relative(
        candidate_t1["coordinate_rmse_angstrom"],
        baseline_t1["coordinate_rmse_angstrom"],
    )
    add("coordinate_rmse_ratio", ratio, "<=1.02", ratio <= 1.02, "T1", "Geo")
    ratio = _relative(
        candidate_t1["matching_mean_angstrom"],
        baseline_t1["matching_mean_angstrom"],
    )
    add("matching_ratio", ratio, "<=1.00", ratio <= 1.0, "T1", "Geo")
    point_change = (
        candidate_t1["stability_mean_percent"]
        - baseline_t1["stability_mean_percent"]
    )
    add(
        "stability_point_change",
        point_change,
        ">=0.00",
        point_change >= 0.0,
        "T1",
        "Stab",
    )

    t1_dyn = dynamics["summary"]["T1"]
    baseline_gap = abs(1.0 - t1_dyn["epoch5"]["step_amplitude_ratio"])
    candidate_gap = abs(1.0 - t1_dyn["candidate"]["step_amplitude_ratio"])
    gap_ratio = _relative(candidate_gap, baseline_gap)
    add(
        "step_amplitude_ideal_gap_ratio",
        gap_ratio,
        "<=0.95",
        gap_ratio <= 0.95,
        "T1",
        "Dyn",
    )

    baseline_bond = baseline_phys["summary"]["T1"]["neuralmd"]
    candidate_bond = candidate_phys["summary"]["T1"]["neuralmd"]
    ratio = _relative(
        candidate_bond["bond_length_mae_angstrom"],
        baseline_bond["bond_length_mae_angstrom"],
    )
    add("bond_length_mae_ratio", ratio, "<=1.05", ratio <= 1.05, "T1", "Phys")

    for scenario in ("T1", "T2", "T3"):
        baseline_extreme = baseline_phys["summary"][scenario]["neuralmd"][
            "extreme_bond_event_percent"
        ]
        candidate_extreme = candidate_phys["summary"][scenario]["neuralmd"][
            "extreme_bond_event_percent"
        ]
        change = candidate_extreme - baseline_extreme
        add(
            "extreme_bond_event_point_change",
            change,
            "<=0.10",
            change <= 0.10,
            scenario,
            "Phys",
        )

    for scenario in ("T2", "T3"):
        candidate_metrics = candidate_geo["summary"][scenario]["neuralmd"]
        baseline_metrics = baseline_geo["summary"][scenario]["neuralmd"]
        ratio = _relative(
            candidate_metrics["coordinate_rmse_angstrom"],
            baseline_metrics["coordinate_rmse_angstrom"],
        )
        add(
            "coordinate_rmse_safety_ratio",
            ratio,
            "<=1.05",
            ratio <= 1.05,
            scenario,
            "Geo",
        )
        candidate_dyn = dynamics["summary"][scenario]["candidate"]
        baseline_dyn = dynamics["summary"][scenario]["epoch5"]
        for metric in (
            "rmsf_profile_mae_angstrom",
            "rg_wasserstein_angstrom",
            "pair_distance_wasserstein_angstrom",
            "step_displacement_wasserstein_angstrom",
        ):
            ratio = _relative(candidate_dyn[metric], baseline_dyn[metric])
            add(metric + "_safety_ratio", ratio, "<=1.05", ratio <= 1.05, scenario, "Dyn")
        candidate_gap = abs(1.0 - candidate_dyn["step_amplitude_ratio"])
        baseline_gap = abs(1.0 - baseline_dyn["step_amplitude_ratio"])
        ratio = _relative(candidate_gap, baseline_gap)
        add(
            "step_amplitude_ideal_gap_safety_ratio",
            ratio,
            "<=1.05",
            ratio <= 1.05,
            scenario,
            "Dyn",
        )

    return {
        "status": "E17 validation-only project gate; not official competition score",
        "passed": all(check["passed"] for check in checks),
        "failed_checks": [check["name"] + ":" + check["scenario"] for check in checks if not check["passed"]],
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-geo", type=Path, required=True)
    parser.add_argument("--candidate-geo", type=Path, required=True)
    parser.add_argument("--dynamics", type=Path, required=True)
    parser.add_argument("--baseline-phys", type=Path, required=True)
    parser.add_argument("--candidate-phys", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_gate(
        _load(args.baseline_geo),
        _load(args.candidate_geo),
        _load(args.dynamics),
        _load(args.baseline_phys),
        _load(args.candidate_phys),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
