from scripts.plot_dense_equivariant_effect_gate import load_plot_rows


def test_load_plot_rows_uses_frozen_seed_and_scenario_values():
    report = {
        "gate": {
            "paired_seed_macro": [
                {"seed": 0, "control": 2.0, "candidate": 1.8},
                {"seed": 42, "control": 2.1, "candidate": 2.0},
            ],
            "mean_summary": {
                arm: {
                    scenario: {
                        "coordinate_rmse_angstrom": base + offset,
                        "rmse_slope_angstrom_per_frame": 0.1 + offset,
                    }
                    for scenario, offset in (("T1", 0.0), ("T2", 1.0), ("T3", 2.0))
                }
                for arm, base in (("control", 1.0), ("candidate", 0.9))
            },
        }
    }

    paired, scenarios = load_plot_rows(report)

    assert paired[0]["candidate"] == 1.8
    assert [row["scenario"] for row in scenarios] == ["T1", "T2", "T3"]
    assert scenarios[2]["control_coordinate_rmse"] == 3.0
    assert scenarios[2]["candidate_coordinate_rmse"] == 2.9
