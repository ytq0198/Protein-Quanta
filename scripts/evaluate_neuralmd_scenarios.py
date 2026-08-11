"""Evaluate NeuralMD on the three competition-aligned trajectory scenarios."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from protein_quanta.scenarios import competition_scenarios, scenario_time_grid
from scripts.evaluate_neuralmd_checkpoint import (
    _aggregate_comparisons,
    _model_arguments,
    _read_split,
)
from scripts.smoke_neuralmd import (
    _checkpoint_architecture,
    _load_checkpoint,
    _load_official_sample,
    _scenario_rollout_comparison,
)


REPORT_STATUS = "competition-aligned proxy; not official score"


def _validate_sample_ids(sample_ids):
    sample_ids = list(sample_ids)
    if not sample_ids:
        raise ValueError("at least one sample ID is required")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample IDs contain duplicates")
    return sample_ids


def _aggregate_scenarios(samples):
    """Aggregate scalar metrics separately for every scenario and model."""
    if not samples:
        raise ValueError("at least one sample is required")
    scenario_names = samples[0]["scenarios"].keys()
    return {
        scenario_name: _aggregate_comparisons(
            [
                {"comparison": sample["scenarios"][scenario_name]["comparison"]}
                for sample in samples
            ]
        )
        for scenario_name in scenario_names
    }


def _upstream_commit(upstream):
    try:
        return subprocess.check_output(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _scenario_record(scenario):
    return {
        "name": scenario.name,
        "observed_frames_zero_based_inclusive": [
            scenario.observed_start,
            scenario.observed_end,
        ],
        "initializer_frames_zero_based": list(scenario.initializer_indices),
        "target_frames_zero_based_inclusive": [
            scenario.target_start,
            scenario.target_end,
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--ode-step-size", type=float, default=5.0)
    parser.add_argument("--scaling", type=float, default=100.0)
    parser.add_argument("--contact-cutoff", type=float, default=4.5)
    parser.add_argument("--trajectory-dir", type=Path)
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("max-samples must be positive")

    started = perf_counter()
    sample_ids = _validate_sample_ids(_read_split(args.split))
    if args.max_samples is not None:
        sample_ids = sample_ids[: args.max_samples]
    scenarios = competition_scenarios()
    architecture = _checkpoint_architecture(args.checkpoint)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    sys.path.insert(0, str(args.upstream))
    sys.path.insert(0, str(args.upstream / "examples"))
    from models.NeuralMD_Binding01_2nd_ODE import NeuralMD_Binding01
    from torchdiffeq import odeint

    model = NeuralMD_Binding01(_model_arguments(architecture))
    checkpoint = _load_checkpoint(model, args.checkpoint)
    model = model.to(device).eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    samples = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for index, sample_id in enumerate(sample_ids, start=1):
        sample_started = perf_counter()
        data, reference = _load_official_sample(args.upstream, args.h5, sample_id)
        official_coordinates = data.ligand_trajectory_pos.transpose(0, 1).numpy()
        if official_coordinates.shape[0] != 100:
            raise ValueError(f"{sample_id}: expected exactly 100 trajectory frames")
        preprocessing_max_abs_diff = float(
            np.max(np.abs(official_coordinates - reference.coordinates))
        )

        data = data.to(device)
        ligand_batch = torch.zeros(
            data.ligand_x.shape[0], dtype=torch.long, device=device
        )
        residue_batch = torch.zeros(
            data.protein_backbone_residue.shape[0], dtype=torch.long, device=device
        )
        condition = (
            data.ligand_x,
            ligand_batch,
            data.ligand_mass,
            data.protein_pos[data.mask_n],
            data.protein_pos[data.mask_ca],
            data.protein_pos[data.mask_c],
            data.protein_backbone_residue,
            residue_batch,
        )
        scenario_results = {}
        for scenario in scenarios:
            first_frame, second_frame = scenario.initializer_indices
            position = data.ligand_trajectory_pos[:, first_frame, :]
            velocity = data.ligand_trajectory_pos[:, second_frame, :] - position
            time_grid = torch.as_tensor(
                scenario_time_grid(scenario, args.scaling),
                device=device,
            )
            with torch.no_grad():
                predicted_velocity, predicted_position = odeint(
                    model,
                    (velocity, position),
                    time_grid,
                    condition=condition,
                    method="euler",
                    options={"step_size": args.ode_step_size / args.scaling},
                )
            prediction = predicted_position.detach().cpu().numpy()
            truth = official_coordinates[first_frame : scenario.target_end + 1]
            if not np.isfinite(prediction).all():
                raise FloatingPointError(
                    f"{sample_id}/{scenario.name}: non-finite prediction"
                )
            scenario_result = {
                **_scenario_record(scenario),
                "local_rollout_frames": int(prediction.shape[0]),
                "evaluated_target_frames": len(scenario.target_indices),
                "comparison": _scenario_rollout_comparison(
                    prediction,
                    truth,
                    observed_local_frames=2,
                    contact_cutoff=args.contact_cutoff,
                ),
            }
            if args.trajectory_dir is not None:
                args.trajectory_dir.mkdir(parents=True, exist_ok=True)
                trajectory_path = (
                    args.trajectory_dir / f"{sample_id}_{scenario.name}.npz"
                )
                np.savez_compressed(
                    trajectory_path,
                    prediction=prediction,
                    truth=truth,
                    velocity=predicted_velocity.detach().cpu().numpy(),
                )
                scenario_result["trajectory_output"] = str(
                    trajectory_path.resolve()
                )
            scenario_results[scenario.name] = scenario_result

        samples.append(
            {
                "sample_id": sample_id,
                "ligand_heavy_atoms": int(data.ligand_x.shape[0]),
                "protein_backbone_atoms": int(data.protein_pos.shape[0]),
                "preprocessing_max_abs_diff_angstrom": preprocessing_max_abs_diff,
                "runtime_seconds": perf_counter() - sample_started,
                "scenarios": scenario_results,
            }
        )
        print(f"[{index}/{len(sample_ids)}] {sample_id} complete", flush=True)

    report = {
        "protocol": {
            "h5": str(args.h5.resolve()),
            "split": str(args.split.resolve()),
            "sample_ids": sample_ids,
            "sample_count": len(sample_ids),
            "scenarios": [_scenario_record(scenario) for scenario in scenarios],
            "ode_method": "euler",
            "ode_step_size": args.ode_step_size,
            "scaling": args.scaling,
            "aggregation": "unweighted mean of per-complex scalar metrics",
            "contact_cutoff_angstrom": args.contact_cutoff,
            "status": REPORT_STATUS,
        },
        "runtime": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch": torch.__version__,
            "parameter_count": parameter_count,
            "peak_gpu_memory_mib": (
                torch.cuda.max_memory_allocated(device) / 1024**2
                if device.type == "cuda"
                else None
            ),
            "total_seconds": perf_counter() - started,
        },
        "upstream": {
            "path": str(args.upstream.resolve()),
            "commit": _upstream_commit(args.upstream),
        },
        "checkpoint": checkpoint,
        "model_architecture": architecture,
        "summary": _aggregate_scenarios(samples),
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
