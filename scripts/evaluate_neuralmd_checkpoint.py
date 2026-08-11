"""Evaluate an official NeuralMD checkpoint on a named MISATO split."""

import argparse
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch

from scripts.smoke_neuralmd import (
    _checkpoint_architecture,
    _load_checkpoint,
    _load_official_sample,
    _rollout_comparison,
)


def _aggregate_comparisons(samples):
    """Compute unweighted means of scalar metrics across complexes."""
    if not samples:
        raise ValueError("at least one sample is required")
    model_names = samples[0]["comparison"].keys()
    summary = {}
    for model_name in model_names:
        first = samples[0]["comparison"][model_name]
        scalar_metrics = [
            key
            for key, value in first.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        summary[model_name] = {
            metric: float(
                np.mean(
                    [sample["comparison"][model_name][metric] for sample in samples]
                )
            )
            for metric in scalar_metrics
        }
    return summary


def _model_arguments(architecture):
    return SimpleNamespace(
        model_3d_ligand="FrameNet01",
        model_3d_protein="FrameNetProtein03",
        emb_dim=128,
        NeuralMD_velocity_refined_value_coefficient=architecture[
            "velocity_refined_value_coefficient"
        ],
        use_MLP_velocity=False,
        FrameNet_cutoff=5.0,
        FrameNet_num_layers=4,
        FrameNet_complex_layer=1,
        FrameNet_num_radial=architecture["frame_net_num_radial"],
        FrameNet_rbf_type="RBF_repredding_01",
        FrameNet_gamma=None,
        FrameNet_readout="mean",
    )


def _read_split(path):
    sample_ids = [line.strip() for line in Path(path).read_text().splitlines()]
    sample_ids = [sample_id for sample_id in sample_ids if sample_id]
    if not sample_ids:
        raise ValueError("split file contains no sample IDs")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("split file contains duplicate sample IDs")
    return sample_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rollout-frames", type=int, default=100)
    parser.add_argument("--ode-step-size", type=float, default=5.0)
    parser.add_argument("--scaling", type=float, default=100.0)
    parser.add_argument("--contact-cutoff", type=float, default=4.5)
    parser.add_argument("--trajectory-dir", type=Path)
    args = parser.parse_args()
    if args.rollout_frames < 3:
        raise ValueError("rollout-frames must be at least 3")

    started = perf_counter()
    sample_ids = _read_split(args.split)
    architecture = _checkpoint_architecture(args.checkpoint)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    import sys

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
        if args.rollout_frames > official_coordinates.shape[0]:
            raise ValueError(
                f"{sample_id}: rollout-frames exceeds available trajectory frames"
            )
        preprocessing_max_abs_diff = float(
            np.max(np.abs(official_coordinates - reference.coordinates))
        )

        data = data.to(device)
        ligand_batch = torch.zeros(
            data.ligand_x.shape[0], dtype=torch.long, device=device
        )
        residue_batch = torch.zeros(
            data.protein_backbone_residue.shape[0],
            dtype=torch.long,
            device=device,
        )
        position = data.ligand_trajectory_pos[:, 0, :]
        velocity = data.ligand_trajectory_pos[:, 1, :] - position
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
        time_grid = (
            torch.arange(args.rollout_frames, dtype=torch.float32, device=device)
            / args.scaling
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
        truth = official_coordinates[: args.rollout_frames]
        row = {
            "sample_id": sample_id,
            "ligand_heavy_atoms": int(position.shape[0]),
            "protein_backbone_atoms": int(data.protein_pos.shape[0]),
            "preprocessing_max_abs_diff_angstrom": preprocessing_max_abs_diff,
            "runtime_seconds": perf_counter() - sample_started,
            "comparison": _rollout_comparison(
                prediction, truth, contact_cutoff=args.contact_cutoff
            ),
        }
        if args.trajectory_dir is not None:
            args.trajectory_dir.mkdir(parents=True, exist_ok=True)
            trajectory_path = args.trajectory_dir / f"{sample_id}.npz"
            np.savez_compressed(
                trajectory_path,
                prediction=prediction,
                truth=truth,
                velocity=predicted_velocity.detach().cpu().numpy(),
            )
            row["trajectory_output"] = str(trajectory_path.resolve())
        samples.append(row)
        print(f"[{index}/{len(sample_ids)}] {sample_id} complete", flush=True)

    report = {
        "protocol": {
            "h5": str(args.h5.resolve()),
            "split": str(args.split.resolve()),
            "sample_ids": sample_ids,
            "sample_count": len(sample_ids),
            "observed_frames": 2,
            "rollout_frames": args.rollout_frames,
            "evaluated_frames": args.rollout_frames - 2,
            "ode_method": "euler",
            "ode_step_size": args.ode_step_size,
            "scaling": args.scaling,
            "aggregation": "unweighted mean of per-complex metrics",
            "contact_cutoff_angstrom": args.contact_cutoff,
            "status": "reproduction proxy; not an official competition score",
        },
        "runtime": {
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "torch": torch.__version__,
            "parameter_count": parameter_count,
            "peak_gpu_memory_mib": (
                torch.cuda.max_memory_allocated(device) / 1024**2
                if device.type == "cuda"
                else None
            ),
            "total_seconds": perf_counter() - started,
        },
        "checkpoint": checkpoint,
        "model_architecture": architecture,
        "summary": _aggregate_comparisons(samples),
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
