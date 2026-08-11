"""Smoke-test the pinned NeuralMD loader, model construction, and one forward pass."""

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import torch

from protein_quanta.misato import load_ligand_trajectory
from protein_quanta.baselines import linear_rollout, static_rollout
from scripts.evaluate_naive_baselines import _evaluate


def _checkpoint_architecture(path: Path):
    """Infer architecture switches that are encoded by state-dict structure."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "binding_model" not in payload:
        raise ValueError("checkpoint must contain a binding_model state dict")
    state_dict = payload["binding_model"]
    radial_means = state_dict.get("ligand_model.radial_emb.means")
    if radial_means is None or radial_means.ndim != 1:
        raise ValueError("checkpoint is missing ligand radial-basis metadata")
    return {
        "frame_net_num_radial": int(radial_means.shape[0]),
        "velocity_refined_value_coefficient": (
            1
            if any(
                key.startswith("refined_velocity_model.")
                for key in state_dict
            )
            else 0
        ),
    }


def _load_checkpoint(model, path: Path):
    """Load the official NeuralMD binding-model weights and record provenance."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "binding_model" not in payload:
        raise ValueError("checkpoint must contain a binding_model state dict")
    incompatibility = model.load_state_dict(payload["binding_model"])
    return {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "missing_keys": list(incompatibility.missing_keys),
        "unexpected_keys": list(incompatibility.unexpected_keys),
    }


def _scenario_rollout_comparison(
    prediction, truth, observed_local_frames, contact_cutoff
):
    """Compare a local scenario rollout after its observed prefix."""
    prediction = np.asarray(prediction)
    truth = np.asarray(truth)
    if prediction.shape != truth.shape:
        raise ValueError("prediction and truth must have matching shapes")
    if truth.ndim != 3 or truth.shape[-1] != 3:
        raise ValueError("rollout trajectories must contain XYZ frames")
    if observed_local_frames < 2:
        raise ValueError("at least two observed frames are required")
    if truth.shape[0] <= observed_local_frames:
        raise ValueError("rollout trajectories must contain target frames")
    history = truth[:observed_local_frames]
    target = truth[observed_local_frames:]
    horizon = target.shape[0]
    return {
        "neuralmd": _evaluate(
            prediction[observed_local_frames:],
            target,
            contact_cutoff=contact_cutoff,
        ),
        "static": _evaluate(
            static_rollout(history, horizon),
            target,
            contact_cutoff=contact_cutoff,
        ),
        "linear": _evaluate(
            linear_rollout(history, horizon),
            target,
            contact_cutoff=contact_cutoff,
        ),
    }


def _rollout_comparison(prediction, truth, contact_cutoff):
    """Compare a full predicted trajectory after two observed frames."""
    return _scenario_rollout_comparison(
        prediction,
        truth,
        observed_local_frames=2,
        contact_cutoff=contact_cutoff,
    )


def _load_official_sample(upstream: Path, h5_path: Path, sample_id: str):
    sys.path.insert(0, str(upstream))
    sys.path.insert(0, str(upstream / "examples"))
    from NeuralMD.datasets.MISATO.dataset_MISATO_semi_flexible import (
        parse_MISATO_data,
    )

    utils = upstream / "NeuralMD" / "datasets" / "MISATO" / "utils"
    with (utils / "atoms_residue_map.pickle").open("rb") as handle:
        residue_map = pickle.load(handle)
    with (utils / "atoms_type_map.pickle").open("rb") as handle:
        protein_atom_map = pickle.load(handle)
    with (utils / "atoms_name_map_for_pdb.pickle").open("rb") as handle:
        residue_atom_map = pickle.load(handle)

    atom_name_map = {
        1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 11: "Na",
        12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl",
        19: "K", 20: "Ca", 34: "Se", 35: "Br", 53: "I",
    }
    periodic_table = pd.read_csv(upstream / "NeuralMD" / "datasets" / "periodic_table.csv")
    mass_map = {
        index: periodic_table.loc[index - 1]["AtomicMass"]
        for index in range(1, 119)
    }

    with h5py.File(h5_path, "r") as handle:
        group = handle[sample_id]
        reference = load_ligand_trajectory(group)
        official = parse_MISATO_data(
            group,
            atom_index2name_dict=atom_name_map,
            atom_num2atom_mass=mass_map,
            residue_index2name_dict=residue_map,
            protein_atom_index2standard_name_dict=protein_atom_map,
            atom_reisdue2standard_atom_name_dict=residue_atom_map,
        )
    return official, reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--sample-id", default="10GS")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--rollout-frames", type=int, default=1)
    parser.add_argument("--ode-step-size", type=float, default=5.0)
    parser.add_argument("--scaling", type=float, default=100.0)
    parser.add_argument("--contact-cutoff", type=float, default=4.5)
    parser.add_argument("--trajectory-output", type=Path)
    args = parser.parse_args()
    if args.rollout_frames <= 0:
        raise ValueError("rollout-frames must be positive")

    data, reference = _load_official_sample(args.upstream, args.h5, args.sample_id)
    official_coordinates = data.ligand_trajectory_pos.transpose(0, 1).numpy()
    preprocessing_max_abs_diff = float(
        np.max(np.abs(official_coordinates - reference.coordinates))
    )

    from models.NeuralMD_Binding01_2nd_ODE import NeuralMD_Binding01

    architecture = (
        _checkpoint_architecture(args.checkpoint)
        if args.checkpoint is not None
        else {
            "frame_net_num_radial": 96,
            "velocity_refined_value_coefficient": 1,
        }
    )
    model_args = SimpleNamespace(
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
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = NeuralMD_Binding01(model_args)
    checkpoint = (
        _load_checkpoint(model, args.checkpoint)
        if args.checkpoint is not None
        else None
    )
    model = model.to(device).eval()
    data = data.to(device)
    ligand_batch = torch.zeros(data.ligand_x.shape[0], dtype=torch.long, device=device)
    residue_batch = torch.zeros(
        data.protein_backbone_residue.shape[0], dtype=torch.long, device=device
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
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        acceleration, output_velocity = model(
            torch.tensor(0.0, device=device),
            (velocity, position),
            condition,
        )

    rollout = None
    if args.rollout_frames > 1:
        if args.rollout_frames > official_coordinates.shape[0]:
            raise ValueError("rollout-frames exceeds available trajectory frames")
        from torchdiffeq import odeint

        time_grid = torch.arange(
            args.rollout_frames, dtype=torch.float32, device=device
        ) / args.scaling
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
        rollout = {
            "frames": args.rollout_frames,
            "observed_frames": 2,
            "evaluated_frames": args.rollout_frames - 2,
            "ode_method": "euler",
            "ode_step_size": args.ode_step_size,
            "scaling": args.scaling,
            "comparison": _rollout_comparison(
                prediction,
                truth,
                contact_cutoff=args.contact_cutoff,
            ),
        }
        if args.trajectory_output is not None:
            args.trajectory_output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.trajectory_output,
                prediction=prediction,
                truth=truth,
                velocity=predicted_velocity.detach().cpu().numpy(),
            )
            rollout["trajectory_output"] = str(args.trajectory_output.resolve())

    report = {
        "sample_id": args.sample_id,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "ligand_heavy_atoms": int(position.shape[0]),
        "trajectory_frames": int(data.ligand_trajectory_pos.shape[1]),
        "protein_backbone_atoms": int(data.protein_pos.shape[0]),
        "preprocessing_max_abs_diff": preprocessing_max_abs_diff,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "model_architecture": architecture,
        "acceleration_shape": list(acceleration.shape),
        "velocity_shape": list(output_velocity.shape),
        "outputs_finite": bool(
            torch.isfinite(acceleration).all() and torch.isfinite(output_velocity).all()
        ),
        "peak_gpu_memory_mib": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else None
        ),
        "checkpoint_loaded": checkpoint is not None,
        "checkpoint": checkpoint,
        "rollout": rollout,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
