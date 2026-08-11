"""Scan a predeclared Static-anchor decay on saved NeuralMD trajectories."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from protein_quanta.anchoring import anchored_residual_rollout
from scripts.evaluate_naive_baselines import _evaluate


SCALAR_METRICS = (
    "coordinate_mae_angstrom",
    "coordinate_rmse_angstrom",
    "matching_mean_angstrom",
    "stability_mean_percent",
    "aligned_rmsd_mean_angstrom",
    "radius_of_gyration_mae_angstrom",
    "rmsf_mae_angstrom",
    "contact_map_agreement_mean",
)


def _reference_candidate(candidates):
    references = [candidate for candidate in candidates if candidate["beta"] == 0]
    if len(references) != 1:
        raise ValueError("candidates must contain exactly one beta 0 reference")
    return references[0]


def _is_eligible(
    candidate,
    reference,
    rmsf_tolerance=1.05,
    coordinate_tolerance=1.02,
):
    summary = candidate["summary"]
    baseline = reference["summary"]
    return bool(
        summary["rmsf_mae_angstrom"]
        <= rmsf_tolerance * baseline["rmsf_mae_angstrom"]
        and summary["coordinate_rmse_angstrom"]
        <= coordinate_tolerance * baseline["coordinate_rmse_angstrom"]
    )


def select_beta(
    candidates,
    rmsf_tolerance=1.05,
    coordinate_tolerance=1.02,
):
    """Select beta using validation Stability with an anti-collapse gate."""
    reference = _reference_candidate(candidates)
    eligible = [
        candidate
        for candidate in candidates
        if _is_eligible(
            candidate,
            reference,
            rmsf_tolerance=rmsf_tolerance,
            coordinate_tolerance=coordinate_tolerance,
        )
    ]
    selected = max(
        eligible,
        key=lambda candidate: (
            candidate["summary"]["stability_mean_percent"],
            -candidate["summary"]["matching_mean_angstrom"],
            -candidate["beta"],
        ),
    )
    return float(selected["beta"])


def resolve_selected_beta(candidates, selection_allowed, **selection_kwargs):
    """Select on validation or preserve the sole frozen nonzero test beta."""
    _reference_candidate(candidates)
    if selection_allowed:
        return select_beta(candidates, **selection_kwargs)
    frozen = [candidate for candidate in candidates if candidate["beta"] != 0]
    if len(frozen) != 1:
        raise ValueError("test evaluation requires exactly one frozen nonzero beta")
    return float(frozen[0]["beta"])


def _load_trajectory(path):
    with np.load(path) as payload:
        if "prediction" not in payload or "truth" not in payload:
            raise ValueError(f"{path} must contain prediction and truth")
        prediction = np.asarray(payload["prediction"])
        truth = np.asarray(payload["truth"])
    if prediction.shape != truth.shape:
        raise ValueError(f"{path} prediction and truth shapes differ")
    if prediction.ndim != 3 or prediction.shape[0] < 3 or prediction.shape[-1] != 3:
        raise ValueError(f"{path} trajectories must have shape (frames, atoms, 3)")
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError(f"{path} trajectories must contain finite values")
    return prediction, truth


def _evaluate_candidate(
    trajectory_dir,
    sample_ids,
    beta,
    decay_scale_frames,
    contact_cutoff,
):
    samples = []
    for sample_id in sample_ids:
        prediction, truth = _load_trajectory(
            Path(trajectory_dir) / f"{sample_id}.npz"
        )
        anchored = anchored_residual_rollout(
            prediction,
            truth[:2],
            beta=beta,
            decay_scale_frames=decay_scale_frames,
        )
        metrics = _evaluate(
            anchored[2:], truth[2:], contact_cutoff=contact_cutoff
        )
        samples.append({"sample_id": sample_id, **metrics})
    summary = {
        metric: float(np.mean([sample[metric] for sample in samples]))
        for metric in SCALAR_METRICS
    }
    return {"beta": float(beta), "summary": summary, "samples": samples}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-label", required=True)
    parser.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=[0, 0.25, 0.5, 1, 2, 4, 8],
    )
    parser.add_argument("--decay-scale-frames", type=float, default=98)
    parser.add_argument("--contact-cutoff", type=float, default=4.5)
    parser.add_argument("--rmsf-tolerance", type=float, default=1.05)
    parser.add_argument("--coordinate-tolerance", type=float, default=1.02)
    args = parser.parse_args()

    if args.split_label.lower() == "test" and len(args.betas) > 2:
        raise ValueError("test evaluation must not scan multiple beta values")
    reference_report = json.loads(
        args.reference_report.read_text(encoding="utf-8")
    )
    sample_ids = reference_report["protocol"]["sample_ids"]
    candidates = [
        _evaluate_candidate(
            args.trajectory_dir,
            sample_ids,
            beta,
            args.decay_scale_frames,
            args.contact_cutoff,
        )
        for beta in args.betas
    ]
    reference = _reference_candidate(candidates)
    for candidate in candidates:
        candidate["eligible"] = _is_eligible(
            candidate,
            reference,
            rmsf_tolerance=args.rmsf_tolerance,
            coordinate_tolerance=args.coordinate_tolerance,
        )
    selection_allowed = args.split_label.lower() != "test"
    selected_beta = resolve_selected_beta(
        candidates,
        selection_allowed=selection_allowed,
        rmsf_tolerance=args.rmsf_tolerance,
        coordinate_tolerance=args.coordinate_tolerance,
    )
    report = {
        "protocol": {
            "status": "reproduction proxy; not an official competition score",
            "split_label": args.split_label,
            "selection_allowed": selection_allowed,
            "trajectory_dir": str(args.trajectory_dir.resolve()),
            "reference_report": str(args.reference_report.resolve()),
            "reference_report_sha256": _sha256(args.reference_report),
            "sample_ids": sample_ids,
            "sample_count": len(sample_ids),
            "observed_frames": 2,
            "decay_scale_frames": args.decay_scale_frames,
            "contact_cutoff_angstrom": args.contact_cutoff,
            "rmsf_tolerance": args.rmsf_tolerance,
            "coordinate_tolerance": args.coordinate_tolerance,
            "selection_rule": (
                "maximum Stability among eligible validation candidates; "
                "tie-break by lower Matching then lower beta"
                if selection_allowed
                else "frozen nonzero beta supplied from validation; no test selection"
            ),
        },
        "selected_beta": selected_beta,
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "selected_beta": selected_beta,
        "summary": next(
            candidate["summary"]
            for candidate in candidates
            if candidate["beta"] == selected_beta
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
