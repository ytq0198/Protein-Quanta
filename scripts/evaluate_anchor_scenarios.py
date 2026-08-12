"""Evaluate a frozen Static-anchor residual on saved competition scenarios."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from protein_quanta.anchoring import (
    anchored_residual_rollout,
    validate_scenario_betas,
)
from scripts.evaluate_naive_baselines import _evaluate
from scripts.evaluate_neuralmd_scenarios import _aggregate_scenarios
from scripts.smoke_neuralmd import _scenario_rollout_comparison


def _anchored_scenario_comparison(
    prediction,
    truth,
    beta,
    decay_scale_frames,
    contact_cutoff,
):
    comparison = _scenario_rollout_comparison(
        prediction,
        truth,
        observed_local_frames=2,
        contact_cutoff=contact_cutoff,
    )
    anchored = anchored_residual_rollout(
        prediction,
        truth[:2],
        beta=beta,
        decay_scale_frames=decay_scale_frames,
    )
    comparison["anchored"] = _evaluate(
        anchored[2:], truth[2:], contact_cutoff=contact_cutoff
    )
    return comparison


def _load_trajectory(path):
    with np.load(path) as payload:
        prediction = np.asarray(payload["prediction"])
        truth = np.asarray(payload["truth"])
    if prediction.shape != truth.shape:
        raise ValueError(f"{path}: prediction and truth shapes differ")
    if prediction.ndim != 3 or prediction.shape[0] < 3 or prediction.shape[-1] != 3:
        raise ValueError(f"{path}: expected (frames, atoms, 3)")
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError(f"{path}: trajectories must be finite")
    return prediction, truth


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_scenario_betas(values):
    beta_by_scenario = {}
    for token in values or []:
        if "=" not in token:
            raise ValueError("scenario beta must use SCENARIO=BETA syntax")
        scenario, raw_beta = token.split("=", 1)
        if not scenario or not raw_beta:
            raise ValueError("scenario name and beta must be non-empty")
        if scenario in beta_by_scenario:
            raise ValueError(f"duplicate scenario beta: {scenario}")
        try:
            beta_by_scenario[scenario] = float(raw_beta)
        except ValueError as error:
            raise ValueError(f"invalid beta for {scenario}: {raw_beta}") from error
    return beta_by_scenario


def _resolve_anchor_policy(scenario_names, beta, scenario_beta_tokens):
    if beta is not None and scenario_beta_tokens:
        raise ValueError("--beta and --scenario-betas are mutually exclusive")
    if scenario_beta_tokens:
        beta_by_scenario = validate_scenario_betas(
            _parse_scenario_betas(scenario_beta_tokens), scenario_names
        )
        policy_type = "scenario-conditioned"
    else:
        scalar_beta = 4.0 if beta is None else float(beta)
        beta_by_scenario = validate_scenario_betas(
            {name: scalar_beta for name in scenario_names}, scenario_names
        )
        policy_type = "scalar"
    return {"type": policy_type, "beta_by_scenario": beta_by_scenario}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--scenario-betas", nargs="+")
    parser.add_argument("--decay-scale-frames", type=float, default=98.0)
    parser.add_argument("--contact-cutoff", type=float, default=4.5)
    parser.add_argument("--split-label", default="validation")
    parser.add_argument("--selection-allowed", action="store_true")
    parser.add_argument("--policy-source", default="unspecified")
    args = parser.parse_args()

    reference = json.loads(args.reference_report.read_text(encoding="utf-8"))
    sample_ids = reference["protocol"]["sample_ids"]
    scenario_records = reference["protocol"]["scenarios"]
    scenario_names = [record["name"] for record in scenario_records]
    anchor_policy = _resolve_anchor_policy(
        scenario_names,
        beta=args.beta,
        scenario_beta_tokens=args.scenario_betas,
    )
    samples = []
    for sample_id in sample_ids:
        scenarios = {}
        for scenario_name in scenario_names:
            path = args.trajectory_dir / f"{sample_id}_{scenario_name}.npz"
            prediction, truth = _load_trajectory(path)
            scenarios[scenario_name] = {
                "comparison": _anchored_scenario_comparison(
                    prediction,
                    truth,
                    beta=anchor_policy["beta_by_scenario"][scenario_name],
                    decay_scale_frames=args.decay_scale_frames,
                    contact_cutoff=args.contact_cutoff,
                )
            }
        samples.append({"sample_id": sample_id, "scenarios": scenarios})

    report = {
        "protocol": {
            "status": "internal combination proxy; not official score",
            "split_label": args.split_label,
            "selection_allowed": args.selection_allowed,
            "anchor_policy": anchor_policy,
            "policy_source": args.policy_source,
            "decay_scale_frames": args.decay_scale_frames,
            "contact_cutoff_angstrom": args.contact_cutoff,
            "sample_ids": sample_ids,
            "sample_count": len(sample_ids),
            "scenarios": scenario_records,
            "trajectory_dir": str(args.trajectory_dir.resolve()),
            "reference_report": str(args.reference_report.resolve()),
            "reference_report_sha256": _sha256(args.reference_report),
        },
        "summary": _aggregate_scenarios(samples),
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
