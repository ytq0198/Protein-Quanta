"""Create a compact, auditable summary from raw dense-effect artifacts."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    summary = {
        "status": report["status"],
        "report_sha256": sha256(args.report),
        "gate": report["gate"],
        "training": {},
    }
    for seed in (0, 42, 123):
        path = args.training_dir / f"dense-equivariant-effect-seed-{seed}.json"
        training = json.loads(path.read_text(encoding="utf-8"))
        summary["training"][str(seed)] = {
            "manifest_sha256": sha256(path),
            "status": training["status"],
            "epochs": {
                arm: len(training["training"][arm]["epochs"])
                for arm in ("control", "candidate")
            },
            "clipped": {
                arm: training["training"][arm]["clipped"]
                for arm in ("control", "candidate")
            },
            "nonfinite": {
                arm: training["training"][arm]["nonfinite"]
                for arm in ("control", "candidate")
            },
            "parameter_l2": training["parameter_l2"],
            "checkpoint_sha256": training["checkpoint_sha256"],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
