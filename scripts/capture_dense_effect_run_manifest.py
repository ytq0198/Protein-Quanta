"""Capture immutable file identities for a sealed dense-effect run."""

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": value.hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--training-script", type=Path, required=True)
    parser.add_argument("--evaluation-script", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--development-ids", type=Path, required=True)
    parser.add_argument("--holdout-ids", type=Path, required=True)
    parser.add_argument("--topology-report", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    files = {
        name: digest(path)
        for name, path in (
            ("training_script", args.training_script),
            ("evaluation_script", args.evaluation_script),
            ("config", args.config),
            ("development_ids", args.development_ids),
            ("holdout_ids", args.holdout_ids),
            ("topology_report", args.topology_report),
            ("dataset", args.dataset),
        )
    }
    manifest = {
        "status": "sealed before holdout target evaluation",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "files": files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
