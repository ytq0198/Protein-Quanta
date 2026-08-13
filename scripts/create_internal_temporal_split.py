"""Create an auditable train-only development split."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from protein_quanta.internal_split import deterministic_hash_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--development-output", type=Path, required=True)
    parser.add_argument("--holdout-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--holdout-count", type=int, default=16)
    parser.add_argument("--salt", default="protein-quanta-temporal-v1")
    args = parser.parse_args()
    raw = args.source.read_bytes()
    identifiers = [line.strip() for line in raw.decode().splitlines() if line.strip()]
    development, holdout = deterministic_hash_split(
        identifiers, args.holdout_count, args.salt
    )
    for path, values in (
        (args.development_output, development),
        (args.holdout_output, holdout),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(values) + "\n", encoding="utf-8")
    manifest = {
        "status": "deterministic train-only development split",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_count": len(identifiers),
        "development_count": len(development),
        "holdout_count": len(holdout),
        "salt": args.salt,
        "official_validation_accessed": False,
        "official_test_accessed": False,
        "limitation": "hash split is sample-disjoint but not yet protein-homology or ligand-scaffold grouped",
        "development_ids": development,
        "holdout_ids": holdout,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in manifest if not key.endswith("_ids")}, indent=2))


if __name__ == "__main__":
    main()
