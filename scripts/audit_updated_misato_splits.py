"""Write a traceable audit of the updated competition split protocol."""

import argparse
import json
from pathlib import Path

from protein_quanta.split_audit import audit_filtered_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--peptides", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--misato-source-commit")
    parser.add_argument("--neuralmd-source-commit")
    args = parser.parse_args()

    report = audit_filtered_splits(
        {"train": args.train, "val": args.val, "test": args.test},
        args.peptides,
    )
    report["status"] = (
        "public MISATO split audit after NeuralMD peptide exclusion; "
        "no trajectory data or hidden evaluation data accessed"
    )
    report["source_revisions"] = {
        "misato_dataset": args.misato_source_commit,
        "neuralmd": args.neuralmd_source_commit,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report["checks"], ensure_ascii=False))


if __name__ == "__main__":
    main()
