"""Join the downloaded full MISATO HDF5 file to the official filtered splits."""

import argparse
import json
from pathlib import Path

from protein_quanta.dataset_acceptance import audit_full_misato_acceptance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--peptides", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-split", type=int, default=16)
    parser.add_argument("--chunk-frames", type=int, default=25)
    parser.add_argument("--observed-md5")
    parser.add_argument("--expected-md5")
    args = parser.parse_args()

    report = audit_full_misato_acceptance(
        args.h5,
        {"train": args.train, "val": args.val, "test": args.test},
        args.peptides,
        samples_per_split=args.samples_per_split,
        chunk_frames=args.chunk_frames,
        observed_md5=args.observed_md5,
        expected_md5=args.expected_md5,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report["checks"], ensure_ascii=False, indent=2))
    if not report["checks"]["all_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
