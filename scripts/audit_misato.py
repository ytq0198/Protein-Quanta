"""Write a JSON schema and integrity report for a MISATO MD HDF5 file."""

import argparse
import json
from pathlib import Path

from protein_quanta.misato import audit_misato_h5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--chunk-frames", type=int, default=16)
    args = parser.parse_args()

    report = audit_misato_h5(
        args.input,
        max_samples=args.max_samples,
        chunk_frames=args.chunk_frames,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"audited {report['audited_complex_count']}/{report['complex_count']} "
        f"complexes: {report['valid_complex_count']} valid, "
        f"{report['invalid_complex_count']} invalid"
    )


if __name__ == "__main__":
    main()
