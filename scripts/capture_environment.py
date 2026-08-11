"""Write a privacy-minimized reproducibility environment snapshot."""

import argparse
import json
from pathlib import Path

from protein_quanta.environment import collect_environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    report = collect_environment()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

