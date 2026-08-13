"""Reaggregate fixed-epoch temporal results without unpublished task weights."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_temporal_multiseed_rope import aggregate_results


def summarize(source):
    aggregate, promotion = aggregate_results(source["results"])
    return {
        "status": "updated-guide equal-macro reaggregation; not official competition scores",
        "protocol": {
            "scenario_aggregation": "unweighted macro mean of T1/T2/T3",
            "reason": "the updated guide states that exact scenario weights will be released later",
            "checkpoint_policy": "fixed final epoch inherited from source; no validation epoch selection",
            "test_accessed": source["protocol"]["test_accessed"],
        },
        "aggregate": aggregate,
        "promotion": promotion,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.source.read_bytes()
    source = json.loads(raw)
    report = summarize(source)
    report["source"] = {
        "path": args.source.as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
