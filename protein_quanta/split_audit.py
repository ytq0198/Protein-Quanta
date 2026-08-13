"""Integrity checks for the updated MISATO/NeuralMD split protocol."""

from __future__ import annotations

import hashlib
from pathlib import Path


def _read_ids(path: Path) -> list[str]:
    values = [
        line.strip().upper()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not values:
        raise ValueError(f"empty ID file: {path}")
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _id_set_sha256(values: set[str]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(values)).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def audit_filtered_splits(split_paths, peptides_path):
    """Audit raw splits before and after NeuralMD peptide exclusion.

    The report deliberately stores counts and hashes rather than the public IDs.
    This makes the experiment protocol traceable without copying source data into
    the project repository.
    """

    required = {"train", "val", "test"}
    if set(split_paths) != required:
        raise ValueError("split_paths must contain exactly train, val and test")

    peptide_rows = _read_ids(Path(peptides_path))
    peptides = set(peptide_rows)
    split_sets = {}
    report = {
        "peptides": {
            "row_count": len(peptide_rows),
            "unique_count": len(peptides),
            "duplicate_rows": len(peptide_rows) - len(peptides),
            "source_sha256": _sha256(Path(peptides_path)),
            "normalized_id_set_sha256": _id_set_sha256(peptides),
        },
        "splits": {},
    }

    for name in ("train", "val", "test"):
        path = Path(split_paths[name])
        rows = _read_ids(path)
        unique = set(rows)
        retained = unique - peptides
        excluded = unique & peptides
        split_sets[name] = unique
        report["splits"][name] = {
            "raw_row_count": len(rows),
            "raw_unique_count": len(unique),
            "duplicate_rows": len(rows) - len(unique),
            "excluded_peptide_count": len(excluded),
            "retained_count": len(retained),
            "source_sha256": _sha256(path),
            "retained_id_set_sha256": _id_set_sha256(retained),
            "excluded_id_set_sha256": _id_set_sha256(excluded),
        }

    overlap = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = split_sets[left] & split_sets[right]
        overlap[f"{left}_{right}"] = {
            "count": len(shared),
            "id_set_sha256": _id_set_sha256(shared),
        }
    assigned = set().union(*split_sets.values())
    report["cross_split_overlap"] = overlap
    report["peptide_ids_present_in_any_split"] = len(peptides & assigned)
    report["peptide_ids_outside_all_splits"] = len(peptides - assigned)
    report["checks"] = {
        "no_duplicate_rows": all(
            row["duplicate_rows"] == 0 for row in report["splits"].values()
        ),
        "splits_disjoint": all(row["count"] == 0 for row in overlap.values()),
        "updated_guide_counts_match": {
            name: report["splits"][name]["retained_count"] == expected
            for name, expected in {"train": 13066, "val": 1357, "test": 1357}.items()
        },
    }
    report["checks"]["all_passed"] = (
        report["checks"]["no_duplicate_rows"]
        and report["checks"]["splits_disjoint"]
        and all(report["checks"]["updated_guide_counts_match"].values())
    )
    return report
