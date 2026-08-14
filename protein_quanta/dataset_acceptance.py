"""Acceptance checks joining official MISATO splits to a downloaded HDF5 file."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import h5py

from protein_quanta.misato import audit_misato_h5


def read_normalized_ids(path: Path) -> list[str]:
    values = [
        line.strip().upper()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not values:
        raise ValueError(f"empty ID file: {path}")
    return values


def id_set_sha256(values: set[str]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(values)).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def evenly_spaced_ids(values: Sequence[str], count: int) -> list[str]:
    """Choose deterministic endpoints and quantiles without random-state ambiguity."""
    ordered = sorted(set(value.upper() for value in values))
    if count <= 0:
        raise ValueError("count must be positive")
    if count >= len(ordered):
        return ordered
    if count == 1:
        return [ordered[len(ordered) // 2]]
    indices = [(index * (len(ordered) - 1)) // (count - 1) for index in range(count)]
    return [ordered[index] for index in indices]


def audit_full_misato_acceptance(
    h5_path: Path,
    split_paths: Mapping[str, Path],
    peptides_path: Path,
    samples_per_split: int = 16,
    chunk_frames: int = 25,
    observed_md5: str | None = None,
    expected_md5: str | None = None,
) -> dict:
    required = {"train", "val", "test"}
    if set(split_paths) != required:
        raise ValueError("split_paths must contain exactly train, val and test")

    raw_splits = {
        name: set(read_normalized_ids(path)) for name, path in split_paths.items()
    }
    peptides = set(read_normalized_ids(peptides_path))
    retained_splits = {name: values - peptides for name, values in raw_splits.items()}
    raw_union = set().union(*raw_splits.values())
    retained_union = set().union(*retained_splits.values())

    with h5py.File(h5_path, "r") as handle:
        h5_ids = {
            name.upper()
            for name, value in handle.items()
            if isinstance(value, h5py.Group)
        }

    sampled_by_split = {
        name: evenly_spaced_ids(values, samples_per_split)
        for name, values in retained_splits.items()
    }
    selected_ids = [
        sample_id
        for name in ("train", "val", "test")
        for sample_id in sampled_by_split[name]
    ]
    schema_audit = audit_misato_h5(
        h5_path,
        selected_sample_ids=selected_ids,
        chunk_frames=chunk_frames,
    )

    missing_raw = raw_union - h5_ids
    extra_h5 = h5_ids - raw_union
    missing_retained = retained_union - h5_ids
    expected_retained_counts = {"train": 13066, "val": 1357, "test": 1357}
    if (observed_md5 is None) != (expected_md5 is None):
        raise ValueError("observed_md5 and expected_md5 must be supplied together")
    normalized_observed_md5 = observed_md5.lower() if observed_md5 else None
    normalized_expected_md5 = expected_md5.lower() if expected_md5 else None

    checks = {
        "hdf5_opens": True,
        "raw_split_union_matches_hdf5_groups": not missing_raw and not extra_h5,
        "all_retained_ids_present": not missing_retained,
        "updated_guide_counts_match": all(
            len(retained_splits[name]) == expected
            for name, expected in expected_retained_counts.items()
        ),
        "sampled_schema_and_finite_values_pass": schema_audit["invalid_complex_count"] == 0,
    }
    if normalized_expected_md5 is not None:
        checks["md5_matches_published_value"] = (
            normalized_observed_md5 == normalized_expected_md5
        )
    checks["all_passed"] = all(checks.values())

    return {
        "status": "full MISATO local-file acceptance audit; public IDs only",
        "hdf5": {
            "path": str(Path(h5_path).resolve()),
            "byte_count": Path(h5_path).stat().st_size,
            "group_count": len(h5_ids),
            "group_id_set_sha256": id_set_sha256(h5_ids),
            "observed_md5": normalized_observed_md5,
            "expected_md5": normalized_expected_md5,
        },
        "raw_split_union": {
            "count": len(raw_union),
            "id_set_sha256": id_set_sha256(raw_union),
        },
        "retained_splits": {
            name: {
                "count": len(values),
                "id_set_sha256": id_set_sha256(values),
            }
            for name, values in retained_splits.items()
        },
        "membership_differences": {
            "raw_ids_missing_from_hdf5_count": len(missing_raw),
            "raw_ids_missing_from_hdf5_sha256": id_set_sha256(missing_raw),
            "extra_hdf5_group_count": len(extra_h5),
            "extra_hdf5_group_sha256": id_set_sha256(extra_h5),
            "retained_ids_missing_from_hdf5_count": len(missing_retained),
            "retained_ids_missing_from_hdf5_sha256": id_set_sha256(missing_retained),
        },
        "sampling": {
            "method": "sorted-ID endpoints and evenly spaced quantiles within each split",
            "samples_per_split": samples_per_split,
            "sample_id_set_sha256": id_set_sha256(set(selected_ids)),
        },
        "sampled_hdf5_audit": schema_audit,
        "checks": checks,
    }
