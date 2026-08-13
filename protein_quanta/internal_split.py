"""Deterministic train-only development splits."""

import hashlib


def deterministic_hash_split(identifiers, holdout_count, salt):
    """Rank unique IDs by salted SHA256 and return development/holdout lists."""
    identifiers = [value.strip() for value in identifiers if value.strip()]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("identifiers must be unique")
    if not 0 < holdout_count < len(identifiers):
        raise ValueError("holdout_count must leave nonempty development and holdout sets")
    if not salt:
        raise ValueError("salt must be nonempty")
    ranked = sorted(
        identifiers,
        key=lambda value: hashlib.sha256(f"{salt}:{value}".encode()).hexdigest(),
    )
    holdout = set(ranked[:holdout_count])
    development = [value for value in identifiers if value not in holdout]
    held_out = [value for value in identifiers if value in holdout]
    return development, held_out
