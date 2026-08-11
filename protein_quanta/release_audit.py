"""Read-only checks for a safe, evidence-backed public research release."""

import json
from pathlib import Path, PurePosixPath


FORBIDDEN_SUFFIXES = {".pth", ".pt", ".ckpt", ".h5", ".hdf5", ".npz", ".npy"}
SECRET_MARKERS = (
    "github" + "_pat_",
    "gh" + "p_",
    "h" + "f_",
)
MAX_TRACKED_BYTES = 5 * 1024 * 1024


def _relative_path(root, value):
    path = Path(value)
    if path.is_absolute():
        return None
    normalized = Path(PurePosixPath(str(value).replace("\\", "/")))
    if ".." in normalized.parts:
        return None
    return root / normalized


def _manifest_evidence_paths(manifest):
    evidence = manifest.get("evidence", {})
    for value in evidence.values():
        if isinstance(value, str) and value.startswith(
            ("reports/", "docs/", "configs/", "protein_quanta/", "scripts/")
        ):
            yield value


def audit_release(root, tracked_paths):
    """Audit tracked files without changing the workspace."""
    root = Path(root).resolve()
    errors = []
    warnings = []
    normalized_paths = sorted({str(PurePosixPath(value)) for value in tracked_paths})

    for relative in normalized_paths:
        full_path = _relative_path(root, relative)
        if full_path is None:
            errors.append(f"unsafe tracked path: {relative}")
            continue
        if not full_path.exists():
            errors.append(f"tracked path is missing: {relative}")
            continue
        if full_path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"model/data artifact must not be tracked: {relative}")
        size = full_path.stat().st_size
        if size > MAX_TRACKED_BYTES:
            errors.append(f"tracked file exceeds 5 MiB release limit: {relative}")
        if size <= MAX_TRACKED_BYTES and full_path.suffix.lower() in {
            ".md",
            ".txt",
            ".json",
            ".py",
            ".sh",
            ".yml",
            ".yaml",
            ".toml",
        }:
            text = full_path.read_text(encoding="utf-8", errors="replace")
            if any(marker in text for marker in SECRET_MARKERS):
                errors.append(f"secret-like marker found in tracked text: {relative}")

    manifest_path = root / "configs" / "frozen_candidate.json"
    if not manifest_path.exists():
        errors.append("missing configs/frozen_candidate.json")
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            errors.append(f"invalid frozen candidate manifest: {error}")
        else:
            selection_claim = manifest.get("selection", {}).get(
                "official_score_claimed"
            )
            physical_claim = manifest.get("physical_diagnostics", {}).get(
                "official_score_claimed"
            )
            if selection_claim is not False or physical_claim is not False:
                errors.append("frozen candidate must not claim an official score")
            for relative in _manifest_evidence_paths(manifest):
                evidence_path = _relative_path(root, relative)
                if evidence_path is None or not evidence_path.exists():
                    errors.append(f"manifest evidence is missing: {relative}")

    if not any((root / name).exists() for name in ("LICENSE", "LICENSE.txt", "LICENSE.md")):
        warnings.append("project-level LICENSE is missing; team decision required")

    return {
        "schema_version": 1,
        "status": "read-only project release audit; not a competition score",
        "passed": not errors,
        "tracked_file_count": len(normalized_paths),
        "errors": errors,
        "warnings": warnings,
    }
