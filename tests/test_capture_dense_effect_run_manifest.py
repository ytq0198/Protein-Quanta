import hashlib
import json
import subprocess
import sys
from pathlib import Path


def test_manifest_captures_all_file_hashes(tmp_path):
    files = []
    for index in range(7):
        path = tmp_path / f"input-{index}.txt"
        path.write_text(f"value-{index}", encoding="utf-8")
        files.append(path)
    output = tmp_path / "manifest.json"
    arguments = [
        sys.executable,
        "scripts/capture_dense_effect_run_manifest.py",
        "--source-commit", "abcdef0",
        "--training-script", str(files[0]),
        "--evaluation-script", str(files[1]),
        "--config", str(files[2]),
        "--development-ids", str(files[3]),
        "--holdout-ids", str(files[4]),
        "--topology-report", str(files[5]),
        "--dataset", str(files[6]),
        "--output", str(output),
    ]

    subprocess.run(arguments, check=True, capture_output=True, text=True)
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["source_commit"] == "abcdef0"
    assert report["status"] == "sealed before holdout target evaluation"
    assert len(report["files"]) == 7
    assert report["files"]["dataset"]["sha256"] == hashlib.sha256(
        files[6].read_bytes()
    ).hexdigest()
