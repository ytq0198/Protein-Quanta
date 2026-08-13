"""Pure-filesystem barrier for opening a frozen effect-gate holdout."""

import hashlib
import json


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_training_artifacts(config, training_dir, checkpoint_dir):
    """Verify all frozen artifacts before the caller may read the holdout."""
    verified = {}
    for seed in config["pairing"]["seeds"]:
        report_path = training_dir / f"seed_{seed}.json"
        if not report_path.is_file():
            raise FileNotFoundError(f"missing training manifest: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "training_complete_holdout_unread":
            raise ValueError(f"seed {seed} is not a completed sealed training run")
        if report.get("seed") != seed:
            raise ValueError(f"training manifest seed mismatch: {report_path}")
        verified[seed] = {"manifest": str(report_path), "checkpoints": {}}
        for arm in ("control", "candidate"):
            checkpoint = checkpoint_dir / f"{arm}_seed_{seed}_final.pth"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
            actual = sha256(checkpoint)
            expected = report.get("checkpoint_sha256", {}).get(arm)
            if actual != expected:
                raise ValueError(f"checkpoint hash mismatch: {checkpoint}")
            verified[seed]["checkpoints"][arm] = {
                "path": str(checkpoint),
                "sha256": actual,
            }
    return verified
