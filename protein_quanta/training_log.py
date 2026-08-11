"""Parsers for the metrics emitted by the pinned NeuralMD training script."""

import re


_EPOCH = re.compile(r"^epoch\s+(\d+)\s*$")
_LOSS = re.compile(r"^loss pos:\s*([-+0-9.eE]+)")
_OPTIMIZER = re.compile(
    r"^optimizer_stats grad_norm_mean:\s*([-+0-9.eE]+)\s+"
    r"grad_norm_max:\s*([-+0-9.eE]+)\s+"
    r"clipped_batches:\s*(\d+)\s+"
    r"skipped_nonfinite_batches:\s*(\d+)"
)
_VALIDATION = {
    "val_coordinate_mae": re.compile(
        r"^MAE train:\s*[-+0-9.eE]+\s+val:\s*([-+0-9.eE]+)"
    ),
    "val_coordinate_rmse": re.compile(
        r"^RMSE train:\s*[-+0-9.eE]+\s+val:\s*([-+0-9.eE]+)"
    ),
    "val_matching": re.compile(
        r"^hr MAE train:\s*[-+0-9.eE]+\s+val:\s*([-+0-9.eE]+)"
    ),
    "val_stability": re.compile(
        r"^Stability train:\s*[-+0-9.eE]+\s+val:\s*([-+0-9.eE]+)"
    ),
}


def parse_neuralmd_training_log(text):
    """Return one metric dictionary per completed or partial training epoch."""
    rows = []
    current = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = _EPOCH.match(line)
        if match:
            current = {"epoch": int(match.group(1))}
            rows.append(current)
            continue
        if current is None:
            continue
        match = _LOSS.match(line)
        if match:
            current["loss_pos"] = float(match.group(1))
            continue
        match = _OPTIMIZER.match(line)
        if match:
            current.update(
                {
                    "grad_norm_mean": float(match.group(1)),
                    "grad_norm_max": float(match.group(2)),
                    "clipped_batches": int(match.group(3)),
                    "skipped_nonfinite_batches": int(match.group(4)),
                }
            )
            continue
        for key, pattern in _VALIDATION.items():
            match = pattern.match(line)
            if match:
                current[key] = float(match.group(1))
                break
    return rows
