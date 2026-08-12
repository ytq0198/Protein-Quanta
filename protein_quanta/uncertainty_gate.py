"""Low-capacity uncertainty gate for Static-residual anchor strength."""

import numpy as np


FEATURE_NAMES = (
    "log_atom_count",
    "log_observed_speed",
    "log_prediction_pair_drift",
    "log_prediction_rg_shift",
    "is_T2",
    "is_T3",
)


def _validate_trajectory(value, name):
    value = np.asarray(value, dtype=float)
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (frames, atoms, 3)")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain finite values")
    return value


def _radius_of_gyration(frame):
    centered = frame - frame.mean(axis=0, keepdims=True)
    return float(np.sqrt(np.mean(np.sum(np.square(centered), axis=-1))))


def extract_gate_features(prediction, history, scenario):
    """Extract inference-time scalar features invariant to rigid transforms."""
    prediction = _validate_trajectory(prediction, "prediction")
    history = _validate_trajectory(history, "history")
    if history.shape[0] != 2:
        raise ValueError("history must contain exactly two frames")
    if prediction.shape[0] < 3 or prediction.shape[1:] != history.shape[1:]:
        raise ValueError("prediction and history dimensions are incompatible")
    if scenario not in {"T1", "T2", "T3"}:
        raise ValueError("scenario must be one of T1, T2, or T3")

    observed_speed = np.sqrt(np.mean(np.square(history[-1] - history[-2])))
    differences = prediction[:, :, None, :] - prediction[:, None, :, :]
    distances = np.linalg.norm(differences, axis=-1)
    pair_drift = np.sqrt(np.mean(np.square(distances - distances[0])))
    rg_shift = abs(
        _radius_of_gyration(prediction[-1])
        - _radius_of_gyration(history[-1])
    )
    raw = (
        prediction.shape[1],
        observed_speed,
        pair_drift,
        rg_shift,
    )
    transformed = [float(np.log1p(max(float(value), 0.0))) for value in raw]
    transformed.extend((float(scenario == "T2"), float(scenario == "T3")))
    return dict(zip(FEATURE_NAMES, transformed))


def strong_anchor_label(baseline, candidate):
    """Return whether a strong anchor is both beneficial and within guards."""
    coordinate_change = 100.0 * (
        candidate["coordinate_rmse_angstrom"]
        / baseline["coordinate_rmse_angstrom"]
        - 1.0
    )
    rmsf_change = 100.0 * (
        candidate["rmsf_mae_angstrom"] / baseline["rmsf_mae_angstrom"] - 1.0
    )
    return bool(
        coordinate_change <= 2.0
        and rmsf_change <= 5.0
        and candidate["matching_mean_angstrom"]
        < baseline["matching_mean_angstrom"]
        and candidate["stability_mean_percent"]
        > baseline["stability_mean_percent"]
    )


def _sigmoid(values):
    values = np.clip(np.asarray(values, dtype=float), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-values))


def fit_logistic_gate(features, labels, l2=10.0, max_iter=100, tolerance=1e-10):
    """Fit deterministic balanced logistic regression using Newton updates."""
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if features.ndim != 2 or labels.shape != (features.shape[0],):
        raise ValueError("features and labels have incompatible shapes")
    if not np.isfinite(features).all() or not np.isfinite(labels).all():
        raise ValueError("features and labels must be finite")
    if not set(np.unique(labels)).issubset({0.0, 1.0}):
        raise ValueError("labels must be binary")
    if l2 < 0:
        raise ValueError("l2 must be non-negative")
    positives = float(labels.sum())
    negatives = float(labels.size - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("both label classes are required")

    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (features - mean) / scale
    design = np.column_stack([np.ones(features.shape[0]), standardized])
    class_weights = np.where(
        labels == 1,
        labels.size / (2.0 * positives),
        labels.size / (2.0 * negatives),
    )
    coefficients = np.zeros(design.shape[1], dtype=float)
    regularizer = np.diag([0.0] + [float(l2)] * features.shape[1])
    iterations = 0
    for iterations in range(1, max_iter + 1):
        probability = np.clip(_sigmoid(design @ coefficients), 1e-8, 1 - 1e-8)
        gradient = design.T @ (class_weights * (probability - labels))
        gradient += regularizer @ coefficients
        curvature = class_weights * probability * (1.0 - probability)
        hessian = design.T @ (design * curvature[:, None]) + regularizer
        hessian += np.eye(hessian.shape[0]) * 1e-8
        step = np.linalg.solve(hessian, gradient)
        coefficients -= step
        if np.linalg.norm(step) <= tolerance:
            break
    return {
        "mean": mean,
        "scale": scale,
        "coefficients": coefficients,
        "l2": float(l2),
        "iterations": iterations,
    }


def predict_gate_probability(model, features):
    features = np.asarray(features, dtype=float)
    if features.ndim == 1:
        features = features[None, :]
    standardized = (features - model["mean"]) / model["scale"]
    design = np.column_stack([np.ones(features.shape[0]), standardized])
    return _sigmoid(design @ model["coefficients"])


def _serializable_model(model):
    return {
        "mean": model["mean"].tolist(),
        "scale": model["scale"].tolist(),
        "coefficients": model["coefficients"].tolist(),
        "l2": model["l2"],
        "iterations": model["iterations"],
    }


def grouped_leave_one_out(records, l2=10.0):
    """Predict every record from a model excluding its complete sample group."""
    if not records:
        raise ValueError("at least one record is required")
    keys = [(row["sample_id"], row["scenario"]) for row in records]
    if len(keys) != len(set(keys)):
        raise ValueError("sample/scenario records must be unique")
    groups = sorted({row["sample_id"] for row in records})
    folds = []
    predictions = []
    for held_out in groups:
        training = [row for row in records if row["sample_id"] != held_out]
        testing = [row for row in records if row["sample_id"] == held_out]
        train_features = np.stack([row["features"] for row in training])
        train_labels = np.asarray([row["label"] for row in training])
        model = fit_logistic_gate(train_features, train_labels, l2=l2)
        probabilities = predict_gate_probability(
            model, np.stack([row["features"] for row in testing])
        )
        for row, probability in zip(testing, probabilities):
            predictions.append(
                {
                    "sample_id": row["sample_id"],
                    "scenario": row["scenario"],
                    "label": int(row["label"]),
                    "probability": float(probability),
                    "selected_beta": 8.0 if probability >= 0.5 else 1.0,
                }
            )
        folds.append(
            {
                "held_out_sample_id": held_out,
                "held_out_record_count": len(testing),
                "training_sample_ids": sorted(
                    {row["sample_id"] for row in training}
                ),
                "training_record_count": len(training),
                "training_positive_count": int(train_labels.sum()),
                "model": _serializable_model(model),
            }
        )
    return {"folds": folds, "predictions": predictions}

