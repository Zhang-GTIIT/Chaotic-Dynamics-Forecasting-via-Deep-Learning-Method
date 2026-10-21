from __future__ import annotations

import math

import numpy as np


def first_crossing(curve: np.ndarray, threshold: float) -> int | None:
    indices = np.flatnonzero(np.asarray(curve) >= float(threshold))
    return None if indices.size == 0 else int(indices[0] + 1)


def spectral_entropy(values: np.ndarray, window: int = 1200) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[:, -min(window, values.shape[1]) :, :]
    values = values - values.mean(axis=1, keepdims=True)
    spectrum = np.abs(np.fft.rfft(values, axis=1)) ** 2
    spectrum = spectrum[:, 1:, :]
    probabilities = spectrum / np.maximum(spectrum.sum(axis=1, keepdims=True), 1.0e-18)
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1.0e-18)), axis=1)
    normalizer = math.log(max(2, spectrum.shape[1]))
    return float(np.mean(entropy / normalizer))


def compute_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    train_std: np.ndarray,
    evaluation: dict,
    validation_rollout_loss: float | None = None,
) -> dict:
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    train_std = np.asarray(train_std, dtype=np.float64).reshape(1, 1, -1)
    if prediction.shape != truth.shape or prediction.ndim != 3:
        raise ValueError(f"Prediction/truth shape mismatch: {prediction.shape}, {truth.shape}")
    if train_std.shape[-1] != prediction.shape[-1] or np.any(train_std <= 1.0e-8):
        raise ValueError("Invalid train_std")
    normalized_squared_error = np.square((prediction - truth) / (train_std + 1.0e-8))
    per_trajectory = np.sqrt(np.mean(normalized_squared_error, axis=2))
    mean_curve = np.mean(per_trajectory, axis=0)
    median_curve = np.median(per_trajectory, axis=0)
    p10_curve = np.percentile(per_trajectory, 10, axis=0)
    p90_curve = np.percentile(per_trajectory, 90, axis=0)
    steps = int(mean_curve.size)

    late = min(int(evaluation["late_window_steps"]), steps)
    pred_temporal_std = np.std(prediction[:, -late:, :], axis=1)
    truth_temporal_std = np.std(truth[:, -late:, :], axis=1)
    late_std_ratio = float(np.mean(pred_temporal_std) / max(float(np.mean(truth_temporal_std)), 1.0e-12))
    truth_entropy = spectral_entropy(truth)
    prediction_entropy = spectral_entropy(prediction)
    entropy_ratio = float(prediction_entropy / max(truth_entropy, 1.0e-12))

    h05 = first_crossing(mean_curve, 0.5)
    h10 = first_crossing(mean_curve, 1.0)
    hsqrt2 = first_crossing(mean_curve, math.sqrt(2.0))
    amplitude_flag = bool(late_std_ratio < float(evaluation["amplitude_collapse_threshold"]))
    periodic_flag = bool(entropy_ratio < float(evaluation["spectral_entropy_ratio_threshold"]))
    start = max(1, int(evaluation["plateau_start_step"])) - 1
    stop = min(steps, int(evaluation["plateau_end_step"]))
    plateau_fraction = 0.0
    if stop > start:
        window = mean_curve[start:stop]
        plateau_fraction = float(
            np.mean(
                (window >= float(evaluation["plateau_nrmse_min"]))
                & (window <= float(evaluation["plateau_nrmse_max"]))
            )
        )
    plateau_flag = bool(
        h10 is not None
        and h10 < 300
        and plateau_fraction > float(evaluation["plateau_fraction_threshold"])
    )
    invalid = bool(amplitude_flag or periodic_flag or plateau_flag)

    def at(step: int) -> float | None:
        return float(mean_curve[step - 1]) if steps >= step else None

    auc_horizon = min(1200, steps)
    summary = {
        "forecast_steps": steps,
        "H0.5": h05,
        "H1.0": h10,
        "Hsqrt2": hsqrt2,
        "Hsqrt2_rank": int(hsqrt2 if hsqrt2 is not None else steps + 1),
        "NRMSE100": at(100),
        "NRMSE200": at(200),
        "NRMSE500": at(500),
        "NRMSE1000": at(1000),
        "NRMSE1200": at(1200),
        "NRMSE3000": at(3000),
        "final_NRMSE": float(mean_curve[-1]),
        "AUC1200": float(np.trapezoid(mean_curve[:auc_horizon], np.arange(1, auc_horizon + 1))),
        "late_std_ratio": late_std_ratio,
        "truth_spectral_entropy": truth_entropy,
        "prediction_spectral_entropy": prediction_entropy,
        "spectral_entropy_ratio": entropy_ratio,
        "plateau_fraction_300_1200": plateau_fraction,
        "amplitude_collapse_flag": amplitude_flag,
        "periodic_collapse_flag": periodic_flag,
        "NRMSE_1_plateau_flag": plateau_flag,
        "invalid_dynamics_flag": invalid,
        "validation_rollout_loss": None if validation_rollout_loss is None else float(validation_rollout_loss),
    }
    crossing_per_trajectory = np.full(prediction.shape[0], np.nan, dtype=np.float64)
    for index, curve in enumerate(per_trajectory):
        crossing = first_crossing(curve, math.sqrt(2.0))
        if crossing is not None:
            crossing_per_trajectory[index] = crossing
    return {
        "summary": summary,
        "forecast_step": np.arange(1, steps + 1, dtype=np.int64),
        "mean_nrmse": mean_curve,
        "median_nrmse": median_curve,
        "p10_nrmse": p10_curve,
        "p90_nrmse": p90_curve,
        "per_trajectory_nrmse": per_trajectory,
        "sqrt2_crossing_per_trajectory": crossing_per_trajectory,
    }


def selection_key(summary: dict) -> tuple:
    steps = int(summary["forecast_steps"])
    hsqrt2 = int(summary["Hsqrt2"] if summary["Hsqrt2"] is not None else steps + 1)
    h1 = int(summary["H1.0"] if summary["H1.0"] is not None else steps + 1)
    auc = float(summary["AUC1200"])
    nrmse1200 = float(summary["NRMSE1200"] if summary["NRMSE1200"] is not None else summary["final_NRMSE"])
    validation_loss = float(summary["validation_rollout_loss"] if summary["validation_rollout_loss"] is not None else float("inf"))
    return (
        int(not bool(summary["invalid_dynamics_flag"])),
        hsqrt2,
        h1,
        -auc,
        -nrmse1200,
        -validation_loss,
    )


def is_better(candidate: dict, incumbent: dict | None) -> bool:
    return incumbent is None or selection_key(candidate) > selection_key(incumbent)
