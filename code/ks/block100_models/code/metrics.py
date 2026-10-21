"""Shared NRMSE and rollout-selection definitions for all three models."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _validate_inputs(
    prediction: np.ndarray, truth: np.ndarray, train_std: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    train_std = np.asarray(train_std, dtype=np.float64).reshape(-1)
    if prediction.shape != truth.shape or prediction.ndim != 3:
        raise ValueError(
            "prediction and truth must have identical shape "
            f"[N_trajectories,N_steps,D]; found {prediction.shape} and {truth.shape}"
        )
    if train_std.shape != (prediction.shape[2],):
        raise ValueError(
            f"train_std must have shape ({prediction.shape[2]},), found {train_std.shape}"
        )
    if prediction.shape[0] == 0 or prediction.shape[1] == 0:
        raise ValueError("NRMSE requires at least one trajectory and one forecast step")
    if not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(truth)):
        raise ValueError("prediction and truth must contain only finite values")
    if not np.all(np.isfinite(train_std)) or np.any(train_std <= 1.0e-8):
        raise ValueError("Every train_std component must be finite and greater than 1e-8")
    return prediction, truth, train_std


def first_crossing(curve: np.ndarray, threshold: float) -> dict:
    """Return a one-based crossing step, or null when the threshold is not reached."""
    curve = np.asarray(curve, dtype=np.float64).reshape(-1)
    indices = np.flatnonzero(curve >= float(threshold))
    if indices.size == 0:
        return {"threshold": float(threshold), "reached": False, "crossing_step": None}
    return {
        "threshold": float(threshold),
        "reached": True,
        "crossing_step": int(indices[0]) + 1,
    }


def per_trajectory_first_crossing(
    nrmse_per_trajectory: np.ndarray, threshold: float
) -> np.ndarray:
    curves = np.asarray(nrmse_per_trajectory, dtype=np.float64)
    if curves.ndim != 2:
        raise ValueError("nrmse_per_trajectory must have shape [N_trajectories,N_steps]")
    crossing_steps = np.full(curves.shape[0], np.nan, dtype=np.float64)
    for trajectory_index, curve in enumerate(curves):
        indices = np.flatnonzero(curve >= float(threshold))
        if indices.size:
            crossing_steps[trajectory_index] = float(indices[0] + 1)
    return crossing_steps


def compute_nrmse_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    train_std: np.ndarray,
    time_step: float,
    lyapunov_exponent: float,
    late_window_steps: int = 200,
    collapse_std_ratio_threshold: float = 0.5,
    validation_mse: float | None = None,
) -> dict:
    prediction, truth, train_std = _validate_inputs(prediction, truth, train_std)
    safe_std = np.maximum(train_std, 1.0e-8)
    normalized_squared_error = np.square(
        (prediction - truth) / safe_std.reshape(1, 1, -1)
    )
    nrmse_per_trajectory = np.sqrt(normalized_squared_error.mean(axis=2))
    mean_nrmse = nrmse_per_trajectory.mean(axis=0)
    median_nrmse = np.median(nrmse_per_trajectory, axis=0)
    p10_nrmse = np.percentile(nrmse_per_trajectory, 10, axis=0)
    p90_nrmse = np.percentile(nrmse_per_trajectory, 90, axis=0)

    forecast_step = np.arange(1, prediction.shape[1] + 1, dtype=np.int64)
    physical_time = forecast_step.astype(np.float64) * float(time_step)
    lyapunov_time = physical_time * float(lyapunov_exponent)
    prediction_std = prediction.std(axis=(0, 2))
    truth_std = truth.std(axis=(0, 2))
    std_ratio = prediction_std / (truth_std + 1.0e-8)

    window = min(int(late_window_steps), prediction.shape[1])
    prediction_temporal_std = prediction[:, -window:, :].std(axis=1).mean()
    truth_temporal_std = truth[:, -window:, :].std(axis=1).mean()
    late_std_ratio = float(prediction_temporal_std / (truth_temporal_std + 1.0e-8))
    collapse_flag = bool(late_std_ratio < float(collapse_std_ratio_threshold))

    horizon = min(1200, prediction.shape[1])
    if horizon == 1:
        nrmse_auc = float(mean_nrmse[0])
    else:
        nrmse_auc = float(np.trapz(mean_nrmse[:horizon], forecast_step[:horizon]))

    crossings = {
        "first_crossing_0_5": first_crossing(mean_nrmse, 0.5),
        "first_crossing_1_0": first_crossing(mean_nrmse, 1.0),
        "first_crossing_sqrt2": first_crossing(mean_nrmse, math.sqrt(2.0)),
    }
    selected_steps = {}
    for step in (100, 200, 500, 1000, 1200):
        selected_steps[f"nrmse_at_step_{step}"] = (
            float(mean_nrmse[step - 1]) if step <= mean_nrmse.size else None
        )

    summary = {
        **crossings,
        **selected_steps,
        "final_nrmse": float(mean_nrmse[-1]),
        "mean_nrmse_over_first_1200": float(mean_nrmse[:horizon].mean()),
        "nrmse_auc_over_first_1200": nrmse_auc,
        "late_window_steps": int(window),
        "late_std_ratio": late_std_ratio,
        "collapse_std_ratio_threshold": float(collapse_std_ratio_threshold),
        "collapse_flag": collapse_flag,
        "validation_mse": None if validation_mse is None else float(validation_mse),
        "forecast_steps": int(prediction.shape[1]),
        "num_trajectories": int(prediction.shape[0]),
    }
    return {
        "forecast_step": forecast_step,
        "physical_time": physical_time,
        "lyapunov_time": lyapunov_time,
        "mean_nrmse": mean_nrmse,
        "median_nrmse": median_nrmse,
        "p10_nrmse": p10_nrmse,
        "p90_nrmse": p90_nrmse,
        "prediction_std": prediction_std,
        "truth_std": truth_std,
        "std_ratio": std_ratio,
        "nrmse_per_trajectory": nrmse_per_trajectory,
        "sqrt2_crossing_per_trajectory": per_trajectory_first_crossing(
            nrmse_per_trajectory, math.sqrt(2.0)
        ),
        "summary": summary,
    }


def _crossing_rank(crossing: dict, forecast_steps: int) -> int:
    if crossing.get("reached", False):
        return int(crossing["crossing_step"])
    return int(forecast_steps) + 1


def rollout_selection_key(summary: dict) -> tuple:
    """Higher tuple is better under the required anti-collapse ordering."""
    forecast_steps = int(summary["forecast_steps"])
    validation_mse = summary.get("validation_mse")
    if validation_mse is None:
        validation_mse = float("inf")
    return (
        int(not bool(summary["collapse_flag"])),
        _crossing_rank(summary["first_crossing_1_0"], forecast_steps),
        _crossing_rank(summary["first_crossing_sqrt2"], forecast_steps),
        -float(summary["nrmse_auc_over_first_1200"]),
        -float(validation_mse),
    )


def is_better_rollout(candidate: dict, incumbent: dict | None) -> bool:
    return incumbent is None or rollout_selection_key(candidate) > rollout_selection_key(incumbent)
