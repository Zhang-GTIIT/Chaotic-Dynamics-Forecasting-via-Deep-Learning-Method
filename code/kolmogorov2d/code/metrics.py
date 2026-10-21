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


def compute_periodic_flow_physical_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    grid_size: int,
    late_window_steps: int = 200,
    acceptable_ratio_min: float = 0.8,
    acceptable_ratio_max: float = 1.2,
) -> dict:
    """Compare late-window kinetic energy and enstrophy in physical space."""
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    grid = int(grid_size)
    if prediction.shape != truth.shape or prediction.ndim != 3:
        raise ValueError("prediction and truth must have matching [N,T,D] shapes")
    if prediction.shape[2] != grid * grid:
        raise ValueError("Periodic-flow dimension does not match grid_size")
    window = min(int(late_window_steps), prediction.shape[1])

    def statistics(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fields = values[:, -window:, :].reshape(
            values.shape[0], window, grid, grid
        )
        spectrum = np.fft.rfft2(fields, axes=(-2, -1), norm="ortho")
        power = np.abs(spectrum) ** 2
        row_modes = np.fft.fftfreq(grid, d=1.0 / grid)[:, None]
        column_modes = np.fft.rfftfreq(grid, d=1.0 / grid)[None, :]
        k_squared = row_modes**2 + column_modes**2
        inverse_k_squared = np.zeros_like(k_squared)
        inverse_k_squared[k_squared > 0.0] = 1.0 / k_squared[k_squared > 0.0]
        weights = np.ones((1, grid // 2 + 1), dtype=np.float64)
        if grid % 2 == 0:
            weights[:, 1:-1] = 2.0
        else:
            weights[:, 1:] = 2.0
        kinetic = 0.5 * (
            power * inverse_k_squared * weights
        ).sum(axis=(-2, -1)) / float(grid * grid)
        enstrophy = 0.5 * np.square(fields).mean(axis=(-2, -1))
        return kinetic.mean(axis=1), enstrophy.mean(axis=1)

    prediction_energy, prediction_enstrophy = statistics(prediction)
    truth_energy, truth_enstrophy = statistics(truth)
    energy_ratio = prediction_energy / np.maximum(truth_energy, 1.0e-12)
    enstrophy_ratio = prediction_enstrophy / np.maximum(truth_enstrophy, 1.0e-12)
    energy_median = float(np.median(energy_ratio))
    enstrophy_median = float(np.median(enstrophy_ratio))
    ratio_min = float(acceptable_ratio_min)
    ratio_max = float(acceptable_ratio_max)
    physical_health_flag = bool(
        ratio_min <= energy_median <= ratio_max
        and ratio_min <= enstrophy_median <= ratio_max
    )
    return {
        "late_energy_ratio_median": energy_median,
        "late_energy_ratio_p10": float(np.quantile(energy_ratio, 0.10)),
        "late_energy_ratio_p90": float(np.quantile(energy_ratio, 0.90)),
        "late_enstrophy_ratio_median": enstrophy_median,
        "late_enstrophy_ratio_p10": float(np.quantile(enstrophy_ratio, 0.10)),
        "late_enstrophy_ratio_p90": float(np.quantile(enstrophy_ratio, 0.90)),
        "physical_ratio_min": ratio_min,
        "physical_ratio_max": ratio_max,
        "physical_health_flag": physical_health_flag,
        "physical_log_error": float(
            abs(math.log(max(energy_median, 1.0e-12)))
            + abs(math.log(max(enstrophy_median, 1.0e-12)))
        ),
    }


def compute_nrmse_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    train_std: np.ndarray,
    time_step: float,
    lyapunov_exponent: float,
    late_window_steps: int = 200,
    collapse_std_ratio_threshold: float = 0.5,
    spectral_entropy_ratio_threshold: float | None = None,
    validation_mse: float | None = None,
    train_mean: np.ndarray | None = None,
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
    if train_mean is None:
        center = np.zeros(prediction.shape[2], dtype=np.float64)
    else:
        center = np.asarray(train_mean, dtype=np.float64).reshape(-1)
        if center.shape != (prediction.shape[2],):
            raise ValueError(
                f"train_mean must have shape ({prediction.shape[2]},), found {center.shape}"
            )
    prediction_anomaly = prediction - center.reshape(1, 1, -1)
    truth_anomaly = truth - center.reshape(1, 1, -1)
    anomaly_numerator = np.sum(prediction_anomaly * truth_anomaly, axis=(0, 2))
    anomaly_denominator = np.sqrt(
        np.sum(np.square(prediction_anomaly), axis=(0, 2))
        * np.sum(np.square(truth_anomaly), axis=(0, 2))
    )
    anomaly_correlation = anomaly_numerator / np.maximum(anomaly_denominator, 1.0e-30)

    window = min(int(late_window_steps), prediction.shape[1])
    prediction_temporal_std = prediction[:, -window:, :].std(axis=1).mean()
    truth_temporal_std = truth[:, -window:, :].std(axis=1).mean()
    late_std_ratio = float(prediction_temporal_std / (truth_temporal_std + 1.0e-8))
    late_mean_nrmse = float(mean_nrmse[-window:].mean())
    late_anomaly_correlation = float(anomaly_correlation[-window:].mean())
    near_one_plateau_flag = bool(
        0.85 <= late_mean_nrmse <= 1.20
        and late_anomaly_correlation <= 0.10
    )
    sub_sqrt2_plateau_flag = bool(
        not first_crossing(mean_nrmse, math.sqrt(2.0))["reached"]
        and late_mean_nrmse < 1.30
        and late_anomaly_correlation <= 0.10
    )
    amplitude_collapse_flag = bool(
        late_std_ratio < float(collapse_std_ratio_threshold)
    )

    spectral_summary = {}
    if spectral_entropy_ratio_threshold is None:
        collapse_flag = amplitude_collapse_flag
    else:
        spectral_window = min(1000, prediction.shape[1])
        prediction_entropy = _mean_temporal_spectral_entropy(
            prediction[:, -spectral_window:, :]
        )
        truth_entropy = _mean_temporal_spectral_entropy(
            truth[:, -spectral_window:, :]
        )
        spectral_entropy_ratio = float(
            prediction_entropy / (truth_entropy + 1.0e-12)
        )
        periodic_collapse_flag = bool(
            spectral_entropy_ratio < float(spectral_entropy_ratio_threshold)
        )
        dynamics_collapse_flag = bool(
            amplitude_collapse_flag or periodic_collapse_flag
        )
        collapse_flag = dynamics_collapse_flag
        spectral_summary = {
            "amplitude_collapse_flag": amplitude_collapse_flag,
            "periodic_collapse_flag": periodic_collapse_flag,
            "dynamics_collapse_flag": dynamics_collapse_flag,
            "spectral_entropy_window_steps": int(spectral_window),
            "truth_spectral_entropy": truth_entropy,
            "prediction_spectral_entropy": prediction_entropy,
            "spectral_entropy_ratio": spectral_entropy_ratio,
            "spectral_entropy_ratio_threshold": float(
                spectral_entropy_ratio_threshold
            ),
        }

    horizon = min(1200, prediction.shape[1])
    if horizon == 1:
        nrmse_auc = float(mean_nrmse[0])
    else:
        nrmse_auc = float(
            np.trapezoid(mean_nrmse[:horizon], forecast_step[:horizon])
        )

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
        "late_mean_nrmse": late_mean_nrmse,
        "late_anomaly_correlation": late_anomaly_correlation,
        "near_one_plateau_flag": near_one_plateau_flag,
        "sub_sqrt2_plateau_flag": sub_sqrt2_plateau_flag,
        "collapse_std_ratio_threshold": float(collapse_std_ratio_threshold),
        "collapse_flag": collapse_flag,
        **spectral_summary,
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
        "anomaly_correlation": anomaly_correlation,
        "nrmse_per_trajectory": nrmse_per_trajectory,
        "sqrt2_crossing_per_trajectory": per_trajectory_first_crossing(
            nrmse_per_trajectory, math.sqrt(2.0)
        ),
        "summary": summary,
    }


def _mean_temporal_spectral_entropy(values: np.ndarray) -> float:
    """Mean normalized rFFT entropy over trajectories and observed channels."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] < 3:
        raise ValueError(
            "Spectral entropy requires shape [N_trajectories,N_steps,D] with N_steps >= 3"
        )
    centered = values - values.mean(axis=1, keepdims=True)
    power = np.square(np.abs(np.fft.rfft(centered, axis=1)))[:, 1:, :]
    number_of_bins = int(power.shape[1])
    if number_of_bins <= 1:
        raise ValueError("Spectral entropy requires at least two non-DC bins")
    power_sum = power.sum(axis=1, keepdims=True)
    probabilities = power / np.maximum(power_sum, 1.0e-30)
    entropy = -np.sum(
        probabilities * np.log(probabilities + 1.0e-30), axis=1
    ) / math.log(number_of_bins)
    entropy = np.where(power_sum[:, 0, :] > 1.0e-30, entropy, 0.0)
    return float(entropy.mean())


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
    amplitude_threshold = max(
        float(summary.get("collapse_std_ratio_threshold", 0.8)), 1.0e-12
    )
    amplitude_health = min(
        max(float(summary.get("late_std_ratio", 0.0)) / amplitude_threshold, 0.0),
        1.0,
    )
    entropy_threshold = summary.get("spectral_entropy_ratio_threshold")
    if entropy_threshold is None:
        spectral_health = 1.0
    else:
        spectral_health = min(
            max(
                float(summary.get("spectral_entropy_ratio", 0.0))
                / max(float(entropy_threshold), 1.0e-12),
                0.0,
            ),
            1.0,
        )
    dynamics_health = min(amplitude_health, spectral_health)
    physical_health = bool(summary.get("physical_health_flag", True))
    physical_log_error = float(summary.get("physical_log_error", 0.0))
    joint_health = bool(physical_health and not bool(summary["collapse_flag"]))
    return (
        int(joint_health),
        int(not bool(summary["collapse_flag"])),
        int(physical_health),
        -physical_log_error,
        dynamics_health,
        int(not bool(summary.get("near_one_plateau_flag", False))),
        int(not bool(summary.get("sub_sqrt2_plateau_flag", False))),
        _crossing_rank(summary["first_crossing_1_0"], forecast_steps),
        _crossing_rank(summary["first_crossing_sqrt2"], forecast_steps),
        -float(summary["nrmse_auc_over_first_1200"]),
        -float(validation_mse),
    )


def is_better_rollout(candidate: dict, incumbent: dict | None) -> bool:
    return incumbent is None or rollout_selection_key(candidate) > rollout_selection_key(incumbent)
