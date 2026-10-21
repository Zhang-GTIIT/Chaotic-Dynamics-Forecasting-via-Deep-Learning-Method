from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def make_plots(
    prediction: np.ndarray,
    truth: np.ndarray,
    metrics: dict,
    checkpoint: dict,
    output: Path,
    ds: int,
    time_step: float,
    lyapunov_exponent: float,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    step = metrics["forecast_step"]
    mean_nrmse = metrics["mean_nrmse"]
    title_suffix = f", ds={ds}"

    figure, axes = plt.subplots(4, 1, figsize=(13, 14), sharex=True)
    extent = [1, truth.shape[1], 0, truth.shape[2] - 1]
    vmin = float(min(truth[0].min(), prediction[0].min()))
    vmax = float(max(truth[0].max(), prediction[0].max()))
    axes[0].imshow(truth[0].T, aspect="auto", origin="lower", extent=extent, cmap="turbo", vmin=vmin, vmax=vmax)
    axes[0].set_title("Ground Truth" + title_suffix)
    axes[1].imshow(prediction[0].T, aspect="auto", origin="lower", extent=extent, cmap="turbo", vmin=vmin, vmax=vmax)
    axes[1].set_title("Prediction" + title_suffix)
    axes[2].imshow(np.abs(prediction[0] - truth[0]).T, aspect="auto", origin="lower", extent=extent, cmap="magma")
    axes[2].set_title("Absolute Error" + title_suffix)
    axes[3].plot(step, mean_nrmse)
    for threshold in (0.5, 1.0, math.sqrt(2.0)):
        axes[3].axhline(threshold, linestyle="--", linewidth=1)
    axes[3].set_title("Mean NRMSE" + title_suffix)
    axes[3].set_xlabel("Forecast Step")
    for axis in axes[:3]:
        axis.set_ylabel("State index")
    axes[3].set_ylabel("NRMSE")
    _save(figure, output / "summary_4in1.png")

    for logarithmic, name in ((False, "linear"), (True, "log")):
        figure, axis = plt.subplots(figsize=(10, 6))
        axis.plot(step, np.maximum(mean_nrmse, 1.0e-10), label="Mean NRMSE")
        for threshold, label in ((0.5, "0.5"), (1.0, "1.0"), (math.sqrt(2.0), "sqrt(2)")):
            axis.axhline(threshold, linestyle="--", label=label)
        if logarithmic:
            axis.set_yscale("log")
        axis.set_xlabel("Forecast Step")
        axis.set_ylabel("Mean NRMSE")
        axis.set_title(f"NRMSE vs Forecast Step ({name})" + title_suffix)
        axis.grid(alpha=0.3)
        axis.legend()
        _save(figure, output / f"nrmse_step_{name}.png")

        figure, axis = plt.subplots(figsize=(10, 6))
        lyapunov_time = step * float(time_step) * float(lyapunov_exponent)
        axis.plot(lyapunov_time, np.maximum(mean_nrmse, 1.0e-10), label="Mean NRMSE")
        for threshold, label in ((0.5, "0.5"), (1.0, "1.0"), (math.sqrt(2.0), "sqrt(2)")):
            axis.axhline(threshold, linestyle="--", label=label)
        if logarithmic:
            axis.set_yscale("log")
        axis.set_xlabel("Lyapunov Time")
        axis.set_ylabel("Mean NRMSE")
        axis.set_title(f"NRMSE vs Lyapunov Time ({name})" + title_suffix)
        axis.grid(alpha=0.3)
        axis.legend()
        _save(figure, output / f"nrmse_lyapunov_{name}.png")

    figure, axis = plt.subplots(figsize=(12, 7))
    channels = min(8, truth.shape[2])
    for channel in range(channels):
        offset = channel * 5.0
        axis.plot(step, truth[0, :, channel] + offset, linewidth=1, label=f"truth ch{channel}" if channel == 0 else None)
        axis.plot(step, prediction[0, :, channel] + offset, linewidth=1, linestyle="--", label=f"prediction ch{channel}" if channel == 0 else None)
    axis.set_xlabel("Forecast Step")
    axis.set_ylabel("Offset channel value")
    axis.set_title("Channel Time Series" + title_suffix)
    axis.legend()
    axis.grid(alpha=0.2)
    _save(figure, output / "channel_timeseries.png")

    truth_centered = truth - truth.mean(axis=1, keepdims=True)
    prediction_centered = prediction - prediction.mean(axis=1, keepdims=True)
    truth_power = np.mean(np.abs(np.fft.rfft(truth_centered, axis=1)) ** 2, axis=(0, 2))
    prediction_power = np.mean(np.abs(np.fft.rfft(prediction_centered, axis=1)) ** 2, axis=(0, 2))
    frequency = np.fft.rfftfreq(truth.shape[1], d=time_step)
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.loglog(frequency[1:], truth_power[1:], label="Truth")
    axis.loglog(frequency[1:], prediction_power[1:], label="Prediction")
    axis.set_xlabel("Temporal Frequency")
    axis.set_ylabel("Mean Power")
    axis.set_title("Temporal Power Spectrum" + title_suffix)
    axis.grid(alpha=0.3)
    axis.legend()
    _save(figure, output / "power_spectrum.png")

    history = checkpoint.get("training_history", [])
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["validation_loss"] for row in history]
    for logarithmic, name in ((False, "linear"), (True, "log")):
        figure, axis = plt.subplots(figsize=(10, 6))
        axis.plot(epochs, train_loss, label="Train")
        axis.plot(epochs, val_loss, label="Validation")
        if logarithmic:
            axis.set_yscale("log")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Closed-loop MSE")
        axis.set_title(f"Training and Validation Loss ({name})" + title_suffix)
        axis.grid(alpha=0.3)
        axis.legend()
        _save(figure, output / f"loss_{name}.png")

    rollout_rows = [row for row in history if row.get("rollout")]
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    if rollout_rows:
        rollout_epochs = [row["epoch"] for row in rollout_rows]
        h1 = [row["rollout"]["H1.0"] or 1201 for row in rollout_rows]
        hsqrt2 = [row["rollout"]["Hsqrt2"] or 1201 for row in rollout_rows]
        late_std = [row["rollout"]["late_std_ratio"] for row in rollout_rows]
        axes[0].plot(rollout_epochs, h1, marker="o", label="H1.0")
        axes[0].plot(rollout_epochs, hsqrt2, marker="o", label="Hsqrt2")
        axes[1].plot(rollout_epochs, late_std, marker="o", label="late std ratio")
        axes[1].axhline(0.55, linestyle="--", color="red")
    axes[0].set_ylabel("Crossing step")
    axes[1].set_ylabel("Ratio")
    axes[1].set_xlabel("Epoch")
    axes[0].set_title("Validation Long-horizon Diagnostics" + title_suffix)
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    _save(figure, output / "validation_long_horizon.png")

    summary = metrics["summary"]
    figure, axis = plt.subplots(figsize=(8, 6))
    values = [summary["truth_spectral_entropy"], summary["prediction_spectral_entropy"]]
    axis.bar(["Truth", "Prediction"], values)
    axis.text(0.5, 0.95, f"ratio={summary['spectral_entropy_ratio']:.3f}", transform=axis.transAxes, ha="center", va="top")
    axis.set_ylabel("Normalized Spectral Entropy")
    axis.set_title("Temporal Spectral Entropy" + title_suffix)
    axis.grid(axis="y", alpha=0.3)
    _save(figure, output / "spectral_entropy.png")

    crossings = metrics["sqrt2_crossing_per_trajectory"]
    reached = crossings[np.isfinite(crossings)]
    figure, axis = plt.subplots(figsize=(10, 6))
    if reached.size:
        axis.hist(reached, bins=min(30, max(5, int(np.sqrt(reached.size)))))
    axis.text(0.98, 0.95, f"unreached={np.count_nonzero(~np.isfinite(crossings))}/{crossings.size}", transform=axis.transAxes, ha="right", va="top")
    axis.set_xlabel("First sqrt(2) Crossing Step")
    axis.set_ylabel("Trajectory Count")
    axis.set_title("Trajectory Crossing Distribution" + title_suffix)
    axis.grid(alpha=0.3)
    _save(figure, output / "trajectory_crossing_distribution.png")

