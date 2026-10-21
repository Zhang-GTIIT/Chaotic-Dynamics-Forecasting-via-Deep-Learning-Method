"""Load one trained checkpoint and run its formal autoregressive rollout."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from data_loader import (
    build_evaluation_windows,
    expected_data_path,
    load_h5_data,
    normalize_data,
    resolve_project_path,
    validate_pred_len,
)
from metrics import compute_nrmse_metrics, compute_periodic_flow_physical_metrics
from model import build_model, model_display_name
from rollout import rollout_model
from train_transformer import (
    build_model_signature,
    checkpoint_directory,
    load_torch_checkpoint,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "code" / "config.yaml"
MODEL_TYPES = ("easy_attention_sgu", "self_attention", "lstm")
CHECKPOINT_NAMES = ("latest", "best_val", "best_rollout")


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _validate_checkpoint_metadata(
    checkpoint: dict,
    config: dict,
    model_type: str,
    ds: int,
    sim_time: int,
    seq_len: int,
    pred_len: int,
) -> None:
    expected = {
        "model_type": model_type,
        "ds": int(ds),
        "sim_time": int(sim_time),
        "seq_len": int(seq_len),
        "pred_len": int(pred_len),
        "stride": int(config["training"]["stride"]),
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(
                f"Checkpoint mismatch for {key}: found {checkpoint.get(key)!r}, "
                f"expected {value!r}"
            )


def _validate_model_signature(checkpoint: dict, model_signature: dict) -> None:
    if checkpoint.get("format_version") != 2 or "model_signature" not in checkpoint:
        raise ValueError(
            "Legacy checkpoint format is incompatible with the fixed project. "
            "Old s64/p3 or format_version=1 checkpoints cannot be used for prediction."
        )
    if checkpoint["model_signature"] != model_signature:
        raise ValueError("Checkpoint model_signature differs from the current model signature")


def _load_evaluation_data(
    config: dict,
    checkpoint: dict,
    ds: int,
    sim_time: int,
) -> tuple[np.ndarray, np.ndarray, str, Path]:
    spatial_resolution = int(config["data_generation"]["spatial_resolution"])
    test_data_path = config["prediction"].get("test_data_path")
    if test_data_path:
        path = resolve_project_path(test_data_path)
        data = load_h5_data(path, spatial_resolution, ds)
        trajectory_ids = np.arange(data.shape[0], dtype=np.int64)
        return data, trajectory_ids, "independent_test", path

    path = expected_data_path(config, ds, sim_time)
    data = load_h5_data(path, spatial_resolution, ds)
    trajectory_ids = np.asarray(
        checkpoint.get("validation_trajectory_ids", []), dtype=np.int64
    )
    if trajectory_ids.size == 0:
        raise ValueError("Checkpoint has no held-out validation trajectory IDs")
    if np.any(trajectory_ids < 0) or np.any(trajectory_ids >= data.shape[0]):
        raise ValueError("Checkpoint validation trajectory IDs are invalid for the HDF5 data")
    train_ids = np.asarray(checkpoint.get("train_trajectory_ids", []), dtype=np.int64)
    if np.intersect1d(train_ids, trajectory_ids).size:
        raise ValueError("Checkpoint contains overlapping train and validation trajectory IDs")
    return data, trajectory_ids, "held_out_validation", path


def _result_directories(
    model_type: str,
    ds: int,
    sim_time: int,
    seq_len: int,
    pred_len: int,
    checkpoint_name: str,
) -> tuple[Path, Path]:
    relative = (
        Path(model_type)
        / f"ds{ds}"
        / f"t{sim_time}"
        / f"s{seq_len}_p{pred_len}"
        / checkpoint_name
    )
    return PROJECT_ROOT / "results" / "data" / relative, PROJECT_ROOT / "results" / "plots" / relative


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _save_stepwise_csv(metrics: dict, path: Path) -> None:
    columns = [
        "forecast_step",
        "physical_time",
        "lyapunov_time",
        "mean_nrmse",
        "median_nrmse",
        "p10_nrmse",
        "p90_nrmse",
        "prediction_std",
        "truth_std",
        "std_ratio",
        "anomaly_correlation",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(zip(*(metrics[column] for column in columns)))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _block_boundaries(num_steps: int, model_type: str, pred_len: int) -> list[int]:
    if model_type == "lstm":
        return []
    return list(range(int(pred_len), int(num_steps), int(pred_len)))


def _decorate_nrmse_axis(
    axis,
    metrics: dict,
    model_type: str,
    pred_len: int,
    x_kind: str,
) -> None:
    summary = metrics["summary"]
    thresholds = ((0.5, "0.5"), (1.0, "1.0"), (math.sqrt(2.0), "sqrt(2)"))
    for value, label in thresholds:
        axis.axhline(value, linestyle="--", linewidth=1.0, label=f"NRMSE={label}")

    crossing = summary["first_crossing_sqrt2"]
    if crossing["reached"]:
        index = int(crossing["crossing_step"]) - 1
        if x_kind == "step":
            x_value = metrics["forecast_step"][index]
        else:
            x_value = metrics["lyapunov_time"][index]
        axis.axvline(
            x_value,
            color="tab:red",
            linestyle=":",
            linewidth=1.5,
            label="First sqrt(2) crossing",
        )

    boundaries = _block_boundaries(
        len(metrics["forecast_step"]), model_type, pred_len
    )
    for boundary_index, boundary in enumerate(boundaries):
        if x_kind == "step":
            x_value = boundary
        else:
            x_value = (
                boundary
                * (metrics["physical_time"][0] / metrics["forecast_step"][0])
                * (metrics["lyapunov_time"][0] / metrics["physical_time"][0])
            )
        axis.axvline(
            x_value,
            color="0.5",
            alpha=0.16,
            linewidth=0.8,
            label="Block boundary" if boundary_index == 0 else None,
        )


def _plot_summary_4in1(
    truth: np.ndarray,
    prediction: np.ndarray,
    metrics: dict,
    plot_path: Path,
    model_type: str,
    pred_len: int,
    title: str,
) -> None:
    display_truth = truth[0]
    display_prediction = prediction[0]
    absolute_error = np.abs(display_prediction - display_truth)
    shared_min = min(float(display_truth.min()), float(display_prediction.min()))
    shared_max = max(float(display_truth.max()), float(display_prediction.max()))
    num_steps, spatial_dim = display_truth.shape
    extent = [1, num_steps, 0, spatial_dim]

    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    image = axes[0, 0].imshow(
        display_truth.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=shared_min,
        vmax=shared_max,
        cmap="viridis",
    )
    axes[0, 0].set_title("Ground Truth")
    figure.colorbar(image, ax=axes[0, 0])
    image = axes[0, 1].imshow(
        display_prediction.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=shared_min,
        vmax=shared_max,
        cmap="viridis",
    )
    axes[0, 1].set_title("Prediction")
    figure.colorbar(image, ax=axes[0, 1])
    image = axes[1, 0].imshow(
        absolute_error.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="magma",
    )
    axes[1, 0].set_title("Absolute Error")
    figure.colorbar(image, ax=axes[1, 0])

    axes[1, 1].plot(metrics["forecast_step"], metrics["mean_nrmse"], label="Mean NRMSE")
    axes[1, 1].fill_between(
        metrics["forecast_step"],
        metrics["p10_nrmse"],
        metrics["p90_nrmse"],
        alpha=0.2,
        label="10th-90th percentile",
    )
    _decorate_nrmse_axis(axes[1, 1], metrics, model_type, pred_len, "step")
    axes[1, 1].set_title("Mean NRMSE")
    axes[1, 1].set_ylabel("NRMSE")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend(fontsize=8)

    for axis in axes.flat:
        axis.set_xlabel("Forecast Step")
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.set_ylabel("Observed Channel")
        for boundary in _block_boundaries(num_steps, model_type, pred_len):
            axis.axvline(boundary, color="white", alpha=0.18, linewidth=0.7)
    figure.suptitle(title)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_nrmse_curve(
    metrics: dict,
    plot_path: Path,
    model_type: str,
    pred_len: int,
    x_kind: str,
    logarithmic: bool,
    floor: float,
    ds: int | None = None,
) -> None:
    x = metrics["forecast_step"] if x_kind == "step" else metrics["lyapunov_time"]
    lower = metrics["p10_nrmse"]
    upper = metrics["p90_nrmse"]
    mean = metrics["mean_nrmse"]
    if logarithmic:
        lower = np.maximum(lower, floor)
        upper = np.maximum(upper, floor)
        mean = np.maximum(mean, floor)
    figure, axis = plt.subplots(figsize=(11, 6.5))
    axis.plot(x, mean, linewidth=1.8, label="Mean NRMSE")
    axis.fill_between(x, lower, upper, alpha=0.2, label="10th-90th percentile")
    _decorate_nrmse_axis(axis, metrics, model_type, pred_len, x_kind)
    if logarithmic:
        axis.set_yscale("log")
    axis.set_xlabel("Forecast Step" if x_kind == "step" else "Lyapunov Time")
    axis.set_ylabel("NRMSE")
    title = (
        f"{model_display_name(model_type)} NRMSE vs "
        f"{'Forecast Step' if x_kind == 'step' else 'Lyapunov Time'}"
    )
    axis.set_title(title + (f", ds={ds}" if ds is not None else ""))
    if model_type == "lstm" and x_kind == "step":
        axis.set_xlim(left=1)
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_loss(history: list[dict], path: Path, logarithmic: bool) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    if history:
        epochs = np.asarray([row["epoch"] for row in history], dtype=np.int64)
        train_loss = np.asarray([row["train_loss"] for row in history], dtype=np.float64)
        val_loss = np.asarray([row["validation_loss"] for row in history], dtype=np.float64)
        if logarithmic:
            train_loss = np.maximum(train_loss, np.finfo(np.float64).tiny)
            val_loss = np.maximum(val_loss, np.finfo(np.float64).tiny)
            axis.set_yscale("log")
        axis.plot(epochs, train_loss, label="Training")
        axis.plot(epochs, val_loss, label="Validation")
        axis.legend()
    else:
        axis.text(0.5, 0.5, "No training history in checkpoint", ha="center", va="center")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE Loss")
    axis.set_title(f"Training/Validation Loss ({'log' if logarithmic else 'linear'})")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _add_curriculum_markers(axis) -> None:
    labels = {20: "H=10 → H=20", 50: "H=20 → H=50", 100: "H=50 → H=100"}
    for epoch, label in labels.items():
        axis.axvline(epoch, color="0.35", linestyle=":", linewidth=1.0)
        axis.annotate(
            label,
            xy=(epoch, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(3, -3),
            textcoords="offset points",
            rotation=90,
            ha="left",
            va="top",
            fontsize=8,
        )


def _plot_lstm_current_h_loss(
    history: list[dict], path: Path, logarithmic: bool, ds: int
) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    epochs = np.asarray([row["epoch"] for row in history], dtype=np.int64)
    train_loss = np.asarray(
        [row["train_current_h_loss"] for row in history], dtype=np.float64
    )
    validation_loss = np.asarray(
        [row["validation_current_h_loss"] for row in history], dtype=np.float64
    )
    if logarithmic:
        floor = np.finfo(np.float64).tiny
        train_loss = np.maximum(train_loss, floor)
        validation_loss = np.maximum(validation_loss, floor)
        axis.set_yscale("log")
    axis.plot(epochs, train_loss, label="Training (current H)")
    axis.plot(epochs, validation_loss, label="Validation (current H)")
    _add_curriculum_markers(axis)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Closed-loop MSE Loss")
    axis.set_title(
        f"Training/Validation Current-H Loss ({'log' if logarithmic else 'linear'}), ds={ds}"
    )
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_lstm_validation_100_loss(
    history: list[dict], path: Path, ds: int
) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    epochs = np.asarray([row["epoch"] for row in history], dtype=np.int64)
    values = np.asarray(
        [row["validation_100_loss"] for row in history], dtype=np.float64
    )
    axis.plot(epochs, values, label="Validation (H=100)")
    _add_curriculum_markers(axis)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("100-step Closed-loop MSE Loss")
    axis.set_title(f"Fixed 100-step Validation Loss, ds={ds}")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_channels(
    truth: np.ndarray, prediction: np.ndarray, path: Path, ds: int | None = None
) -> None:
    spatial_dim = truth.shape[2]
    displayed = np.unique(
        np.rint(np.linspace(0, spatial_dim - 1, min(spatial_dim, 8))).astype(np.int64)
    )
    figure, axes = plt.subplots(
        displayed.size, 1, figsize=(13, 2.8 * displayed.size), sharex=True
    )
    axes = np.atleast_1d(axes)
    steps = np.arange(1, truth.shape[1] + 1)
    for channel, axis in zip(displayed, axes):
        axis.plot(steps, truth[0, :, channel], label="Truth", linewidth=1.2)
        axis.plot(steps, prediction[0, :, channel], label="Prediction", linewidth=1.0)
        axis.set_ylabel(f"Channel {channel}")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[-1].set_xlabel("Forecast Step")
    figure.suptitle(
        "Observed-channel Truth and Prediction (trajectory 1)"
        + (f", ds={ds}" if ds is not None else "")
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_power_spectrum(
    truth: np.ndarray,
    prediction: np.ndarray,
    time_step: float,
    path: Path,
    ds: int | None = None,
) -> None:
    truth_power = np.abs(np.fft.rfft(truth, axis=1)) ** 2
    prediction_power = np.abs(np.fft.rfft(prediction, axis=1)) ** 2
    truth_mean = truth_power.mean(axis=(0, 2))
    prediction_mean = prediction_power.mean(axis=(0, 2))
    frequency = np.fft.rfftfreq(truth.shape[1], d=float(time_step))
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.semilogy(frequency, np.maximum(truth_mean, 1.0e-30), label="Ground Truth")
    axis.semilogy(frequency, np.maximum(prediction_mean, 1.0e-30), label="Prediction")
    axis.set_xlabel("Temporal Frequency")
    axis.set_ylabel("Mean Power")
    axis.set_title(
        "Average Temporal Power Spectrum" + (f", ds={ds}" if ds is not None else "")
    )
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_std_ratio(metrics: dict, path: Path, ds: int | None = None) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(metrics["forecast_step"], metrics["std_ratio"], label="Prediction/Truth std")
    axis.axhline(1.0, color="black", linestyle="--", label="Ratio=1")
    axis.axhline(
        metrics["summary"]["collapse_std_ratio_threshold"],
        color="tab:red",
        linestyle=":",
        label="Collapse threshold",
    )
    axis.set_xlabel("Forecast Step")
    axis.set_ylabel("Standard-deviation Ratio")
    axis.set_title(
        "Prediction/Truth Standard-deviation Ratio"
        + (f", ds={ds}" if ds is not None else "")
    )
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_sqrt2_crossings(
    metrics: dict, path: Path, ds: int | None = None
) -> None:
    crossings = metrics["sqrt2_crossing_per_trajectory"]
    reached = crossings[np.isfinite(crossings)]
    unreached = int(np.count_nonzero(~np.isfinite(crossings)))
    figure, axis = plt.subplots(figsize=(10, 6))
    if reached.size:
        bins = min(30, max(5, int(np.sqrt(reached.size))))
        axis.hist(reached, bins=bins, alpha=0.8)
    axis.text(
        0.98,
        0.95,
        f"Unreached: {unreached}/{crossings.size}",
        transform=axis.transAxes,
        ha="right",
        va="top",
    )
    axis.set_xlabel("First Forecast Step Reaching sqrt(2)")
    axis.set_ylabel("Trajectory Count")
    axis.set_title(
        "Per-trajectory First sqrt(2) Crossing"
        + (f", ds={ds}" if ds is not None else "")
    )
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_plots(
    truth: np.ndarray,
    prediction: np.ndarray,
    metrics: dict,
    checkpoint: dict,
    plot_dir: Path,
    model_type: str,
    pred_len: int,
    config: dict,
    ds: int,
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    title = (
        f"{model_display_name(model_type)} | ds={ds} | "
        f"epoch={checkpoint['epoch']} | P={pred_len}"
    )
    _plot_summary_4in1(
        truth,
        prediction,
        metrics,
        plot_dir / "summary_4in1.png",
        model_type,
        pred_len,
        title,
    )
    floor = float(config["prediction"]["nrmse_plot_floor"])
    lstm_ds = int(ds) if model_type == "lstm" else None
    _plot_nrmse_curve(
        metrics, plot_dir / "nrmse_step_linear.png", model_type, pred_len, "step", False, floor, lstm_ds
    )
    _plot_nrmse_curve(
        metrics, plot_dir / "nrmse_step_log.png", model_type, pred_len, "step", True, floor, lstm_ds
    )
    _plot_nrmse_curve(
        metrics,
        plot_dir / "nrmse_lyapunov_linear.png",
        model_type,
        pred_len,
        "lyapunov",
        False,
        floor,
        lstm_ds,
    )
    _plot_nrmse_curve(
        metrics,
        plot_dir / "nrmse_lyapunov_log.png",
        model_type,
        pred_len,
        "lyapunov",
        True,
        floor,
        lstm_ds,
    )
    history = list(checkpoint.get("training_history", []))
    if model_type == "lstm":
        _plot_lstm_current_h_loss(
            history, plot_dir / "loss_current_h_linear.png", False, ds
        )
        _plot_lstm_current_h_loss(
            history, plot_dir / "loss_current_h_log.png", True, ds
        )
        _plot_lstm_validation_100_loss(
            history, plot_dir / "validation_100_loss.png", ds
        )
    else:
        _plot_loss(history, plot_dir / "loss_linear.png", False)
        _plot_loss(history, plot_dir / "loss_log.png", True)
    _plot_channels(truth, prediction, plot_dir / "channel_timeseries.png", lstm_ds)
    _plot_power_spectrum(
        truth,
        prediction,
        float(config["data_generation"]["time_step"]),
        plot_dir / "power_spectrum.png",
        lstm_ds,
    )
    _plot_std_ratio(metrics, plot_dir / "std_ratio.png", lstm_ds)
    _plot_sqrt2_crossings(
        metrics, plot_dir / "sqrt2_crossing_distribution.png", lstm_ds
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained KS checkpoint")
    parser.add_argument("--model", choices=MODEL_TYPES, required=True)
    parser.add_argument("--ds", type=int, choices=(1, 2, 4, 8, 16, 32), required=True)
    parser.add_argument("--sim-time", type=int, default=10000)
    parser.add_argument("--pred-len", type=int, default=100)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--trajectories", type=int, default=None)
    parser.add_argument("--checkpoint", choices=CHECKPOINT_NAMES, default="best_rollout")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=(
            "Override data_generation.save_dir without editing config.yaml. "
            "Useful when moving between Windows and Linux servers."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config = copy.deepcopy(load_config())
    if args.model == "lstm":
        configured_ds = [
            int(value) for value in config["data_generation"]["downsampling_factors"]
        ]
        if args.ds not in configured_ds:
            config["data_generation"]["downsampling_factors"] = configured_ds + [args.ds]
    if args.data_dir is not None:
        config["data_generation"]["save_dir"] = str(
            Path(args.data_dir).expanduser().resolve()
        )
    validate_pred_len(config, args.pred_len)
    if args.model == "lstm":
        mapping = config["lstm_training"]["seq_len_by_ds"]
        expected_seq_len = mapping.get(args.ds, mapping.get(str(args.ds)))
        if expected_seq_len is None:
            raise ValueError(f"No LSTM seq_len is configured for ds={args.ds}")
        seq_len = int(expected_seq_len)
    else:
        seq_len = int(config["training"]["seq_len"])
    total_steps = args.steps or int(config["prediction"]["total_pred_steps"])
    minimum_steps = int(config["prediction"]["minimum_required_steps"])
    if total_steps < minimum_steps:
        raise ValueError(
            f"Prediction requires at least {minimum_steps} forecast steps; received {total_steps}"
        )
    requested_trajectories = args.trajectories or int(
        config["prediction"]["num_eval_trajectories"]
    )
    if requested_trajectories <= 0:
        raise ValueError("--trajectories must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = checkpoint_directory(
        args.model, args.ds, args.sim_time, seq_len, args.pred_len
    ) / f"{args.checkpoint}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = load_torch_checkpoint(checkpoint_path, device)
    if args.model == "lstm":
        checkpoint_seq_len = int(
            checkpoint.get("model_signature", {}).get(
                "seq_len", checkpoint.get("seq_len", -1)
            )
        )
        if checkpoint_seq_len != seq_len:
            raise ValueError(
                f"LSTM checkpoint seq_len={checkpoint_seq_len} differs from "
                f"configured ds{args.ds} seq_len={seq_len}"
            )
        seq_len = checkpoint_seq_len
        config["training"]["seq_len"] = seq_len
    _validate_checkpoint_metadata(
        checkpoint,
        config,
        args.model,
        args.ds,
        args.sim_time,
        seq_len,
        args.pred_len,
    )
    data, available_ids, data_source, data_path = _load_evaluation_data(
        config, checkpoint, args.ds, args.sim_time
    )
    input_dim = int(data.shape[2])
    model_signature = build_model_signature(
        config,
        args.model,
        args.ds,
        args.sim_time,
        input_dim,
        args.pred_len,
    )
    if args.model == "lstm":
        model_signature["training_mode"] = str(
            config["lstm_training"]["training_mode"]
        )
    _validate_model_signature(checkpoint, model_signature)
    data_mean = np.asarray(checkpoint["data_mean"], dtype=np.float32).reshape(-1)
    data_std = np.asarray(checkpoint["data_std"], dtype=np.float32).reshape(-1)
    if data_mean.size != input_dim or data_std.size != input_dim:
        raise ValueError(
            f"Checkpoint normalization dimension mismatch: input_dim={input_dim}, "
            f"data_mean={data_mean.size}, data_std={data_std.size}"
        )
    if not np.all(np.isfinite(data_mean)) or not np.all(np.isfinite(data_std)):
        raise ValueError("Checkpoint data_mean/data_std contain non-finite values")
    if np.any(data_std <= 1.0e-8):
        raise ValueError("Checkpoint data_std contains a component <= 1e-8")

    initial_raw, truth, evaluated_ids, starts = build_evaluation_windows(
        data=data,
        trajectory_ids=available_ids,
        seq_len=seq_len,
        forecast_steps=total_steps,
        max_trajectories=requested_trajectories,
    )
    initial_normalized = normalize_data(initial_raw, data_mean, data_std)
    initial_tensor = torch.from_numpy(initial_normalized).to(
        device=device, dtype=torch.float32
    )
    model = build_model(args.model, input_dim, config, args.pred_len).to(
        device=device, dtype=torch.float32
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    prediction_normalized = rollout_model(
        model_type=args.model,
        model=model,
        initial_window=initial_tensor,
        total_steps=total_steps,
        seq_len=seq_len,
        pred_len=args.pred_len,
    )
    prediction = prediction_normalized.cpu().numpy() * data_std + data_mean

    metrics = compute_nrmse_metrics(
        prediction=prediction,
        truth=truth,
        train_std=data_std,
        time_step=float(config["data_generation"]["time_step"]),
        lyapunov_exponent=float(config["prediction"]["lyapunov_exponent"]),
        late_window_steps=int(config["rollout_validation"]["late_window_steps"]),
        collapse_std_ratio_threshold=float(
            config["rollout_validation"]["collapse_std_ratio_threshold"]
        ),
        spectral_entropy_ratio_threshold=(
            float(config["lstm_training"]["spectral_entropy_ratio_threshold"])
            if args.model == "lstm"
            else None
        ),
        validation_mse=float(checkpoint["best_validation_loss"]),
        train_mean=data_mean,
    )
    if config.get("experiment", {}).get("system") == "kolmogorov2d":
        metrics["summary"].update(
            compute_periodic_flow_physical_metrics(
                prediction,
                truth,
                grid_size=int(config["spatial_adapter"]["grid_size"]),
                late_window_steps=int(
                    config["rollout_validation"]["late_window_steps"]
                ),
            )
        )
    data_dir, plot_dir = _result_directories(
        args.model,
        args.ds,
        args.sim_time,
        seq_len,
        args.pred_len,
        args.checkpoint,
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    _save_stepwise_csv(metrics, data_dir / "stepwise_metrics.csv")
    np.savez_compressed(
        data_dir / "per_trajectory_nrmse.npz",
        nrmse_per_trajectory=metrics["nrmse_per_trajectory"],
        sqrt2_crossing_step=metrics["sqrt2_crossing_per_trajectory"],
        trajectory_ids=evaluated_ids,
        start_indices=starts,
    )
    np.savez_compressed(
        data_dir / "predictions_truth.npz",
        prediction=prediction.astype(np.float32),
        truth=truth.astype(np.float32),
        initial_window=initial_raw.astype(np.float32),
        trajectory_ids=evaluated_ids,
        start_indices=starts,
    )
    summary = {
        "model_type": args.model,
        "model_label": model_display_name(args.model),
        "rollout_type": (
            f"Block autoregression, P={args.pred_len}"
            if args.model != "lstm"
            else "Stateful one-step autoregression"
        ),
        "ds": int(args.ds),
        "sim_time": int(args.sim_time),
        "seq_len": seq_len,
        "pred_len": int(args.pred_len),
        "checkpoint": args.checkpoint,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "training_mode": checkpoint.get("training_mode"),
        "data_source": data_source,
        "data_path": str(data_path),
        "trajectory_ids": evaluated_ids.tolist(),
        "start_indices": starts.tolist(),
        "time_step": float(config["data_generation"]["time_step"]),
        "lyapunov_exponent": float(config["prediction"]["lyapunov_exponent"]),
        **metrics["summary"],
    }
    _atomic_json(summary, data_dir / "summary.json")
    save_plots(
        truth,
        prediction,
        metrics,
        checkpoint,
        plot_dir,
        args.model,
        args.pred_len,
        config,
        args.ds,
    )
    print(f"Data source: {data_source}")
    print(f"Summary: {data_dir / 'summary.json'}")
    print(f"Plots: {plot_dir}")


if __name__ == "__main__":
    main()
