"""Create the compact, system-specific final result package from a completed prediction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAMES = {
    "easy_attention_sgu": "Easy Attention + SGU",
    "self_attention": "Self Attention",
    "lstm": "LSTM",
}


def load_config() -> dict:
    with (PROJECT_ROOT / "code" / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_checkpoint(path: Path) -> dict:
    with path.open("rb") as handle:
        try:
            return torch.load(handle, map_location="cpu", weights_only=False)
        except TypeError:
            handle.seek(0)
            return torch.load(handle, map_location="cpu")


def read_stepwise(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty metrics file: {path}")
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        for key in rows[0]
    }


def write_curve(path: Path, columns: dict[str, np.ndarray]) -> None:
    names = list(columns)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(names)
        writer.writerows(zip(*(columns[name] for name in names)))


def write_history(path: Path, history: list[dict]) -> None:
    scalar_keys = sorted(
        {
            key
            for row in history
            for key, value in row.items()
            if value is None or isinstance(value, (str, int, float, bool))
        }
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        for row in history:
            writer.writerow({key: row.get(key) for key in scalar_keys})


def crossing_value(summary: dict, key: str) -> float:
    crossing = summary[key]
    return float(crossing["crossing_step"]) if crossing["reached"] else np.nan


def plot_nrmse(metrics: dict, summary: dict, title: str, path: Path) -> None:
    x = metrics["forecast_step"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, metrics["mean_nrmse"], label="Mean NRMSE", linewidth=1.8)
    ax.plot(x, metrics["median_nrmse"], label="Median NRMSE", linewidth=1.2)
    ax.fill_between(x, metrics["p10_nrmse"], metrics["p90_nrmse"], alpha=0.2, label="P10-P90")
    for value, label in ((0.5, "0.5"), (1.0, "1.0"), (math.sqrt(2.0), "sqrt(2)")):
        ax.axhline(value, color="0.35", linestyle="--", linewidth=0.9, label=f"NRMSE={label}")
    crossing = summary["first_crossing_sqrt2"]
    if crossing["reached"]:
        ax.axvline(crossing["crossing_step"], color="tab:red", linestyle=":")
    ax.set(xlabel="Forecast step", ylabel="NRMSE", title=title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def plot_thresholds(summary: dict, title: str, path: Path) -> None:
    keys = ["first_crossing_0_5", "first_crossing_1_0", "first_crossing_sqrt2"]
    labels = ["0.5", "1.0", "sqrt(2)"]
    values = [crossing_value(summary, key) for key in keys]
    maximum = int(summary["forecast_steps"])
    display = [maximum + 1 if np.isnan(value) else value for value in values]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(labels, display, color=["tab:blue", "tab:orange", "tab:red"])
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            "not reached" if np.isnan(value) else str(int(value)),
            ha="center",
            va="bottom",
        )
    ax.set(xlabel="NRMSE threshold", ylabel="First crossing step", title=title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def plot_loss(history: list[dict], title: str, path: Path) -> None:
    epochs = np.asarray([row["epoch"] for row in history])
    train = np.asarray([row["train_loss"] for row in history])
    validation = np.asarray([row["validation_loss"] for row in history])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.semilogy(epochs, np.maximum(train, 1.0e-15), label="Train")
    ax.semilogy(epochs, np.maximum(validation, 1.0e-15), label="Validation")
    ax.set(xlabel="Epoch", ylabel="MSE", title=title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def plot_lorenz(
    truth: np.ndarray,
    prediction: np.ndarray,
    metrics: dict,
    summary: dict,
    history: list[dict],
    title: str,
    output: Path,
) -> None:
    steps = np.arange(1, truth.shape[0] + 1)
    limit = min(800, truth.shape[0])
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for axis, values, label in zip(
        axes,
        (truth[:limit].T, prediction[:limit].T, np.abs(prediction[:limit] - truth[:limit]).T),
        ("Truth", "Prediction", "Absolute error"),
    ):
        image = axis.imshow(values, aspect="auto", origin="lower", cmap="viridis")
        axis.set_ylabel("State index")
        axis.set_title(label)
        fig.colorbar(image, ax=axis)
    axes[-1].set_xlabel("Forecast step")
    fig.suptitle(f"{title}: representative state evolution")
    fig.tight_layout()
    fig.savefig(output / "01_state_evolution.png", dpi=240)
    plt.close(fig)

    selected = np.unique(np.rint(np.linspace(0, truth.shape[1] - 1, 5)).astype(int))
    fig, axes = plt.subplots(selected.size, 1, figsize=(12, 10), sharex=True)
    for axis, index in zip(np.atleast_1d(axes), selected):
        axis.plot(steps, truth[:, index], label="Truth", linewidth=1.1)
        axis.plot(steps, prediction[:, index], label="Prediction", linewidth=0.9)
        axis.set_ylabel(f"x[{index}]")
        axis.grid(alpha=0.25)
    axes[0].legend()
    axes[-1].set_xlabel("Forecast step")
    fig.suptitle(f"{title}: selected variables")
    fig.tight_layout()
    fig.savefig(output / "02_selected_variables.png", dpi=240)
    plt.close(fig)
    plot_nrmse(metrics, summary, f"{title}: NRMSE", output / "03_nrmse.png")
    plot_thresholds(summary, f"{title}: first crossings", output / "04_thresholds.png")
    plot_loss(history, f"{title}: training history", output / "05_training_loss.png")


def flow_diagnostics(vorticity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = vorticity.shape[-1]
    modes = np.fft.fftfreq(n, d=1.0 / n)
    kx = modes.reshape(n, 1)
    ky = modes.reshape(1, n)
    k2 = kx * kx + ky * ky
    safe = np.where(k2 == 0, 1.0, k2)
    omega_hat = np.fft.fft2(vorticity, axes=(-2, -1))
    psi_hat = omega_hat / safe
    psi_hat[..., 0, 0] = 0
    u = np.fft.ifft2(1j * ky * psi_hat, axes=(-2, -1)).real
    v = np.fft.ifft2(-1j * kx * psi_hat, axes=(-2, -1)).real
    energy = 0.5 * np.mean(u * u + v * v, axis=(-2, -1))
    enstrophy = 0.5 * np.mean(vorticity * vorticity, axis=(-2, -1))
    return energy, enstrophy


def radial_spectrum(vorticity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = vorticity.shape[-1]
    modes = np.fft.fftfreq(n, d=1.0 / n)
    kx, ky = np.meshgrid(modes, modes, indexing="ij")
    shells = np.rint(np.sqrt(kx * kx + ky * ky)).astype(int)
    power = np.abs(np.fft.fft2(vorticity)) ** 2 / (n ** 4)
    maximum = n // 2
    spectrum = np.asarray([power[shells == shell].sum() for shell in range(1, maximum + 1)])
    return np.arange(1, maximum + 1), spectrum


def plot_kolmogorov(
    truth_flat: np.ndarray,
    prediction_flat: np.ndarray,
    metrics: dict,
    summary: dict,
    history: list[dict],
    title: str,
    output: Path,
) -> None:
    n = int(round(math.sqrt(truth_flat.shape[1])))
    if n * n != truth_flat.shape[1]:
        raise ValueError("Kolmogorov feature dimension is not a square")
    truth = truth_flat.reshape(-1, n, n)
    prediction = prediction_flat.reshape(-1, n, n)
    selected_steps = np.unique(np.rint(np.linspace(0, truth.shape[0] - 1, 4)).astype(int))
    fig, axes = plt.subplots(3, selected_steps.size, figsize=(4 * selected_steps.size, 10))
    scale = max(float(np.abs(truth[selected_steps]).max()), float(np.abs(prediction[selected_steps]).max()))
    for column, step in enumerate(selected_steps):
        for row, values, label in (
            (0, truth[step], "Truth"),
            (1, prediction[step], "Prediction"),
            (2, prediction[step] - truth[step], "Error"),
        ):
            image = axes[row, column].imshow(values.T, origin="lower", cmap="RdBu_r", vmin=-scale, vmax=scale)
            axes[row, column].set_title(f"{label}, step {step + 1}")
            fig.colorbar(image, ax=axes[row, column], fraction=0.046)
    fig.suptitle(f"{title}: vorticity snapshots")
    fig.tight_layout()
    fig.savefig(output / "01_vorticity_snapshots.png", dpi=220)
    plt.close(fig)

    y_index = n // 2
    limit = min(1000, truth.shape[0])
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fields = (
        truth[:limit, :, y_index].T,
        prediction[:limit, :, y_index].T,
        np.abs(prediction[:limit, :, y_index] - truth[:limit, :, y_index]).T,
    )
    for axis, values, label in zip(axes, fields, ("Truth", "Prediction", "Absolute error")):
        image = axis.imshow(values, aspect="auto", origin="lower", cmap="viridis")
        axis.set_ylabel("x index")
        axis.set_title(label)
        fig.colorbar(image, ax=axis)
    axes[-1].set_xlabel("Forecast step")
    fig.suptitle(f"{title}: vorticity evolution at y=N/2")
    fig.tight_layout()
    fig.savefig(output / "02_vorticity_evolution.png", dpi=220)
    plt.close(fig)

    truth_energy, truth_enstrophy = flow_diagnostics(truth)
    pred_energy, pred_enstrophy = flow_diagnostics(prediction)
    steps = np.arange(1, truth.shape[0] + 1)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(steps, truth_energy, label="Truth")
    axes[0].plot(steps, pred_energy, label="Prediction")
    axes[0].set_ylabel("Kinetic energy")
    axes[1].plot(steps, truth_enstrophy, label="Truth")
    axes[1].plot(steps, pred_enstrophy, label="Prediction")
    axes[1].set_ylabel("Enstrophy")
    axes[1].set_xlabel("Forecast step")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    fig.suptitle(f"{title}: energy and enstrophy")
    fig.tight_layout()
    fig.savefig(output / "03_energy_enstrophy.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for step in selected_steps:
        wave, truth_spectrum = radial_spectrum(truth[step])
        _, pred_spectrum = radial_spectrum(prediction[step])
        ax.loglog(wave, np.maximum(truth_spectrum, 1.0e-15), label=f"Truth step {step + 1}")
        ax.loglog(wave, np.maximum(pred_spectrum, 1.0e-15), "--", label=f"Pred step {step + 1}")
    ax.set(xlabel="Wavenumber shell", ylabel="Vorticity power", title=f"{title}: spectra")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output / "04_energy_spectrum.png", dpi=240)
    plt.close(fig)
    plot_nrmse(metrics, summary, f"{title}: NRMSE", output / "05_nrmse.png")
    plot_loss(history, f"{title}: training history", output / "06_training_loss.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODEL_NAMES), required=True)
    args = parser.parse_args()
    config = load_config()
    sim_time = int(config["data_generation"]["sim_time"])
    seq_len = int(config["training"]["seq_len"])
    pred_len = int(config["training"]["pred_len"])
    relative = Path(args.model) / "ds1" / f"t{sim_time}" / f"s{seq_len}_p{pred_len}"
    source_data = PROJECT_ROOT / "results" / "data" / relative / "best_rollout"
    checkpoint_path = PROJECT_ROOT / "models" / relative / "best_rollout.pt"
    if not source_data.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(f"Prediction/checkpoint missing for {args.model}")
    output = PROJECT_ROOT / "results" / "final" / args.model
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    with (source_data / "summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    metrics = read_stepwise(source_data / "stepwise_metrics.csv")
    arrays = np.load(source_data / "predictions_truth.npz")
    truth = np.asarray(arrays["truth"][0], dtype=np.float32)
    prediction = np.asarray(arrays["prediction"][0], dtype=np.float32)
    initial = np.asarray(arrays["initial_window"][0], dtype=np.float32)
    checkpoint = load_checkpoint(checkpoint_path)
    history = list(checkpoint.get("training_history", []))
    shutil.copy2(source_data / "summary.json", output / "summary.json")
    shutil.copy2(source_data / "per_trajectory_nrmse.npz", output / "nrmse_per_trajectory.npz")
    write_curve(
        output / "nrmse_mean.csv",
        {"forecast_step": metrics["forecast_step"], "mean_nrmse": metrics["mean_nrmse"]},
    )
    write_curve(
        output / "nrmse_median.csv",
        {"forecast_step": metrics["forecast_step"], "median_nrmse": metrics["median_nrmse"]},
    )
    write_history(output / "training_history.csv", history)
    thresholds = {
        key: summary[key]
        for key in ("first_crossing_0_5", "first_crossing_1_0", "first_crossing_sqrt2")
    }
    atomic_json(thresholds, output / "thresholds.json")
    np.savez_compressed(
        output / "representative_prediction.npz",
        truth=truth,
        prediction=prediction,
        initial_window=initial,
        trajectory_id=np.asarray(arrays["trajectory_ids"][0]),
        start_index=np.asarray(arrays["start_indices"][0]),
    )
    shutil.copy2(PROJECT_ROOT / "code" / "config.yaml", output / "config_used.yaml")
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    checkpoint_info = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": digest.hexdigest(),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "epoch": int(checkpoint["epoch"]),
        "model_type": checkpoint["model_type"],
        "model_signature": checkpoint["model_signature"],
        "best_validation_loss": float(checkpoint["best_validation_loss"]),
        "best_rollout_metrics": checkpoint.get("best_rollout_metrics"),
    }
    atomic_json(checkpoint_info, output / "checkpoint_info.json")
    logs = [
        PROJECT_ROOT / "server_logs" / f"{args.model}_training.log",
        PROJECT_ROOT / "server_logs" / f"{args.model}_prediction.log",
    ]
    with (output / "run.log").open("w", encoding="utf-8", newline="\n") as target:
        for log in logs:
            target.write(f"===== {log.name} =====\n")
            if log.exists():
                target.write(log.read_text(encoding="utf-8", errors="replace"))
                target.write("\n")
    title = f"{config['experiment']['display_name']} — {MODEL_NAMES[args.model]}"
    if config["experiment"]["system"] == "lorenz96":
        plot_lorenz(truth, prediction, metrics, summary, history, title, output)
        expected_plots = 5
    else:
        plot_kolmogorov(truth, prediction, metrics, summary, history, title, output)
        expected_plots = 6
    plot_count = len(list(output.glob("*.png")))
    if plot_count != expected_plots:
        raise RuntimeError(f"Expected {expected_plots} final plots, found {plot_count}")
    print(f"Finalized {args.model}: {output}")


if __name__ == "__main__":
    main()

