"""Compare the three completed models using only already-generated final results."""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from finalize_results import flow_diagnostics, radial_spectrum


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS = ("easy_attention_sgu", "self_attention", "lstm")
LABELS = {
    "easy_attention_sgu": "Easy Attention + SGU",
    "self_attention": "Self Attention",
    "lstm": "LSTM",
}


def load_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {key: np.asarray([float(row[key]) for row in rows]) for key in rows[0]}


def load_all() -> tuple[dict, dict]:
    summaries = {}
    curves = {}
    for model in MODELS:
        directory = PROJECT_ROOT / "results" / "final" / model
        with (directory / "summary.json").open("r", encoding="utf-8") as handle:
            summaries[model] = json.load(handle)
        mean = load_csv(directory / "nrmse_mean.csv")
        median = load_csv(directory / "nrmse_median.csv")
        curves[model] = {
            "forecast_step": mean["forecast_step"],
            "mean_nrmse": mean["mean_nrmse"],
            "median_nrmse": median["median_nrmse"],
        }
    lengths = {len(curves[model]["forecast_step"]) for model in MODELS}
    if len(lengths) != 1:
        raise ValueError("All models must use the same forecast horizon")
    ids = [summaries[model]["trajectory_ids"] for model in MODELS]
    starts = [summaries[model]["start_indices"] for model in MODELS]
    if any(value != ids[0] for value in ids[1:]) or any(value != starts[0] for value in starts[1:]):
        raise ValueError("Model comparisons do not use identical test trajectories/windows")
    return summaries, curves


def comparison_nrmse(curves: dict, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for model in MODELS:
        ax.plot(
            curves[model]["forecast_step"],
            curves[model]["mean_nrmse"],
            label=LABELS[model],
            linewidth=1.6,
        )
    for value, label in ((0.5, "0.5"), (1.0, "1.0"), (math.sqrt(2.0), "sqrt(2)")):
        ax.axhline(value, color="0.35", linestyle="--", linewidth=0.8, label=f"NRMSE={label}")
    ax.set(xlabel="Forecast step", ylabel="Mean NRMSE", title=title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def comparison_thresholds(summaries: dict, title: str, path: Path) -> None:
    keys = ("first_crossing_0_5", "first_crossing_1_0", "first_crossing_sqrt2")
    x = np.arange(3)
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    for offset, model in enumerate(MODELS):
        maximum = int(summaries[model]["forecast_steps"]) + 1
        values = [
            summaries[model][key]["crossing_step"]
            if summaries[model][key]["reached"]
            else maximum
            for key in keys
        ]
        bars = ax.bar(x + (offset - 1) * width, values, width, label=LABELS[model])
        for bar, key in zip(bars, keys):
            if not summaries[model][key]["reached"]:
                bar.set_hatch("//")
    ax.set_xticks(x, ["0.5", "1.0", "sqrt(2)"])
    ax.set(xlabel="NRMSE threshold", ylabel="First crossing step", title=title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def lorenz_extra(output: Path, title: str) -> None:
    fig, axes = plt.subplots(len(MODELS), 1, figsize=(11, 9), sharex=True)
    for axis, model in zip(axes, MODELS):
        arrays = np.load(PROJECT_ROOT / "results" / "final" / model / "representative_prediction.npz")
        truth = arrays["truth"]
        prediction = arrays["prediction"]
        steps = np.arange(1, truth.shape[0] + 1)
        axis.plot(steps, truth[:, 0], label="Truth", linewidth=1.0)
        axis.plot(steps, prediction[:, 0], label=LABELS[model], linewidth=0.9)
        axis.set_ylabel("x[0]")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[-1].set_xlabel("Forecast step")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output / "03_representative_predictions.png", dpi=240)
    plt.close(fig)


def kolmogorov_extra(output: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    spectrum_errors = []
    for model in MODELS:
        arrays = np.load(PROJECT_ROOT / "results" / "final" / model / "representative_prediction.npz")
        truth_flat = arrays["truth"]
        prediction_flat = arrays["prediction"]
        n = int(round(math.sqrt(truth_flat.shape[1])))
        truth = truth_flat.reshape(-1, n, n)
        prediction = prediction_flat.reshape(-1, n, n)
        truth_energy, truth_enstrophy = flow_diagnostics(truth)
        pred_energy, pred_enstrophy = flow_diagnostics(prediction)
        steps = np.arange(1, truth.shape[0] + 1)
        axes[0].plot(
            steps,
            np.abs(pred_energy - truth_energy) / np.maximum(np.abs(truth_energy), 1.0e-12),
            label=LABELS[model],
        )
        axes[1].plot(
            steps,
            np.abs(pred_enstrophy - truth_enstrophy) / np.maximum(np.abs(truth_enstrophy), 1.0e-12),
            label=LABELS[model],
        )
        _, truth_spectrum = radial_spectrum(truth[-1])
        _, pred_spectrum = radial_spectrum(prediction[-1])
        spectrum_errors.append(
            float(np.mean(np.abs(pred_spectrum - truth_spectrum) / np.maximum(truth_spectrum, 1.0e-12)))
        )
    axes[0].set_ylabel("Relative energy error")
    axes[1].set_ylabel("Relative enstrophy error")
    axes[1].set_xlabel("Forecast step")
    for axis in axes:
        axis.set_yscale("log")
        axis.grid(alpha=0.3)
        axis.legend()
    fig.suptitle(f"{title}: invariant diagnostics")
    fig.tight_layout()
    fig.savefig(output / "03_energy_enstrophy_error.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar([LABELS[model] for model in MODELS], spectrum_errors)
    ax.set(ylabel="Mean relative spectral error", title=f"{title}: final spectrum error")
    ax.tick_params(axis="x", rotation=12)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "04_spectrum_error.png", dpi=240)
    plt.close(fig)


def main() -> None:
    with (PROJECT_ROOT / "code" / "config.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    summaries, curves = load_all()
    output = PROJECT_ROOT / "results" / "final" / "comparison"
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    system_name = config["experiment"]["display_name"]
    comparison_nrmse(curves, f"{system_name}: three-model NRMSE", output / "01_nrmse_comparison.png")
    comparison_thresholds(
        summaries, f"{system_name}: threshold crossings", output / "02_threshold_comparison.png"
    )
    if config["experiment"]["system"] == "lorenz96":
        lorenz_extra(output, f"{system_name}: representative x[0]")
        expected = 3
    else:
        kolmogorov_extra(output, system_name)
        expected = 4
    columns = [
        "model",
        "validation_mse",
        "crossing_0_5",
        "crossing_1_0",
        "crossing_sqrt2",
        "final_nrmse",
        "mean_nrmse_first_1200",
        "nrmse_auc_first_1200",
        "collapse_flag",
        "late_std_ratio",
    ]
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for model in MODELS:
            summary = summaries[model]
            writer.writerow(
                {
                    "model": model,
                    "validation_mse": summary["validation_mse"],
                    "crossing_0_5": summary["first_crossing_0_5"]["crossing_step"],
                    "crossing_1_0": summary["first_crossing_1_0"]["crossing_step"],
                    "crossing_sqrt2": summary["first_crossing_sqrt2"]["crossing_step"],
                    "final_nrmse": summary["final_nrmse"],
                    "mean_nrmse_first_1200": summary["mean_nrmse_over_first_1200"],
                    "nrmse_auc_first_1200": summary["nrmse_auc_over_first_1200"],
                    "collapse_flag": summary["collapse_flag"],
                    "late_std_ratio": summary["late_std_ratio"],
                }
            )
    plot_count = len(list(output.glob("*.png")))
    if plot_count != expected:
        raise RuntimeError(f"Expected {expected} comparison plots, found {plot_count}")
    print(f"Comparison complete: {output}")


if __name__ == "__main__":
    main()

