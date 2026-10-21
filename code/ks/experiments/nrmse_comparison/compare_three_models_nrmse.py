from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = OUTPUT_DIR.parent

MODEL_STYLES = {
    "Easy Attention + SGU": {"color": "#0072B2", "linewidth": 2.2},
    "Self Attention": {"color": "#D55E00", "linewidth": 2.2},
    "LSTM": {"color": "#009E73", "linewidth": 2.2},
}

CSV_PATHS = {
    16: {
        "Easy Attention + SGU": RESULTS_ROOT
        / "easy_attention_feature_gate_ds16"
        / "results_data"
        / "stepwise_metrics.csv",
        "Self Attention": RESULTS_ROOT
        / "self_attention_ds16"
        / "results_data"
        / "stepwise_metrics.csv",
        "LSTM": RESULTS_ROOT
        / "lstm_ds16"
        / "results_data"
        / "stepwise_metrics.csv",
    },
    32: {
        "Easy Attention + SGU": RESULTS_ROOT
        / "easy_attention_feature_gate_ds32"
        / "data"
        / "stepwise_metrics.csv",
        "Self Attention": RESULTS_ROOT
        / "self_attention_ds32"
        / "data"
        / "stepwise_metrics.csv",
        "LSTM": RESULTS_ROOT
        / "lstm_ds32"
        / "results_data"
        / "stepwise_metrics.csv",
    },
}


def load_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing metrics file: {path}")

    steps: list[int] = []
    values: list[float] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"forecast_step", "mean_nrmse"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Required columns {sorted(required)} not found in {path}")
        for row in reader:
            steps.append(int(row["forecast_step"]))
            values.append(float(row["mean_nrmse"]))

    step_array = np.asarray(steps, dtype=np.int64)
    value_array = np.asarray(values, dtype=np.float64)
    if step_array.size != 3000:
        raise ValueError(f"Expected 3000 forecast steps in {path}, got {step_array.size}")
    if not np.array_equal(step_array, np.arange(1, 3001, dtype=np.int64)):
        raise ValueError(f"forecast_step must be exactly 1..3000 in {path}")
    if not np.isfinite(value_array).all():
        raise ValueError(f"Non-finite mean_nrmse values found in {path}")
    return step_array, value_array


def plot_ds(ds: int) -> Path:
    curves = {name: load_curve(path) for name, path in CSV_PATHS[ds].items()}
    reference_steps = next(iter(curves.values()))[0]
    for name, (steps, _) in curves.items():
        if not np.array_equal(steps, reference_steps):
            raise ValueError(f"Forecast-step mismatch for {name}, ds={ds}")

    fig, ax = plt.subplots(figsize=(12, 7), dpi=180)
    for name in ("Easy Attention + SGU", "Self Attention", "LSTM"):
        steps, mean_nrmse = curves[name]
        ax.plot(steps, mean_nrmse, label=name, **MODEL_STYLES[name])

    thresholds = (
        (0.5, "NRMSE = 0.5"),
        (1.0, "NRMSE = 1.0"),
        (math.sqrt(2.0), r"NRMSE = $\sqrt{2}$"),
    )
    for value, label in thresholds:
        ax.axhline(value, color="#666666", linestyle="--", linewidth=1.0, alpha=0.65)
        ax.text(
            2990,
            value,
            label,
            ha="right",
            va="bottom",
            fontsize=9,
            color="#555555",
        )

    ax.set_title(f"Three-Model Mean NRMSE Comparison, ds={ds}", fontsize=16)
    ax.set_xlabel("Forecast step", fontsize=12)
    ax.set_ylabel("Mean NRMSE", fontsize=12)
    ax.set_xlim(1, 3000)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=True, fontsize=10)
    fig.tight_layout()

    output_path = OUTPUT_DIR / f"three_models_nrmse_ds{ds}.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_bar_ds(ds: int) -> Path:
    curves = {name: load_curve(path) for name, path in CSV_PATHS[ds].items()}
    model_names = ("Easy Attention + SGU", "Self Attention", "LSTM")
    sqrt_two = math.sqrt(2.0)
    crossing_steps: list[int] = []
    reached_flags: list[bool] = []
    forecast_limit = 0
    for name in model_names:
        _, mean_nrmse = curves[name]
        forecast_limit = max(forecast_limit, int(mean_nrmse.size))
        crossing_indices = np.flatnonzero(mean_nrmse >= sqrt_two)
        reached = crossing_indices.size > 0
        reached_flags.append(reached)
        crossing_steps.append(
            int(crossing_indices[0]) + 1 if reached else int(mean_nrmse.size)
        )

    fig, ax = plt.subplots(figsize=(10, 7), dpi=180)
    positions = np.arange(len(model_names), dtype=np.float64)
    bars = ax.bar(
        positions,
        crossing_steps,
        width=0.62,
        color=[MODEL_STYLES[name]["color"] for name in model_names],
        edgecolor="#333333",
        linewidth=0.8,
    )
    for bar, step, reached in zip(bars, crossing_steps, reached_flags):
        if not reached:
            bar.set_hatch("//")
        label = str(step) if reached else f"Not reached\n(>{forecast_limit})"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_title(
        rf"First NRMSE = $\sqrt{{2}}$ Crossing Step, ds={ds}",
        fontsize=16,
    )
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel(r"First crossing of NRMSE = $\sqrt{2}$ (forecast step)", fontsize=12)
    ax.set_xticks(positions, model_names)
    ax.set_ylim(0.0, max(crossing_steps) * 1.16)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    output_path = OUTPUT_DIR / f"three_models_nrmse_bar_ds{ds}.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    generated = [plot_ds(ds) for ds in (16, 32)]
    generated.extend(plot_bar_ds(ds) for ds in (16, 32))
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
