"""Compare already-generated results without rerunning any model prediction."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_TYPES = ("self_attention", "easy_attention_sgu", "lstm")


def _result_directory(
    model_type: str,
    ds: int,
    sim_time: int,
    seq_len: int,
    pred_len: int,
    checkpoint_name: str,
) -> Path:
    return (
        PROJECT_ROOT
        / "results"
        / "data"
        / model_type
        / f"ds{ds}"
        / f"t{sim_time}"
        / f"s{seq_len}_p{pred_len}"
        / checkpoint_name
    )


def _legend_label(model_type: str, pred_len: int) -> str:
    if model_type == "self_attention":
        return f"Self Attention\nBlock autoregression, P={pred_len}"
    if model_type == "easy_attention_sgu":
        return f"Easy Attention + SGU\nBlock autoregression, P={pred_len}"
    return "LSTM\nStateful one-step autoregression"


def _load_model_result(
    model_type: str,
    ds: int,
    sim_time: int,
    seq_len: int,
    pred_len: int,
    checkpoint_name: str,
) -> dict:
    directory = _result_directory(
        model_type, ds, sim_time, seq_len, pred_len, checkpoint_name
    )
    csv_path = directory / "stepwise_metrics.csv"
    summary_path = directory / "summary.json"
    if not csv_path.exists() or not summary_path.exists():
        raise FileNotFoundError(
            f"Existing prediction results are required; missing {csv_path} or {summary_path}"
        )
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    expected = {
        "model_type": model_type,
        "ds": int(ds),
        "sim_time": int(sim_time),
        "seq_len": int(seq_len),
        "pred_len": int(pred_len),
        "checkpoint": checkpoint_name,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(
                f"Result metadata mismatch for {key} in {summary_path}: "
                f"found {summary.get(key)!r}, expected {value!r}"
            )
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Stepwise metric file is empty: {csv_path}")
    columns = (
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
    )
    metrics = {
        column: np.asarray([float(row[column]) for row in rows], dtype=np.float64)
        for column in columns
    }
    expected_steps = np.arange(1, len(rows) + 1, dtype=np.float64)
    if not np.array_equal(metrics["forecast_step"], expected_steps):
        raise ValueError(f"forecast_step must be one-based and consecutive in {csv_path}")
    return {"summary": summary, "metrics": metrics}


def _plot_curves(
    results: dict,
    x_column: str,
    logarithmic: bool,
    output_path: Path,
    pred_len: int,
) -> None:
    figure, axis = plt.subplots(figsize=(12, 7))
    for model_type in MODEL_TYPES:
        metrics = results[model_type]["metrics"]
        curve = metrics["mean_nrmse"]
        if logarithmic:
            curve = np.maximum(curve, 1.0e-10)
        axis.plot(
            metrics[x_column],
            curve,
            linewidth=1.8,
            label=_legend_label(model_type, pred_len),
        )
    for threshold, label in ((0.5, "0.5"), (1.0, "1.0"), (math.sqrt(2.0), "sqrt(2)")):
        axis.axhline(threshold, color="0.35", linestyle="--", alpha=0.55, label=f"NRMSE={label}")
    if logarithmic:
        axis.set_yscale("log")
    axis.set_xlabel("Forecast Step" if x_column == "forecast_step" else "Lyapunov Time")
    axis.set_ylabel("Mean NRMSE")
    axis.set_title(
        f"Three-model NRMSE vs "
        f"{'Forecast Step' if x_column == 'forecast_step' else 'Lyapunov Time'}"
    )
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_crossings(results: dict, output_path: Path, pred_len: int) -> None:
    crossing_keys = (
        ("first_crossing_0_5", "0.5"),
        ("first_crossing_1_0", "1.0"),
        ("first_crossing_sqrt2", "sqrt(2)"),
    )
    x = np.arange(len(crossing_keys), dtype=np.float64)
    width = 0.24
    figure, axis = plt.subplots(figsize=(12, 7))
    for model_index, model_type in enumerate(MODEL_TYPES):
        summary = results[model_type]["summary"]
        maximum = int(summary["forecast_steps"]) + 1
        values = []
        reached_flags = []
        for key, _ in crossing_keys:
            crossing = summary[key]
            reached_flags.append(bool(crossing["reached"]))
            values.append(
                int(crossing["crossing_step"]) if crossing["reached"] else maximum
            )
        positions = x + (model_index - 1) * width
        bars = axis.bar(
            positions,
            values,
            width=width,
            label=_legend_label(model_type, pred_len),
        )
        for bar, reached in zip(bars, reached_flags):
            if not reached:
                bar.set_hatch("//")
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    "NR",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    axis.set_xticks(x, [label for _, label in crossing_keys])
    axis.set_xlabel("NRMSE Threshold")
    axis.set_ylabel("First Crossing Forecast Step")
    axis.set_title("First-crossing Comparison (NR = not reached)")
    axis.grid(axis="y", alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_selected_steps(results: dict, output_path: Path, pred_len: int) -> None:
    selected_steps = (100, 200, 500, 1000, 1200)
    x = np.arange(len(selected_steps), dtype=np.float64)
    width = 0.24
    figure, axis = plt.subplots(figsize=(12, 7))
    for model_index, model_type in enumerate(MODEL_TYPES):
        summary = results[model_type]["summary"]
        values = [summary.get(f"nrmse_at_step_{step}") for step in selected_steps]
        values = [np.nan if value is None else float(value) for value in values]
        axis.bar(
            x + (model_index - 1) * width,
            values,
            width=width,
            label=_legend_label(model_type, pred_len),
        )
    axis.set_xticks(x, [str(step) for step in selected_steps])
    axis.set_xlabel("Forecast Step")
    axis.set_ylabel("Mean NRMSE")
    axis.set_title("Selected-step NRMSE Comparison")
    axis.grid(axis="y", alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare existing three-model KS results")
    parser.add_argument("--ds", type=int, choices=(16, 32), required=True)
    parser.add_argument("--sim-time", type=int, default=10000)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--pred-len", type=int, choices=(100, 120), default=100)
    parser.add_argument(
        "--checkpoint", choices=("latest", "best_val", "best_rollout"), default="best_rollout"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.pred_len > args.seq_len:
        raise ValueError("pred_len must be <= seq_len")
    results = {
        model_type: _load_model_result(
            model_type,
            args.ds,
            args.sim_time,
            args.seq_len,
            args.pred_len,
            args.checkpoint,
        )
        for model_type in MODEL_TYPES
    }
    lengths = {
        len(result["metrics"]["forecast_step"]) for result in results.values()
    }
    if len(lengths) != 1:
        raise ValueError("All three results must contain the same number of forecast steps")

    output_dir = (
        PROJECT_ROOT
        / "results"
        / "plots"
        / "comparisons"
        / f"ds{args.ds}"
        / f"t{args.sim_time}"
        / f"s{args.seq_len}_p{args.pred_len}"
        / args.checkpoint
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_curves(
        results,
        "forecast_step",
        False,
        output_dir / "three_models_nrmse_step_linear.png",
        args.pred_len,
    )
    _plot_curves(
        results,
        "forecast_step",
        True,
        output_dir / "three_models_nrmse_step_log.png",
        args.pred_len,
    )
    _plot_curves(
        results,
        "lyapunov_time",
        False,
        output_dir / "three_models_nrmse_lyapunov_linear.png",
        args.pred_len,
    )
    _plot_curves(
        results,
        "lyapunov_time",
        True,
        output_dir / "three_models_nrmse_lyapunov_log.png",
        args.pred_len,
    )
    _plot_crossings(
        results, output_dir / "three_models_first_crossings.png", args.pred_len
    )
    _plot_selected_steps(
        results, output_dir / "three_models_selected_step_nrmse.png", args.pred_len
    )
    print(f"Comparison plots: {output_dir}")


if __name__ == "__main__":
    main()
