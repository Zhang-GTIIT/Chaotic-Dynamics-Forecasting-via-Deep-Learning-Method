from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT / "results" / "data" / "lstm"
PLOT_ROOT = PROJECT / "results" / "plots" / "lstm"
DS_VALUES = (1, 2, 4, 8, 16, 32)


def seq_len_for(ds: int) -> int:
    return 256 if ds == 32 else 128


def result_dir(ds: int) -> Path:
    return DATA_ROOT / f"ds{ds}" / "t10000" / f"s{seq_len_for(ds)}_p100" / "best_rollout"


def crossing_step(value: dict):
    return value.get("crossing_step") if value.get("reached", False) else None


summaries = {}
curves = {}
rows = []
for ds in DS_VALUES:
    directory = result_dir(ds)
    with (directory / "summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    table = np.genfromtxt(directory / "stepwise_metrics.csv", delimiter=",", names=True)
    summaries[ds] = summary
    curves[ds] = (table["forecast_step"], table["mean_nrmse"])
    rows.append(
        {
            "ds": ds,
            "observed_dimension": 64 // ds,
            "seq_len": seq_len_for(ds),
            "H0.5": crossing_step(summary["first_crossing_0_5"]),
            "H1.0": crossing_step(summary["first_crossing_1_0"]),
            "Hsqrt2": crossing_step(summary["first_crossing_sqrt2"]),
            "NRMSE_100": summary["nrmse_at_step_100"],
            "NRMSE_200": summary["nrmse_at_step_200"],
            "NRMSE_500": summary["nrmse_at_step_500"],
            "NRMSE_1000": summary["nrmse_at_step_1000"],
            "NRMSE_1200": summary["nrmse_at_step_1200"],
            "final_NRMSE": summary["final_nrmse"],
            "collapse_flag": summary["collapse_flag"],
        }
    )

summary_path = DATA_ROOT / "lstm_all_ds_summary.csv"
temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
with temporary.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, summary_path)

all_plot_dir = PLOT_ROOT / "all_ds"
all_plot_dir.mkdir(parents=True, exist_ok=True)


def save_crossing_plot(key: str, xlabel: str, filename: str, use_observed_dim: bool = False) -> None:
    x = np.asarray([64 // ds if use_observed_dim else ds for ds in DS_VALUES])
    y = np.asarray(
        [
            crossing_step(summaries[ds][key])
            if crossing_step(summaries[ds][key]) is not None
            else summaries[ds]["forecast_steps"] + 1
            for ds in DS_VALUES
        ],
        dtype=np.float64,
    )
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.plot(x, y, marker="o")
    axis.set_xlabel(xlabel)
    axis.set_ylabel("First-crossing Forecast Step")
    axis.set_title(f"{key.replace('_', ' ')} vs {xlabel}")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(all_plot_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(figure)


save_crossing_plot("first_crossing_sqrt2", "Downsampling Factor", "Hsqrt2_vs_ds.png")
save_crossing_plot("first_crossing_1_0", "Downsampling Factor", "H1_vs_ds.png")
save_crossing_plot(
    "first_crossing_sqrt2",
    "Observed Dimension",
    "Hsqrt2_vs_observed_dimension.png",
    True,
)

for logarithmic, filename in (
    (False, "NRMSE_all_ds_linear.png"),
    (True, "NRMSE_all_ds_log.png"),
):
    figure, axis = plt.subplots(figsize=(11, 6.5))
    for ds in DS_VALUES:
        steps, nrmse = curves[ds]
        axis.plot(steps, nrmse, label=f"ds={ds}")
    if logarithmic:
        axis.set_yscale("log")
    axis.set_xlim(left=1)
    axis.set_xlabel("Forecast Step")
    axis.set_ylabel("Mean NRMSE")
    axis.set_title(f"LSTM NRMSE Across Downsampling Factors ({'log' if logarithmic else 'linear'})")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(all_plot_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(figure)

print(f"SUMMARY_READY {summary_path} {all_plot_dir}")
