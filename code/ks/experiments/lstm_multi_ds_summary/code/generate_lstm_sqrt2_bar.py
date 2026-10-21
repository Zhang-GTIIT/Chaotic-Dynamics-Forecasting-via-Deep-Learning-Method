from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DS_VALUES = (1, 2, 4, 8, 16, 32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the first mean-NRMSE sqrt(2) crossing for LSTM ds=1..32."
    )
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = {int(row["ds"]): row for row in csv.DictReader(handle)}
    missing = [ds for ds in DS_VALUES if ds not in rows]
    if missing:
        raise ValueError(f"Missing ds rows in {path}: {missing}")
    return rows


def main() -> None:
    args = parse_args()
    rows = load_rows(args.summary_csv)

    forecast_limit = 3000
    crossing_steps: list[int | None] = []
    for ds in DS_VALUES:
        raw_value = rows[ds]["Hsqrt2"].strip()
        crossing_steps.append(int(float(raw_value)) if raw_value else None)

    display_heights = [step if step is not None else forecast_limit for step in crossing_steps]
    colors = ["#4C78A8", "#72A0C1", "#59A14F", "#F2CF5B", "#F28E2B", "#E15759"]

    figure, axis = plt.subplots(figsize=(10.5, 6.5))
    bars = axis.bar(
        [str(ds) for ds in DS_VALUES],
        display_heights,
        color=colors,
        edgecolor="#333333",
        linewidth=0.8,
        width=0.68,
    )

    for bar, step in zip(bars, crossing_steps):
        if step is None:
            bar.set_hatch("//")
            label = f"Not reached\n(>{forecast_limit})"
        else:
            label = str(step)
        axis.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    axis.set_xlabel("Downsampling Factor (ds)", fontsize=12)
    axis.set_ylabel("First crossing of NRMSE = sqrt(2) (forecast step)", fontsize=12)
    axis.set_title("LSTM First NRMSE = sqrt(2) Crossing Step, ds=1-32", fontsize=14)
    axis.set_ylim(0, forecast_limit * 1.12)
    axis.grid(axis="y", alpha=0.3, linestyle="--")
    axis.set_axisbelow(True)

    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
