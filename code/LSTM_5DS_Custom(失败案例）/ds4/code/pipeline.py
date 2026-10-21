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

from common import PROJECT_ROOT, atomic_json, atomic_text, candidate_root, load_config
from metrics import selection_key
from predict import predict_candidate
from train import train_candidate


DS_ORDER = (8, 4, 2, 1, 32)


def needs_fallback(ds: int, summary: dict) -> bool:
    if bool(summary["invalid_dynamics_flag"]):
        return True
    if int(ds) == 32:
        h1 = summary.get("H1.0")
        return h1 is not None and int(h1) < 300
    hsqrt2 = summary.get("Hsqrt2")
    return hsqrt2 is not None and int(hsqrt2) < 600


def choose_best(summaries: list[dict]) -> dict:
    valid = [summary for summary in summaries if not bool(summary["invalid_dynamics_flag"])]
    pool = valid if valid else summaries
    selected = max(pool, key=selection_key)
    result = dict(selected)
    if not valid:
        result["status"] = "FAILED_DYNAMICS"
    return result


def install_best(ds: int, selected: dict) -> Path:
    source = candidate_root(ds, int(selected["candidate"]))
    destination = PROJECT_ROOT / f"ds{ds}" / "best"
    if destination.exists():
        raise FileExistsError(f"Best directory already exists: {destination}")
    shutil.copytree(source, destination)
    atomic_json(destination / "summary.json", selected)
    atomic_text(destination / "READY_TO_SYNC", "ready\n")
    return destination


def update_master() -> None:
    rows = []
    for ds in DS_ORDER:
        path = PROJECT_ROOT / f"ds{ds}" / "best" / "summary.json"
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    columns = [
        "ds", "D", "status", "candidate", "model_type", "seq_len", "hidden", "layers",
        "final_training_horizon", "BPTT_segment", "H0.5", "H1.0", "Hsqrt2",
        "NRMSE1200", "NRMSE3000", "AUC1200", "late_std_ratio",
        "spectral_entropy_ratio", "invalid_dynamics_flag",
    ]
    with (PROJECT_ROOT / "MASTER_SUMMARY.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})
    lines = ["LSTM_5DS_Custom master summary", ""]
    for row in rows:
        lines.append(
            f"ds={row['ds']} D={row['D']} status={row['status']} candidate={row['candidate']} "
            f"seq={row['seq_len']} hidden={row['hidden']} layers={row['layers']} "
            f"H1={row['H1.0']} Hsqrt2={row['Hsqrt2']} NRMSE1200={row['NRMSE1200']} "
            f"NRMSE3000={row['NRMSE3000']} invalid={row['invalid_dynamics_flag']}"
        )
    atomic_text(PROJECT_ROOT / "MASTER_SUMMARY.txt", "\n".join(lines) + "\n")


def final_comparison() -> None:
    output = PROJECT_ROOT / "comparison"
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    curves = {}
    for ds in DS_ORDER:
        best = PROJECT_ROOT / f"ds{ds}" / "best"
        summary = json.loads((best / "summary.json").read_text(encoding="utf-8"))
        summaries.append(summary)
        if not summary["invalid_dynamics_flag"]:
            data = np.genfromtxt(best / "results" / "stepwise_metrics.csv", delimiter=",", names=True)
            curves[ds] = (data["forecast_step"], data["mean_nrmse"])

    valid = [row for row in summaries if not row["invalid_dynamics_flag"]]
    dimensions = [row["D"] for row in valid]
    ds_values = [row["ds"] for row in valid]
    hsqrt2 = [row["Hsqrt2"] if row["Hsqrt2"] is not None else 3001 for row in valid]
    h1 = [row["H1.0"] if row["H1.0"] is not None else 3001 for row in valid]

    def scatter(x, y, xlabel, ylabel, title, filename):
        figure, axis = plt.subplots(figsize=(9, 6))
        axis.plot(x, y, marker="o")
        for xv, yv, ds in zip(x, y, ds_values):
            axis.annotate(f"ds={ds}", (xv, yv))
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.3)
        figure.tight_layout()
        figure.savefig(output / filename, dpi=220, bbox_inches="tight")
        plt.close(figure)

    scatter(ds_values, hsqrt2, "ds", "Hsqrt2", "Hsqrt2 vs ds", "Hsqrt2_vs_ds.png")
    scatter(ds_values, h1, "ds", "H1.0", "H1.0 vs ds", "H1_vs_ds.png")
    scatter(dimensions, hsqrt2, "Observed dimension", "Hsqrt2", "Hsqrt2 vs dimension", "Hsqrt2_vs_dimension.png")
    for logarithmic, filename in ((False, "NRMSE_all_5ds_linear.png"), (True, "NRMSE_all_5ds_log.png")):
        figure, axis = plt.subplots(figsize=(11, 7))
        for ds, (step, curve) in curves.items():
            axis.plot(step, np.maximum(curve, 1.0e-10), label=f"ds={ds}")
        axis.axhline(math.sqrt(2.0), linestyle="--", color="black", label="sqrt(2)")
        if logarithmic:
            axis.set_yscale("log")
        axis.set_xlabel("Forecast Step")
        axis.set_ylabel("Mean NRMSE")
        axis.set_title("Valid LSTM candidates: NRMSE")
        axis.grid(alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / filename, dpi=220, bbox_inches="tight")
        plt.close(figure)


def run() -> None:
    config = load_config()
    for ds in DS_ORDER:
        atomic_text(PROJECT_ROOT / "LIVE_STATUS.txt", f"stage: training\ncurrent ds: {ds}\ncurrent candidate: 1\n")
        train_candidate(ds, 1)
        primary = predict_candidate(ds, 1)
        summaries = [primary]
        if needs_fallback(ds, primary):
            atomic_text(PROJECT_ROOT / "LIVE_STATUS.txt", f"stage: fallback\ncurrent ds: {ds}\ncurrent candidate: 2\n")
            train_candidate(ds, 2)
            summaries.append(predict_candidate(ds, 2))
        selected = choose_best(summaries)
        install_best(ds, selected)
        update_master()
        atomic_text(
            PROJECT_ROOT / "LIVE_STATUS.txt",
            f"stage: ds_complete\ncurrent ds: {ds}\nbest candidate: {selected['candidate']}\nstatus: {selected['status']}\nready_to_sync: true\n",
        )
    final_comparison()
    update_master()
    atomic_text(PROJECT_ROOT / "TASK_COMPLETED.txt", "All five ds experiments completed.\n")
    atomic_text(PROJECT_ROOT / "LIVE_STATUS.txt", "stage: complete\nall_ds_complete: true\n")


if __name__ == "__main__":
    run()
