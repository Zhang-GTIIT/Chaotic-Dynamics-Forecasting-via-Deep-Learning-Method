from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from common import atomic_json, candidate_config, candidate_root, load_config
from data import evaluation_windows
from metrics import compute_metrics
from model import autoregressive_rollout, build_model
from plots import make_plots


def predict_candidate(ds: int, candidate: int) -> dict:
    config = load_config()
    settings = candidate_config(config, ds, candidate)
    root = candidate_root(ds, candidate)
    checkpoint_path = root / "checkpoints" / "best_rollout.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    device = torch.device("cuda")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if int(checkpoint["ds"]) != int(ds) or int(checkpoint["candidate"]) != int(candidate):
        raise ValueError("Checkpoint metadata mismatch")
    mean = np.asarray(checkpoint["data_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["data_std"], dtype=np.float32)
    model = build_model(settings).to(device=device, dtype=torch.float32)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    trajectory_ids = np.arange(int(config["evaluation"]["test_trajectories"]), dtype=np.int64)
    windows = evaluation_windows(
        ds=ds,
        test=True,
        trajectory_ids=trajectory_ids,
        seq_len=int(settings["seq_len"]),
        forecast_steps=int(config["evaluation"]["prediction_steps"]),
    )
    normalized = (windows.initial - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    prediction_normalized = autoregressive_rollout(
        model,
        torch.from_numpy(np.ascontiguousarray(normalized)).to(device=device, dtype=torch.float32),
        int(config["evaluation"]["prediction_steps"]),
    )
    prediction = prediction_normalized.cpu().numpy() * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)
    metrics = compute_metrics(
        prediction,
        windows.truth,
        std,
        config["evaluation"],
        checkpoint.get("best_validation_loss"),
    )
    summary = {
        "status": "SUCCESS" if not metrics["summary"]["invalid_dynamics_flag"] else "FAILED_DYNAMICS",
        "ds": int(ds),
        "D": int(settings["D"]),
        "candidate": int(candidate),
        "seq_len": int(settings["seq_len"]),
        "hidden": int(settings["hidden_size"]),
        "layers": int(settings["num_layers"]),
        "model_type": "ResidualDeltaLSTM",
        "prediction_type": "stateful one-step delta autoregression",
        "training_method": "continuous closed-loop curriculum with TBPTT",
        "final_training_horizon": int(settings["phases"][-1]["horizon"]),
        "BPTT_segment": int(settings["tbptt_segment"]),
        "best_epoch": int(checkpoint["epoch"]),
        "data_source": "independent_test_seed_4242",
        "test_trajectories": len(trajectory_ids),
        **metrics["summary"],
    }
    results_dir = root / "results"
    plots_dir = root / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "summary.json", summary)
    atomic_json(results_dir / "summary.json", summary)
    with (results_dir / "stepwise_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["forecast_step", "physical_time", "lyapunov_time", "mean_nrmse", "median_nrmse", "p10_nrmse", "p90_nrmse"])
        for index, step in enumerate(metrics["forecast_step"]):
            writer.writerow(
                [
                    int(step),
                    float(step) * float(config["data"]["time_step"]),
                    float(step) * float(config["data"]["time_step"]) * float(config["evaluation"]["lyapunov_exponent"]),
                    float(metrics["mean_nrmse"][index]),
                    float(metrics["median_nrmse"][index]),
                    float(metrics["p10_nrmse"][index]),
                    float(metrics["p90_nrmse"][index]),
                ]
            )
    np.savez_compressed(
        results_dir / "per_trajectory_nrmse.npz",
        nrmse_per_trajectory=metrics["per_trajectory_nrmse"].astype(np.float32),
        sqrt2_crossing_step=metrics["sqrt2_crossing_per_trajectory"],
        trajectory_ids=windows.trajectory_ids,
        start_indices=windows.start_indices,
    )
    np.savez_compressed(
        results_dir / "predictions_truth.npz",
        prediction=prediction.astype(np.float32),
        truth=windows.truth.astype(np.float32),
        initial_window=windows.initial.astype(np.float32),
        trajectory_ids=windows.trajectory_ids,
        start_indices=windows.start_indices,
    )
    make_plots(
        prediction,
        windows.truth,
        metrics,
        checkpoint,
        plots_dir,
        ds,
        float(config["data"]["time_step"]),
        float(config["evaluation"]["lyapunov_exponent"]),
    )
    summary_line = json.dumps(summary, ensure_ascii=False)
    (root / "logs" / "prediction.log").write_text(summary_line + "\n", encoding="utf-8")
    print(summary_line, flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=int, choices=(1, 2, 4, 8, 32), required=True)
    parser.add_argument("--candidate", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    predict_candidate(args.ds, args.candidate)


if __name__ == "__main__":
    main()
