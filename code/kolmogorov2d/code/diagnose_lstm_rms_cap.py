"""Diagnose hard RMS caps on the archived no-relaxation LSTM-v2 weights."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from data_loader import build_evaluation_windows  # noqa: E402
from metrics import (  # noqa: E402
    compute_nrmse_metrics,
    compute_periodic_flow_physical_metrics,
)
from model import build_model  # noqa: E402
from rollout import stateful_lstm_rollout  # noqa: E402


def main() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "code/config.yaml").read_text())
    checkpoint_path = Path(
        "/root/autodl-tmp/Chaos_2Systems_3Models/baseline_diagnostics/"
        "kolmogorov_lstm_v2_failed_physics_20260907_0949/"
        "models_lstm/ds1/t5000/s128_p100/best_rollout.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    with h5py.File(PROJECT_ROOT / "data/kolmogorov2d_test.h5", "r") as handle:
        data = np.asarray(handle["data"], dtype=np.float32).reshape(15, 5000, -1)
    initial, truth, _, _ = build_evaluation_windows(
        data, np.arange(15), 128, 1200, max_trajectories=15
    )
    data_mean = np.asarray(checkpoint["data_mean"], dtype=np.float32)
    data_std = np.asarray(checkpoint["data_std"], dtype=np.float32)
    normalized_initial = (initial - data_mean) / data_std
    device = torch.device("cuda")

    for cap in (1.15, 1.20, 1.25):
        candidate_config = copy.deepcopy(config)
        candidate_config["lstm"]["stability_override"] = {
            "field_rms_max": cap,
            "rms_relaxation_strength": 0.0,
            "rms_reference_window": 64,
        }
        model = build_model("lstm", data.shape[2], candidate_config, 100).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        chunks = []
        for start in range(0, normalized_initial.shape[0], 5):
            batch = torch.from_numpy(normalized_initial[start : start + 5]).to(device)
            chunks.append(stateful_lstm_rollout(model, batch, 1200).cpu().numpy())
        normalized_prediction = np.concatenate(chunks, axis=0)
        prediction = normalized_prediction * data_std + data_mean
        metrics = compute_nrmse_metrics(
            prediction,
            truth,
            data_std,
            time_step=0.01,
            lyapunov_exponent=1.0,
            late_window_steps=200,
            collapse_std_ratio_threshold=0.8,
            spectral_entropy_ratio_threshold=0.75,
            train_mean=data_mean,
        )["summary"]
        metrics.update(
            compute_periodic_flow_physical_metrics(prediction, truth, 64, 200)
        )
        print(
            json.dumps(
                {
                    "cap": cap,
                    "first_crossing_sqrt2": metrics["first_crossing_sqrt2"],
                    "late_std_ratio": metrics["late_std_ratio"],
                    "collapse_flag": metrics["collapse_flag"],
                    "spectral_entropy_ratio": metrics["spectral_entropy_ratio"],
                    "energy_ratio": metrics["late_energy_ratio_median"],
                    "enstrophy_ratio": metrics["late_enstrophy_ratio_median"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del model, chunks, normalized_prediction, prediction
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
