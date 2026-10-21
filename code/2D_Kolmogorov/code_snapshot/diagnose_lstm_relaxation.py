"""Evaluate LSTM RMS-relaxation strengths without changing checkpoint weights."""

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
from metrics import compute_nrmse_metrics  # noqa: E402
from model import build_model  # noqa: E402
from rollout import stateful_lstm_rollout  # noqa: E402


def physical_ratios(prediction: np.ndarray, truth: np.ndarray) -> dict:
    grid = 64

    def statistics(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fields = values[:, -200:, :].reshape(values.shape[0], 200, grid, grid)
        spectrum = np.fft.rfft2(fields, axes=(-2, -1), norm="ortho")
        power = np.abs(spectrum) ** 2
        row_modes = np.fft.fftfreq(grid, d=1.0 / grid)[:, None]
        column_modes = np.fft.rfftfreq(grid, d=1.0 / grid)[None, :]
        k_squared = row_modes**2 + column_modes**2
        inverse_k_squared = np.zeros_like(k_squared)
        inverse_k_squared[k_squared > 0.0] = 1.0 / k_squared[k_squared > 0.0]
        weights = np.ones((1, grid // 2 + 1))
        weights[:, 1:-1] = 2.0
        kinetic = 0.5 * (power * inverse_k_squared * weights).sum(axis=(-2, -1))
        kinetic = kinetic.mean(axis=1) / float(grid * grid)
        enstrophy = (0.5 * np.square(fields).mean(axis=(-2, -1))).mean(axis=1)
        return kinetic, enstrophy

    prediction_energy, prediction_enstrophy = statistics(prediction)
    truth_energy, truth_enstrophy = statistics(truth)
    energy_ratio = prediction_energy / truth_energy
    enstrophy_ratio = prediction_enstrophy / truth_enstrophy
    return {
        "energy_ratio_median": float(np.median(energy_ratio)),
        "enstrophy_ratio_median": float(np.median(enstrophy_ratio)),
    }


def main() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "code" / "config.yaml").read_text())
    checkpoint_path = (
        PROJECT_ROOT / "models/lstm/ds1/t5000/s128_p100/best_rollout.pt"
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

    for strength in (0.001, 0.002, 0.004):
        candidate_config = copy.deepcopy(config)
        candidate_config["lstm"]["stability_override"][
            "rms_relaxation_strength"
        ] = strength
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
            time_step=float(config["data_generation"]["time_step"]),
            lyapunov_exponent=float(config["prediction"]["lyapunov_exponent"]),
            late_window_steps=200,
            collapse_std_ratio_threshold=0.8,
            spectral_entropy_ratio_threshold=0.75,
            train_mean=data_mean,
        )
        result = {
            "strength": strength,
            "first_crossing_sqrt2": metrics["summary"]["first_crossing_sqrt2"],
            "late_std_ratio": metrics["summary"]["late_std_ratio"],
            "collapse_flag": metrics["summary"]["collapse_flag"],
            "final_nrmse": metrics["summary"]["final_nrmse"],
            "spectral_entropy_ratio": metrics["summary"]["spectral_entropy_ratio"],
        }
        result.update(physical_ratios(prediction, truth))
        print(json.dumps(result, sort_keys=True), flush=True)
        del model, chunks, normalized_prediction, prediction
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
