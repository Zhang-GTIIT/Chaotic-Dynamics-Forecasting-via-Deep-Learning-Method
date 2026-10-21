from __future__ import annotations

import math

import numpy as np
import torch

from common import candidate_config, load_config
from generate_data import ETDRK4KS, initial_conditions
from metrics import compute_metrics
from model import autoregressive_rollout, build_model


def main() -> None:
    config = load_config()
    for ds in (8, 4, 2, 1, 32):
        for candidate in (1, 2):
            settings = candidate_config(config, ds, candidate)
            model = build_model(settings)
            batch = 2
            history = torch.randn(batch, int(settings["seq_len"]), int(settings["D"]))
            prediction = autoregressive_rollout(model, history, 7)
            assert prediction.shape == (batch, 7, int(settings["D"]))
            assert torch.isfinite(prediction).all()
    rng = np.random.default_rng(1)
    truth = rng.normal(size=(3, 1200, 8))
    prediction = truth + 0.1 * rng.normal(size=truth.shape)
    metrics = compute_metrics(prediction, truth, np.ones(8), config["evaluation"])
    assert metrics["summary"]["forecast_steps"] == 1200
    solver = ETDRK4KS(64, 22.0, 0.1)
    spectral = np.fft.rfft(initial_conditions(3, 64, 42), axis=-1)
    for _ in range(20):
        spectral = solver.step(spectral)
    assert np.isfinite(spectral).all()
    print("SMOKE_CHECK_OK")


if __name__ == "__main__":
    main()
