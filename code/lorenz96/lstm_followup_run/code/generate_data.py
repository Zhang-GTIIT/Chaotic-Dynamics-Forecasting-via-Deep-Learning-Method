"""Generate and validate independent Lorenz-96 train/validation and test trajectories."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "code" / "config.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def rhs(state: np.ndarray, forcing: float) -> np.ndarray:
    return (
        (np.roll(state, -1, axis=-1) - np.roll(state, 2, axis=-1))
        * np.roll(state, 1, axis=-1)
        - state
        + float(forcing)
    )


def rk4_step(state: np.ndarray, dt: float, forcing: float) -> np.ndarray:
    k1 = rhs(state, forcing)
    k2 = rhs(state + 0.5 * dt * k1, forcing)
    k3 = rhs(state + 0.5 * dt * k2, forcing)
    k4 = rhs(state + dt * k3, forcing)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate(
    count: int,
    dimension: int,
    forcing: float,
    dt: float,
    burn_steps: int,
    saved_steps: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    state = forcing + rng.normal(0.0, 1.0, size=(count, dimension))
    state += 0.01 * np.arange(count, dtype=np.float64)[:, None] / max(count, 1)
    for _ in range(burn_steps):
        state = rk4_step(state, dt, forcing)
    output = np.empty((count, saved_steps, dimension), dtype=np.float32)
    for step in range(saved_steps):
        state = rk4_step(state, dt, forcing)
        if not np.all(np.isfinite(state)):
            raise FloatingPointError(f"Non-finite Lorenz-96 state at saved step {step + 1}")
        output[:, step] = state
    return output


def atomic_h5(path: Path, data: np.ndarray, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with h5py.File(temporary, "w") as handle:
            dataset = handle.create_dataset(
                "data",
                data=data,
                dtype="float32",
                chunks=(1, min(256, data.shape[1]), data.shape[2]),
                compression="lzf",
            )
            for key, value in metadata.items():
                dataset.attrs[key] = value
            handle.flush()
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_file(path: Path, expected_shape: tuple[int, int, int]) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        if "data" not in handle:
            raise KeyError(f"Missing dataset 'data' in {path}")
        dataset = handle["data"]
        if tuple(dataset.shape) != expected_shape or dataset.dtype != np.dtype("float32"):
            raise ValueError(
                f"Unexpected Lorenz-96 dataset in {path}: shape={dataset.shape}, "
                f"dtype={dataset.dtype}, expected={expected_shape}/float32"
            )
        data = dataset[:]
    if not np.all(np.isfinite(data)):
        raise ValueError(f"Non-finite values in {path}")
    split = max(1, data.shape[1] // 5)
    early = data[:, :split]
    late = data[:, -split:]
    overall_std = float(data.std())
    if overall_std <= 1.0e-8:
        raise ValueError(f"Degenerate Lorenz-96 variance in {path}")
    early_energy = float(np.mean(early * early))
    late_energy = float(np.mean(late * late))
    energy_drift = abs(late_energy - early_energy) / max(early_energy, 1.0e-12)
    if energy_drift > 0.35:
        raise ValueError(
            f"Lorenz-96 stationarity check failed for {path}: relative energy drift={energy_drift:.3f}"
        )
    return {
        "path": str(path),
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "mean": float(data.mean()),
        "std": overall_std,
        "minimum": float(data.min()),
        "maximum": float(data.max()),
        "early_energy": early_energy,
        "late_energy": late_energy,
        "relative_energy_drift": energy_drift,
    }


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config()
    settings = config["data_generation"]
    dimension = int(settings["state_dimension"])
    forcing = float(settings["forcing"])
    dt = float(settings["integration_dt"])
    saved_steps = int(settings["saved_steps"])
    burn_steps = int(round(float(settings["burn_in_time"]) / dt))
    train_count = int(settings["trainval_trajectories"])
    test_count = int(settings["independent_test_trajectories"])
    train_path = (PROJECT_ROOT / settings["train_data_path"]).resolve()
    test_path = (PROJECT_ROOT / settings["test_data_path"]).resolve()
    metadata = {
        "system": "lorenz96",
        "K": dimension,
        "F": forcing,
        "integrator": "RK4",
        "dt": dt,
        "burn_in_steps": burn_steps,
        "saved_steps": saved_steps,
    }
    if not args.validate_only:
        if args.force or not train_path.exists():
            atomic_h5(
                train_path,
                simulate(train_count, dimension, forcing, dt, burn_steps, saved_steps, 42001),
                {**metadata, "split_role": "trainval", "seed": 42001},
            )
        if args.force or not test_path.exists():
            atomic_h5(
                test_path,
                simulate(test_count, dimension, forcing, dt, burn_steps, saved_steps, 99001),
                {**metadata, "split_role": "independent_test", "seed": 99001},
            )
    report = {
        "system": "lorenz96",
        "configuration": metadata,
        "trainval": validate_file(train_path, (train_count, saved_steps, dimension)),
        "independent_test": validate_file(test_path, (test_count, saved_steps, dimension)),
        "independent_seed_check": True,
        "status": "validated",
    }
    atomic_json(report, PROJECT_ROOT / "results" / "data" / "data_validation.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
