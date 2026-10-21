from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from common import PROJECT_ROOT, atomic_json, ds_dimension


DATA_DIR = PROJECT_ROOT / "data"


def data_path(ds: int, test: bool = False) -> Path:
    suffix = "_test" if test else ""
    return DATA_DIR / f"ks_ds{int(ds)}_t10000{suffix}.h5"


def validate_h5(ds: int, test: bool = False) -> dict:
    path = data_path(ds, test)
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        if "data" not in handle:
            raise KeyError(f"Missing data dataset: {path}")
        dataset = handle["data"]
        expected = (100, 100000, ds_dimension(ds))
        if tuple(dataset.shape) != expected:
            raise ValueError(f"Expected {expected}, found {dataset.shape}: {path}")
        if dataset.dtype != np.float32:
            raise ValueError(f"Expected float32, found {dataset.dtype}: {path}")
        sample = np.asarray(dataset[0, :10])
        if not np.all(np.isfinite(sample)):
            raise ValueError(f"Non-finite data: {path}")
        return {"path": str(path), "shape": list(expected), "dtype": str(dataset.dtype), "is_virtual": bool(dataset.is_virtual)}


def load_training_array(ds: int) -> np.ndarray:
    validate_h5(ds, test=False)
    with h5py.File(data_path(ds, False), "r") as handle:
        array = np.asarray(handle["data"][:], dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise ValueError("Training array contains non-finite values")
    return array


def split_ids(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    path = DATA_DIR / "split_manifest_seed42.json"
    permutation = np.random.default_rng(seed).permutation(100).astype(np.int64)
    expected = {
        "seed": int(seed),
        "train_trajectory_ids": permutation[:80].tolist(),
        "validation_trajectory_ids": permutation[80:].tolist(),
    }
    if path.exists():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError(f"Incompatible split manifest: {path}")
    else:
        atomic_json(path, expected)
    return permutation[:80], permutation[80:]


def train_statistics(array: np.ndarray, train_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    channels = int(array.shape[-1])
    count = 0
    mean = np.zeros(channels, dtype=np.float64)
    m2 = np.zeros(channels, dtype=np.float64)
    for trajectory_id in train_ids:
        trajectory = np.asarray(array[int(trajectory_id)], dtype=np.float64)
        local_count = int(trajectory.shape[0])
        local_mean = trajectory.mean(axis=0)
        local_m2 = np.square(trajectory - local_mean).sum(axis=0)
        if count == 0:
            mean, m2, count = local_mean, local_m2, local_count
        else:
            total = count + local_count
            delta = local_mean - mean
            mean += delta * (local_count / total)
            m2 += local_m2 + np.square(delta) * count * local_count / total
            count = total
    std = np.sqrt(m2 / count)
    if np.any(std <= 1.0e-8) or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise ValueError("Invalid train-only normalization statistics")
    return mean.astype(np.float32), std.astype(np.float32)


def normalized_batch(
    array: np.ndarray,
    trajectory_ids: np.ndarray,
    starts: np.ndarray,
    length: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    batch = np.empty((len(starts), int(length), array.shape[-1]), dtype=np.float32)
    for index, (trajectory_id, start) in enumerate(zip(trajectory_ids, starts)):
        batch[index] = array[int(trajectory_id), int(start) : int(start) + int(length)]
    return np.ascontiguousarray((batch - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1))


def random_specs(
    rng: np.random.Generator,
    allowed_ids: np.ndarray,
    batch_size: int,
    total_steps: int,
    seq_len: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    trajectory_ids = rng.choice(allowed_ids, size=int(batch_size), replace=True)
    max_start = int(total_steps) - int(seq_len) - int(horizon)
    if max_start < 0:
        raise ValueError("Sequence and horizon exceed trajectory length")
    starts = rng.integers(0, max_start + 1, size=int(batch_size), dtype=np.int64)
    return trajectory_ids.astype(np.int64), starts


def fixed_specs(
    allowed_ids: np.ndarray,
    total_steps: int,
    seq_len: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    max_start = int(total_steps) - int(seq_len) - int(horizon)
    if max_start < 0:
        raise ValueError("Sequence and horizon exceed trajectory length")
    count = len(allowed_ids)
    starts = np.rint(np.linspace(0, max_start, count)).astype(np.int64)
    return np.asarray(allowed_ids, dtype=np.int64), starts


@dataclass
class EvaluationWindows:
    initial: np.ndarray
    truth: np.ndarray
    trajectory_ids: np.ndarray
    start_indices: np.ndarray


def evaluation_windows(
    ds: int,
    test: bool,
    trajectory_ids: np.ndarray,
    seq_len: int,
    forecast_steps: int,
) -> EvaluationWindows:
    path = data_path(ds, test)
    validate_h5(ds, test)
    max_start = 100000 - int(seq_len) - int(forecast_steps)
    if max_start < 0:
        raise ValueError("Forecast does not fit in trajectory")
    starts = np.rint(np.linspace(0, max_start, len(trajectory_ids))).astype(np.int64)
    initial = np.empty((len(trajectory_ids), seq_len, ds_dimension(ds)), dtype=np.float32)
    truth = np.empty((len(trajectory_ids), forecast_steps, ds_dimension(ds)), dtype=np.float32)
    with h5py.File(path, "r") as handle:
        dataset = handle["data"]
        for index, (trajectory_id, start) in enumerate(zip(trajectory_ids, starts)):
            values = np.asarray(
                dataset[int(trajectory_id), int(start) : int(start) + int(seq_len) + int(forecast_steps)],
                dtype=np.float32,
            )
            initial[index] = values[:seq_len]
            truth[index] = values[seq_len:]
    return EvaluationWindows(initial, truth, np.asarray(trajectory_ids, dtype=np.int64), starts)
