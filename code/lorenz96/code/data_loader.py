"""HDF5 loading, trajectory-safe splits, normalization, and window sampling."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULT_DATA_DIR = PROJECT_ROOT / "results" / "data"


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve configured paths without depending on the process working directory."""
    path = Path(path_value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def validate_pred_len(config: dict, pred_len: int) -> None:
    seq_len = int(config["training"]["seq_len"])
    allowed = [int(value) for value in config["training"]["allowed_pred_lens"]]
    if pred_len not in allowed:
        raise ValueError(f"pred_len must be one of {allowed}; received {pred_len}")
    if pred_len > seq_len:
        raise ValueError(f"pred_len ({pred_len}) must be <= seq_len ({seq_len})")


def expected_data_path(config: dict, ds: int, sim_time: int) -> Path:
    configured = [int(value) for value in config["data_generation"]["downsampling_factors"]]
    if ds not in configured:
        raise ValueError(f"ds must be one of {configured}; received {ds}")
    explicit_path = config["data_generation"].get("train_data_path")
    if explicit_path:
        return resolve_project_path(explicit_path)
    data_dir = resolve_project_path(config["data_generation"]["save_dir"])
    return data_dir / f"trajectories_t{sim_time}.h5"


def inspect_h5_data(path: Path, spatial_resolution: int, ds: int) -> dict:
    """Validate the on-disk array without changing or downsampling it."""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Expected HDF5 data file does not exist: {path}")

    expected_dim = int(spatial_resolution) // int(ds)
    with h5py.File(path, "r") as handle:
        if "data" not in handle:
            raise KeyError(f"HDF5 file has no 'data' dataset: {path}")
        dataset = handle["data"]
        if dataset.ndim not in (3, 4):
            raise ValueError(
                "Expected data.shape=[num_trajectories,total_steps,D] or "
                "[num_trajectories,total_steps,Nx,Ny], "
                f"but found {dataset.shape} in {path}"
            )
        feature_dim = int(np.prod(dataset.shape[2:]))
        if feature_dim != expected_dim:
            raise ValueError(
                f"Invalid feature dimension for ds={ds}: expected "
                f"{spatial_resolution}//{ds}={expected_dim}, found trailing shape "
                f"{dataset.shape[2:]} (product={feature_dim}) in {path}."
            )
        if int(dataset.shape[0]) < 2:
            raise ValueError(f"At least two trajectories are required in {path}")
        return {
            "path": str(path),
            "shape": tuple(int(value) for value in dataset.shape),
            "dtype": str(dataset.dtype),
        }


def load_h5_data(path: Path, spatial_resolution: int, ds: int) -> np.ndarray:
    metadata = inspect_h5_data(path, spatial_resolution, ds)
    with h5py.File(path, "r") as handle:
        data = handle["data"][:]
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 4:
        data = data.reshape(data.shape[0], data.shape[1], -1)
    if not np.all(np.isfinite(data)):
        raise ValueError(f"Non-finite values found in HDF5 data: {metadata['path']}")
    return data


def split_manifest_path(sim_time: int, random_seed: int) -> Path:
    return RESULT_DATA_DIR / f"split_manifest_t{sim_time}_seed{random_seed}.json"


def _validate_manifest(
    manifest: dict,
    num_trajectories: int,
    train_val_split: float,
    random_seed: int,
    sim_time: int,
) -> tuple[np.ndarray, np.ndarray]:
    expected = {
        "num_trajectories": int(num_trajectories),
        "train_val_split": float(train_val_split),
        "random_seed": int(random_seed),
        "sim_time": int(sim_time),
    }
    for key, value in expected.items():
        if key not in manifest or manifest[key] != value:
            raise ValueError(
                f"Existing split manifest is incompatible: key {key!r} is "
                f"{manifest.get(key)!r}, expected {value!r}. It will not be overwritten."
            )

    train_ids = np.asarray(manifest.get("train_trajectory_ids", []), dtype=np.int64)
    val_ids = np.asarray(manifest.get("validation_trajectory_ids", []), dtype=np.int64)
    combined = np.concatenate([train_ids, val_ids])
    expected_train_count = int(num_trajectories * train_val_split)
    if train_ids.size == 0 or val_ids.size == 0:
        raise ValueError("Existing split manifest has an empty train or validation split")
    if train_ids.size != expected_train_count:
        raise ValueError(
            f"Existing split manifest has {train_ids.size} train trajectories; "
            f"expected {expected_train_count}"
        )
    if combined.size != num_trajectories:
        raise ValueError("Existing split manifest does not contain every trajectory exactly once")
    if np.unique(combined).size != num_trajectories:
        raise ValueError("Existing split manifest contains duplicate trajectory IDs")
    if not np.array_equal(np.sort(combined), np.arange(num_trajectories)):
        raise ValueError("Existing split manifest contains out-of-range trajectory IDs")
    if np.intersect1d(train_ids, val_ids).size:
        raise ValueError("Existing split manifest leaks trajectories between train and validation")
    expected_permutation = np.random.default_rng(random_seed).permutation(num_trajectories)
    if not np.array_equal(train_ids, expected_permutation[:expected_train_count]) or not np.array_equal(
        val_ids, expected_permutation[expected_train_count:]
    ):
        raise ValueError(
            "Existing split manifest IDs do not match the deterministic seed-based partition"
        )
    return train_ids, val_ids


def get_or_create_trajectory_split(
    num_trajectories: int,
    train_val_split: float,
    random_seed: int,
    sim_time: int,
) -> tuple[np.ndarray, np.ndarray, Path]:
    """Create one model/ds/pred_len-independent trajectory partition."""
    manifest_path = split_manifest_path(sim_time, random_seed)
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        train_ids, val_ids = _validate_manifest(
            manifest, num_trajectories, train_val_split, random_seed, sim_time
        )
        return train_ids, val_ids, manifest_path

    train_count = int(num_trajectories * train_val_split)
    if train_count <= 0 or train_count >= num_trajectories:
        raise ValueError(
            f"train_val_split={train_val_split} produces an empty train or validation split"
        )
    permutation = np.random.default_rng(random_seed).permutation(num_trajectories)
    train_ids = permutation[:train_count].astype(np.int64)
    val_ids = permutation[train_count:].astype(np.int64)
    manifest = {
        "format_version": 1,
        "num_trajectories": int(num_trajectories),
        "train_val_split": float(train_val_split),
        "random_seed": int(random_seed),
        "sim_time": int(sim_time),
        "train_trajectory_ids": train_ids.tolist(),
        "validation_trajectory_ids": val_ids.tolist(),
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, manifest_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return train_ids, val_ids, manifest_path


def compute_train_statistics(
    data: np.ndarray, train_trajectory_ids: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel population mean/std from train trajectories only."""
    running_count = 0
    running_mean = np.zeros(data.shape[2], dtype=np.float64)
    running_m2 = np.zeros(data.shape[2], dtype=np.float64)

    for trajectory_id in np.asarray(train_trajectory_ids, dtype=np.int64):
        trajectory = np.asarray(data[int(trajectory_id)], dtype=np.float64)
        local_count = int(trajectory.shape[0])
        local_mean = trajectory.mean(axis=0)
        local_m2 = np.square(trajectory - local_mean).sum(axis=0)
        if running_count == 0:
            running_mean = local_mean
            running_m2 = local_m2
            running_count = local_count
            continue
        total_count = running_count + local_count
        delta = local_mean - running_mean
        running_mean = running_mean + delta * (local_count / total_count)
        running_m2 = (
            running_m2
            + local_m2
            + np.square(delta) * running_count * local_count / total_count
        )
        running_count = total_count

    if running_count == 0:
        raise ValueError("Cannot compute normalization statistics from an empty train split")
    data_std = np.sqrt(running_m2 / running_count)
    if not np.all(np.isfinite(running_mean)) or not np.all(np.isfinite(data_std)):
        raise ValueError("Train-only normalization statistics contain non-finite values")
    bad_channels = np.flatnonzero(data_std <= 1.0e-8)
    if bad_channels.size:
        raise ValueError(
            "Train-only data_std must exceed 1e-8 for every channel; invalid channels: "
            + ", ".join(str(int(index)) for index in bad_channels)
        )
    return running_mean.astype(np.float32), data_std.astype(np.float32)


def normalize_data(data: np.ndarray, data_mean: np.ndarray, data_std: np.ndarray) -> np.ndarray:
    data_mean = np.asarray(data_mean, dtype=np.float32).reshape(1, 1, -1)
    data_std = np.asarray(data_std, dtype=np.float32).reshape(1, 1, -1)
    if np.any(data_std <= 1.0e-8):
        raise ValueError("data_std must exceed 1e-8 before normalization")
    return np.ascontiguousarray((np.asarray(data, dtype=np.float32) - data_mean) / data_std)


@dataclass
class TrainingDataBundle:
    data_path: Path
    raw_data: np.ndarray
    normalized_data: np.ndarray
    train_trajectory_ids: np.ndarray
    validation_trajectory_ids: np.ndarray
    data_mean: np.ndarray
    data_std: np.ndarray
    manifest_path: Path


def prepare_training_data(config: dict, ds: int, sim_time: int) -> TrainingDataBundle:
    path = expected_data_path(config, ds, sim_time)
    data = load_h5_data(path, int(config["data_generation"]["spatial_resolution"]), ds)
    train_ids, val_ids, manifest_path = get_or_create_trajectory_split(
        num_trajectories=data.shape[0],
        train_val_split=float(config["training"]["train_val_split"]),
        random_seed=int(config["training"]["random_seed"]),
        sim_time=sim_time,
    )
    data_mean, data_std = compute_train_statistics(data, train_ids)
    normalized = normalize_data(data, data_mean, data_std)
    return TrainingDataBundle(
        data_path=path,
        raw_data=data,
        normalized_data=normalized,
        train_trajectory_ids=train_ids,
        validation_trajectory_ids=val_ids,
        data_mean=data_mean,
        data_std=data_std,
        manifest_path=manifest_path,
    )


def _legal_start_count(max_start: int, stride: int) -> int:
    regular_count = max_start // stride + 1
    return regular_count + int(max_start % stride != 0)


def _slot_to_start(slot: np.ndarray | int, max_start: int, stride: int):
    regular_count = max_start // stride + 1
    values = np.asarray(slot)
    starts = values * stride
    starts = np.where(values >= regular_count, max_start, starts)
    return starts.astype(np.int64)


class RandomWindowDataset(Dataset):
    """Fixed-size epoch sampler without a materialized list of all legal windows."""

    def __init__(
        self,
        data: np.ndarray,
        trajectory_ids: Sequence[int],
        seq_len: int,
        pred_len: int,
        stride: int,
        num_samples: int,
        random_seed: int,
    ) -> None:
        self.data = data
        self.trajectory_ids = np.asarray(trajectory_ids, dtype=np.int64)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.stride = int(stride)
        self.num_samples = int(num_samples)
        self.random_seed = int(random_seed)
        self.max_start = int(data.shape[1] - self.seq_len - self.pred_len)
        if self.trajectory_ids.size == 0:
            raise ValueError("RandomWindowDataset requires at least one trajectory")
        if self.stride <= 0 or self.num_samples <= 0:
            raise ValueError("stride and num_samples must be positive")
        if self.max_start < 0:
            raise ValueError(
                f"Trajectory length {data.shape[1]} is smaller than "
                f"seq_len+pred_len={self.seq_len + self.pred_len}"
            )
        self._sample_trajectory_ids = np.empty(self.num_samples, dtype=np.int64)
        self._sample_starts = np.empty(self.num_samples, dtype=np.int64)
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        rng = np.random.default_rng(self.random_seed + 1_000_003 * int(epoch))
        positions = rng.integers(0, self.trajectory_ids.size, size=self.num_samples)
        self._sample_trajectory_ids = self.trajectory_ids[positions]
        slot_count = _legal_start_count(self.max_start, self.stride)
        slots = rng.integers(0, slot_count, size=self.num_samples)
        self._sample_starts = _slot_to_start(slots, self.max_start, self.stride)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        trajectory_id = int(self._sample_trajectory_ids[index])
        start = int(self._sample_starts[index])
        src = self.data[trajectory_id, start : start + self.seq_len]
        target = self.data[
            trajectory_id,
            start + self.seq_len : start + self.seq_len + self.pred_len,
        ]
        return torch.from_numpy(src), torch.from_numpy(target)


def _even_start_positions(max_start: int, stride: int, count: int) -> np.ndarray:
    slot_count = _legal_start_count(max_start, stride)
    if count <= 0 or count > slot_count:
        raise ValueError(f"Requested {count} starts from {slot_count} legal positions")
    if count == 1:
        slots = np.asarray([0], dtype=np.int64)
    else:
        slots = np.rint(np.linspace(0, slot_count - 1, count)).astype(np.int64)
    return _slot_to_start(slots, max_start, stride)


class FixedWindowDataset(Dataset):
    """One immutable, bounded validation window set covering trajectories and time."""

    def __init__(
        self,
        data: np.ndarray,
        trajectory_ids: Sequence[int],
        seq_len: int,
        pred_len: int,
        stride: int,
        max_samples: int,
    ) -> None:
        self.data = data
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.stride = int(stride)
        trajectory_ids = np.asarray(trajectory_ids, dtype=np.int64)
        max_start = int(data.shape[1] - self.seq_len - self.pred_len)
        if trajectory_ids.size == 0:
            raise ValueError("FixedWindowDataset requires at least one trajectory")
        if max_start < 0:
            raise ValueError("Trajectory is too short for the fixed validation windows")
        legal_per_trajectory = _legal_start_count(max_start, self.stride)
        sample_count = min(int(max_samples), legal_per_trajectory * trajectory_ids.size)
        if sample_count < trajectory_ids.size:
            raise ValueError(
                f"max_samples={max_samples} cannot cover all {trajectory_ids.size} "
                "validation trajectories"
            )

        base, remainder = divmod(sample_count, trajectory_ids.size)
        sampled_trajectories: list[np.ndarray] = []
        sampled_starts: list[np.ndarray] = []
        for position, trajectory_id in enumerate(trajectory_ids):
            count = base + int(position < remainder)
            starts = _even_start_positions(max_start, self.stride, count)
            sampled_trajectories.append(np.full(count, trajectory_id, dtype=np.int64))
            sampled_starts.append(starts)
        self.trajectory_ids = np.concatenate(sampled_trajectories)
        self.starts = np.concatenate(sampled_starts)

    def __len__(self) -> int:
        return int(self.starts.size)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        trajectory_id = int(self.trajectory_ids[index])
        start = int(self.starts[index])
        src = self.data[trajectory_id, start : start + self.seq_len]
        target = self.data[
            trajectory_id,
            start + self.seq_len : start + self.seq_len + self.pred_len,
        ]
        return torch.from_numpy(src), torch.from_numpy(target)


def build_evaluation_windows(
    data: np.ndarray,
    trajectory_ids: Iterable[int],
    seq_len: int,
    forecast_steps: int,
    max_trajectories: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select deterministic windows spread across trajectories and valid start times."""
    trajectory_ids = np.asarray(list(trajectory_ids), dtype=np.int64)
    if trajectory_ids.size == 0:
        raise ValueError("No trajectories are available for evaluation")
    if max_trajectories is not None and trajectory_ids.size > int(max_trajectories):
        positions = np.rint(
            np.linspace(0, trajectory_ids.size - 1, int(max_trajectories))
        ).astype(np.int64)
        trajectory_ids = trajectory_ids[positions]
    max_start = int(data.shape[1] - int(seq_len) - int(forecast_steps))
    if max_start < 0:
        raise ValueError(
            f"Trajectory length {data.shape[1]} is smaller than required "
            f"seq_len+forecast_steps={seq_len + forecast_steps}"
        )
    fractions = np.linspace(0.0, 1.0, trajectory_ids.size, endpoint=True)
    starts = np.rint(fractions * max_start).astype(np.int64)
    initial_windows = np.stack(
        [
            data[int(trajectory_id), int(start) : int(start) + int(seq_len)]
            for trajectory_id, start in zip(trajectory_ids, starts)
        ]
    )
    truths = np.stack(
        [
            data[
                int(trajectory_id),
                int(start) + int(seq_len) : int(start) + int(seq_len) + int(forecast_steps),
            ]
            for trajectory_id, start in zip(trajectory_ids, starts)
        ]
    )
    return (
        np.ascontiguousarray(initial_windows, dtype=np.float32),
        np.ascontiguousarray(truths, dtype=np.float32),
        trajectory_ids,
        starts,
    )


def lstm_chunk_advance(seq_len: int, pred_len: int) -> int:
    advance = int(seq_len) + int(pred_len) - 1
    if advance <= 0:
        raise ValueError(
            f"Invalid LSTM chunk advance: seq_len={seq_len}, pred_len={pred_len}"
        )
    return advance


def lstm_chunk_count(hidden_state_propagation: int, seq_len: int, pred_len: int) -> int:
    advance = lstm_chunk_advance(seq_len, pred_len)
    return max(1, int(hidden_state_propagation) // advance)


def sample_lstm_group_spec(
    rng: np.random.Generator,
    trajectory_ids: Sequence[int],
    batch_size: int,
    total_steps: int,
    seq_len: int,
    pred_len: int,
    hidden_state_propagation: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample lanes whose successive chunks remain on their assigned trajectory."""
    trajectory_ids = np.asarray(trajectory_ids, dtype=np.int64)
    chunks = lstm_chunk_count(hidden_state_propagation, seq_len, pred_len)
    chunk_advance = lstm_chunk_advance(seq_len, pred_len)
    min_predict_on = int(seq_len)
    max_predict_on = int(total_steps) - int(pred_len) - (chunks - 1) * chunk_advance
    if max_predict_on < min_predict_on:
        raise ValueError(
            "Trajectory is too short for LSTM hidden-state propagation: "
            f"total_steps={total_steps}, seq_len={seq_len}, pred_len={pred_len}, "
            f"hidden_state_propagation={hidden_state_propagation}"
        )
    lane_trajectory_ids = trajectory_ids[
        rng.integers(0, trajectory_ids.size, size=int(batch_size))
    ]
    relative_max = max_predict_on - min_predict_on
    slot_count = _legal_start_count(relative_max, int(stride))
    slots = rng.integers(0, slot_count, size=int(batch_size))
    initial_predict_on = min_predict_on + _slot_to_start(slots, relative_max, int(stride))
    return lane_trajectory_ids, initial_predict_on


def extract_lstm_chunk(
    data: np.ndarray,
    lane_trajectory_ids: Sequence[int],
    initial_predict_on: Sequence[int],
    chunk_index: int,
    seq_len: int,
    pred_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build teacher-forced inputs of length seq_len+pred_len-1 for one group chunk.

    For the first predict_on=p, the input covers
    [p-seq_len, ..., p+pred_len-2], so hidden state is passed on after processing
    p+pred_len-2. The next predict_on is p+seq_len+pred_len-1 and its first
    input is p+pred_len-1. Consecutive chunks are therefore strictly contiguous.
    """
    chunk_advance = lstm_chunk_advance(seq_len, pred_len)
    predict_on = (
        np.asarray(initial_predict_on, dtype=np.int64)
        + int(chunk_index) * chunk_advance
    )
    inputs = []
    targets = []
    for trajectory_id, point in zip(lane_trajectory_ids, predict_on):
        trajectory = data[int(trajectory_id)]
        inputs.append(trajectory[int(point) - seq_len : int(point) + pred_len - 1])
        targets.append(trajectory[int(point) : int(point) + pred_len])
    return (
        np.ascontiguousarray(np.stack(inputs), dtype=np.float32),
        np.ascontiguousarray(np.stack(targets), dtype=np.float32),
    )
