from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import h5py
import numpy as np

from common import PROJECT_ROOT, atomic_json, atomic_text, load_config


DATA_DIR = PROJECT_ROOT / "data"
STATUS_PATH = PROJECT_ROOT / "LIVE_STATUS.txt"


class ETDRK4KS:
    """Batched periodic Kuramoto-Sivashinsky ETDRK4 solver."""

    def __init__(self, n: int, length: float, dt: float) -> None:
        self.n = int(n)
        self.length = float(length)
        self.dt = float(dt)
        k = 2.0 * np.pi * np.fft.rfftfreq(self.n, d=self.length / self.n)
        linear = k * k - k**4
        self.e = np.exp(self.dt * linear)
        self.e2 = np.exp(0.5 * self.dt * linear)
        roots = np.exp(1j * np.pi * (np.arange(1, 17) - 0.5) / 16.0)
        lr = self.dt * linear[:, None] + roots[None, :]
        self.q = self.dt * np.real(np.mean((np.exp(lr / 2.0) - 1.0) / lr, axis=1))
        self.f1 = self.dt * np.real(
            np.mean((-4.0 - lr + np.exp(lr) * (4.0 - 3.0 * lr + lr**2)) / lr**3, axis=1)
        )
        self.f2 = self.dt * np.real(
            np.mean((2.0 + lr + np.exp(lr) * (-2.0 + lr)) / lr**3, axis=1)
        )
        self.f3 = self.dt * np.real(
            np.mean((-4.0 - 3.0 * lr - lr**2 + np.exp(lr) * (4.0 - lr)) / lr**3, axis=1)
        )
        self.g = -0.5j * k
        self.g[-1] = 0.0

    def nonlinear(self, v: np.ndarray) -> np.ndarray:
        state = np.fft.irfft(v, n=self.n, axis=-1)
        return self.g[None, :] * np.fft.rfft(state * state, axis=-1)

    def step(self, v: np.ndarray) -> np.ndarray:
        nv = self.nonlinear(v)
        a = self.e2[None, :] * v + self.q[None, :] * nv
        na = self.nonlinear(a)
        b = self.e2[None, :] * v + self.q[None, :] * na
        nb = self.nonlinear(b)
        c = self.e2[None, :] * a + self.q[None, :] * (2.0 * nb - nv)
        nc = self.nonlinear(c)
        return (
            self.e[None, :] * v
            + self.f1[None, :] * nv
            + 2.0 * self.f2[None, :] * (na + nb)
            + self.f3[None, :] * nc
        )


def initial_conditions(count: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    grid = 2.0 * np.pi * np.arange(n, dtype=np.float64) / n
    states = np.zeros((count, n), dtype=np.float64)
    for mode in range(1, 9):
        sine = rng.uniform(-1.0, 1.0, size=(count, 1))
        cosine = rng.uniform(-1.0, 1.0, size=(count, 1))
        scale = 1.0 / np.sqrt(float(mode))
        states += scale * (sine * np.sin(mode * grid) + cosine * np.cos(mode * grid))
    states -= states.mean(axis=1, keepdims=True)
    states /= np.maximum(states.std(axis=1, keepdims=True), 1.0e-8)
    states += rng.uniform(-0.07, 0.07, size=(count, 1))
    return states


def _status(label: str, completed: int, total: int, started: float) -> None:
    elapsed = time.time() - started
    rate = completed / elapsed if elapsed > 0 else 0.0
    remaining = (total - completed) / rate if rate > 0 else float("inf")
    atomic_text(
        STATUS_PATH,
        (
            "stage: data_generation\n"
            f"dataset: {label}\n"
            f"completed_steps: {completed}\n"
            f"total_steps: {total}\n"
            f"elapsed_seconds: {elapsed:.1f}\n"
            f"estimated_remaining_seconds: {remaining:.1f}\n"
        ),
    )


def generate_full(path: Path, label: str, seed: int, config: dict) -> None:
    data_cfg = config["data"]
    count = int(data_cfg[f"{label}_source_trajectories"])
    n = int(data_cfg["spatial_resolution"])
    total_steps = int(data_cfg["total_steps"])
    transient_steps = int(round(float(data_cfg["transient_time"]) / float(data_cfg["time_step"])))
    if path.exists():
        with h5py.File(path, "r") as handle:
            dataset = handle.get("data")
            if dataset is not None and tuple(dataset.shape) == (count, total_steps, n) and bool(handle.attrs.get("complete", False)):
                print(f"Reusing complete full-field dataset: {path}")
                return
        raise ValueError(f"Existing incompatible data file will not be overwritten: {path}")

    solver = ETDRK4KS(n, float(data_cfg["domain_length"]), float(data_cfg["time_step"]))
    initial = initial_conditions(count, n, seed)
    spectral = np.fft.rfft(initial, axis=-1)
    for _ in range(transient_steps):
        spectral = solver.step(spectral)
    post_transient = np.fft.irfft(spectral, n=n, axis=-1)
    if not np.all(np.isfinite(post_transient)):
        raise FloatingPointError("KS solver became non-finite during transient integration")

    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    chunk_steps = 1000
    started = time.time()
    with h5py.File(temporary, "w") as handle:
        dataset = handle.create_dataset(
            "data",
            shape=(count, total_steps, n),
            dtype=np.float32,
            chunks=(1, chunk_steps, n),
            compression=str(data_cfg["h5_compression"]),
            compression_opts=int(data_cfg["h5_compression_level"]),
            shuffle=True,
        )
        handle.create_dataset("initial_conditions", data=initial.astype(np.float32))
        handle.create_dataset("post_transient_state", data=post_transient.astype(np.float32))
        handle.attrs.update(
            {
                "equation": "u_t + u*u_x + u_xx + u_xxxx = 0",
                "solver": "ETDRK4",
                "periodic_boundary": True,
                "spatial_resolution": n,
                "domain_length": float(data_cfg["domain_length"]),
                "time_step": float(data_cfg["time_step"]),
                "sim_time": int(data_cfg["sim_time"]),
                "transient_time": float(data_cfg["transient_time"]),
                "seed": int(seed),
                "trajectory_count": count,
                "complete": False,
            }
        )
        for start in range(0, total_steps, chunk_steps):
            stop = min(total_steps, start + chunk_steps)
            block = np.empty((count, stop - start, n), dtype=np.float32)
            for offset in range(stop - start):
                if start + offset > 0:
                    spectral = solver.step(spectral)
                state = np.fft.irfft(spectral, n=n, axis=-1)
                if not np.all(np.isfinite(state)):
                    raise FloatingPointError(f"KS solver became non-finite at output step {start + offset}")
                block[:, offset, :] = state.astype(np.float32)
            dataset[:, start:stop, :] = block
            handle.flush()
            _status(label, stop, total_steps, started)
        handle.attrs["complete"] = True
        handle.flush()
    os.replace(temporary, path)


def create_vds(full_path: Path, destination: Path, ds: int, config: dict) -> None:
    count = int(config["data"]["train_source_trajectories"] if "_test" not in destination.stem else config["data"]["test_source_trajectories"])
    total_steps = int(config["data"]["total_steps"])
    n = int(config["data"]["spatial_resolution"])
    expected = (count, total_steps, n // int(ds))
    if destination.exists():
        with h5py.File(destination, "r") as handle:
            if "data" in handle and tuple(handle["data"].shape) == expected and bool(handle.attrs.get("complete", False)):
                return
        raise ValueError(f"Existing incompatible VDS will not be overwritten: {destination}")
    layout = h5py.VirtualLayout(shape=expected, dtype=np.float32)
    source = h5py.VirtualSource(str(full_path.resolve()), "data", shape=(count, total_steps, n))
    layout[:, :, :] = source[:, :, :: int(ds)]
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with h5py.File(temporary, "w", libver="latest") as handle:
        handle.create_virtual_dataset("data", layout, fillvalue=np.nan)
        handle.attrs.update({"source_full_field": str(full_path.resolve()), "downsampling_factor": int(ds), "complete": True})
        handle.flush()
    os.replace(temporary, destination)


def validate_pairing(config: dict) -> dict:
    data_dir = PROJECT_ROOT / "data"
    total_steps = int(config["data"]["total_steps"])
    probes = [(0, 0), (17, 12345), (99, total_steps - 1)]
    shapes: dict[str, list[int]] = {}
    for suffix, full_name in [("", "full_train_t10000.h5"), ("_test", "full_test_t10000.h5")]:
        full_path = data_dir / full_name
        with h5py.File(full_path, "r") as full_handle:
            full = full_handle["data"]
            for ds in (1, 2, 4, 8, 32):
                path = data_dir / f"ks_ds{ds}_t10000{suffix}.h5"
                with h5py.File(path, "r") as handle:
                    reduced = handle["data"]
                    shapes[path.name] = list(map(int, reduced.shape))
                    for trajectory, step in probes:
                        if not np.array_equal(full[trajectory, step, ::ds], reduced[trajectory, step]):
                            raise ValueError(f"Strict full-field pairing check failed: {path}")
    return shapes


def smoke() -> None:
    solver = ETDRK4KS(64, 22.0, 0.1)
    state = initial_conditions(3, 64, 42)
    spectral = np.fft.rfft(state, axis=-1)
    for _ in range(50):
        spectral = solver.step(spectral)
    output = np.fft.irfft(spectral, n=64, axis=-1)
    if output.shape != (3, 64) or not np.all(np.isfinite(output)):
        raise RuntimeError("ETDRK4 smoke check failed")
    print("ETDRK4_SMOKE_OK", float(output.std()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        smoke()
        return
    config = load_config()
    data_dir = PROJECT_ROOT / "data"
    train_full = data_dir / "full_train_t10000.h5"
    test_full = data_dir / "full_test_t10000.h5"
    generate_full(train_full, "train", int(config["data"]["train_seed"]), config)
    generate_full(test_full, "test", int(config["data"]["test_seed"]), config)
    for ds in (1, 2, 4, 8, 32):
        create_vds(train_full, data_dir / f"ks_ds{ds}_t10000.h5", ds, config)
        create_vds(test_full, data_dir / f"ks_ds{ds}_t10000_test.h5", ds, config)
    shapes = validate_pairing(config)
    atomic_json(
        data_dir / "data_manifest.json",
        {
            "generator": "batched ETDRK4",
            "equation": "u_t + u*u_x + u_xx + u_xxxx = 0",
            "periodic_boundary": True,
            "train_seed": int(config["data"]["train_seed"]),
            "test_seed": int(config["data"]["test_seed"]),
            "shapes": shapes,
            "derived_storage": "HDF5 virtual datasets selecting full_field[:,:,::ds]",
        },
    )
    atomic_text(STATUS_PATH, "stage: data_ready\nData generated: true\n")
    print("DATA_GENERATION_COMPLETE", shapes)


if __name__ == "__main__":
    main()
