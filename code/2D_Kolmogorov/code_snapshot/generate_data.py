"""Pseudo-spectral 2D Kolmogorov-flow generator with resolution and physics checks."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "code" / "config.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class KolmogorovSolver:
    def __init__(
        self, grid_size: int, viscosity: float, drag: float, forcing_wavenumber: int, device: torch.device
    ) -> None:
        self.n = int(grid_size)
        self.nu = float(viscosity)
        self.drag = float(drag)
        self.nf = int(forcing_wavenumber)
        self.device = device
        modes = torch.fft.fftfreq(self.n, d=1.0 / self.n, device=device)
        self.kx = modes.reshape(1, self.n, 1)
        self.ky = modes.reshape(1, 1, self.n)
        self.k2 = self.kx.square() + self.ky.square()
        self.safe_k2 = torch.where(self.k2 == 0, torch.ones_like(self.k2), self.k2)
        cutoff = self.n // 3
        self.dealias = ((self.kx.abs() <= cutoff) & (self.ky.abs() <= cutoff)).to(torch.float32)
        coordinates = 2.0 * math.pi * torch.arange(self.n, device=device) / self.n
        y = coordinates.reshape(1, 1, self.n)
        self.forcing = -float(self.nf) * torch.cos(float(self.nf) * y)
        self.dx = 2.0 * math.pi / self.n

    def velocity(self, omega: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        omega_hat = torch.fft.fft2(omega)
        psi_hat = omega_hat / self.safe_k2
        psi_hat[..., 0, 0] = 0.0
        # Convention: u=psi_y, v=-psi_x, omega=-Laplacian(psi).
        u = torch.fft.ifft2(1j * self.ky * psi_hat).real
        v = torch.fft.ifft2(-1j * self.kx * psi_hat).real
        return u, v

    def rhs(self, omega: torch.Tensor) -> torch.Tensor:
        omega_hat = torch.fft.fft2(omega)
        psi_hat = omega_hat / self.safe_k2
        psi_hat[..., 0, 0] = 0.0
        u = torch.fft.ifft2(1j * self.ky * psi_hat).real
        v = torch.fft.ifft2(-1j * self.kx * psi_hat).real
        omega_x = torch.fft.ifft2(1j * self.kx * omega_hat).real
        omega_y = torch.fft.ifft2(1j * self.ky * omega_hat).real
        nonlinear_hat = torch.fft.fft2(u * omega_x + v * omega_y) * self.dealias
        nonlinear = torch.fft.ifft2(nonlinear_hat).real
        diffusion = torch.fft.ifft2(-self.nu * self.k2 * omega_hat).real
        return -nonlinear + diffusion - self.drag * omega + self.forcing

    def project(self, omega: torch.Tensor) -> torch.Tensor:
        omega_hat = torch.fft.fft2(omega) * self.dealias
        omega_hat[..., 0, 0] = 0.0
        return torch.fft.ifft2(omega_hat).real

    def step(self, omega: torch.Tensor, dt: float) -> torch.Tensor:
        k1 = self.rhs(omega)
        k2 = self.rhs(omega + 0.5 * dt * k1)
        k3 = self.rhs(omega + 0.5 * dt * k2)
        k4 = self.rhs(omega + dt * k3)
        return self.project(omega + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))

    def cfl(self, omega: torch.Tensor, dt: float) -> float:
        u, v = self.velocity(omega)
        speed = torch.sqrt(u.square() + v.square()).amax()
        return float((speed * float(dt) / self.dx).item())

    def diagnostics(self, omega: torch.Tensor) -> dict:
        u, v = self.velocity(omega)
        omega_hat = torch.fft.fft2(omega)
        radial = torch.sqrt(self.k2)
        high = radial >= (self.n / 4.0)
        power = omega_hat.abs().square()
        return {
            "energy": float((0.5 * (u.square() + v.square()).mean()).item()),
            "enstrophy": float((0.5 * omega.square().mean()).item()),
            "high_mode_enstrophy_fraction": float(
                (power[:, high[0]].sum() / torch.clamp(power.sum(), min=1.0e-30)).item()
            ),
            "mean_vorticity_abs": float(omega.mean(dim=(-2, -1)).abs().max().item()),
        }


def initial_vorticity(
    count: int, solver: KolmogorovSolver, seed: int
) -> torch.Tensor:
    generator = torch.Generator(device=solver.device)
    generator.manual_seed(int(seed))
    noise = torch.randn(
        (count, solver.n, solver.n), generator=generator, device=solver.device, dtype=torch.float32
    )
    noise_hat = torch.fft.fft2(noise)
    low_pass = ((solver.kx.abs() <= 5) & (solver.ky.abs() <= 5)).to(torch.float32)
    smooth = torch.fft.ifft2(noise_hat * low_pass).real
    smooth = smooth / torch.clamp(smooth.std(dim=(-2, -1), keepdim=True), min=1.0e-6)
    coordinates = 2.0 * math.pi * torch.arange(solver.n, device=solver.device) / solver.n
    base = (
        -solver.nf
        / (solver.nu * solver.nf * solver.nf + solver.drag)
        * torch.cos(solver.nf * coordinates)
    ).reshape(1, 1, solver.n)
    return solver.project(base + 2.0 * smooth)


@torch.no_grad()
def short_run_diagnostics(
    n: int, settings: dict, device: torch.device, seed: int
) -> dict:
    solver = KolmogorovSolver(
        n,
        float(settings["viscosity"]),
        float(settings["linear_drag"]),
        int(settings["forcing_wavenumber"]),
        device,
    )
    dt = float(settings["integration_dt"])
    state = initial_vorticity(4, solver, seed)
    burn_steps = int(round(4.0 / dt))
    sample_steps = int(round(2.0 / dt))
    maximum_cfl = 0.0
    for step in range(burn_steps + sample_steps):
        state = solver.step(state, dt)
        if step % 50 == 0:
            maximum_cfl = max(maximum_cfl, solver.cfl(state, dt))
        if not torch.isfinite(state).all():
            raise FloatingPointError(f"Non-finite state in N={n} resolution check at step {step}")
    result = solver.diagnostics(state)
    result.update({"grid_size": n, "maximum_cfl": maximum_cfl})
    return result


def resolution_check(settings: dict, device: torch.device) -> dict:
    diagnostics = {
        str(n): short_run_diagnostics(int(n), settings, device, 314159)
        for n in settings["candidate_grid_sizes"]
    }
    coarse = diagnostics["32"]
    fine = diagnostics["64"]
    energy_difference = abs(coarse["energy"] - fine["energy"]) / max(fine["energy"], 1.0e-12)
    enstrophy_difference = abs(coarse["enstrophy"] - fine["enstrophy"]) / max(
        fine["enstrophy"], 1.0e-12
    )
    coarse_resolved = (
        coarse["maximum_cfl"] < 0.8
        and coarse["high_mode_enstrophy_fraction"] < 0.12
        and energy_difference < 0.30
        and enstrophy_difference < 0.40
    )
    selected = 32 if coarse_resolved else 64
    return {
        "candidate_diagnostics": diagnostics,
        "relative_energy_difference_32_vs_64": energy_difference,
        "relative_enstrophy_difference_32_vs_64": enstrophy_difference,
        "selection_rule": {
            "maximum_cfl_32_lt": 0.8,
            "high_mode_fraction_32_lt": 0.12,
            "relative_energy_difference_lt": 0.30,
            "relative_enstrophy_difference_lt": 0.40,
        },
        "selected_grid_size": selected,
        "status": "passed",
    }


def update_selected_grid(config: dict, grid_size: int) -> None:
    expected_trainval = 75 if int(grid_size) == 64 else 100
    expected_test = 15 if int(grid_size) == 64 else 20
    changed = (
        int(config["data_generation"]["grid_size"]) != int(grid_size)
        or int(config["data_generation"]["spatial_resolution"]) != int(grid_size) ** 2
        or int(config["data_generation"]["trainval_trajectories"]) != expected_trainval
        or int(config["data_generation"]["independent_test_trajectories"]) != expected_test
    )
    if not changed:
        return
    config["data_generation"]["grid_size"] = int(grid_size)
    config["data_generation"]["spatial_resolution"] = int(grid_size) ** 2
    config["data_generation"]["trainval_trajectories"] = expected_trainval
    config["data_generation"]["num_initial_conditions"] = expected_trainval
    config["data_generation"]["independent_test_trajectories"] = expected_test
    config["prediction"]["num_eval_trajectories"] = expected_test
    temporary = CONFIG_PATH.with_suffix(".yaml.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, CONFIG_PATH)


@torch.no_grad()
def simulate_all(settings: dict, grid_size: int, device: torch.device) -> tuple[np.ndarray, dict]:
    train_count = int(settings["trainval_trajectories"])
    test_count = int(settings["independent_test_trajectories"])
    total_count = train_count + test_count
    saved_steps = int(settings["saved_steps"])
    data_dt = float(settings["data_dt"])
    dt = float(settings["integration_dt"])
    substeps = int(round(data_dt / dt))
    if substeps <= 0 or not math.isclose(substeps * dt, data_dt, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("data_dt must be an integer multiple of integration_dt")
    solver = KolmogorovSolver(
        grid_size,
        float(settings["viscosity"]),
        float(settings["linear_drag"]),
        int(settings["forcing_wavenumber"]),
        device,
    )
    train_initial = initial_vorticity(train_count, solver, 20260905)
    test_initial = initial_vorticity(test_count, solver, 20261905)
    state = torch.cat([train_initial, test_initial], dim=0)
    burn_steps = int(round(float(settings["burn_in_time"]) / dt))
    maximum_cfl = 0.0
    for step in range(burn_steps):
        state = solver.step(state, dt)
        if step % 100 == 0:
            maximum_cfl = max(maximum_cfl, solver.cfl(state, dt))
        if step % 500 == 0:
            print(f"burn-in {step}/{burn_steps} cfl={maximum_cfl:.4f}", flush=True)
        if not torch.isfinite(state).all():
            raise FloatingPointError(f"Non-finite state during burn-in at step {step}")
    output = np.empty((total_count, saved_steps, grid_size, grid_size), dtype=np.float32)
    for frame in range(saved_steps):
        for _ in range(substeps):
            state = solver.step(state, dt)
        if frame % 100 == 0:
            maximum_cfl = max(maximum_cfl, solver.cfl(state, dt))
        if frame % 250 == 0:
            print(f"saved frame {frame}/{saved_steps} cfl={maximum_cfl:.4f}", flush=True)
        if not torch.isfinite(state).all():
            raise FloatingPointError(f"Non-finite state at saved frame {frame}")
        output[:, frame] = state.cpu().numpy()
    if maximum_cfl >= 0.8:
        raise RuntimeError(f"CFL stability check failed: maximum CFL={maximum_cfl:.4f}")
    diagnostics = solver.diagnostics(state)
    diagnostics.update(
        {
            "maximum_cfl": maximum_cfl,
            "burn_in_steps": burn_steps,
            "substeps_per_saved_frame": substeps,
        }
    )
    return output, diagnostics


def atomic_h5(path: Path, data: np.ndarray, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with h5py.File(temporary, "w") as handle:
            dataset = handle.create_dataset(
                "data",
                data=data,
                dtype="float32",
                chunks=(1, min(32, data.shape[1]), data.shape[2], data.shape[3]),
                compression="lzf",
            )
            for key, value in metadata.items():
                dataset.attrs[key] = value
            handle.flush()
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_file(path: Path, expected_shape: tuple[int, int, int, int]) -> dict:
    with h5py.File(path, "r") as handle:
        dataset = handle["data"]
        if tuple(dataset.shape) != expected_shape or dataset.dtype != np.dtype("float32"):
            raise ValueError(
                f"Unexpected dataset {path}: {dataset.shape}/{dataset.dtype}, "
                f"expected {expected_shape}/float32"
            )
        sample_ids = np.unique(np.linspace(0, dataset.shape[0] - 1, min(8, dataset.shape[0])).astype(int))
        sample_times = np.unique(np.linspace(0, dataset.shape[1] - 1, 20).astype(int))
        sample = np.stack([dataset[int(i), sample_times] for i in sample_ids])
    if not np.all(np.isfinite(sample)):
        raise ValueError(f"Non-finite sampled values in {path}")
    early = sample[:, :5]
    late = sample[:, -5:]
    early_enstrophy = float(0.5 * np.mean(early * early))
    late_enstrophy = float(0.5 * np.mean(late * late))
    drift = abs(late_enstrophy - early_enstrophy) / max(early_enstrophy, 1.0e-12)
    if drift > 0.50:
        raise ValueError(f"Stationarity check failed for {path}: enstrophy drift={drift:.3f}")
    return {
        "path": str(path),
        "shape": list(expected_shape),
        "dtype": "float32",
        "sample_mean": float(sample.mean()),
        "sample_std": float(sample.std()),
        "sample_minimum": float(sample.min()),
        "sample_maximum": float(sample.max()),
        "early_enstrophy": early_enstrophy,
        "late_enstrophy": late_enstrophy,
        "relative_enstrophy_drift": drift,
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
    parser.add_argument("--resolution-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Kolmogorov generation requires the configured CUDA server")
    device = torch.device("cuda")
    config = load_config()
    settings = config["data_generation"]
    resolution_path = PROJECT_ROOT / "results" / "data" / "resolution_check.json"
    if resolution_path.exists() and not args.force:
        with resolution_path.open("r", encoding="utf-8") as handle:
            resolution = json.load(handle)
    else:
        resolution = resolution_check(settings, device)
        atomic_json(resolution, resolution_path)
    grid_size = int(resolution["selected_grid_size"])
    update_selected_grid(config, grid_size)
    if args.resolution_only:
        print(json.dumps(resolution, indent=2))
        return
    config = load_config()
    settings = config["data_generation"]
    train_count = int(settings["trainval_trajectories"])
    test_count = int(settings["independent_test_trajectories"])
    saved_steps = int(settings["saved_steps"])
    train_path = (PROJECT_ROOT / settings["train_data_path"]).resolve()
    test_path = (PROJECT_ROOT / settings["test_data_path"]).resolve()
    generation_diagnostics = None
    if not args.validate_only and (args.force or not train_path.exists() or not test_path.exists()):
        all_data, generation_diagnostics = simulate_all(settings, grid_size, device)
        metadata = {
            "system": "kolmogorov2d",
            "grid_size": grid_size,
            "reynolds_number": float(settings["reynolds_number"]),
            "viscosity": float(settings["viscosity"]),
            "linear_drag": float(settings["linear_drag"]),
            "forcing_wavenumber": int(settings["forcing_wavenumber"]),
            "vorticity_forcing": "-n_f*cos(n_f*y)",
            "streamfunction_convention": "u=psi_y,v=-psi_x,omega=-Laplacian(psi)",
            "integrator": "RK4 pseudo-spectral",
            "dealiasing": "2/3 rule",
            "integration_dt": float(settings["integration_dt"]),
            "data_dt": float(settings["data_dt"]),
        }
        atomic_h5(
            train_path,
            all_data[:train_count],
            {**metadata, "split_role": "trainval", "seed": 20260905},
        )
        atomic_h5(
            test_path,
            all_data[train_count:],
            {**metadata, "split_role": "independent_test", "seed": 20261905},
        )
        del all_data
    report = {
        "system": "kolmogorov2d",
        "resolution_check": resolution,
        "generation_diagnostics": generation_diagnostics,
        "trainval": validate_file(
            train_path, (train_count, saved_steps, grid_size, grid_size)
        ),
        "independent_test": validate_file(
            test_path, (test_count, saved_steps, grid_size, grid_size)
        ),
        "status": "validated",
    }
    atomic_json(report, PROJECT_ROOT / "results" / "data" / "data_validation.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
