"""Training framework for Self Attention and Easy Attention + SGU."""

from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_loader import (
    FixedWindowDataset,
    RandomWindowDataset,
    TrainingDataBundle,
    build_evaluation_windows,
    prepare_training_data,
    validate_pred_len,
)
from metrics import (
    compute_nrmse_metrics,
    compute_periodic_flow_physical_metrics,
    is_better_rollout,
)
from model import build_model
from rollout import rollout_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = PROJECT_ROOT / "models"
LOG_ROOT = PROJECT_ROOT / "results" / "logs"


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict:
    state = {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(checkpoint: dict) -> None:
    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])
    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if torch.cuda.is_available() and "cuda_rng_state_all" in checkpoint:
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in checkpoint["cuda_rng_state_all"]]
        )


def atomic_torch_save(payload: dict, path: Path) -> None:
    """Use a binary file object and atomic replace for Windows Unicode paths."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary_path.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def load_torch_checkpoint(path: Path, device: torch.device) -> dict:
    with path.open("rb") as handle:
        try:
            return torch.load(handle, map_location=device, weights_only=False)
        except TypeError:
            handle.seek(0)
            return torch.load(handle, map_location=device)


def atomic_json_write(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def effective_config(config: dict, model_type: str, pred_len: int) -> dict:
    snapshot = copy.deepcopy(config)
    snapshot["training"]["model_type"] = model_type
    snapshot["training"]["pred_len"] = int(pred_len)
    return snapshot


def attention_settings(config: dict, model_type: str) -> dict:
    settings = copy.deepcopy(config["attention_training"])
    overrides = config.get("attention_training_by_model", {}).get(model_type, {})
    settings.update(copy.deepcopy(overrides))
    return settings


def forecast_loss_settings(config: dict, model_type: str) -> dict:
    settings = copy.deepcopy(config["loss"])
    overrides = config.get("loss_by_model", {}).get(model_type, {})
    settings.update(copy.deepcopy(overrides))
    return settings


def build_model_signature(
    config: dict,
    model_type: str,
    ds: int,
    sim_time: int,
    input_dim: int,
    pred_len: int,
) -> dict:
    loss_settings = forecast_loss_settings(config, model_type)
    signature = {
        "signature_version": 2,
        "model_type": model_type,
        "ds": int(ds),
        "sim_time": int(sim_time),
        "input_dim": int(input_dim),
        "spatial_resolution": int(config["data_generation"]["spatial_resolution"]),
        "domain_length": float(config["data_generation"]["domain_length"]),
        "time_step": float(config["data_generation"]["time_step"]),
        "seq_len": int(config["training"]["seq_len"]),
        "pred_len": int(pred_len),
        "stride": int(config["training"]["stride"]),
        "loss_type": str(config["loss"]["type"]),
        "horizon_weighting": str(config["loss"]["horizon_weighting"]),
        "variance_weight": float(loss_settings["variance_weight"]),
        "mean_weight": float(loss_settings["mean_weight"]),
        "temporal_variance_weight": float(
            loss_settings.get("temporal_variance_weight", 0.0)
        ),
        "increment_weight": float(loss_settings.get("increment_weight", 0.0)),
        "increment_energy_weight": float(
            loss_settings.get("increment_energy_weight", 0.0)
        ),
        "spatial_spectrum_weight": float(
            loss_settings.get("spatial_spectrum_weight", 0.0)
        ),
        "kinetic_energy_weight": float(
            loss_settings.get("kinetic_energy_weight", 0.0)
        ),
        "enstrophy_weight": float(loss_settings.get("enstrophy_weight", 0.0)),
        "spatial_mean_weight": float(
            loss_settings.get("spatial_mean_weight", 0.0)
        ),
        "prediction_parameterization": str(
            config["model"]["prediction_parameterization"]
        ),
        "architecture_revision": "kolmogorov_physics_featured_long_horizon_v5",
        "spatial_adapter": copy.deepcopy(config.get("spatial_adapter", {})),
    }
    if model_type in {"easy_attention_sgu", "self_attention"}:
        signature.update(
            {
                "d_model": int(config["model"]["d_model"]),
                "nhead": int(config["model"]["nhead"]),
                "num_layers": int(config["model"]["num_layers"]),
                "dim_feedforward": int(config["model"]["dim_feedforward"]),
                "dropout": float(config["model"]["dropout"]),
                "ffn_activation": str(config["model"]["ffn_activation"]),
                "output_head_activation": str(
                    config["model"]["output_head_activation"]
                ),
                "decoder_head": str(config["model"]["decoder_head"]),
            }
        )
        if model_type == "easy_attention_sgu":
            signature["alpha_noise_std"] = float(
                config["easy_attention_sgu"]["alpha_noise_std"]
            )
    elif model_type == "lstm":
        signature["architecture_revision"] = (
            "kolmogorov_lstm_recent_increment_physics_v5"
        )
        signature["training_mode"] = str(
            config["lstm_training"]["training_mode"]
        )
        signature.update(
            {
                "hidden_size": int(config["lstm"]["hidden_size"]),
                "num_layers": int(config["lstm"]["num_layers"]),
                "dropout_keep_prob": float(config["lstm"]["dropout_keep_prob"]),
                "zoneout_keep_prob": float(config["lstm"]["zoneout_keep_prob"]),
                "residual_connections": bool(
                    config["lstm"]["residual_connections"]
                ),
                "hidden_state_propagation": int(
                    config["lstm"]["hidden_state_propagation"]
                ),
                "dtype": str(config["lstm"]["dtype"]),
                "state_increment_prediction": bool(
                    config["lstm"]["state_increment_prediction"]
                ),
                "lstm_stability_override": copy.deepcopy(
                    config["lstm"].get("stability_override", {})
                ),
            }
        )
    else:
        raise ValueError(f"Unknown model_type for model signature: {model_type}")
    return signature


def build_resume_signature(
    config: dict,
    model_type: str,
    ds: int,
    sim_time: int,
    input_dim: int,
    pred_len: int,
) -> dict:
    model_signature = build_model_signature(
        config, model_type, ds, sim_time, input_dim, pred_len
    )
    if model_type in {"easy_attention_sgu", "self_attention"}:
        settings = attention_settings(config, model_type)
        return {
            "model_signature": model_signature,
            "optimizer": "AdamW",
            "batch_size": int(settings["batch_size"]),
            "learning_rate": float(settings["learning_rate"]),
            "weight_decay": float(settings["weight_decay"]),
            "gradient_clip": float(settings["gradient_clip"]),
            "scheduler": str(settings["scheduler"]),
            "eta_min": float(settings["eta_min"]),
            "num_epochs": int(settings["num_epochs"]),
            "max_train_batches_per_epoch": int(
                settings["max_train_batches_per_epoch"]
            ),
            "max_val_batches_per_epoch": int(settings["max_val_batches_per_epoch"]),
            "training_mode": str(settings.get("training_mode", "one_block")),
            "block_curriculum": copy.deepcopy(
                settings.get("block_curriculum", [])
            ),
            "batch_size_by_blocks": copy.deepcopy(
                settings.get("batch_size_by_blocks", {})
            ),
            "pointwise_steps_cap": int(settings.get("pointwise_steps_cap", pred_len)),
            "train_val_split": float(config["training"]["train_val_split"]),
            "random_seed": int(config["training"]["random_seed"]),
        }
    if model_type == "lstm":
        settings = config["lstm_training"]
        return {
            "model_signature": model_signature,
            "optimizer": "Adam",
            "batch_size": int(settings["batch_size"]),
            "learning_rate": float(settings["learning_rate"]),
            "weight_decay": float(settings["weight_decay"]),
            "gradient_clip": float(settings["gradient_clip"]),
            "num_epochs": int(settings["num_epochs"]),
            "max_batch_groups_per_epoch": int(
                settings["max_batch_groups_per_epoch"]
            ),
            "train_val_split": float(config["training"]["train_val_split"]),
            "random_seed": int(config["training"]["random_seed"]),
        }
    raise ValueError(f"Unknown model_type for resume signature: {model_type}")


def checkpoint_directory(
    model_type: str, ds: int, sim_time: int, seq_len: int, pred_len: int
) -> Path:
    return (
        MODEL_ROOT
        / model_type
        / f"ds{ds}"
        / f"t{sim_time}"
        / f"s{seq_len}_p{pred_len}"
    )


def progress_path(
    model_type: str, ds: int, sim_time: int, seq_len: int, pred_len: int
) -> Path:
    return LOG_ROOT / (
        f"{model_type}_ds{ds}_t{sim_time}_s{seq_len}_p{pred_len}_progress.json"
    )


def validate_checkpoint_compatibility(
    checkpoint: dict,
    model_type: str,
    ds: int,
    sim_time: int,
    seq_len: int,
    pred_len: int,
    stride: int,
    resume_signature: dict,
    bundle: TrainingDataBundle,
) -> None:
    if (
        checkpoint.get("format_version") != 2
        or "model_signature" not in checkpoint
        or "resume_signature" not in checkpoint
    ):
        raise ValueError(
            "Legacy checkpoint format is incompatible with the fixed project. "
            "Old s64/p3 or format_version=1 checkpoints cannot be resumed."
        )
    expected_metadata = {
        "model_type": model_type,
        "ds": int(ds),
        "sim_time": int(sim_time),
        "seq_len": int(seq_len),
        "pred_len": int(pred_len),
        "stride": int(stride),
    }
    for key, expected in expected_metadata.items():
        if checkpoint.get(key) != expected:
            raise ValueError(
                f"Checkpoint configuration mismatch for {key}: "
                f"found {checkpoint.get(key)!r}, expected {expected!r}"
            )
    if checkpoint["model_signature"] != resume_signature["model_signature"]:
        raise ValueError("Checkpoint model_signature differs from the current model signature")
    if checkpoint["resume_signature"] != resume_signature:
        raise ValueError("Checkpoint resume_signature differs from the current training signature")
    if not np.array_equal(
        np.asarray(checkpoint.get("train_trajectory_ids"), dtype=np.int64),
        bundle.train_trajectory_ids,
    ):
        raise ValueError("Checkpoint train trajectory IDs differ from the shared manifest")
    if not np.array_equal(
        np.asarray(checkpoint.get("validation_trajectory_ids"), dtype=np.int64),
        bundle.validation_trajectory_ids,
    ):
        raise ValueError("Checkpoint validation trajectory IDs differ from the shared manifest")
    if not np.allclose(
        np.asarray(checkpoint.get("data_mean"), dtype=np.float32),
        bundle.data_mean,
        rtol=0.0,
        atol=1.0e-7,
    ):
        raise ValueError("Checkpoint train-only data_mean differs from the current data")
    if not np.allclose(
        np.asarray(checkpoint.get("data_std"), dtype=np.float32),
        bundle.data_std,
        rtol=0.0,
        atol=1.0e-7,
    ):
        raise ValueError("Checkpoint train-only data_std differs from the current data")


def build_checkpoint_payload(
    *,
    model_type: str,
    ds: int,
    sim_time: int,
    seq_len: int,
    pred_len: int,
    stride: int,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler_state_dict: dict | None,
    bundle: TrainingDataBundle,
    best_validation_loss: float,
    rollout_selection_metrics: dict | None,
    best_rollout_metrics: dict | None,
    training_history: list[dict],
    model_signature: dict,
    resume_signature: dict,
    config_snapshot: dict,
    patience_counter: int,
    training_complete: bool,
) -> dict:
    return {
        "format_version": 2,
        "model_type": model_type,
        "ds": int(ds),
        "sim_time": int(sim_time),
        "seq_len": int(seq_len),
        "pred_len": int(pred_len),
        "stride": int(stride),
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler_state_dict,
        "train_trajectory_ids": bundle.train_trajectory_ids.tolist(),
        "validation_trajectory_ids": bundle.validation_trajectory_ids.tolist(),
        "data_mean": bundle.data_mean.copy(),
        "data_std": bundle.data_std.copy(),
        "best_validation_loss": float(best_validation_loss),
        "rollout_selection_metrics": copy.deepcopy(rollout_selection_metrics),
        "best_rollout_metrics": copy.deepcopy(best_rollout_metrics),
        "training_history": copy.deepcopy(training_history),
        "model_signature": copy.deepcopy(model_signature),
        "resume_signature": copy.deepcopy(resume_signature),
        "config_snapshot": copy.deepcopy(config_snapshot),
        "patience_counter": int(patience_counter),
        "training_complete": bool(training_complete),
        "data_path": str(bundle.data_path),
        "split_manifest_path": str(bundle.manifest_path),
        "rng_state": capture_rng_state(),
    }


def restore_checkpoint_rng(checkpoint: dict) -> None:
    rng_state = checkpoint.get("rng_state", {})
    restore_rng_state(rng_state)


def _batch_diagnostics(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, float, float]:
    per_horizon = (prediction - target).square().mean(dim=(0, 2))
    middle = per_horizon.shape[0] // 2
    return (
        float(per_horizon[0].detach().item()),
        float(per_horizon[middle].detach().item()),
        float(per_horizon[-1].detach().item()),
    )


def distribution_regularized_forecast_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    pointwise_steps: int,
    variance_weight: float,
    mean_weight: float,
    previous_state: torch.Tensor | None = None,
    temporal_variance_weight: float = 0.0,
    increment_weight: float = 0.0,
    increment_energy_weight: float = 0.0,
    spatial_spectrum_weight: float = 0.0,
    kinetic_energy_weight: float = 0.0,
    enstrophy_weight: float = 0.0,
    spatial_mean_weight: float = 0.0,
    spatial_grid_size: int | None = None,
    normalization_mean: torch.Tensor | None = None,
    normalization_std: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pointwise loss plus temporal and periodic-flow structure preservation."""
    pointwise_horizon = int(pointwise_steps)
    if pointwise_horizon <= 0 or pointwise_horizon > prediction.shape[1]:
        raise ValueError("pointwise_steps must fit inside the forecast horizon")
    pointwise_mse = F.mse_loss(
        prediction[:, :pointwise_horizon, :],
        target[:, :pointwise_horizon, :],
    )
    prediction_mean = prediction.mean(dim=(0, 1))
    target_mean = target.mean(dim=(0, 1))
    prediction_std = prediction.std(dim=(0, 1), unbiased=False)
    target_std = target.std(dim=(0, 1), unbiased=False)
    mean_loss = F.mse_loss(prediction_mean, target_mean)
    component_std_loss = F.mse_loss(prediction_std, target_std)
    prediction_energy = prediction.square().mean(dim=2)
    target_energy = target.square().mean(dim=2)
    field_energy_loss = F.mse_loss(prediction_energy, target_energy)
    variance_loss = component_std_loss + field_energy_loss
    if prediction.shape[1] > 1:
        temporal_variance_loss = F.mse_loss(
            prediction.std(dim=1, unbiased=False),
            target.std(dim=1, unbiased=False),
        )
    else:
        temporal_variance_loss = prediction.new_zeros(())
    if previous_state is None:
        increment_loss = prediction.new_zeros(())
        increment_energy_loss = prediction.new_zeros(())
    else:
        previous_state = previous_state.reshape(prediction.shape[0], 1, prediction.shape[2])
        prediction_delta = torch.diff(
            torch.cat((previous_state, prediction), dim=1), dim=1
        )
        target_delta = torch.diff(torch.cat((previous_state, target), dim=1), dim=1)
        increment_loss = F.mse_loss(
            prediction_delta[:, :pointwise_horizon, :],
            target_delta[:, :pointwise_horizon, :],
        )
        prediction_delta_rms = prediction_delta.square().mean(dim=2).add(1.0e-12).sqrt()
        target_delta_rms = target_delta.square().mean(dim=2).add(1.0e-12).sqrt()
        increment_energy_loss = F.mse_loss(
            prediction_delta_rms, target_delta_rms
        )
    spectrum_loss = prediction.new_zeros(())
    kinetic_energy_loss = prediction.new_zeros(())
    enstrophy_loss = prediction.new_zeros(())
    spatial_mean_loss = prediction.new_zeros(())
    physics_enabled = any(
        float(weight) > 0.0
        for weight in (
            spatial_spectrum_weight,
            kinetic_energy_weight,
            enstrophy_weight,
            spatial_mean_weight,
        )
    )
    if physics_enabled:
        if spatial_grid_size is None:
            raise ValueError("spatial_grid_size is required for periodic-flow losses")
        if normalization_mean is None or normalization_std is None:
            raise ValueError("Train normalization statistics are required for physical losses")
        grid_size = int(spatial_grid_size)
        if prediction.shape[2] != grid_size * grid_size:
            raise ValueError("Forecast dimension does not match spatial_grid_size")
        physical_prediction = prediction * normalization_std + normalization_mean
        physical_target = target * normalization_std + normalization_mean
        sample_count = min(8, int(prediction.shape[1]))
        sample_indices = torch.linspace(
            0,
            prediction.shape[1] - 1,
            sample_count,
            device=prediction.device,
        ).round().long()
        prediction_fields = physical_prediction.index_select(1, sample_indices).reshape(
            prediction.shape[0], sample_count, grid_size, grid_size
        )
        target_fields = physical_target.index_select(1, sample_indices).reshape(
            target.shape[0], sample_count, grid_size, grid_size
        )
        prediction_spectrum = torch.fft.rfft2(prediction_fields, norm="ortho")
        target_spectrum = torch.fft.rfft2(target_fields, norm="ortho")
        prediction_power = prediction_spectrum.abs().square()
        target_power = target_spectrum.abs().square()
        if float(spatial_spectrum_weight) > 0.0:
            spectrum_loss = F.smooth_l1_loss(
                torch.log1p(prediction_power), torch.log1p(target_power)
            )
        if float(kinetic_energy_weight) > 0.0:
            row_modes = torch.fft.fftfreq(
                grid_size, d=1.0 / grid_size, device=prediction.device
            )[:, None]
            column_modes = torch.fft.rfftfreq(
                grid_size, d=1.0 / grid_size, device=prediction.device
            )[None, :]
            inverse_k_squared = torch.where(
                row_modes.square() + column_modes.square() > 0.0,
                (row_modes.square() + column_modes.square()).reciprocal(),
                torch.zeros_like(row_modes + column_modes),
            )
            rfft_weights = torch.ones_like(column_modes)
            if grid_size % 2 == 0:
                rfft_weights[:, 1:-1] = 2.0
            else:
                rfft_weights[:, 1:] = 2.0
            normalization = float(grid_size * grid_size)
            prediction_kinetic = 0.5 * (
                prediction_power * inverse_k_squared * rfft_weights
            ).sum(dim=(-2, -1)) / normalization
            target_kinetic = 0.5 * (
                target_power * inverse_k_squared * rfft_weights
            ).sum(dim=(-2, -1)) / normalization
            kinetic_energy_loss = F.smooth_l1_loss(
                torch.log(prediction_kinetic + 1.0e-8),
                torch.log(target_kinetic + 1.0e-8),
            )
        if float(enstrophy_weight) > 0.0:
            prediction_enstrophy = 0.5 * prediction_fields.square().mean(dim=(-2, -1))
            target_enstrophy = 0.5 * target_fields.square().mean(dim=(-2, -1))
            enstrophy_loss = F.smooth_l1_loss(
                torch.log(prediction_enstrophy + 1.0e-8),
                torch.log(target_enstrophy + 1.0e-8),
            )
        if float(spatial_mean_weight) > 0.0:
            spatial_mean_loss = F.mse_loss(
                prediction_fields.mean(dim=(-2, -1)),
                target_fields.mean(dim=(-2, -1)),
            )
    objective = (
        pointwise_mse
        + float(variance_weight) * variance_loss
        + float(mean_weight) * mean_loss
        + float(temporal_variance_weight) * temporal_variance_loss
        + float(increment_weight) * increment_loss
        + float(increment_energy_weight) * increment_energy_loss
        + float(spatial_spectrum_weight) * spectrum_loss
        + float(kinetic_energy_weight) * kinetic_energy_loss
        + float(enstrophy_weight) * enstrophy_loss
        + float(spatial_mean_weight) * spatial_mean_loss
    )
    return objective, pointwise_mse, variance_loss, mean_loss


def block_closed_loop_prediction(
    model: torch.nn.Module,
    initial_window: torch.Tensor,
    seq_len: int,
    pred_len: int,
    num_blocks: int,
) -> torch.Tensor:
    """Feed complete predicted blocks back into the next block during training."""
    window = initial_window
    blocks = []
    for _ in range(int(num_blocks)):
        block = model(window)
        expected = (window.shape[0], int(pred_len), window.shape[2])
        if tuple(block.shape) != expected:
            raise RuntimeError(
                f"Attention block shape mismatch: found {tuple(block.shape)}, expected {expected}"
            )
        blocks.append(block)
        window = torch.cat((window, block), dim=1)[:, -int(seq_len) :, :]
    return torch.cat(blocks, dim=1)


def run_attention_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip: float,
    seq_len: int,
    pred_len: int,
    num_blocks: int,
    loss_steps: int,
    pointwise_steps_cap: int,
    variance_weight: float,
    mean_weight: float,
    temporal_variance_weight: float,
    increment_weight: float,
    increment_energy_weight: float,
    spatial_spectrum_weight: float,
    kinetic_energy_weight: float,
    enstrophy_weight: float,
    spatial_mean_weight: float,
    spatial_grid_size: int,
    normalization_mean: torch.Tensor,
    normalization_std: torch.Tensor,
) -> dict:
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "optimization_loss": 0.0,
        "variance_loss": 0.0,
        "mean_loss": 0.0,
        "horizon_1": 0.0,
        "horizon_mid": 0.0,
        "horizon_last": 0.0,
    }
    sample_count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for src, target in loader:
            src = src.to(device=device, dtype=torch.float32, non_blocking=True)
            target = target.to(device=device, dtype=torch.float32, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = block_closed_loop_prediction(
                model, src, seq_len, pred_len, num_blocks
            )
            if int(loss_steps) <= 0 or int(loss_steps) > prediction.shape[1]:
                raise ValueError("loss_steps must fit inside the generated horizon")
            prediction_for_loss = prediction[:, : int(loss_steps), :]
            target_for_loss = target[:, : int(loss_steps), :]
            pointwise_steps = min(int(pointwise_steps_cap), int(loss_steps))
            objective, pointwise_mse, variance_loss, mean_loss = (
                distribution_regularized_forecast_loss(
                    prediction_for_loss,
                    target_for_loss,
                    pointwise_steps,
                    variance_weight,
                    mean_weight,
                    previous_state=src[:, -1, :],
                    temporal_variance_weight=temporal_variance_weight,
                    increment_weight=increment_weight,
                    increment_energy_weight=increment_energy_weight,
                    spatial_spectrum_weight=spatial_spectrum_weight,
                    kinetic_energy_weight=kinetic_energy_weight,
                    enstrophy_weight=enstrophy_weight,
                    spatial_mean_weight=spatial_mean_weight,
                    spatial_grid_size=spatial_grid_size,
                    normalization_mean=normalization_mean,
                    normalization_std=normalization_std,
                )
            )
            if not torch.isfinite(objective):
                raise FloatingPointError("Non-finite attention training objective")
            if training:
                objective.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
                optimizer.step()

            horizon_1, horizon_mid, horizon_last = _batch_diagnostics(
                prediction_for_loss, target_for_loss
            )
            batch_size = int(src.shape[0])
            totals["loss"] += float(pointwise_mse.detach().item()) * batch_size
            totals["optimization_loss"] += float(objective.detach().item()) * batch_size
            totals["variance_loss"] += float(variance_loss.detach().item()) * batch_size
            totals["mean_loss"] += float(mean_loss.detach().item()) * batch_size
            totals["horizon_1"] += horizon_1 * batch_size
            totals["horizon_mid"] += horizon_mid * batch_size
            totals["horizon_last"] += horizon_last * batch_size
            sample_count += batch_size
    if sample_count == 0:
        raise RuntimeError("No samples were processed in the attention epoch")
    return {key: value / sample_count for key, value in totals.items()}


def evaluate_rollout_validation(
    model_type: str,
    model: torch.nn.Module,
    bundle: TrainingDataBundle,
    config: dict,
    device: torch.device,
    pred_len: int,
    validation_mse: float,
) -> dict:
    seq_len = int(config["training"]["seq_len"])
    forecast_steps = int(config["rollout_validation"]["forecast_steps"])
    initial, truth_normalized, _, _ = build_evaluation_windows(
        bundle.normalized_data,
        bundle.validation_trajectory_ids,
        seq_len=seq_len,
        forecast_steps=forecast_steps,
    )
    initial_tensor = torch.from_numpy(initial).to(device=device, dtype=torch.float32)
    model.eval()
    prediction_normalized = rollout_model(
        model_type=model_type,
        model=model,
        initial_window=initial_tensor,
        total_steps=forecast_steps,
        seq_len=seq_len,
        pred_len=pred_len,
    )
    prediction = prediction_normalized.cpu().numpy() * bundle.data_std + bundle.data_mean
    truth = truth_normalized * bundle.data_std + bundle.data_mean
    metrics = compute_nrmse_metrics(
        prediction=prediction,
        truth=truth,
        train_std=bundle.data_std,
        time_step=float(config["data_generation"]["time_step"]),
        lyapunov_exponent=float(config["prediction"]["lyapunov_exponent"]),
        late_window_steps=int(config["rollout_validation"]["late_window_steps"]),
        collapse_std_ratio_threshold=float(
            config["rollout_validation"]["collapse_std_ratio_threshold"]
        ),
        spectral_entropy_ratio_threshold=float(
            config["rollout_validation"].get(
                "spectral_entropy_ratio_threshold",
                config["lstm_training"]["spectral_entropy_ratio_threshold"],
            )
        ),
        validation_mse=validation_mse,
        train_mean=bundle.data_mean,
    )
    if config.get("experiment", {}).get("system") == "kolmogorov2d":
        metrics["summary"].update(
            compute_periodic_flow_physical_metrics(
                prediction,
                truth,
                grid_size=int(config["spatial_adapter"]["grid_size"]),
                late_window_steps=int(
                    config["rollout_validation"]["late_window_steps"]
                ),
            )
        )
    return metrics["summary"]


def _write_progress(
    path: Path,
    model_type: str,
    ds: int,
    sim_time: int,
    seq_len: int,
    pred_len: int,
    epoch: int,
    num_epochs: int,
    train_metrics: dict,
    val_metrics: dict,
    best_validation_loss: float,
    patience_counter: int,
    rollout_summary: dict | None,
    status: str,
) -> None:
    atomic_json_write(
        {
            "status": status,
            "model_type": model_type,
            "ds": int(ds),
            "sim_time": int(sim_time),
            "seq_len": int(seq_len),
            "pred_len": int(pred_len),
            "epoch": int(epoch),
            "num_epochs": int(num_epochs),
            "train_loss": float(train_metrics["loss"]),
            "validation_loss": float(val_metrics["loss"]),
            "best_validation_loss": float(best_validation_loss),
            "patience_counter": int(patience_counter),
            "rollout_selection_metrics": rollout_summary,
        },
        path,
    )


def _validate_attention_settings(config: dict, model_type: str, pred_len: int) -> None:
    if model_type not in {"easy_attention_sgu", "self_attention"}:
        raise ValueError("Attention trainer only accepts easy_attention_sgu or self_attention")
    validate_pred_len(config, pred_len)
    if config["loss"]["type"] != "mse" or config["loss"]["horizon_weighting"] != "uniform":
        raise ValueError("Attention training requires uniform MSE over every forecast horizon")
    settings = attention_settings(config, model_type)
    loss_settings = forecast_loss_settings(config, model_type)
    if bool(settings["use_amp"]):
        raise ValueError("Formal attention training is fp32 and requires use_amp=false")
    if settings["scheduler"] != "cosine":
        raise ValueError("Formal attention training requires the cosine scheduler")
    if settings.get("training_mode") != "model_specific_closed_loop_curriculum":
        raise ValueError(
            "Attention training_mode must be 'model_specific_closed_loop_curriculum'"
        )
    curriculum = settings.get("block_curriculum", [])
    if not curriculum or int(curriculum[0]["start_epoch"]) != 1:
        raise ValueError("Attention block_curriculum must start at epoch 1")
    if any(
        int(phase["loss_steps"]) <= 0
        or int(phase["loss_steps"]) > int(phase["num_blocks"]) * int(pred_len)
        for phase in curriculum
    ):
        raise ValueError("Each loss_steps value must fit its generated horizon")
    for key in (
        "variance_weight",
        "mean_weight",
        "temporal_variance_weight",
        "increment_weight",
        "increment_energy_weight",
        "spatial_spectrum_weight",
    ):
        if float(loss_settings.get(key, 0.0)) < 0.0:
            raise ValueError(f"loss setting {key} must be non-negative")


def _attention_blocks_for_epoch(epoch: int, settings: dict) -> int:
    for phase in settings["block_curriculum"]:
        if int(phase["start_epoch"]) <= int(epoch) <= int(phase["end_epoch"]):
            return int(phase["num_blocks"])
    raise ValueError(f"No attention block curriculum phase covers epoch {epoch}")


def _attention_loss_steps_for_epoch(epoch: int, settings: dict) -> int:
    for phase in settings["block_curriculum"]:
        if int(phase["start_epoch"]) <= int(epoch) <= int(phase["end_epoch"]):
            return int(phase["loss_steps"])
    raise ValueError(f"No attention loss-horizon phase covers epoch {epoch}")


def _batch_size_for_blocks(settings: dict, num_blocks: int) -> int:
    mapping = settings["batch_size_by_blocks"]
    value = mapping.get(int(num_blocks), mapping.get(str(num_blocks)))
    if value is None or int(value) <= 0:
        raise ValueError(f"No positive attention batch size for {num_blocks} blocks")
    return int(value)


def train_attention_model(
    config: dict,
    model_type: str,
    ds: int,
    sim_time: int,
    pred_len: int,
    resume: bool = True,
) -> None:
    _validate_attention_settings(config, model_type, pred_len)
    seq_len = int(config["training"]["seq_len"])
    stride = int(config["training"]["stride"])
    random_seed = int(config["training"]["random_seed"])
    settings = attention_settings(config, model_type)
    num_epochs = int(settings["num_epochs"])
    adaptive = settings.get("adaptive_early_stopping", {})
    effective_num_epochs = min(
        num_epochs, int(adaptive.get("max_epochs", num_epochs))
    )
    minimum_epochs = int(adaptive.get("minimum_epochs", 1))
    relative_min_delta = float(adaptive.get("relative_min_delta", 0.0))
    reset_on_rollout = bool(adaptive.get("reset_on_rollout_improvement", True))
    rollout_patience_epochs = int(adaptive.get("rollout_patience_epochs", 0))
    max_train_batches = int(settings["max_train_batches_per_epoch"])
    max_val_batches = int(settings["max_val_batches_per_epoch"])
    patience = int(
        adaptive.get("patience_epochs", settings["early_stopping_patience"])
    )
    gradient_clip = float(settings["gradient_clip"])
    loss_settings = forecast_loss_settings(config, model_type)
    variance_weight = float(loss_settings["variance_weight"])
    mean_weight = float(loss_settings["mean_weight"])
    temporal_variance_weight = float(
        loss_settings.get("temporal_variance_weight", 0.0)
    )
    increment_weight = float(loss_settings.get("increment_weight", 0.0))
    increment_energy_weight = float(
        loss_settings.get("increment_energy_weight", 0.0)
    )
    spatial_spectrum_weight = float(
        loss_settings.get("spatial_spectrum_weight", 0.0)
    )
    kinetic_energy_weight = float(loss_settings.get("kinetic_energy_weight", 0.0))
    enstrophy_weight = float(loss_settings.get("enstrophy_weight", 0.0))
    spatial_mean_weight = float(loss_settings.get("spatial_mean_weight", 0.0))
    spatial_grid_size = int(config["spatial_adapter"]["grid_size"])
    pointwise_steps_cap = int(settings.get("pointwise_steps_cap", pred_len))
    if pointwise_steps_cap <= 0:
        raise ValueError("pointwise_steps_cap must be positive")
    snapshot = effective_config(config, model_type, pred_len)

    set_random_seed(random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = prepare_training_data(config, ds, sim_time)
    input_dim = int(bundle.normalized_data.shape[2])
    normalization_mean = torch.as_tensor(
        bundle.data_mean, device=device, dtype=torch.float32
    ).reshape(1, 1, input_dim)
    normalization_std = torch.as_tensor(
        bundle.data_std, device=device, dtype=torch.float32
    ).reshape(1, 1, input_dim)
    model_signature = build_model_signature(
        config, model_type, ds, sim_time, input_dim, pred_len
    )
    resume_signature = build_resume_signature(
        config, model_type, ds, sim_time, input_dim, pred_len
    )
    block_counts = sorted(
        {int(phase["num_blocks"]) for phase in settings["block_curriculum"]}
    )
    train_datasets = {}
    train_loaders = {}
    validation_loaders = {}
    for blocks in block_counts:
        phase_batch_size = _batch_size_for_blocks(settings, blocks)
        train_dataset = RandomWindowDataset(
            data=bundle.normalized_data,
            trajectory_ids=bundle.train_trajectory_ids,
            seq_len=seq_len,
            pred_len=pred_len * blocks,
            stride=stride,
            num_samples=max_train_batches * phase_batch_size,
            random_seed=random_seed,
        )
        validation_dataset = FixedWindowDataset(
            data=bundle.normalized_data,
            trajectory_ids=bundle.validation_trajectory_ids,
            seq_len=seq_len,
            pred_len=pred_len * blocks,
            stride=stride,
            max_samples=max_val_batches * phase_batch_size,
        )
        train_datasets[blocks] = train_dataset
        train_loaders[blocks] = DataLoader(
            train_dataset,
            batch_size=phase_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        validation_loaders[blocks] = DataLoader(
            validation_dataset,
            batch_size=phase_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

    model = build_model(model_type, input_dim, config, pred_len).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=effective_num_epochs,
        eta_min=float(settings["eta_min"]),
    )
    directory = checkpoint_directory(model_type, ds, sim_time, seq_len, pred_len)
    latest_path = directory / "latest.pt"
    best_val_path = directory / "best_val.pt"
    best_rollout_path = directory / "best_rollout.pt"
    status_path = progress_path(model_type, ds, sim_time, seq_len, pred_len)

    start_epoch = 0
    best_validation_loss = float("inf")
    best_rollout_metrics = None
    latest_rollout_metrics = None
    patience_counter = 0
    early_stopping_reference_loss = float("inf")
    last_rollout_improvement_epoch = 0
    history: list[dict] = []
    if resume and latest_path.exists():
        checkpoint = load_torch_checkpoint(latest_path, device)
        validate_checkpoint_compatibility(
            checkpoint,
            model_type,
            ds,
            sim_time,
            seq_len,
            pred_len,
            stride,
            resume_signature,
            bundle,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scheduler.T_max = effective_num_epochs
        start_epoch = int(checkpoint["epoch"])
        best_validation_loss = float(checkpoint["best_validation_loss"])
        best_rollout_metrics = checkpoint.get("best_rollout_metrics")
        latest_rollout_metrics = checkpoint.get("rollout_selection_metrics")
        patience_counter = int(checkpoint.get("patience_counter", 0))
        history = list(checkpoint.get("training_history", []))
        replay_rollout = None
        patience_counter = 0
        early_stopping_reference_loss = float("inf")
        last_rollout_improvement_epoch = 0
        for row in history:
            row_epoch = int(row["epoch"])
            row_loss = float(row["validation_loss"])
            if row_loss < early_stopping_reference_loss * (1.0 - relative_min_delta):
                early_stopping_reference_loss = row_loss
                patience_counter = 0
            elif row_epoch >= minimum_epochs:
                patience_counter += 1
            row_rollout = row.get("rollout_selection_metrics")
            if row_rollout is not None and is_better_rollout(row_rollout, replay_rollout):
                replay_rollout = row_rollout
                last_rollout_improvement_epoch = row_epoch
                if reset_on_rollout:
                    patience_counter = 0
        restore_checkpoint_rng(checkpoint)
        if checkpoint.get("training_complete", False):
            return

    rollout_enabled = bool(config["rollout_validation"]["enabled"])
    rollout_every = int(config["rollout_validation"]["every_epochs"])
    for epoch_index in range(start_epoch, effective_num_epochs):
        epoch = epoch_index + 1
        num_blocks = _attention_blocks_for_epoch(epoch, settings)
        loss_steps = _attention_loss_steps_for_epoch(epoch, settings)
        train_dataset = train_datasets[num_blocks]
        train_dataset.set_epoch(epoch_index)
        train_metrics = run_attention_epoch(
            model,
            train_loaders[num_blocks],
            device,
            optimizer,
            gradient_clip,
            seq_len,
            pred_len,
            num_blocks,
            loss_steps,
            pointwise_steps_cap,
            variance_weight,
            mean_weight,
            temporal_variance_weight,
            increment_weight,
            increment_energy_weight,
            spatial_spectrum_weight,
            kinetic_energy_weight,
            enstrophy_weight,
            spatial_mean_weight,
            spatial_grid_size,
            normalization_mean,
            normalization_std,
        )
        val_metrics = run_attention_epoch(
            model,
            validation_loaders[num_blocks],
            device,
            None,
            gradient_clip,
            seq_len,
            pred_len,
            num_blocks,
            loss_steps,
            pointwise_steps_cap,
            variance_weight,
            mean_weight,
            temporal_variance_weight,
            increment_weight,
            increment_energy_weight,
            spatial_spectrum_weight,
            kinetic_energy_weight,
            enstrophy_weight,
            spatial_mean_weight,
            spatial_grid_size,
            normalization_mean,
            normalization_std,
        )
        scheduler.step()

        validation_improved = val_metrics["loss"] < best_validation_loss
        if validation_improved:
            best_validation_loss = float(val_metrics["loss"])
        significant_validation_improvement = (
            val_metrics["loss"]
            < early_stopping_reference_loss * (1.0 - relative_min_delta)
        )
        if significant_validation_improvement:
            early_stopping_reference_loss = float(val_metrics["loss"])
            patience_counter = 0
        elif epoch >= minimum_epochs:
            patience_counter += 1

        rollout_improved = False
        if rollout_enabled and epoch % rollout_every == 0:
            latest_rollout_metrics = evaluate_rollout_validation(
                model_type,
                model,
                bundle,
                config,
                device,
                pred_len,
                validation_mse=float(val_metrics["loss"]),
            )
            rollout_improved = is_better_rollout(
                latest_rollout_metrics, best_rollout_metrics
            )
            if rollout_improved:
                best_rollout_metrics = copy.deepcopy(latest_rollout_metrics)
                last_rollout_improvement_epoch = epoch
                if reset_on_rollout:
                    patience_counter = 0

        history_row = {
            "epoch": epoch,
            "training_mode": str(settings["training_mode"]),
            "closed_loop_blocks": int(num_blocks),
            "closed_loop_horizon": int(loss_steps),
            "pointwise_horizon": int(min(pointwise_steps_cap, loss_steps)),
            "generated_closed_loop_horizon": int(num_blocks * pred_len),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(train_metrics["loss"]),
            "validation_loss": float(val_metrics["loss"]),
            "train_optimization_loss": float(train_metrics["optimization_loss"]),
            "validation_optimization_loss": float(val_metrics["optimization_loss"]),
            "train_variance_loss": float(train_metrics["variance_loss"]),
            "validation_variance_loss": float(val_metrics["variance_loss"]),
            "train_mean_loss": float(train_metrics["mean_loss"]),
            "validation_mean_loss": float(val_metrics["mean_loss"]),
            "train_horizon_1_mse": float(train_metrics["horizon_1"]),
            "train_horizon_mid_mse": float(train_metrics["horizon_mid"]),
            "train_horizon_last_mse": float(train_metrics["horizon_last"]),
            "validation_horizon_1_mse": float(val_metrics["horizon_1"]),
            "validation_horizon_mid_mse": float(val_metrics["horizon_mid"]),
            "validation_horizon_last_mse": float(val_metrics["horizon_last"]),
            "rollout_selection_metrics": copy.deepcopy(latest_rollout_metrics)
            if epoch % rollout_every == 0
            else None,
        }
        history.append(history_row)
        training_complete = bool(
            epoch == effective_num_epochs
            or (epoch >= minimum_epochs and patience_counter >= patience)
            or (
                rollout_patience_epochs > 0
                and epoch >= minimum_epochs
                and last_rollout_improvement_epoch > 0
                and epoch - last_rollout_improvement_epoch >= rollout_patience_epochs
            )
        )
        payload = build_checkpoint_payload(
            model_type=model_type,
            ds=ds,
            sim_time=sim_time,
            seq_len=seq_len,
            pred_len=pred_len,
            stride=stride,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler_state_dict=scheduler.state_dict(),
            bundle=bundle,
            best_validation_loss=best_validation_loss,
            rollout_selection_metrics=latest_rollout_metrics,
            best_rollout_metrics=best_rollout_metrics,
            training_history=history,
            model_signature=model_signature,
            resume_signature=resume_signature,
            config_snapshot=snapshot,
            patience_counter=patience_counter,
            training_complete=training_complete,
        )
        if validation_improved:
            atomic_torch_save(payload, best_val_path)
        if rollout_improved:
            atomic_torch_save(payload, best_rollout_path)
        atomic_torch_save(payload, latest_path)
        _write_progress(
            status_path,
            model_type,
            ds,
            sim_time,
            seq_len,
            pred_len,
            epoch,
            effective_num_epochs,
            train_metrics,
            val_metrics,
            best_validation_loss,
            patience_counter,
            latest_rollout_metrics,
            "complete" if training_complete else "running",
        )
        print(
            f"Epoch {epoch}/{effective_num_epochs} | blocks={num_blocks} | "
            f"loss_steps={loss_steps} | "
            f"train={train_metrics['loss']:.8e} | "
            f"val={val_metrics['loss']:.8e} | lr={optimizer.param_groups[0]['lr']:.8e}"
        )
        if training_complete:
            break
