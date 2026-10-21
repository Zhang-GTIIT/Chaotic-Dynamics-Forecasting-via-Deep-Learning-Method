"""Closed-loop curriculum training for the state-increment StatefulLSTM."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data_loader import TrainingDataBundle, prepare_training_data, validate_pred_len
from metrics import is_better_rollout
from model import build_model
from train_transformer import (
    atomic_json_write,
    atomic_torch_save,
    build_checkpoint_payload,
    build_model_signature,
    build_resume_signature,
    checkpoint_directory,
    distribution_regularized_forecast_loss,
    effective_config,
    evaluate_rollout_validation,
    forecast_loss_settings,
    load_torch_checkpoint,
    progress_path,
    restore_checkpoint_rng,
    set_random_seed,
    validate_checkpoint_compatibility,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_MODE = "kolmogorov_closed_loop_finetune_v5"


def lstm_seq_len(config: dict, ds: int) -> int:
    mapping = config["lstm_training"]["seq_len_by_ds"]
    value = mapping.get(int(ds), mapping.get(str(int(ds))))
    if value is None:
        raise ValueError(f"No LSTM seq_len is configured for ds={ds}")
    seq_len = int(value)
    if seq_len <= 0:
        raise ValueError(f"LSTM seq_len for ds={ds} must be positive")
    return seq_len


def lstm_effective_config(config: dict, ds: int, pred_len: int) -> dict:
    result = copy.deepcopy(config)
    configured_ds = [
        int(value) for value in result["data_generation"]["downsampling_factors"]
    ]
    if int(ds) not in configured_ds:
        result["data_generation"]["downsampling_factors"] = configured_ds + [int(ds)]
    result["training"]["seq_len"] = lstm_seq_len(result, ds)
    result["training"]["model_type"] = "lstm"
    result["training"]["pred_len"] = int(pred_len)
    return result


def curriculum_phase(epoch: int, settings: dict) -> dict:
    for phase in settings["curriculum"]:
        if int(phase["start_epoch"]) <= int(epoch) <= int(phase["end_epoch"]):
            return phase
    raise ValueError(f"No LSTM curriculum phase covers epoch {epoch}")


def curriculum_horizon(epoch: int, settings: dict) -> int:
    return int(curriculum_phase(epoch, settings)["horizon"])


def learning_rate_for_epoch(epoch: int, settings: dict) -> float:
    phase = curriculum_phase(epoch, settings)
    if int(phase["horizon"]) < 100:
        return float(phase["learning_rate"])
    start_lr = float(phase["learning_rate"])
    eta_min = float(settings["eta_min"])
    final_epoch = int(settings["num_epochs"])
    start_epoch = int(phase["start_epoch"])
    progress = (int(epoch) - start_epoch) / max(final_epoch - start_epoch, 1)
    progress = min(1.0, max(0.0, progress))
    return eta_min + 0.5 * (start_lr - eta_min) * (
        1.0 + math.cos(math.pi * progress)
    )


def micro_batch_size_for_horizon(settings: dict, horizon: int) -> int:
    mapping = settings["micro_batch_size_by_horizon"]
    value = mapping.get(int(horizon), mapping.get(str(int(horizon))))
    if value is None:
        raise ValueError(f"No micro batch size configured for horizon H={horizon}")
    micro_batch_size = int(value)
    effective_batch_size = int(settings["batch_size"])
    if micro_batch_size <= 0 or micro_batch_size > effective_batch_size:
        raise ValueError("micro batch size must be in [1,effective batch size]")
    if effective_batch_size % micro_batch_size != 0:
        raise ValueError("effective batch size must be divisible by micro batch size")
    return micro_batch_size


def _validate_lstm_settings(config: dict, pred_len: int) -> None:
    validate_pred_len(config, pred_len)
    settings = config["lstm_training"]
    if settings.get("training_mode") != TRAINING_MODE:
        raise ValueError(f"LSTM training_mode must be {TRAINING_MODE!r}")
    if config["loss"]["type"] != "mse" or config["loss"]["horizon_weighting"] != "uniform":
        raise ValueError("LSTM training requires uniform MSE")
    if config["lstm"]["dtype"] != "float32":
        raise ValueError("LSTM requires dtype='float32'")
    if bool(settings["use_amp"]):
        raise ValueError("LSTM requires use_amp=false")
    if int(settings["num_epochs"]) <= 0:
        raise ValueError("LSTM num_epochs must be positive")
    if int(pred_len) != 100:
        raise ValueError("The formal closed-loop LSTM requires pred_len=100")
    if float(config["loss"]["variance_weight"]) < 0.0:
        raise ValueError("loss.variance_weight must be non-negative")
    if float(config["loss"]["mean_weight"]) < 0.0:
        raise ValueError("loss.mean_weight must be non-negative")
    phases = settings["curriculum"]
    if int(phases[0]["start_epoch"]) != 1:
        raise ValueError("LSTM curriculum must start at epoch 1")
    if int(phases[-1]["end_epoch"]) != int(settings["num_epochs"]):
        raise ValueError("LSTM curriculum must end at num_epochs")
    for previous, current in zip(phases, phases[1:]):
        if int(current["start_epoch"]) != int(previous["end_epoch"]) + 1:
            raise ValueError("LSTM curriculum phases must be contiguous")
    for horizon in {int(phase["horizon"]) for phase in phases}:
        micro_batch_size_for_horizon(settings, horizon)


def closed_loop_prediction(
    model: torch.nn.Module,
    history: torch.Tensor,
    horizon: int,
    training: bool,
) -> torch.Tensor:
    """Warm up with real history, then feed predictions back without detaching."""
    if history.ndim != 3:
        raise ValueError("history must have shape [B,seq_len,D]")
    if int(horizon) <= 0:
        raise ValueError("horizon must be positive")
    warmup_output, hidden_state = model(history, None, training=training)
    next_step = warmup_output[:, -1:, :]
    predictions = [next_step]
    for _ in range(1, int(horizon)):
        output, hidden_state = model(next_step, hidden_state, training=training)
        next_step = output[:, -1:, :]
        predictions.append(next_step)
    return torch.cat(predictions, dim=1)


def _extract_batch(
    data: np.ndarray,
    trajectory_ids: np.ndarray,
    starts: np.ndarray,
    seq_len: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    history_offsets = np.arange(int(seq_len), dtype=np.int64)
    target_offsets = np.arange(int(horizon), dtype=np.int64) + int(seq_len)
    history_indices = starts[:, None] + history_offsets[None, :]
    target_indices = starts[:, None] + target_offsets[None, :]
    histories = data[trajectory_ids[:, None], history_indices]
    targets = data[trajectory_ids[:, None], target_indices]
    return (
        np.ascontiguousarray(histories, dtype=np.float32),
        np.ascontiguousarray(targets, dtype=np.float32),
    )


def _fixed_validation_specs(
    trajectory_ids: np.ndarray,
    total_steps: int,
    seq_len: int,
    batch_size: int,
    num_groups: int,
    max_horizon: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    max_start = int(total_steps) - int(seq_len) - int(max_horizon)
    if max_start < 0:
        raise ValueError("Data is too short for fixed 100-step LSTM validation")
    specs = []
    total_slots = int(batch_size) * int(num_groups)
    for group in range(int(num_groups)):
        positions = np.arange(batch_size, dtype=np.int64) + group * batch_size
        lane_ids = trajectory_ids[positions % trajectory_ids.size]
        starts = np.floor(
            positions.astype(np.float64) * (max_start + 1) / max(total_slots, 1)
        ).astype(np.int64)
        specs.append((lane_ids.copy(), starts))
    return specs


def _run_batches(
    model: torch.nn.Module,
    data: np.ndarray,
    specs: list[tuple[np.ndarray, np.ndarray]],
    seq_len: int,
    horizon: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip: float,
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
    pointwise_steps_cap: int,
    micro_batch_size: int,
) -> dict:
    training = optimizer is not None
    model.train(training)
    totals = {"mse": 0.0, "optimization_loss": 0.0, "variance_loss": 0.0, "mean_loss": 0.0}
    sample_count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for trajectory_ids, starts in specs:
            effective_batch_size = int(trajectory_ids.size)
            if training:
                optimizer.zero_grad(set_to_none=True)
            for chunk_start in range(0, effective_batch_size, int(micro_batch_size)):
                chunk_stop = min(chunk_start + int(micro_batch_size), effective_batch_size)
                chunk_ids = trajectory_ids[chunk_start:chunk_stop]
                chunk_starts = starts[chunk_start:chunk_stop]
                histories_np, targets_np = _extract_batch(
                    data, chunk_ids, chunk_starts, seq_len, horizon
                )
                histories = torch.from_numpy(histories_np).to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                targets = torch.from_numpy(targets_np).to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                prediction = closed_loop_prediction(model, histories, horizon, training)
                objective, pointwise_mse, variance_loss, mean_loss = (
                    distribution_regularized_forecast_loss(
                        prediction,
                        targets,
                        min(int(pointwise_steps_cap), int(horizon)),
                        variance_weight,
                        mean_weight,
                        previous_state=histories[:, -1, :],
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
                    raise FloatingPointError(
                        f"Non-finite closed-loop loss at horizon H={horizon}"
                    )
                count = int(histories.shape[0])
                if training:
                    (objective * (count / effective_batch_size)).backward()
                totals["mse"] += float(pointwise_mse.detach().item()) * count
                totals["optimization_loss"] += float(objective.detach().item()) * count
                totals["variance_loss"] += float(variance_loss.detach().item()) * count
                totals["mean_loss"] += float(mean_loss.detach().item()) * count
                sample_count += count
                del prediction, histories, targets, objective
            if training:
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=float(gradient_clip)
                )
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError("Non-finite LSTM gradient norm")
                optimizer.step()
    if sample_count == 0:
        raise RuntimeError("No LSTM batches were processed")
    return {key: value / sample_count for key, value in totals.items()}


def _random_training_specs(
    rng: np.random.Generator,
    trajectory_ids: np.ndarray,
    total_steps: int,
    seq_len: int,
    horizon: int,
    batch_size: int,
    num_groups: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    max_start = int(total_steps) - int(seq_len) - int(horizon)
    if max_start < 0:
        raise ValueError(
            f"Data is too short for seq_len={seq_len} and horizon={horizon}"
        )
    specs = []
    for _ in range(int(num_groups)):
        lane_ids = rng.choice(trajectory_ids, size=int(batch_size), replace=True)
        starts = rng.integers(0, max_start + 1, size=int(batch_size), dtype=np.int64)
        specs.append((lane_ids.astype(np.int64, copy=False), starts))
    return specs


def _write_lstm_progress(
    path: Path,
    ds: int,
    sim_time: int,
    seq_len: int,
    pred_len: int,
    epoch: int,
    num_epochs: int,
    horizon: int,
    learning_rate: float,
    train_current_h_loss: float,
    validation_current_h_loss: float,
    validation_100_loss: float,
    best_validation_loss: float,
    patience_counter: int,
    rollout_summary: dict | None,
    status: str,
) -> None:
    atomic_json_write(
        {
            "status": status,
            "model_type": "lstm",
            "training_mode": TRAINING_MODE,
            "ds": int(ds),
            "sim_time": int(sim_time),
            "seq_len": int(seq_len),
            "pred_len": int(pred_len),
            "epoch": int(epoch),
            "num_epochs": int(num_epochs),
            "current_horizon": int(horizon),
            "learning_rate": float(learning_rate),
            "train_current_h_loss": float(train_current_h_loss),
            "validation_current_h_loss": float(validation_current_h_loss),
            "validation_100_loss": float(validation_100_loss),
            "best_validation_100_loss": (
                None if not math.isfinite(best_validation_loss) else float(best_validation_loss)
            ),
            "patience_counter": int(patience_counter),
            "rollout_selection_metrics": rollout_summary,
        },
        path,
    )


def train_lstm_model(
    config: dict,
    ds: int,
    sim_time: int,
    pred_len: int,
    resume: bool = True,
) -> None:
    _validate_lstm_settings(config, pred_len)
    config = lstm_effective_config(config, ds, pred_len)
    model_type = "lstm"
    seq_len = int(config["training"]["seq_len"])
    stride = int(config["training"]["stride"])
    random_seed = int(config["training"]["random_seed"])
    settings = config["lstm_training"]
    batch_size = int(settings["batch_size"])
    num_epochs = int(settings["num_epochs"])
    adaptive = settings.get("adaptive_early_stopping", {})
    effective_num_epochs = min(
        num_epochs, int(adaptive.get("max_epochs", num_epochs))
    )
    h100_start_epoch = min(
        int(phase["start_epoch"])
        for phase in settings["curriculum"]
        if int(phase["horizon"]) >= 100
    )
    longest_horizon = max(int(phase["horizon"]) for phase in settings["curriculum"])
    long_horizon_start_epoch = min(
        int(phase["start_epoch"])
        for phase in settings["curriculum"]
        if int(phase["horizon"]) == longest_horizon
    )
    minimum_epochs = max(
        h100_start_epoch, int(adaptive.get("minimum_epochs", h100_start_epoch))
    )
    relative_min_delta = float(adaptive.get("relative_min_delta", 0.0))
    reset_on_rollout = bool(adaptive.get("reset_on_rollout_improvement", True))
    num_groups = int(settings["max_batch_groups_per_epoch"])
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
    snapshot = effective_config(config, model_type, pred_len)

    set_random_seed(random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle: TrainingDataBundle = prepare_training_data(config, ds, sim_time)
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
    model_signature["training_mode"] = TRAINING_MODE
    resume_signature = build_resume_signature(
        config, model_type, ds, sim_time, input_dim, pred_len
    )
    resume_signature.update(
        {
            "training_mode": TRAINING_MODE,
            "seq_len_by_ds": copy.deepcopy(settings["seq_len_by_ds"]),
            "curriculum": copy.deepcopy(settings["curriculum"]),
            "h100_learning_rate": float(settings["h100_learning_rate"]),
            "eta_min": float(settings["eta_min"]),
            "micro_batch_size_by_horizon": copy.deepcopy(
                settings["micro_batch_size_by_horizon"]
            ),
            "initial_weights_path": settings.get("initial_weights_path"),
        }
    )
    model = build_model(model_type, input_dim, config, pred_len).to(
        device=device, dtype=torch.float32
    )
    initial_weights_path = settings.get("initial_weights_path")
    if not resume and initial_weights_path:
        source_path = Path(str(initial_weights_path))
        if not source_path.is_absolute():
            source_path = PROJECT_ROOT / source_path
        if not source_path.exists():
            raise FileNotFoundError(
                f"Configured LSTM initial weights do not exist: {source_path}"
            )
        source_checkpoint = load_torch_checkpoint(source_path, device)
        model.load_state_dict(source_checkpoint["model_state_dict"], strict=True)
        print(
            f"Initialized LSTM weights from {source_path} "
            f"(source epoch {source_checkpoint.get('epoch')})",
            flush=True,
        )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
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
        if checkpoint.get("scheduler_state_dict") is not None:
            raise ValueError("LSTM checkpoint unexpectedly contains scheduler state")
        start_epoch = int(checkpoint["epoch"])
        best_validation_loss = float(
            checkpoint.get("best_validation_100_loss_since_epoch_101", float("inf"))
        )
        best_rollout_metrics = checkpoint.get("best_rollout_metrics")
        latest_rollout_metrics = checkpoint.get("rollout_selection_metrics")
        patience_counter = int(checkpoint.get("patience_counter", 0))
        history = list(checkpoint.get("training_history", []))
        replay_rollout = None
        patience_counter = 0
        early_stopping_reference_loss = float("inf")
        for row in history:
            row_epoch = int(row["epoch"])
            if row_epoch >= h100_start_epoch:
                row_loss = float(row["validation_100_loss"])
                if row_loss < early_stopping_reference_loss * (1.0 - relative_min_delta):
                    early_stopping_reference_loss = row_loss
                    patience_counter = 0
                elif row_epoch >= minimum_epochs:
                    patience_counter += 1
            row_rollout = row.get("rollout_selection_metrics")
            if row_rollout is not None and is_better_rollout(row_rollout, replay_rollout):
                replay_rollout = row_rollout
                if reset_on_rollout:
                    patience_counter = 0
        restore_checkpoint_rng(checkpoint)
        if checkpoint.get("training_complete", False):
            return

    fixed_validation_specs = _fixed_validation_specs(
        bundle.validation_trajectory_ids,
        bundle.normalized_data.shape[1],
        seq_len,
        batch_size,
        num_groups,
        max(int(phase["horizon"]) for phase in settings["curriculum"]),
    )
    rollout_enabled = bool(config["rollout_validation"]["enabled"])
    rollout_every = int(config["rollout_validation"]["every_epochs"])

    for epoch_index in range(start_epoch, effective_num_epochs):
        epoch = epoch_index + 1
        horizon = curriculum_horizon(epoch, settings)
        micro_batch_size = micro_batch_size_for_horizon(settings, horizon)
        learning_rate = learning_rate_for_epoch(epoch, settings)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate

        rng = np.random.default_rng(random_seed + 1_000_003 * epoch_index)
        training_specs = _random_training_specs(
            rng,
            bundle.train_trajectory_ids,
            bundle.normalized_data.shape[1],
            seq_len,
            horizon,
            batch_size,
            num_groups,
        )
        train_metrics = _run_batches(
            model,
            bundle.normalized_data,
            training_specs,
            seq_len,
            horizon,
            device,
            optimizer,
            gradient_clip,
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
            pointwise_steps_cap,
            micro_batch_size,
        )
        validation_current_metrics = _run_batches(
            model,
            bundle.normalized_data,
            fixed_validation_specs,
            seq_len,
            horizon,
            device,
            None,
            gradient_clip,
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
            pointwise_steps_cap,
            micro_batch_size,
        )
        validation_100_metrics = (
            validation_current_metrics
            if horizon == 100
            else _run_batches(
                model,
                bundle.normalized_data,
                fixed_validation_specs,
                seq_len,
                100,
                device,
                None,
                gradient_clip,
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
                pointwise_steps_cap,
                micro_batch_size_for_horizon(settings, 100),
            )
        )
        train_current_h_loss = float(train_metrics["mse"])
        validation_current_h_loss = float(validation_current_metrics["mse"])
        validation_100_loss = float(validation_100_metrics["mse"])

        validation_improved = False
        if epoch >= h100_start_epoch:
            validation_improved = validation_100_loss < best_validation_loss
            if validation_improved:
                best_validation_loss = float(validation_100_loss)
            significant_validation_improvement = (
                validation_100_loss
                < early_stopping_reference_loss * (1.0 - relative_min_delta)
            )
            if significant_validation_improvement:
                early_stopping_reference_loss = float(validation_100_loss)
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
                validation_mse=float(validation_100_loss),
            )
            rollout_improved = bool(
                epoch >= long_horizon_start_epoch
                and is_better_rollout(latest_rollout_metrics, best_rollout_metrics)
            )
            if rollout_improved:
                best_rollout_metrics = copy.deepcopy(latest_rollout_metrics)
                if reset_on_rollout:
                    patience_counter = 0

        history_row = {
            "epoch": int(epoch),
            "current_horizon": int(horizon),
            "learning_rate": float(learning_rate),
            "train_current_h_loss": float(train_current_h_loss),
            "validation_current_h_loss": float(validation_current_h_loss),
            "validation_100_loss": float(validation_100_loss),
            "train_loss": float(train_current_h_loss),
            "validation_loss": float(validation_current_h_loss),
            "train_optimization_loss": float(train_metrics["optimization_loss"]),
            "validation_optimization_loss": float(validation_current_metrics["optimization_loss"]),
            "validation_100_optimization_loss": float(validation_100_metrics["optimization_loss"]),
            "train_variance_loss": float(train_metrics["variance_loss"]),
            "validation_variance_loss": float(validation_current_metrics["variance_loss"]),
            "train_mean_loss": float(train_metrics["mean_loss"]),
            "validation_mean_loss": float(validation_current_metrics["mean_loss"]),
            "rollout_selection_metrics": (
                copy.deepcopy(latest_rollout_metrics)
                if epoch % rollout_every == 0
                else None
            ),
        }
        history.append(history_row)
        training_complete = bool(
            epoch == effective_num_epochs
            or (epoch >= minimum_epochs and patience_counter >= patience)
        )
        checkpoint_best_loss = (
            best_validation_loss
            if math.isfinite(best_validation_loss)
            else float(validation_100_loss)
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
            scheduler_state_dict=None,
            bundle=bundle,
            best_validation_loss=checkpoint_best_loss,
            rollout_selection_metrics=latest_rollout_metrics,
            best_rollout_metrics=best_rollout_metrics,
            training_history=history,
            model_signature=model_signature,
            resume_signature=resume_signature,
            config_snapshot=snapshot,
            patience_counter=patience_counter,
            training_complete=training_complete,
        )
        payload.update(
            {
                "training_mode": TRAINING_MODE,
                "current_horizon": int(horizon),
                "validation_100_loss": float(validation_100_loss),
                "best_validation_100_loss_since_epoch_101": float(
                    best_validation_loss
                ),
                "initial_weights_path": settings.get("initial_weights_path"),
            }
        )
        if validation_improved:
            atomic_torch_save(payload, best_val_path)
        if rollout_improved:
            atomic_torch_save(payload, best_rollout_path)
        atomic_torch_save(payload, latest_path)
        _write_lstm_progress(
            status_path,
            ds,
            sim_time,
            seq_len,
            pred_len,
            epoch,
            effective_num_epochs,
            horizon,
            learning_rate,
            train_current_h_loss,
            validation_current_h_loss,
            validation_100_loss,
            best_validation_loss,
            patience_counter,
            latest_rollout_metrics,
            "complete" if training_complete else "running",
        )
        print(
            f"Epoch {epoch}/{effective_num_epochs} | H={horizon} | "
            f"train_current_h={train_current_h_loss:.8e} | "
            f"val_current_h={validation_current_h_loss:.8e} | "
            f"val_100={validation_100_loss:.8e} | lr={learning_rate:.8e}",
            flush=True,
        )
        if training_complete:
            break
