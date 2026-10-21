"""Closed-loop curriculum training for the stateful KS LSTM."""

from __future__ import annotations

import copy
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
    effective_config,
    evaluate_rollout_validation,
    load_torch_checkpoint,
    progress_path,
    restore_checkpoint_rng,
    set_random_seed,
    validate_checkpoint_compatibility,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_MODE = "closed_loop_curriculum"


def _normalized_curriculum(config: dict) -> list[dict[str, int]]:
    return [
        {
            "end_epoch": int(stage["end_epoch"]),
            "horizon": int(stage["horizon"]),
        }
        for stage in config["lstm_training"]["curriculum"]
    ]


def get_lstm_training_horizon(epoch: int, config: dict) -> int:
    """Return the closed-loop horizon assigned to a one-based epoch."""
    epoch = int(epoch)
    if epoch <= 0:
        raise ValueError("epoch must be one-based and positive")
    pred_len = int(config["training"]["pred_len"])
    for stage in _normalized_curriculum(config):
        horizon = int(stage["horizon"])
        if horizon <= 0 or horizon > pred_len:
            raise ValueError(
                f"Invalid LSTM curriculum horizon {horizon}; pred_len={pred_len}"
            )
        if epoch <= int(stage["end_epoch"]):
            return horizon
    raise ValueError(f"No LSTM curriculum stage covers epoch {epoch}")


def _validate_lstm_settings(config: dict, pred_len: int) -> None:
    validate_pred_len(config, pred_len)
    if int(pred_len) != int(config["training"]["pred_len"]):
        raise ValueError("LSTM-v2 requires the configured pred_len=100")
    if int(config["training"]["seq_len"]) != 128:
        raise ValueError("LSTM-v2 requires seq_len=128")
    if int(config["training"]["stride"]) != 1:
        raise ValueError("LSTM-v2 requires stride=1")
    if config["loss"]["type"] != "mse" or config["loss"]["horizon_weighting"] != "uniform":
        raise ValueError("LSTM-v2 training requires uniform MSE")
    if bool(config["lstm"]["residual_connections"]):
        raise ValueError("LSTM-v2 requires residual_connections=false")
    if config["lstm"]["dtype"] != "float32":
        raise ValueError("LSTM-v2 requires dtype='float32'")

    settings = config["lstm_training"]
    if settings.get("training_mode") != TRAINING_MODE:
        raise ValueError(f"LSTM-v2 requires training_mode={TRAINING_MODE!r}")
    if bool(settings["use_amp"]):
        raise ValueError("LSTM-v2 requires use_amp=false")
    if int(settings["fixed_validation_horizon"]) != int(pred_len):
        raise ValueError("fixed_validation_horizon must equal pred_len=100")
    if int(settings["early_stopping_start_epoch"]) != 101:
        raise ValueError("LSTM-v2 early stopping must start at epoch 101")
    curriculum = _normalized_curriculum(config)
    expected = [
        {"end_epoch": 20, "horizon": 10},
        {"end_epoch": 50, "horizon": 20},
        {"end_epoch": 100, "horizon": 50},
        {"end_epoch": 250, "horizon": 100},
    ]
    if curriculum != expected:
        raise ValueError(f"Unexpected LSTM-v2 curriculum: {curriculum!r}")
    if curriculum[-1]["end_epoch"] < int(settings["num_epochs"]):
        raise ValueError("LSTM curriculum does not cover every training epoch")


def closed_loop_lstm_forward(
    model: torch.nn.Module,
    history: torch.Tensor,
    horizon: int,
    training: bool,
) -> torch.Tensor:
    """Warm up on real history, then feed predictions back with a continuous graph."""
    horizon = int(horizon)
    if history.ndim != 3:
        raise ValueError("history must have shape [B,seq_len,D]")
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    outputs, hidden_state = model(history, hidden_state=None, training=training)
    pred = outputs[:, -1:, :]
    predictions = [pred]
    for _ in range(1, horizon):
        outputs, hidden_state = model(pred, hidden_state, training=training)
        pred = outputs[:, -1:, :]
        predictions.append(pred)
    prediction = torch.cat(predictions, dim=1)
    expected = (history.shape[0], horizon, history.shape[2])
    if tuple(prediction.shape) != expected:
        raise RuntimeError(
            f"Closed-loop LSTM returned {tuple(prediction.shape)}; expected {expected}"
        )
    return prediction


def _legal_start_count(
    total_steps: int, seq_len: int, pred_len: int, stride: int
) -> int:
    maximum_start = int(total_steps) - int(seq_len) - int(pred_len)
    if maximum_start < 0:
        raise ValueError(
            f"No legal LSTM window: total_steps={total_steps}, "
            f"seq_len={seq_len}, pred_len={pred_len}"
        )
    return maximum_start // int(stride) + 1


def _sample_training_specs(
    rng: np.random.Generator,
    trajectory_ids: np.ndarray,
    batch_size: int,
    legal_start_count: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    positions = rng.integers(0, trajectory_ids.size, size=int(batch_size))
    lane_ids = trajectory_ids[positions]
    starts = rng.integers(0, int(legal_start_count), size=int(batch_size))
    return lane_ids.astype(np.int64, copy=False), (starts * int(stride)).astype(
        np.int64, copy=False
    )


def _build_fixed_validation_specs(
    trajectory_ids: np.ndarray,
    sample_count: int,
    legal_start_count: int,
    stride: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if int(sample_count) < int(trajectory_ids.size):
        raise ValueError("Fixed validation samples must cover every validation trajectory")
    repeats = (int(sample_count) + trajectory_ids.size - 1) // trajectory_ids.size
    lane_ids = np.tile(trajectory_ids, repeats)[: int(sample_count)].copy()
    rng = np.random.default_rng(int(random_seed))
    rng.shuffle(lane_ids)
    start_slots = rng.integers(0, int(legal_start_count), size=int(sample_count))
    starts = (start_slots * int(stride)).astype(np.int64, copy=False)
    return lane_ids.astype(np.int64, copy=False), starts


def _extract_closed_loop_batch(
    data: np.ndarray,
    lane_ids: np.ndarray,
    starts: np.ndarray,
    seq_len: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    history_offsets = starts[:, None] + np.arange(int(seq_len), dtype=np.int64)[None, :]
    target_offsets = (
        starts[:, None]
        + int(seq_len)
        + np.arange(int(horizon), dtype=np.int64)[None, :]
    )
    history = data[lane_ids[:, None], history_offsets]
    targets = data[lane_ids[:, None], target_offsets]
    return (
        np.ascontiguousarray(history, dtype=np.float32),
        np.ascontiguousarray(targets, dtype=np.float32),
    )


def _horizon_diagnostics(
    prediction: torch.Tensor, target: torch.Tensor
) -> tuple[float, float, float]:
    per_horizon = (prediction - target).square().mean(dim=(0, 2))
    middle = per_horizon.shape[0] // 2
    return (
        float(per_horizon[0].detach().item()),
        float(per_horizon[middle].detach().item()),
        float(per_horizon[-1].detach().item()),
    )


def run_closed_loop_epoch(
    *,
    model: torch.nn.Module,
    data: np.ndarray,
    trajectory_ids: np.ndarray,
    seq_len: int,
    pred_len: int,
    horizon: int,
    stride: int,
    batch_size: int,
    num_batches: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip: float,
    random_seed: int,
    fixed_specs: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    """Run free-running rollout MSE; future truth is used only as supervision."""
    training = optimizer is not None
    model.train(training)
    trajectory_ids = np.asarray(trajectory_ids, dtype=np.int64)
    if trajectory_ids.size == 0:
        raise ValueError("trajectory_ids cannot be empty")
    if int(horizon) <= 0 or int(horizon) > int(pred_len):
        raise ValueError(f"horizon must be in [1,{pred_len}]")
    if int(batch_size) <= 0 or int(num_batches) <= 0:
        raise ValueError("batch_size and num_batches must be positive")

    legal_count = _legal_start_count(data.shape[1], seq_len, pred_len, stride)
    rng = np.random.default_rng(int(random_seed))
    if fixed_specs is not None:
        fixed_ids, fixed_starts = fixed_specs
        expected_samples = int(batch_size) * int(num_batches)
        if fixed_ids.shape != (expected_samples,) or fixed_starts.shape != (expected_samples,):
            raise ValueError("Fixed validation specifications have an unexpected shape")
        if not np.isin(fixed_ids, trajectory_ids).all():
            raise ValueError("Fixed validation specifications contain non-validation IDs")

    totals = {"loss": 0.0, "horizon_1": 0.0, "horizon_mid": 0.0, "horizon_last": 0.0}
    sample_count = 0
    for batch_index in range(int(num_batches)):
        if fixed_specs is None:
            lane_ids, starts = _sample_training_specs(
                rng, trajectory_ids, batch_size, legal_count, stride
            )
        else:
            begin = batch_index * int(batch_size)
            end = begin + int(batch_size)
            lane_ids = fixed_ids[begin:end]
            starts = fixed_starts[begin:end]

        history_np, targets_np = _extract_closed_loop_batch(
            data, lane_ids, starts, seq_len, horizon
        )
        history = torch.from_numpy(history_np).to(device=device, dtype=torch.float32)
        targets = torch.from_numpy(targets_np).to(device=device, dtype=torch.float32)

        if training:
            optimizer.zero_grad(set_to_none=True)
            prediction = closed_loop_lstm_forward(model, history, horizon, training=True)
            loss = F.mse_loss(prediction, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            optimizer.step()
        else:
            with torch.no_grad():
                prediction = closed_loop_lstm_forward(
                    model, history, horizon, training=False
                )
                loss = F.mse_loss(prediction, targets)

        horizon_1, horizon_mid, horizon_last = _horizon_diagnostics(
            prediction, targets
        )
        current_count = int(history.shape[0])
        totals["loss"] += float(loss.detach().item()) * current_count
        totals["horizon_1"] += horizon_1 * current_count
        totals["horizon_mid"] += horizon_mid * current_count
        totals["horizon_last"] += horizon_last * current_count
        sample_count += current_count

    if sample_count == 0:
        raise RuntimeError("No closed-loop LSTM batches were processed")
    return {key: value / sample_count for key, value in totals.items()}


def _write_lstm_progress(
    *,
    path: Path,
    ds: int,
    sim_time: int,
    seq_len: int,
    pred_len: int,
    epoch: int,
    num_epochs: int,
    training_horizon: int,
    fixed_validation_horizon: int,
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
            "model_type": "lstm",
            "training_mode": TRAINING_MODE,
            "ds": int(ds),
            "sim_time": int(sim_time),
            "seq_len": int(seq_len),
            "pred_len": int(pred_len),
            "epoch": int(epoch),
            "num_epochs": int(num_epochs),
            "current_training_horizon": int(training_horizon),
            "fixed_validation_horizon": int(fixed_validation_horizon),
            "train_loss": float(train_metrics["loss"]),
            "validation_loss": float(val_metrics["loss"]),
            "train_horizon_1_mse": float(train_metrics["horizon_1"]),
            "train_horizon_mid_mse": float(train_metrics["horizon_mid"]),
            "train_horizon_last_mse": float(train_metrics["horizon_last"]),
            "validation_horizon_1_mse": float(val_metrics["horizon_1"]),
            "validation_horizon_mid_mse": float(val_metrics["horizon_mid"]),
            "validation_horizon_last_mse": float(val_metrics["horizon_last"]),
            "best_validation_loss": float(best_validation_loss),
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
    model_type = "lstm"
    seq_len = int(config["training"]["seq_len"])
    stride = int(config["training"]["stride"])
    random_seed = int(config["training"]["random_seed"])
    settings = config["lstm_training"]
    batch_size = int(settings["batch_size"])
    num_epochs = int(settings["num_epochs"])
    max_train_batches = int(settings["max_train_batches_per_epoch"])
    max_val_batches = int(settings["max_val_batches_per_epoch"])
    patience = int(settings["early_stopping_patience"])
    early_stopping_start = int(settings["early_stopping_start_epoch"])
    validation_horizon = int(settings["fixed_validation_horizon"])
    rollout_every = int(settings["rollout_every_epochs"])
    gradient_clip = float(settings["gradient_clip"])
    curriculum = _normalized_curriculum(config)
    snapshot = effective_config(config, model_type, pred_len)

    set_random_seed(random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle: TrainingDataBundle = prepare_training_data(config, ds, sim_time)
    input_dim = int(bundle.normalized_data.shape[2])
    model_signature = build_model_signature(
        config, model_type, ds, sim_time, input_dim, pred_len
    )
    resume_signature = build_resume_signature(
        config, model_type, ds, sim_time, input_dim, pred_len
    )
    model = build_model(model_type, input_dim, config, pred_len).to(
        device=device, dtype=torch.float32
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

    legal_count = _legal_start_count(
        bundle.normalized_data.shape[1], seq_len, pred_len, stride
    )
    fixed_validation_specs = _build_fixed_validation_specs(
        bundle.validation_trajectory_ids,
        sample_count=max_val_batches * batch_size,
        legal_start_count=legal_count,
        stride=stride,
        random_seed=random_seed + 97_409,
    )

    start_epoch = 0
    best_validation_loss = float("inf")
    best_rollout_metrics = None
    latest_rollout_metrics = None
    patience_counter = 0
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
            raise ValueError("LSTM-v2 checkpoint unexpectedly contains a scheduler")
        if checkpoint.get("training_mode") != TRAINING_MODE:
            raise ValueError("Cannot resume a teacher-forced LSTM checkpoint")
        start_epoch = int(checkpoint["epoch"])
        best_validation_loss = float(checkpoint["best_validation_loss"])
        best_rollout_metrics = checkpoint.get("best_rollout_metrics")
        latest_rollout_metrics = checkpoint.get("rollout_selection_metrics")
        patience_counter = int(checkpoint.get("patience_counter", 0))
        history = list(checkpoint.get("training_history", []))
        restore_checkpoint_rng(checkpoint)
        if checkpoint.get("training_complete", False):
            return

    rollout_enabled = bool(config["rollout_validation"]["enabled"])
    for epoch_index in range(start_epoch, num_epochs):
        epoch = epoch_index + 1
        training_horizon = get_lstm_training_horizon(epoch, config)
        train_metrics = run_closed_loop_epoch(
            model=model,
            data=bundle.normalized_data,
            trajectory_ids=bundle.train_trajectory_ids,
            seq_len=seq_len,
            pred_len=pred_len,
            horizon=training_horizon,
            stride=stride,
            batch_size=batch_size,
            num_batches=max_train_batches,
            device=device,
            optimizer=optimizer,
            gradient_clip=gradient_clip,
            random_seed=random_seed + 1_000_003 * epoch_index,
        )
        val_metrics = run_closed_loop_epoch(
            model=model,
            data=bundle.normalized_data,
            trajectory_ids=bundle.validation_trajectory_ids,
            seq_len=seq_len,
            pred_len=pred_len,
            horizon=validation_horizon,
            stride=stride,
            batch_size=batch_size,
            num_batches=max_val_batches,
            device=device,
            optimizer=None,
            gradient_clip=gradient_clip,
            random_seed=random_seed + 97_409,
            fixed_specs=fixed_validation_specs,
        )

        validation_improved = val_metrics["loss"] < best_validation_loss
        if validation_improved:
            best_validation_loss = float(val_metrics["loss"])
        if epoch < early_stopping_start:
            patience_counter = 0
        elif validation_improved:
            patience_counter = 0
        else:
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

        history_row = {
            "epoch": epoch,
            "training_horizon": training_horizon,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(train_metrics["loss"]),
            "validation_loss": float(val_metrics["loss"]),
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
        training_complete = epoch == num_epochs or (
            epoch >= early_stopping_start and patience_counter >= patience
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
        payload.update(
            {
                "training_mode": TRAINING_MODE,
                "hidden_size": int(config["lstm"]["hidden_size"]),
                "curriculum": copy.deepcopy(curriculum),
                "current_training_horizon": int(training_horizon),
                "fixed_validation_horizon": int(validation_horizon),
            }
        )
        if validation_improved:
            atomic_torch_save(payload, best_val_path)
        if rollout_improved:
            atomic_torch_save(payload, best_rollout_path)
        atomic_torch_save(payload, latest_path)
        _write_lstm_progress(
            path=status_path,
            ds=ds,
            sim_time=sim_time,
            seq_len=seq_len,
            pred_len=pred_len,
            epoch=epoch,
            num_epochs=num_epochs,
            training_horizon=training_horizon,
            fixed_validation_horizon=validation_horizon,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            best_validation_loss=best_validation_loss,
            patience_counter=patience_counter,
            rollout_summary=latest_rollout_metrics,
            status="complete" if training_complete else "running",
        )
        print(
            f"Epoch {epoch}/{num_epochs} | H={training_horizon} | "
            f"train_closed_loop={train_metrics['loss']:.8e} | "
            f"val_closed_loop_100={val_metrics['loss']:.8e} | "
            f"lr={optimizer.param_groups[0]['lr']:.8e}"
        )
        if training_complete:
            break
