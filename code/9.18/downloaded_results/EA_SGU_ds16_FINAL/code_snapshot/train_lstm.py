"""Stateful teacher-forced LSTM training on trajectory-isolated splits."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data_loader import (
    TrainingDataBundle,
    extract_lstm_chunk,
    lstm_chunk_count,
    prepare_training_data,
    sample_lstm_group_spec,
    validate_pred_len,
)
from metrics import is_better_rollout
from model import build_model
from train_transformer import (
    _write_progress,
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


def _validate_lstm_settings(config: dict, pred_len: int) -> None:
    validate_pred_len(config, pred_len)
    if config["loss"]["type"] != "mse" or config["loss"]["horizon_weighting"] != "uniform":
        raise ValueError("LSTM training requires uniform MSE")
    if bool(config["lstm"]["residual_connections"]):
        raise ValueError("The formal LSTM baseline requires residual_connections=false")
    if config["lstm"]["dtype"] != "float32":
        raise ValueError("The formal LSTM baseline requires dtype='float32'")
    if bool(config["lstm_training"]["use_amp"]):
        raise ValueError("The formal LSTM baseline requires use_amp=false")


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


def run_lstm_epoch(
    model: torch.nn.Module,
    data: np.ndarray,
    trajectory_ids: np.ndarray,
    seq_len: int,
    pred_len: int,
    stride: int,
    hidden_state_propagation: int,
    batch_size: int,
    num_groups: int,
    random_seed: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip: float,
) -> dict:
    """Reset state per group and retain/detach it only along each fixed lane."""
    training = optimizer is not None
    model.train(training)
    rng = np.random.default_rng(int(random_seed))
    num_chunks = lstm_chunk_count(hidden_state_propagation, seq_len, pred_len)
    totals = {"loss": 0.0, "horizon_1": 0.0, "horizon_mid": 0.0, "horizon_last": 0.0}
    observation_count = 0

    for group_index in range(int(num_groups)):
        lane_ids, initial_predict_on = sample_lstm_group_spec(
            rng=rng,
            trajectory_ids=trajectory_ids,
            batch_size=batch_size,
            total_steps=data.shape[1],
            seq_len=seq_len,
            pred_len=pred_len,
            hidden_state_propagation=hidden_state_propagation,
            stride=stride,
        )
        if not training:
            positions = (
                np.arange(batch_size, dtype=np.int64) + group_index * batch_size
            ) % trajectory_ids.size
            lane_ids = trajectory_ids[positions]

        hidden_state = None
        for chunk_index in range(num_chunks):
            inputs_np, targets_np = extract_lstm_chunk(
                data=data,
                lane_trajectory_ids=lane_ids,
                initial_predict_on=initial_predict_on,
                chunk_index=chunk_index,
                seq_len=seq_len,
                pred_len=pred_len,
            )
            inputs = torch.from_numpy(inputs_np).to(device=device, dtype=torch.float32)
            targets = torch.from_numpy(targets_np).to(device=device, dtype=torch.float32)
            if training:
                optimizer.zero_grad(set_to_none=True)
                outputs, hidden_state = model(inputs, hidden_state, training=True)
                prediction = outputs[:, -pred_len:, :]
                loss = F.mse_loss(prediction, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
                optimizer.step()
            else:
                with torch.no_grad():
                    outputs, hidden_state = model(inputs, hidden_state, training=False)
                    prediction = outputs[:, -pred_len:, :]
                    loss = F.mse_loss(prediction, targets)

            horizon_1, horizon_mid, horizon_last = _horizon_diagnostics(
                prediction, targets
            )
            current_count = int(inputs.shape[0])
            totals["loss"] += float(loss.detach().item()) * current_count
            totals["horizon_1"] += horizon_1 * current_count
            totals["horizon_mid"] += horizon_mid * current_count
            totals["horizon_last"] += horizon_last * current_count
            observation_count += current_count
            hidden_state = [(h.detach(), c.detach()) for h, c in hidden_state]

    if observation_count == 0:
        raise RuntimeError("No LSTM batch groups were processed")
    return {key: value / observation_count for key, value in totals.items()}


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
    hidden_state_propagation = int(config["lstm"]["hidden_state_propagation"])
    settings = config["lstm_training"]
    batch_size = int(settings["batch_size"])
    num_epochs = int(settings["num_epochs"])
    max_groups = int(settings["max_batch_groups_per_epoch"])
    patience = int(settings["early_stopping_patience"])
    gradient_clip = float(settings["gradient_clip"])
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
            raise ValueError("LSTM checkpoint unexpectedly contains a scheduler state")
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
    rollout_every = int(config["rollout_validation"]["every_epochs"])
    for epoch_index in range(start_epoch, num_epochs):
        epoch_seed = random_seed + 1_000_003 * epoch_index
        train_metrics = run_lstm_epoch(
            model=model,
            data=bundle.normalized_data,
            trajectory_ids=bundle.train_trajectory_ids,
            seq_len=seq_len,
            pred_len=pred_len,
            stride=stride,
            hidden_state_propagation=hidden_state_propagation,
            batch_size=batch_size,
            num_groups=max_groups,
            random_seed=epoch_seed,
            device=device,
            optimizer=optimizer,
            gradient_clip=gradient_clip,
        )
        val_metrics = run_lstm_epoch(
            model=model,
            data=bundle.normalized_data,
            trajectory_ids=bundle.validation_trajectory_ids,
            seq_len=seq_len,
            pred_len=pred_len,
            stride=stride,
            hidden_state_propagation=hidden_state_propagation,
            batch_size=batch_size,
            num_groups=max_groups,
            random_seed=random_seed + 97_409,
            device=device,
            optimizer=None,
            gradient_clip=gradient_clip,
        )
        epoch = epoch_index + 1
        validation_improved = val_metrics["loss"] < best_validation_loss
        if validation_improved:
            best_validation_loss = float(val_metrics["loss"])
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
        training_complete = epoch == num_epochs or patience_counter >= patience
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
            num_epochs,
            train_metrics,
            val_metrics,
            best_validation_loss,
            patience_counter,
            latest_rollout_metrics,
            "complete" if training_complete else "running",
        )
        print(
            f"Epoch {epoch}/{num_epochs} | train={train_metrics['loss']:.8e} | "
            f"val={val_metrics['loss']:.8e} | lr={optimizer.param_groups[0]['lr']:.8e}"
        )
        if training_complete:
            break
