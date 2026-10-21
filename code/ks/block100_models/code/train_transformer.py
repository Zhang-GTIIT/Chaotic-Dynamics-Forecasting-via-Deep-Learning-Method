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
from metrics import compute_nrmse_metrics, is_better_rollout
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


def build_model_signature(
    config: dict,
    model_type: str,
    ds: int,
    sim_time: int,
    input_dim: int,
    pred_len: int,
) -> dict:
    signature = {
        "signature_version": 1,
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
            }
        )
        if model_type == "easy_attention_sgu":
            signature["alpha_noise_std"] = float(
                config["easy_attention_sgu"]["alpha_noise_std"]
            )
    elif model_type == "lstm":
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
        settings = config["attention_training"]
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


def run_attention_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip: float,
) -> dict:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "horizon_1": 0.0, "horizon_mid": 0.0, "horizon_last": 0.0}
    sample_count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for src, target in loader:
            src = src.to(device=device, dtype=torch.float32, non_blocking=True)
            target = target.to(device=device, dtype=torch.float32, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(src)
            loss = F.mse_loss(prediction, target)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
                optimizer.step()

            horizon_1, horizon_mid, horizon_last = _batch_diagnostics(prediction, target)
            batch_size = int(src.shape[0])
            totals["loss"] += float(loss.detach().item()) * batch_size
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
        validation_mse=validation_mse,
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
    if bool(config["attention_training"]["use_amp"]):
        raise ValueError("Formal attention training is fp32 and requires use_amp=false")
    if config["attention_training"]["scheduler"] != "cosine":
        raise ValueError("Formal attention training requires the cosine scheduler")


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
    settings = config["attention_training"]
    batch_size = int(settings["batch_size"])
    num_epochs = int(settings["num_epochs"])
    max_train_batches = int(settings["max_train_batches_per_epoch"])
    max_val_batches = int(settings["max_val_batches_per_epoch"])
    patience = int(settings["early_stopping_patience"])
    gradient_clip = float(settings["gradient_clip"])
    snapshot = effective_config(config, model_type, pred_len)

    set_random_seed(random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = prepare_training_data(config, ds, sim_time)
    input_dim = int(bundle.normalized_data.shape[2])
    model_signature = build_model_signature(
        config, model_type, ds, sim_time, input_dim, pred_len
    )
    resume_signature = build_resume_signature(
        config, model_type, ds, sim_time, input_dim, pred_len
    )
    train_dataset = RandomWindowDataset(
        data=bundle.normalized_data,
        trajectory_ids=bundle.train_trajectory_ids,
        seq_len=seq_len,
        pred_len=pred_len,
        stride=stride,
        num_samples=max_train_batches * batch_size,
        random_seed=random_seed,
    )
    validation_dataset = FixedWindowDataset(
        data=bundle.normalized_data,
        trajectory_ids=bundle.validation_trajectory_ids,
        seq_len=seq_len,
        pred_len=pred_len,
        stride=stride,
        max_samples=max_val_batches * batch_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
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
        T_max=num_epochs,
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
        train_dataset.set_epoch(epoch_index)
        train_metrics = run_attention_epoch(
            model, train_loader, device, optimizer, gradient_clip
        )
        val_metrics = run_attention_epoch(
            model, validation_loader, device, None, gradient_clip
        )
        scheduler.step()
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
