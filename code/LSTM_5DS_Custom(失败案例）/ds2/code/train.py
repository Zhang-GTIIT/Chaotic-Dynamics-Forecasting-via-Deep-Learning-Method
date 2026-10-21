from __future__ import annotations

import argparse
import copy
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from common import (
    PROJECT_ROOT,
    atomic_json,
    atomic_text,
    atomic_torch_save,
    candidate_config,
    candidate_root,
    gpu_status,
    load_config,
    phase_for_epoch,
    set_seed,
    snapshot_code,
)
from data import (
    evaluation_windows,
    fixed_specs,
    load_training_array,
    normalized_batch,
    random_specs,
    split_ids,
    train_statistics,
)
from metrics import compute_metrics, is_better
from model import autoregressive_rollout, build_model, detach_hidden


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    settings: dict,
    config: dict,
    epoch: int,
    history: list[dict],
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    best_validation_loss: float,
    latest_rollout: dict | None,
    best_rollout: dict | None,
) -> dict:
    return {
        "format_version": 1,
        "model_type": "ResidualDeltaLSTM",
        "prediction_type": "stateful_one_step_delta_autoregression",
        "training_method": "continuous_closed_loop_tbptt",
        "ds": int(settings["ds"]),
        "D": int(settings["D"]),
        "candidate": int(settings["candidate"]),
        "seq_len": int(settings["seq_len"]),
        "hidden_size": int(settings["hidden_size"]),
        "num_layers": int(settings["num_layers"]),
        "feature_order": int(settings["feature_order"]),
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_trajectory_ids": train_ids.tolist(),
        "validation_trajectory_ids": val_ids.tolist(),
        "data_mean": mean,
        "data_std": std,
        "best_validation_loss": float(best_validation_loss),
        "rollout_selection_metrics": copy.deepcopy(latest_rollout),
        "best_rollout_metrics": copy.deepcopy(best_rollout),
        "training_history": copy.deepcopy(history),
        "settings": copy.deepcopy(settings),
        "config_snapshot": copy.deepcopy(config),
        "rng_state": {
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def phase_a_batch(
    model,
    optimizer,
    sequence: torch.Tensor,
    delta_weight: float,
    gradient_clip: float,
) -> float:
    model.train()
    history, target = sequence[:, :-1], sequence[:, -1]
    optimizer.zero_grad(set_to_none=True)
    deltas, _ = model.warmup(history)
    predicted_delta = deltas[:, -1]
    prediction = history[:, -1] + predicted_delta
    true_delta = target - history[:, -1]
    loss = F.mse_loss(prediction, target) + float(delta_weight) * F.mse_loss(predicted_delta, true_delta)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite phase-A loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
    optimizer.step()
    return float(loss.detach())


def closed_loop_batch(
    model,
    optimizer,
    sequence: torch.Tensor,
    seq_len: int,
    horizon: int,
    tbptt_segment: int,
    delta_weight: float,
    gradient_clip: float,
) -> float:
    model.train()
    history = sequence[:, :seq_len]
    targets = sequence[:, seq_len : seq_len + horizon]
    warmup_delta, hidden = model.warmup(history)
    previous2 = history[:, -3]
    previous = history[:, -2]
    current = history[:, -1]
    pending_next = current + warmup_delta[:, -1]
    pending_delta = warmup_delta[:, -1]
    total = 0.0
    total_steps = 0
    for segment_start in range(0, int(horizon), int(tbptt_segment)):
        segment_stop = min(int(horizon), segment_start + int(tbptt_segment))
        optimizer.zero_grad(set_to_none=True)
        predictions = []
        deltas = []
        for global_step in range(segment_start, segment_stop):
            if global_step == 0:
                next_state, delta = pending_next, pending_delta
            else:
                next_state, delta, hidden = model.predict_step(current, previous, previous2, hidden)
            predictions.append(next_state)
            deltas.append(delta)
            previous2, previous, current = previous, current, next_state
        prediction = torch.stack(predictions, dim=1)
        predicted_delta = torch.stack(deltas, dim=1)
        target = targets[:, segment_start:segment_stop]
        target_previous = history[:, -1:] if segment_start == 0 else targets[:, segment_start - 1 : segment_start]
        target_with_previous = torch.cat([target_previous, target], dim=1)
        true_delta = target_with_previous[:, 1:] - target_with_previous[:, :-1]
        loss = F.mse_loss(prediction, target)
        if float(delta_weight) > 0.0:
            loss = loss + float(delta_weight) * F.mse_loss(predicted_delta, true_delta)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite closed-loop loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
        optimizer.step()
        segment_length = segment_stop - segment_start
        total += float(loss.detach()) * segment_length
        total_steps += segment_length
        hidden = detach_hidden(hidden)
        previous2, previous, current = previous2.detach(), previous.detach(), current.detach()
    return total / max(1, total_steps)


@torch.no_grad()
def validation_loss(model, sequence: torch.Tensor, seq_len: int, horizon: int) -> float:
    history = sequence[:, :seq_len]
    target = sequence[:, seq_len : seq_len + horizon]
    prediction = autoregressive_rollout(model, history, horizon)
    return float(F.mse_loss(prediction, target))


@torch.no_grad()
def rollout_validation(
    model,
    ds: int,
    seq_len: int,
    mean: np.ndarray,
    std: np.ndarray,
    val_ids: np.ndarray,
    evaluation: dict,
    device: torch.device,
    val_loss: float,
) -> dict:
    windows = evaluation_windows(
        ds=ds,
        test=False,
        trajectory_ids=val_ids[: int(evaluation["validation_trajectories"])],
        seq_len=seq_len,
        forecast_steps=int(evaluation["validation_rollout_steps"]),
    )
    normalized = (windows.initial - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    history = torch.from_numpy(np.ascontiguousarray(normalized)).to(device=device, dtype=torch.float32)
    predicted_normalized = autoregressive_rollout(model, history, int(evaluation["validation_rollout_steps"]))
    prediction = predicted_normalized.cpu().numpy() * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)
    return compute_metrics(prediction, windows.truth, std, evaluation, val_loss)["summary"]


def write_live_status(
    ds: int,
    candidate: int,
    epoch: int,
    settings: dict,
    phase: dict,
    train_loss: float,
    val_loss: float,
    rollout: dict | None,
) -> None:
    gpu = gpu_status()
    lines = [
        f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "stage: training",
        f"current ds: {ds}",
        f"current candidate: {candidate}",
        f"epoch: {epoch}",
        f"max epochs: {settings['max_epochs']}",
        f"current H: {phase['horizon']}",
        f"current BPTT segment: {min(int(settings['tbptt_segment']), int(phase['horizon']))}",
        f"learning rate: {float(phase['lr']):.8g}",
        f"train loss: {train_loss:.8g}",
        f"validation loss: {val_loss:.8g}",
    ]
    if rollout:
        lines.extend(
            [
                f"H0.5: {rollout['H0.5']}",
                f"H1.0: {rollout['H1.0']}",
                f"Hsqrt2: {rollout['Hsqrt2']}",
                f"late_std_ratio: {rollout['late_std_ratio']:.6g}",
                f"spectral_entropy_ratio: {rollout['spectral_entropy_ratio']:.6g}",
                f"invalid flag: {rollout['invalid_dynamics_flag']}",
            ]
        )
    lines.extend(f"{key}: {value}" for key, value in gpu.items())
    atomic_text(PROJECT_ROOT / "LIVE_STATUS.txt", "\n".join(lines) + "\n")


def train_candidate(ds: int, candidate: int) -> Path:
    config = load_config()
    settings = candidate_config(config, ds, candidate)
    root = candidate_root(ds, candidate)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Candidate directory is not empty: {root}")
    for leaf in ("code_snapshot", "checkpoints", "results", "plots", "logs"):
        (root / leaf).mkdir(parents=True, exist_ok=True)
    snapshot_code(root / "code_snapshot")
    with (root / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(settings, handle, sort_keys=False)

    set_seed(int(config["project"]["random_seed"]) + ds * 100 + candidate)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    array = load_training_array(ds)
    train_ids, val_ids = split_ids(int(config["data"]["split_seed"]))
    mean, std = train_statistics(array, train_ids)
    model = build_model(settings).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(settings.get("phase_a_lr", settings["phases"][0]["lr"])), weight_decay=0.0)
    rng = np.random.default_rng(int(config["project"]["random_seed"]) + ds * 10_000 + candidate)
    history: list[dict] = []
    best_validation = float("inf")
    best_rollout: dict | None = None
    latest_rollout: dict | None = None
    patience = 0
    final_phase_start = int(settings["phases"][-1]["start"])
    validation_ids, validation_starts = fixed_specs(
        val_ids,
        array.shape[1],
        int(settings["seq_len"]),
        max(int(phase["horizon"]) for phase in settings["phases"]),
    )
    started = time.time()

    for epoch in range(1, int(settings["max_epochs"]) + 1):
        phase = phase_for_epoch(settings, epoch)
        horizon = int(phase["horizon"])
        set_optimizer_lr(optimizer, float(phase["lr"]))
        train_losses = []
        for _ in range(int(settings["batches_per_epoch"])):
            trajectory_ids, starts = random_specs(
                rng,
                train_ids,
                int(settings["batch_size"]),
                array.shape[1],
                int(settings["seq_len"]),
                horizon,
            )
            values = normalized_batch(array, trajectory_ids, starts, int(settings["seq_len"]) + horizon, mean, std)
            tensor = torch.from_numpy(values).to(device=device, dtype=torch.float32)
            if bool(phase.get("phase_a", False)):
                loss = phase_a_batch(
                    model,
                    optimizer,
                    tensor,
                    float(settings["phase_a_delta_weight"]),
                    float(settings["gradient_clip"]),
                )
            else:
                loss = closed_loop_batch(
                    model,
                    optimizer,
                    tensor,
                    int(settings["seq_len"]),
                    horizon,
                    min(int(settings["tbptt_segment"]), horizon),
                    float(settings["delta_consistency_weight"]),
                    float(settings["gradient_clip"]),
                )
            train_losses.append(loss)

        val_horizon = min(horizon, int(config["evaluation"]["validation_rollout_steps"]))
        val_values = normalized_batch(
            array,
            validation_ids,
            validation_starts,
            int(settings["seq_len"]) + val_horizon,
            mean,
            std,
        )
        val_tensor = torch.from_numpy(val_values).to(device=device, dtype=torch.float32)
        val_loss = validation_loss(model, val_tensor, int(settings["seq_len"]), val_horizon)
        train_loss = float(np.mean(train_losses))
        rollout_improved = False
        if epoch % int(config["evaluation"]["validation_every_epochs"]) == 0 or epoch == int(settings["max_epochs"]):
            latest_rollout = rollout_validation(
                model,
                ds,
                int(settings["seq_len"]),
                mean,
                std,
                val_ids,
                config["evaluation"],
                device,
                val_loss,
            )
            rollout_improved = is_better(latest_rollout, best_rollout)
            if rollout_improved:
                best_rollout = copy.deepcopy(latest_rollout)

        record = {
            "epoch": epoch,
            "horizon": horizon,
            "tbptt_segment": min(int(settings["tbptt_segment"]), horizon),
            "learning_rate": float(phase["lr"]),
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "rollout": copy.deepcopy(latest_rollout) if epoch % int(config["evaluation"]["validation_every_epochs"]) == 0 else None,
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        payload = checkpoint_payload(
            model,
            optimizer,
            settings,
            config,
            epoch,
            history,
            train_ids,
            val_ids,
            mean,
            std,
            min(best_validation, val_loss),
            latest_rollout,
            best_rollout,
        )
        if val_loss < best_validation:
            best_validation = val_loss
            atomic_torch_save(root / "checkpoints" / "best_val.pt", payload)
        if rollout_improved:
            atomic_torch_save(root / "checkpoints" / "best_rollout.pt", payload)
        atomic_torch_save(root / "checkpoints" / "latest.pt", payload)
        atomic_json(root / "logs" / "progress.json", record)
        write_live_status(ds, candidate, epoch, settings, phase, train_loss, val_loss, latest_rollout)
        rollout_text = ""
        if record["rollout"]:
            rollout_text = (
                f" | H1={record['rollout']['H1.0']} | Hsqrt2={record['rollout']['Hsqrt2']}"
                f" | late_std={record['rollout']['late_std_ratio']:.4f}"
                f" | entropy={record['rollout']['spectral_entropy_ratio']:.4f}"
                f" | invalid={record['rollout']['invalid_dynamics_flag']}"
            )
        log_line = (
            f"ds={ds} candidate={candidate} epoch={epoch}/{settings['max_epochs']} H={horizon} "
            f"train={train_loss:.7e} val={val_loss:.7e} lr={float(phase['lr']):.3e}{rollout_text}"
        )
        print(log_line, flush=True)
        with (root / "logs" / "training.log").open("a", encoding="utf-8", newline="\n") as log_handle:
            log_handle.write(log_line + "\n")
            log_handle.flush()

        if epoch >= final_phase_start + int(config["training"]["final_phase_min_epochs"]) - 1:
            patience = 0 if rollout_improved else patience + int(config["evaluation"]["validation_every_epochs"]) if record["rollout"] else patience
            if patience >= int(config["training"]["early_stopping_patience"]):
                stop_line = f"Adaptive early stop at epoch {epoch}"
                print(stop_line, flush=True)
                with (root / "logs" / "training.log").open("a", encoding="utf-8", newline="\n") as log_handle:
                    log_handle.write(stop_line + "\n")
                break

    with (root / "logs" / "training_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "horizon", "tbptt_segment", "learning_rate", "train_loss", "validation_loss", "elapsed_seconds"])
        writer.writeheader()
        for row in history:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    atomic_json(root / "logs" / "training_history.json", history)
    del array, model, optimizer
    torch.cuda.empty_cache()
    return root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=int, choices=(1, 2, 4, 8, 32), required=True)
    parser.add_argument("--candidate", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    train_candidate(args.ds, args.candidate)


if __name__ == "__main__":
    main()
