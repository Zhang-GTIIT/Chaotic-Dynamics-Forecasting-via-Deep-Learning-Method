"""Normalized-space rollout implementations for attention and LSTM models."""

from __future__ import annotations

from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def block_autoregressive_rollout(
    model: torch.nn.Module,
    initial_window: torch.Tensor,
    total_steps: int,
    seq_len: int,
    pred_len: int,
) -> torch.Tensor:
    """Generate full prediction blocks and feed every generated state back."""
    if initial_window.ndim != 3:
        raise ValueError("initial_window must have shape [B,seq_len,D]")
    if initial_window.shape[1] != int(seq_len):
        raise ValueError(
            f"Expected initial window length {seq_len}, found {initial_window.shape[1]}"
        )
    if int(total_steps) <= 0:
        raise ValueError("total_steps must be positive")
    if int(pred_len) <= 0 or int(pred_len) > int(seq_len):
        raise ValueError("pred_len must be positive and no larger than seq_len")

    window = initial_window
    prediction_blocks: list[torch.Tensor] = []
    generated_steps = 0
    with torch.inference_mode():
        while generated_steps < int(total_steps):
            block = model(window)
            expected_shape = (window.shape[0], int(pred_len), window.shape[2])
            if tuple(block.shape) != expected_shape:
                raise RuntimeError(
                    f"Attention model returned {tuple(block.shape)}; expected {expected_shape}"
                )
            prediction_blocks.append(block)
            window = torch.cat([window, block], dim=1)[:, -int(seq_len) :, :]
            generated_steps += int(pred_len)

    predictions = torch.cat(prediction_blocks, dim=1)
    return predictions[:, : int(total_steps), :]


def stateful_lstm_rollout(
    model: torch.nn.Module,
    initial_window: torch.Tensor,
    total_steps: int,
) -> torch.Tensor:
    """Warm up on the full real history, then retain state during one-step feedback."""
    if initial_window.ndim != 3:
        raise ValueError("initial_window must have shape [B,seq_len,D]")
    if int(total_steps) <= 0:
        raise ValueError("total_steps must be positive")

    predictions: list[torch.Tensor] = []
    with torch.inference_mode():
        warmup_output, hidden_state = model(initial_window, None, training=False)
        next_step = warmup_output[:, -1:, :]
        predictions.append(next_step)
        for _ in range(1, int(total_steps)):
            output, hidden_state = model(next_step, hidden_state, training=False)
            next_step = output[:, -1:, :]
            predictions.append(next_step)
    return torch.cat(predictions, dim=1)


def rollout_model(
    model_type: str,
    model: torch.nn.Module,
    initial_window: torch.Tensor,
    total_steps: int,
    seq_len: int,
    pred_len: int,
) -> torch.Tensor:
    if model_type in {"easy_attention_sgu", "self_attention"}:
        return block_autoregressive_rollout(
            model=model,
            initial_window=initial_window,
            total_steps=total_steps,
            seq_len=seq_len,
            pred_len=pred_len,
        )
    if model_type == "lstm":
        return stateful_lstm_rollout(model, initial_window, total_steps)
    raise ValueError(f"Unknown model_type: {model_type}")
