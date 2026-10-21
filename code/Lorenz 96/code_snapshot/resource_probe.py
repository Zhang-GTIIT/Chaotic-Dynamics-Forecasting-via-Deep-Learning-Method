"""Probe configured full-horizon GPU memory and reduce only batch size if required."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import yaml

from model import build_model
from train_lstm import closed_loop_prediction
from train_transformer import block_closed_loop_prediction


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "code" / "config.yaml"


def attempt(
    model_type: str,
    batch_size: int,
    dimension: int,
    config: dict,
    num_blocks: int = 1,
) -> tuple[bool, str]:
    try:
        model = build_model(model_type, dimension, config, 100).cuda().train()
        history = torch.zeros((batch_size, 128, dimension), device="cuda")
        if model_type == "lstm":
            prediction = closed_loop_prediction(model, history, 100, training=True)
        else:
            prediction = block_closed_loop_prediction(
                model, history, 128, 100, num_blocks
            )
        prediction.square().mean().backward()
        maximum = int(torch.cuda.max_memory_allocated())
        del prediction, history, model
        torch.cuda.empty_cache()
        return True, str(maximum)
    except RuntimeError as error:
        message = str(error)
        if "out of memory" not in message.lower():
            raise
        torch.cuda.empty_cache()
        return False, message


def find_batch(
    model_types: tuple[str, ...],
    configured: int,
    dimension: int,
    config: dict,
    num_blocks: int = 1,
) -> tuple[int, dict]:
    candidate = int(configured)
    attempts = {}
    while candidate >= 1:
        successful = True
        for model_type in model_types:
            torch.cuda.reset_peak_memory_stats()
            ok, detail = attempt(
                model_type, candidate, dimension, config, num_blocks
            )
            attempts[f"{model_type}_blocks_{num_blocks}_batch_{candidate}"] = {
                "ok": ok,
                "detail": detail,
            }
            if not ok:
                successful = False
                break
        if successful:
            return candidate, attempts
        candidate //= 2
    raise RuntimeError(f"No viable CUDA batch size for {model_types}")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal remote runs")
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    dimension = int(config["data_generation"]["spatial_resolution"])
    lstm_configured = int(config["lstm_training"]["batch_size"])
    mapping = config["attention_training"]["batch_size_by_blocks"]
    attention_selected = {}
    attention_attempts = {}
    for blocks in (1, 2, 3):
        configured = int(mapping.get(blocks, mapping.get(str(blocks))))
        selected, attempts = find_batch(
            ("easy_attention_sgu", "self_attention"),
            configured,
            dimension,
            config,
            num_blocks=blocks,
        )
        attention_selected[blocks] = selected
        attention_attempts.update(attempts)
    lstm_selected, lstm_attempts = find_batch(
        ("lstm",), lstm_configured, dimension, config
    )
    config["attention_training"]["batch_size"] = attention_selected[1]
    config["attention_training"]["batch_size_by_blocks"] = attention_selected
    config["lstm_training"]["batch_size"] = lstm_selected
    temporary = CONFIG_PATH.with_suffix(".yaml.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, CONFIG_PATH)
    report = {
        "gpu": torch.cuda.get_device_name(0),
        "dimension": dimension,
        "attention_selected_batch_size_by_blocks": attention_selected,
        "lstm_configured_batch_size": lstm_configured,
        "lstm_selected_batch_size": lstm_selected,
        "adjustment_required": (
            any(
                int(mapping.get(blocks, mapping.get(str(blocks)))) != selected
                for blocks, selected in attention_selected.items()
            )
            or lstm_selected != lstm_configured
        ),
        "attempts": {**attention_attempts, **lstm_attempts},
    }
    path = PROJECT_ROOT / "results" / "data" / "resource_probe.json"
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
