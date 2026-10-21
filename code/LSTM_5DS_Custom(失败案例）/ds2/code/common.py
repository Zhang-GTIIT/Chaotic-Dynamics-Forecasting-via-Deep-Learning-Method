from __future__ import annotations

import copy
import json
import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "code" / "config.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ds_dimension(ds: int) -> int:
    if ds not in (1, 2, 4, 8, 32):
        raise ValueError(f"Unsupported ds={ds}")
    return 64 // ds


def deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key == "inherit":
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def candidate_config(config: dict, ds: int, candidate: int) -> dict:
    entry = config["ds_configs"][int(ds)]
    result = copy.deepcopy(entry["primary"])
    if int(candidate) == 2:
        result = deep_merge(result, entry["fallback"])
        if "final_horizon" in result:
            result["phases"][-1]["horizon"] = int(result.pop("final_horizon"))
        if int(ds) == 32 and "max_lr" in result:
            max_lr = float(result.pop("max_lr"))
            result["phase_a_lr"] = max_lr
            for phase in result["phases"]:
                phase["lr"] = min(float(phase["lr"]), max_lr)
            shift = int(result["phase_a_epochs"]) - int(entry["primary"]["phase_a_epochs"])
            for phase in result["phases"]:
                phase["start"] = int(phase["start"]) + shift
                phase["end"] = int(phase["end"]) + shift
    elif int(candidate) != 1:
        raise ValueError("candidate must be 1 or 2")
    result["ds"] = int(ds)
    result["D"] = ds_dimension(ds)
    result["candidate"] = int(candidate)
    result["max_epochs"] = int(result["phases"][-1]["end"])
    return result


def candidate_root(ds: int, candidate: int) -> Path:
    return PROJECT_ROOT / f"ds{ds}" / f"candidate_{candidate:02d}"


def snapshot_code(destination: Path) -> None:
    source = PROJECT_ROOT / "code"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def gpu_status() -> dict:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).strip().splitlines()[0]
        util, used, total = [int(part.strip()) for part in output.split(",")]
        return {"gpu_utilization_percent": util, "gpu_memory_used_mib": used, "gpu_memory_total_mib": total}
    except Exception as error:
        return {"gpu_status_error": str(error)}


def phase_for_epoch(settings: dict, epoch: int) -> dict:
    if int(epoch) <= int(settings.get("phase_a_epochs", 0)):
        return {"phase_a": True, "horizon": 1, "lr": float(settings["phase_a_lr"]), "start": 1, "end": int(settings["phase_a_epochs"])}
    for phase in settings["phases"]:
        if int(phase["start"]) <= int(epoch) <= int(phase["end"]):
            result = copy.deepcopy(phase)
            result["phase_a"] = False
            if "cosine_min_lr" in result:
                start, end = int(result["start"]), int(result["end"])
                progress = 0.0 if end == start else (int(epoch) - start) / (end - start)
                low, high = float(result["cosine_min_lr"]), float(result["lr"])
                result["lr"] = low + 0.5 * (high - low) * (1.0 + np.cos(np.pi * progress))
            return result
    raise ValueError(f"No training phase configured for epoch {epoch}")

