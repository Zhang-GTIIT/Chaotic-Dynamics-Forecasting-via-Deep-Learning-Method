"""Assemble the checkpoint-free desktop delivery tree and reproducibility manifest."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS = ("easy_attention_sgu", "self_attention", "lstm")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    with (PROJECT_ROOT / "code" / "config.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    delivery = PROJECT_ROOT / "delivery"
    if delivery.exists():
        shutil.rmtree(delivery)
    for relative in ("code_snapshot", "configs", "results", "logs"):
        (delivery / relative).mkdir(parents=True, exist_ok=True)
    for source in sorted((PROJECT_ROOT / "code").iterdir()):
        if source.is_file() and source.suffix in {".py", ".yaml", ".sh"}:
            shutil.copy2(source, delivery / "code_snapshot" / source.name)
    shutil.copy2(PROJECT_ROOT / "code" / "config.yaml", delivery / "configs" / "config.yaml")
    shutil.copytree(
        PROJECT_ROOT / "results" / "final",
        delivery / "results",
        dirs_exist_ok=True,
    )
    for source in sorted((PROJECT_ROOT / "server_logs").glob("*")):
        if source.is_file():
            shutil.copy2(source, delivery / "logs" / source.name)

    data_validation_path = PROJECT_ROOT / "results" / "data" / "data_validation.json"
    with data_validation_path.open("r", encoding="utf-8") as handle:
        data_validation = json.load(handle)
    summaries = {}
    for model in MODELS:
        with (PROJECT_ROOT / "results" / "final" / model / "summary.json").open(
            "r", encoding="utf-8"
        ) as handle:
            summaries[model] = json.load(handle)
    split_manifest = PROJECT_ROOT / "results" / "data" / (
        f"split_manifest_t{config['data_generation']['sim_time']}_"
        f"seed{config['training']['random_seed']}.json"
    )
    with split_manifest.open("r", encoding="utf-8") as handle:
        split = json.load(handle)
    data_files = {}
    for key in ("train_data_path", "test_data_path"):
        path = (PROJECT_ROOT / config["data_generation"][key]).resolve()
        with h5py.File(path, "r") as handle:
            shape = list(handle["data"].shape)
            dtype = str(handle["data"].dtype)
        data_files[key] = {
            "path_on_server": str(path),
            "shape": shape,
            "dtype": dtype,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    manifest = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": config["experiment"],
        "data_generation": config["data_generation"],
        "data_files": data_files,
        "data_validation": data_validation,
        "trajectory_split": split,
        "training": config["training"],
        "model": config["model"],
        "easy_attention_sgu": config["easy_attention_sgu"],
        "lstm": config["lstm"],
        "attention_training": config["attention_training"],
        "lstm_training": config["lstm_training"],
        "loss": config["loss"],
        "rollout_validation": config["rollout_validation"],
        "prediction": config["prediction"],
        "summaries": summaries,
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "code_sha256": {
            path.name: sha256(path)
            for path in sorted((PROJECT_ROOT / "code").glob("*"))
            if path.is_file()
        },
        "raw_data_in_delivery": False,
        "full_checkpoints_in_delivery": False,
    }
    temporary = delivery / "run_manifest.json.tmp"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, delivery / "run_manifest.json")

    system = config["experiment"]["display_name"]
    data = config["data_generation"]
    lines = [
        f"# {system}: three-model chaotic forecasting benchmark",
        "",
        "This folder is the compact, checkpoint-free result package produced on the remote GPU server.",
        "Raw trajectory HDF5 files and full checkpoints remain on the server and are intentionally omitted.",
        "",
        "## Dynamical system and data",
        "",
        f"- System: {config['experiment']['system_label']}",
        f"- Saved time step: {data['time_step']}",
        f"- Saved steps per trajectory: {data['saved_steps']}",
        f"- Train/validation trajectories: {data['trainval_trajectories']} "
        f"({len(split['train_trajectory_ids'])} train, {len(split['validation_trajectory_ids'])} validation)",
        f"- Independent test trajectories: {data['independent_test_trajectories']}",
        f"- Random split seed: {config['training']['random_seed']}",
        "- Normalization: componentwise mean and standard deviation computed from train trajectories only.",
        "",
        "## Models and rollout",
        "",
        "- Easy Attention + SGU: temporal encoder followed by a state-conditioned recurrent increment decoder; 1/2/3-block closed-loop curriculum and 100-step block autoregression.",
        "- Self Attention: self-attention encoder with the same state-conditioned recurrent increment decoder; 1/2/3-block closed-loop curriculum and 100-step block autoregression.",
        "- LSTM: StatefulLSTM with one-step state-increment prediction and closed-loop curriculum; stateful one-step rollout.",
        f"- Shared window: seq_len={config['training']['seq_len']}, pred_len={config['training']['pred_len']}.",
        f"- Loss: uniform pointwise MSE plus lightweight variance ({config['loss']['variance_weight']}) and mean ({config['loss']['mean_weight']}) preservation terms.",
        "- Checkpoint selection continues to use pointwise validation MSE and collapse-aware rollout metrics.",
        "- Threshold crossings are one-based and remain null when not reached.",
        "",
        "## Layout",
        "",
        "- code_snapshot/: exact source/config snapshot used for this run.",
        "- configs/: top-level resolved experiment configuration.",
        "- results/<model>/: metrics, compact representative arrays, checkpoint metadata, logs, and figures.",
        "- results/comparison/: three-model comparison figures and summary.csv.",
        "- logs/: full orchestration, training, and prediction logs.",
        "- run_manifest.json: seeds, data hashes/shapes, software versions, code hashes, configuration, and summaries.",
        "",
        "## Main result summary",
        "",
    ]
    for model in MODELS:
        summary = summaries[model]
        crossing = summary["first_crossing_sqrt2"]
        crossing_text = str(crossing["crossing_step"]) if crossing["reached"] else "not reached"
        lines.append(
            f"- {model}: best validation MSE={summary['validation_mse']:.6g}; "
            f"sqrt(2) crossing={crossing_text}; final NRMSE={summary['final_nrmse']:.6g}; "
            f"collapse={summary['collapse_flag']}."
        )
    (delivery / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Delivery assembled: {delivery}")


if __name__ == "__main__":
    main()
