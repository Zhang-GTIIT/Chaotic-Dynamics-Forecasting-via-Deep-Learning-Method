"""Command-line training entry point for the three-model chaos experiment."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

from data_loader import expected_data_path, inspect_h5_data, validate_pred_len
from train_lstm import train_lstm_model
from train_transformer import train_attention_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "code" / "config.yaml"
MODEL_TYPES = ("easy_attention_sgu", "self_attention", "lstm")


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def check_data(config: dict, requested_ds: int | None, requested_sim_time: int | None) -> None:
    ds_values = (
        [int(requested_ds)]
        if requested_ds is not None
        else [int(value) for value in config["data_generation"]["downsampling_factors"]]
    )
    sim_times = (
        [int(requested_sim_time)]
        if requested_sim_time is not None
        else [int(value) for value in config["data_generation"]["simulation_times"]]
    )
    spatial_resolution = int(config["data_generation"]["spatial_resolution"])
    for sim_time in sim_times:
        for ds in ds_values:
            path = expected_data_path(config, ds, sim_time)
            metadata = inspect_h5_data(path, spatial_resolution, ds)
            print(
                f"OK: {metadata['path']} | shape={metadata['shape']} | "
                f"dtype={metadata['dtype']}"
            )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Three-model chaos-system training")
    parser.add_argument("--model", choices=MODEL_TYPES, default=None)
    parser.add_argument("--ds", type=int, choices=(1, 2, 4, 8, 16, 32), default=None)
    parser.add_argument("--sim-time", type=int, default=None)
    parser.add_argument("--pred-len", type=int, default=None)
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=(
            "Override data_generation.save_dir without editing config.yaml. "
            "Useful when moving between Windows and Linux servers."
        ),
    )
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Only validate configured HDF5 paths and shapes; never train",
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true")
    resume_group.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config = copy.deepcopy(load_config())
    if args.data_dir is not None:
        config["data_generation"]["save_dir"] = str(
            Path(args.data_dir).expanduser().resolve()
        )
    if args.check_data:
        check_data(config, args.ds, args.sim_time)
        return

    model_type = args.model or str(config["training"]["model_type"])
    ds = args.ds or int(config["data_generation"]["downsampling_factors"][0])
    sim_time = args.sim_time or int(config["data_generation"]["sim_time"])
    pred_len = args.pred_len or int(config["training"]["pred_len"])
    validate_pred_len(config, pred_len)
    if model_type == "lstm":
        train_lstm_model(config, ds, sim_time, pred_len, resume=args.resume)
    else:
        train_attention_model(
            config, model_type, ds, sim_time, pred_len, resume=args.resume
        )


if __name__ == "__main__":
    main()
