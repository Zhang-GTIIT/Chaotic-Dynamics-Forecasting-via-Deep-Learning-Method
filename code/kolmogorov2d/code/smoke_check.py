"""Small shape/mechanics check for the system-adapted models; this does not train."""

from pathlib import Path

import torch
import yaml

from model import build_model
from rollout import rollout_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    with (PROJECT_ROOT / "code" / "config.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dimension = int(config["data_generation"]["spatial_resolution"])
    seq_len = int(config["training"]["seq_len"])
    pred_len = int(config["training"]["pred_len"])
    initial = torch.zeros((1, seq_len, dimension), device=device)
    for model_type in ("easy_attention_sgu", "self_attention", "lstm"):
        model = build_model(model_type, dimension, config, pred_len).to(device).eval()
        with torch.no_grad():
            if model_type == "lstm":
                output, _ = model(initial)
                if output.shape != initial.shape:
                    raise RuntimeError(f"LSTM output shape mismatch: {output.shape}")
                rollout = rollout_model(model_type, model, initial, 7, seq_len, pred_len)
                expected = (1, 7, dimension)
            else:
                output = model(initial)
                if output.shape != (1, pred_len, dimension):
                    raise RuntimeError(f"Attention output shape mismatch: {output.shape}")
                rollout = rollout_model(model_type, model, initial, pred_len + 5, seq_len, pred_len)
                expected = (1, pred_len + 5, dimension)
            if tuple(rollout.shape) != expected:
                raise RuntimeError(f"{model_type} rollout shape mismatch: {rollout.shape}")
        del model, output, rollout
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"{model_type}: OK", flush=True)


if __name__ == "__main__":
    main()
