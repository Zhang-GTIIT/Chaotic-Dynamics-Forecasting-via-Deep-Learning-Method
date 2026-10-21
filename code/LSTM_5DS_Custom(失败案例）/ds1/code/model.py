from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


HiddenState = list[tuple[torch.Tensor, torch.Tensor]]


class ResidualDeltaLSTM(nn.Module):
    """LSTM core with configurable derivative features and residual state prediction."""

    def __init__(
        self,
        state_dim: int,
        feature_order: int,
        encoder_dim: int,
        hidden_size: int,
        num_layers: int,
        head_dim: int,
        layer_norm_lstm: bool,
    ) -> None:
        super().__init__()
        if int(feature_order) not in (0, 1, 2):
            raise ValueError("feature_order must be 0, 1, or 2")
        self.state_dim = int(state_dim)
        self.feature_order = int(feature_order)
        self.encoder_dim = int(encoder_dim)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.layer_norm_lstm = bool(layer_norm_lstm)
        feature_dim = self.state_dim * (self.feature_order + 1)
        self.input_encoder = nn.Sequential(
            nn.Linear(feature_dim, self.encoder_dim),
            nn.GELU(),
            nn.LayerNorm(self.encoder_dim),
        )
        self.lstm_layers = nn.ModuleList()
        self.output_norms = nn.ModuleList()
        for layer_index in range(self.num_layers):
            input_size = self.encoder_dim if layer_index == 0 else self.hidden_size
            self.lstm_layers.append(nn.LSTM(input_size, self.hidden_size, num_layers=1, batch_first=True))
            self.output_norms.append(nn.LayerNorm(self.hidden_size) if self.layer_norm_lstm else nn.Identity())
        self.delta_head = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, int(head_dim)),
            nn.GELU(),
            nn.Linear(int(head_dim), self.state_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        for layer in self.lstm_layers:
            for name, parameter in layer.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(parameter)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(parameter)
                elif "bias" in name:
                    nn.init.zeros_(parameter)
                    with torch.no_grad():
                        parameter[self.hidden_size : 2 * self.hidden_size].fill_(1.0)

    def zero_state(self, batch_size: int, device: torch.device) -> HiddenState:
        return [
            (
                torch.zeros(1, batch_size, self.hidden_size, device=device),
                torch.zeros(1, batch_size, self.hidden_size, device=device),
            )
            for _ in range(self.num_layers)
        ]

    def sequence_features(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3 or sequence.shape[-1] != self.state_dim:
            raise ValueError(f"Expected [B,T,{self.state_dim}], found {tuple(sequence.shape)}")
        features = [sequence]
        if self.feature_order >= 1:
            delta = torch.zeros_like(sequence)
            delta[:, 1:] = sequence[:, 1:] - sequence[:, :-1]
            features.append(delta)
        if self.feature_order >= 2:
            delta2 = torch.zeros_like(sequence)
            delta2[:, 2:] = sequence[:, 2:] - 2.0 * sequence[:, 1:-1] + sequence[:, :-2]
            features.append(delta2)
        return torch.cat(features, dim=-1)

    def step_features(
        self,
        current: torch.Tensor,
        previous: torch.Tensor,
        previous2: torch.Tensor,
    ) -> torch.Tensor:
        features = [current]
        if self.feature_order >= 1:
            features.append(current - previous)
        if self.feature_order >= 2:
            features.append(current - 2.0 * previous + previous2)
        return torch.cat(features, dim=-1)

    def core(self, features: torch.Tensor, hidden: Optional[HiddenState] = None) -> tuple[torch.Tensor, HiddenState]:
        if hidden is None:
            hidden = self.zero_state(features.shape[0], features.device)
        encoded = self.input_encoder(features)
        layer_input = encoded
        next_hidden: HiddenState = []
        for index, (layer, norm) in enumerate(zip(self.lstm_layers, self.output_norms)):
            output, state = layer(layer_input, hidden[index])
            layer_input = norm(output)
            next_hidden.append(state)
        return layer_input, next_hidden

    def warmup(self, history: torch.Tensor) -> tuple[torch.Tensor, HiddenState]:
        hidden_sequence, states = self.core(self.sequence_features(history), None)
        delta = self.delta_head(hidden_sequence)
        return delta, states

    def predict_step(
        self,
        current: torch.Tensor,
        previous: torch.Tensor,
        previous2: torch.Tensor,
        hidden: HiddenState,
    ) -> tuple[torch.Tensor, torch.Tensor, HiddenState]:
        features = self.step_features(current, previous, previous2).unsqueeze(1)
        hidden_output, next_hidden = self.core(features, hidden)
        delta = self.delta_head(hidden_output[:, 0])
        return current + delta, delta, next_hidden


def build_model(settings: dict) -> ResidualDeltaLSTM:
    return ResidualDeltaLSTM(
        state_dim=int(settings["D"]),
        feature_order=int(settings["feature_order"]),
        encoder_dim=int(settings["encoder_dim"]),
        hidden_size=int(settings["hidden_size"]),
        num_layers=int(settings["num_layers"]),
        head_dim=int(settings["head_dim"]),
        layer_norm_lstm=bool(settings["layer_norm_lstm"]),
    )


def detach_hidden(hidden: HiddenState) -> HiddenState:
    return [(h.detach(), c.detach()) for h, c in hidden]


@torch.no_grad()
def autoregressive_rollout(model: ResidualDeltaLSTM, history: torch.Tensor, steps: int) -> torch.Tensor:
    model.eval()
    warmup_delta, hidden = model.warmup(history)
    previous2 = history[:, -3]
    previous = history[:, -2]
    current = history[:, -1]
    next_state = current + warmup_delta[:, -1]
    predictions = [next_state]
    previous2, previous, current = previous, current, next_state
    for _ in range(1, int(steps)):
        next_state, _, hidden = model.predict_step(current, previous, previous2, hidden)
        predictions.append(next_state)
        previous2, previous, current = previous, current, next_state
    return torch.stack(predictions, dim=1)
