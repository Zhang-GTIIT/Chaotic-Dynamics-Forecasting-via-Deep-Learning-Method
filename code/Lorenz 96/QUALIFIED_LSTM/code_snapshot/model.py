"""Three comparable forecasting models adapted for smooth chaotic dynamics."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_len, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] > self.encoding.shape[1]:
            raise ValueError(
                f"Expected [batch, sequence, d_model] with sequence <= "
                f"{self.encoding.shape[1]}, found {tuple(x.shape)}"
            )
        return x + self.encoding[:, : x.shape[1]].to(dtype=x.dtype)


class EasyAttentionSGUEncoderLayer(nn.Module):
    """Dense Easy Attention temporal mixing followed by an SGU gate and Post-LN."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        seq_len: int,
        dim_feedforward: int,
        dropout: float,
        alpha_noise_std: float,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.seq_len = int(seq_len)
        self.head_dim = self.d_model // self.nhead

        self.W_V = nn.Linear(d_model, d_model)
        self.W_U = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        identity = torch.eye(seq_len, dtype=torch.float32).unsqueeze(0).repeat(nhead, 1, 1)
        noise = float(alpha_noise_std) * torch.randn(nhead, seq_len, seq_len)
        self.alpha = nn.Parameter(identity + noise)

        self.attention_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.ffn_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape
        if sequence_length != self.seq_len:
            raise ValueError(
                f"Easy Attention alpha is configured for seq_len={self.seq_len}, "
                f"but received sequence length {sequence_length}"
            )

        values = self.W_V(x)
        values = values.view(batch_size, sequence_length, self.nhead, self.head_dim)
        values = values.permute(0, 2, 1, 3)
        temporal_mix = torch.matmul(self.alpha.unsqueeze(0), values)
        temporal_mix = temporal_mix.permute(0, 2, 1, 3).contiguous()
        temporal_mix = temporal_mix.view(batch_size, sequence_length, self.d_model)

        gate = F.gelu(self.W_U(x))
        gated = temporal_mix * gate
        attention_output = self.W_O(gated)
        x = self.norm1(x + self.attention_dropout(attention_output))
        return self.norm2(x + self.ffn_dropout(self.ffn(x)))


class SelfAttentionEncoderLayer(nn.Module):
    """Standard unmasked multi-head scaled dot-product self attention with Post-LN."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.head_dim = self.d_model // self.nhead
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.ffn_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def _split_heads(self, value: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = value.shape
        value = value.view(batch_size, sequence_length, self.nhead, self.head_dim)
        return value.permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        queries = self._split_heads(self.W_Q(x))
        keys = self._split_heads(self.W_K(x))
        values = self._split_heads(self.W_V(x))
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = torch.softmax(scores, dim=-1)
        mixed = torch.matmul(attention, values)
        mixed = mixed.permute(0, 2, 1, 3).contiguous().view(x.shape[0], x.shape[1], self.d_model)
        attention_output = self.W_O(mixed)
        x = self.norm1(x + self.attention_dropout(attention_output))
        return self.norm2(x + self.ffn_dropout(self.ffn(x)))


class _ChaoticAttentionBase(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        seq_len: int,
        pred_len: int,
        encoder_layers: list[nn.Module],
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.input_projection = nn.Linear(self.input_dim, self.d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(self.d_model, self.seq_len)
        self.encoder_layers = nn.ModuleList(encoder_layers)
        self.horizon_embedding = nn.Parameter(
            torch.empty(self.pred_len, self.d_model, dtype=torch.float32)
        )
        self.state_projection = nn.Linear(self.input_dim, self.d_model)
        self.decoder_cell = nn.GRUCell(self.d_model, self.d_model)
        self.output_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.input_dim),
        )
        self.apply(self._initialize_standard_weights)
        for name, parameter in self.decoder_cell.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(parameter)
            elif "bias" in name:
                nn.init.zeros_(parameter)
        nn.init.normal_(self.horizon_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.output_head[-1].weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.output_head[-1].bias)

    @staticmethod
    def _initialize_standard_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input [B,seq_len,D], found {tuple(x.shape)}")
        if x.shape[1] != self.seq_len or x.shape[2] != self.input_dim:
            raise ValueError(
                f"Expected input [B,{self.seq_len},{self.input_dim}], found {tuple(x.shape)}"
            )
        hidden = self.positional_encoding(self.input_projection(x))
        for layer in self.encoder_layers:
            hidden = layer(hidden)
        decoder_hidden = hidden[:, -1, :]
        current_state = x[:, -1, :]
        predictions = []
        for step in range(self.pred_len):
            decoder_input = (
                self.state_projection(current_state)
                + self.horizon_embedding[step].unsqueeze(0)
            )
            decoder_hidden = self.decoder_cell(decoder_input, decoder_hidden)
            current_state = current_state + self.output_head(decoder_hidden)
            predictions.append(current_state)
        return torch.stack(predictions, dim=1)


class EasyAttentionSGUIncrementForecaster(_ChaoticAttentionBase):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.001,
        seq_len: int = 128,
        pred_len: int = 100,
        alpha_noise_std: float = 0.01,
    ) -> None:
        layers = [
            EasyAttentionSGUEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                seq_len=seq_len,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                alpha_noise_std=alpha_noise_std,
            )
            for _ in range(num_layers)
        ]
        super().__init__(input_dim, d_model, seq_len, pred_len, layers)


class SelfAttentionIncrementForecaster(_ChaoticAttentionBase):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.001,
        seq_len: int = 128,
        pred_len: int = 100,
    ) -> None:
        layers = [
            SelfAttentionEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ]
        super().__init__(input_dim, d_model, seq_len, pred_len, layers)


class StatefulLSTM(nn.Module):
    """Three-layer fp32 LSTM with optional input dropout and hidden-state zoneout."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        output_size: int | None = None,
        dropout_keep_prob: float = 0.995,
        zoneout_keep_prob: float = 0.98,
        residual_connections: bool = False,
        state_increment_prediction: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.output_size = int(output_size if output_size is not None else input_size)
        self.dropout_keep_prob = float(dropout_keep_prob)
        self.zoneout_keep_prob = float(zoneout_keep_prob)
        self.residual_connections = bool(residual_connections)
        self.state_increment_prediction = bool(state_increment_prediction)
        if self.state_increment_prediction and self.output_size != self.input_size:
            raise ValueError(
                "State-increment prediction requires output_size == input_size"
            )
        self.lstm_layers = nn.ModuleList(
            [nn.LSTM(self.input_size, self.hidden_size, batch_first=True)]
            + [
                nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
                for _ in range(1, self.num_layers)
            ]
        )
        self.output_projection = nn.Linear(self.hidden_size, self.output_size)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for name, parameter in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(parameter)
            elif "bias" in name:
                nn.init.zeros_(parameter)
        if self.state_increment_prediction:
            nn.init.normal_(self.output_projection.weight, mean=0.0, std=1.0e-3)
            nn.init.zeros_(self.output_projection.bias)

    def zero_state(self, batch_size: int, device: torch.device) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [
            (
                torch.zeros(1, batch_size, self.hidden_size, device=device, dtype=torch.float32),
                torch.zeros(1, batch_size, self.hidden_size, device=device, dtype=torch.float32),
            )
            for _ in range(self.num_layers)
        ]

    def _zoneout(
        self,
        new_h: torch.Tensor,
        new_c: torch.Tensor,
        old_h: torch.Tensor,
        old_c: torch.Tensor,
        training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if training and self.zoneout_keep_prob < 1.0:
            mask_h = torch.bernoulli(torch.full_like(new_h, self.zoneout_keep_prob))
            mask_c = torch.bernoulli(torch.full_like(new_c, self.zoneout_keep_prob))
            new_h = mask_h * new_h + (1.0 - mask_h) * old_h
            new_c = mask_c * new_c + (1.0 - mask_c) * old_c
        return new_h, new_c

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        training: bool | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        if x.ndim != 3 or x.shape[2] != self.input_size:
            raise ValueError(
                f"Expected LSTM input [B,T,{self.input_size}], found {tuple(x.shape)}"
            )
        x = x.to(dtype=torch.float32)
        input_reference = x
        use_training_behavior = self.training if training is None else bool(training)
        if hidden_state is None:
            hidden_state = self.zero_state(x.shape[0], x.device)
        if len(hidden_state) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} hidden-state pairs")
        if use_training_behavior and self.dropout_keep_prob < 1.0:
            x = F.dropout(x, p=1.0 - self.dropout_keep_prob, training=True)

        layer_input = x
        next_states: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_index, layer in enumerate(self.lstm_layers):
            old_h, old_c = hidden_state[layer_index]
            layer_output, (new_h, new_c) = layer(layer_input, (old_h, old_c))
            next_states.append(
                self._zoneout(new_h, new_c, old_h, old_c, use_training_behavior)
            )
            if self.residual_connections and layer_index > 0:
                layer_input = layer_output + layer_input
            else:
                layer_input = layer_output
        output = self.output_projection(layer_input)
        if self.state_increment_prediction:
            output = input_reference + output
        return output, next_states


def build_model(model_type: str, input_dim: int, config: dict, pred_len: int) -> nn.Module:
    if config["model"].get("prediction_parameterization") != "state_increment":
        raise ValueError("model.prediction_parameterization must be 'state_increment'")
    if model_type == "easy_attention_sgu":
        if config["model"].get("decoder_head") != "state_conditioned_recurrent_increment":
            raise ValueError(
                "model.decoder_head must be 'state_conditioned_recurrent_increment'"
            )
        return EasyAttentionSGUIncrementForecaster(
            input_dim=input_dim,
            d_model=int(config["model"]["d_model"]),
            nhead=int(config["model"]["nhead"]),
            num_layers=int(config["model"]["num_layers"]),
            dim_feedforward=int(config["model"]["dim_feedforward"]),
            dropout=float(config["model"]["dropout"]),
            seq_len=int(config["training"]["seq_len"]),
            pred_len=int(pred_len),
            alpha_noise_std=float(config["easy_attention_sgu"]["alpha_noise_std"]),
        )
    if model_type == "self_attention":
        if config["model"].get("decoder_head") != "state_conditioned_recurrent_increment":
            raise ValueError(
                "model.decoder_head must be 'state_conditioned_recurrent_increment'"
            )
        return SelfAttentionIncrementForecaster(
            input_dim=input_dim,
            d_model=int(config["model"]["d_model"]),
            nhead=int(config["model"]["nhead"]),
            num_layers=int(config["model"]["num_layers"]),
            dim_feedforward=int(config["model"]["dim_feedforward"]),
            dropout=float(config["model"]["dropout"]),
            seq_len=int(config["training"]["seq_len"]),
            pred_len=int(pred_len),
        )
    if model_type == "lstm":
        if config["lstm"]["dtype"] != "float32":
            raise ValueError("The formal LSTM configuration requires dtype='float32'")
        return StatefulLSTM(
            input_size=input_dim,
            hidden_size=int(config["lstm"]["hidden_size"]),
            num_layers=int(config["lstm"]["num_layers"]),
            output_size=input_dim,
            dropout_keep_prob=float(config["lstm"]["dropout_keep_prob"]),
            zoneout_keep_prob=float(config["lstm"]["zoneout_keep_prob"]),
            residual_connections=bool(config["lstm"]["residual_connections"]),
            state_increment_prediction=bool(
                config["lstm"]["state_increment_prediction"]
            ),
        )
    raise ValueError(
        "model_type must be one of: easy_attention_sgu, self_attention, lstm"
    )


def model_display_name(model_type: str) -> str:
    names = {
        "easy_attention_sgu": "Easy Attention + SGU",
        "self_attention": "Self Attention",
        "lstm": "LSTM",
    }
    if model_type not in names:
        raise ValueError(f"Unknown model_type: {model_type}")
    return names[model_type]
