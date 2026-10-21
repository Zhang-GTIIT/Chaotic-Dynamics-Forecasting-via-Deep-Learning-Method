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


class PeriodicSpatialEncoder(nn.Module):
    """Encode a periodic 2D field without flattening away local structure."""

    def __init__(self, grid_size: int, d_model: int) -> None:
        super().__init__()
        if int(grid_size) % 8 != 0:
            raise ValueError("PeriodicSpatialEncoder requires grid_size divisible by 8")
        self.grid_size = int(grid_size)
        base = self.grid_size // 8
        self.convolutions = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2, padding_mode="circular"),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, padding_mode="circular"),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, padding_mode="circular"),
            nn.GELU(),
        )
        self.projection = nn.Linear(64 * base * base, int(d_model))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        expected = self.grid_size * self.grid_size
        if states.shape[-1] != expected:
            raise ValueError(
                f"Expected flattened {self.grid_size}x{self.grid_size} field, "
                f"found last dimension {states.shape[-1]}"
            )
        leading = states.shape[:-1]
        fields = states.reshape(-1, 1, self.grid_size, self.grid_size)
        encoded = self.projection(self.convolutions(fields).flatten(1))
        return encoded.reshape(*leading, encoded.shape[-1])


class PeriodicSpatialDecoder(nn.Module):
    """Decode latent increments with periodic convolutions and smooth upsampling."""

    def __init__(self, d_model: int, grid_size: int) -> None:
        super().__init__()
        if int(grid_size) % 8 != 0:
            raise ValueError("PeriodicSpatialDecoder requires grid_size divisible by 8")
        self.grid_size = int(grid_size)
        self.base = self.grid_size // 8
        self.projection = nn.Linear(int(d_model), 64 * self.base * self.base)
        self.conv1 = nn.Conv2d(
            64, 32, kernel_size=3, padding=1, padding_mode="circular"
        )
        self.conv2 = nn.Conv2d(
            32, 16, kernel_size=3, padding=1, padding_mode="circular"
        )
        self.final_conv = nn.Conv2d(
            16, 1, kernel_size=3, padding=1, padding_mode="circular"
        )

    def initialize_small_output(self) -> None:
        nn.init.normal_(self.final_conv.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.final_conv.bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        leading = latent.shape[:-1]
        field = self.projection(latent.reshape(-1, latent.shape[-1]))
        field = field.reshape(-1, 64, self.base, self.base)
        field = F.interpolate(field, scale_factor=2, mode="bilinear", align_corners=False)
        field = F.gelu(self.conv1(field))
        field = F.interpolate(field, scale_factor=2, mode="bilinear", align_corners=False)
        field = F.gelu(self.conv2(field))
        field = F.interpolate(field, scale_factor=2, mode="bilinear", align_corners=False)
        field = self.final_conv(field)
        return field.reshape(*leading, self.grid_size * self.grid_size)


class PeriodicLocalDynamics(nn.Module):
    """Physics-featured full-resolution update path for periodic vorticity."""

    def __init__(self, grid_size: int) -> None:
        super().__init__()
        self.grid_size = int(grid_size)
        row_modes = torch.fft.fftfreq(self.grid_size, d=1.0 / self.grid_size)
        column_modes = torch.fft.rfftfreq(self.grid_size, d=1.0 / self.grid_size)
        k_row = row_modes[:, None]
        k_column = column_modes[None, :]
        k_squared = k_row.square() + k_column.square()
        self.register_buffer("k_row", k_row, persistent=True)
        self.register_buffer("k_column", k_column, persistent=True)
        self.register_buffer(
            "inverse_k_squared",
            torch.where(k_squared > 0.0, k_squared.reciprocal(), torch.zeros_like(k_squared)),
            persistent=True,
        )
        self.register_buffer("k_squared", k_squared, persistent=True)
        self.conv1 = nn.Conv2d(
            4, 48, kernel_size=5, padding=2, padding_mode="circular"
        )
        self.conv2 = nn.Conv2d(
            48, 48, kernel_size=3, padding=1, padding_mode="circular"
        )
        self.final_conv = nn.Conv2d(
            48, 1, kernel_size=3, padding=1, padding_mode="circular"
        )

    def initialize_small_output(self) -> None:
        nn.init.normal_(self.final_conv.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.final_conv.bias)

    @staticmethod
    def _unit_rms(feature: torch.Tensor) -> torch.Tensor:
        scale = feature.square().mean(dim=(-2, -1), keepdim=True).add(1.0e-8).sqrt()
        return feature / scale

    def forward(
        self,
        states: torch.Tensor,
        previous_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        leading = states.shape[:-1]
        field = states.reshape(-1, 1, self.grid_size, self.grid_size)
        if previous_states is None:
            recent_increment = torch.zeros_like(field)
        else:
            recent_increment = (
                states - previous_states
            ).reshape(-1, 1, self.grid_size, self.grid_size)

        omega_hat = torch.fft.rfft2(field[:, 0], norm="ortho")
        stream_hat = omega_hat * self.inverse_k_squared.to(dtype=omega_hat.dtype)
        velocity_column = torch.fft.irfft2(
            1j * self.k_column * stream_hat,
            s=(self.grid_size, self.grid_size),
            norm="ortho",
        )
        velocity_row = torch.fft.irfft2(
            -1j * self.k_row * stream_hat,
            s=(self.grid_size, self.grid_size),
            norm="ortho",
        )
        gradient_column = torch.fft.irfft2(
            1j * self.k_column * omega_hat,
            s=(self.grid_size, self.grid_size),
            norm="ortho",
        )
        gradient_row = torch.fft.irfft2(
            1j * self.k_row * omega_hat,
            s=(self.grid_size, self.grid_size),
            norm="ortho",
        )
        advection = -(
            velocity_column * gradient_column + velocity_row * gradient_row
        ).unsqueeze(1)
        laplacian = torch.fft.irfft2(
            -self.k_squared * omega_hat,
            s=(self.grid_size, self.grid_size),
            norm="ortho",
        ).unsqueeze(1)
        features = torch.cat(
            (
                field,
                self._unit_rms(recent_increment),
                self._unit_rms(advection),
                self._unit_rms(laplacian),
            ),
            dim=1,
        )
        features = F.gelu(self.conv1(features))
        features = F.gelu(self.conv2(features))
        update = self.final_conv(features)
        return update.reshape(*leading, self.grid_size * self.grid_size)


def dealias_periodic_increment(
    increment: torch.Tensor,
    grid_size: int,
    spectral_filter_fraction: float,
) -> torch.Tensor:
    """Project learned increments onto the solver's dealiased Fourier band."""
    fraction = float(spectral_filter_fraction)
    if fraction >= 1.0:
        return increment
    if not 0.0 < fraction <= 1.0:
        raise ValueError("spectral_filter_fraction must be in (0, 1]")
    grid = int(grid_size)
    if increment.shape[-1] != grid * grid:
        raise ValueError("Increment dimension does not match grid_size")
    leading = increment.shape[:-1]
    field = increment.reshape(-1, grid, grid)
    spectrum = torch.fft.rfft2(field, norm="ortho")
    ky = torch.fft.fftfreq(grid, device=field.device).abs()
    kx = torch.fft.rfftfreq(grid, device=field.device).abs()
    cutoff = 0.5 * fraction
    mask = (ky[:, None] <= cutoff) & (kx[None, :] <= cutoff)
    filtered = torch.fft.irfft2(
        spectrum * mask.to(dtype=spectrum.dtype),
        s=(grid, grid),
        norm="ortho",
    )
    return filtered.reshape(*leading, grid * grid)


def stable_periodic_field_update(
    current_state: torch.Tensor,
    raw_increment: torch.Tensor,
    grid_size: int,
    spectral_filter_fraction: float,
    max_normalized_step: float,
    rms_ratio_min: float,
    rms_ratio_max: float,
    field_rms_min: float,
    field_rms_max: float,
    rms_relaxation_strength: float = 0.0,
    rms_relaxation_target: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Dealias and bound a learned update using measured one-step field statistics."""
    limit = float(max_normalized_step)
    increment = limit * torch.tanh(raw_increment / limit)
    increment = dealias_periodic_increment(
        increment,
        grid_size=int(grid_size),
        spectral_filter_fraction=float(spectral_filter_fraction),
    )
    candidate = current_state + increment
    reference_rms = current_state.square().mean(dim=-1, keepdim=True).add(1.0e-8).sqrt()
    candidate_rms = candidate.square().mean(dim=-1, keepdim=True).add(1.0e-8).sqrt()
    raw_ratio = candidate_rms / reference_rms
    stable_ratio = raw_ratio.clamp(
        min=float(rms_ratio_min), max=float(rms_ratio_max)
    )
    candidate = candidate * (stable_ratio / raw_ratio)
    candidate_rms = candidate.square().mean(dim=-1, keepdim=True).add(1.0e-8).sqrt()
    relaxation_strength = float(rms_relaxation_strength)
    if relaxation_strength > 0.0 and rms_relaxation_target is not None:
        target_rms = torch.as_tensor(
            rms_relaxation_target, device=candidate.device, dtype=candidate.dtype
        ).clamp(min=float(field_rms_min), max=float(field_rms_max))
        candidate = candidate * torch.exp(
            -relaxation_strength
            * torch.log((candidate_rms + 1.0e-8) / (target_rms + 1.0e-8))
        )
        candidate_rms = (
            candidate.square().mean(dim=-1, keepdim=True).add(1.0e-8).sqrt()
        )
    bounded_rms = candidate_rms.clamp(
        min=float(field_rms_min), max=float(field_rms_max)
    )
    return candidate * (bounded_rms / candidate_rms)


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
        nhead: int,
        seq_len: int,
        pred_len: int,
        encoder_layers: list[nn.Module],
        dropout: float,
        spatial_grid_size: int | None = None,
        stability_config: dict | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.spatial_grid_size = (
            None if spatial_grid_size is None else int(spatial_grid_size)
        )
        self.stability_config = dict(stability_config or {})
        self.time_homogeneous_decoder = bool(
            self.stability_config.get("time_homogeneous_decoder", False)
        )
        if self.spatial_grid_size is not None:
            if self.input_dim != self.spatial_grid_size * self.spatial_grid_size:
                raise ValueError("input_dim does not match the configured spatial grid")
            self.input_projection = PeriodicSpatialEncoder(
                self.spatial_grid_size, self.d_model
            )
        else:
            self.input_projection = nn.Linear(self.input_dim, self.d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(self.d_model, self.seq_len)
        self.encoder_layers = nn.ModuleList(encoder_layers)
        self.horizon_embedding = nn.Parameter(
            torch.empty(self.pred_len, self.d_model, dtype=torch.float32)
        )
        if self.spatial_grid_size is not None:
            self.state_projection = PeriodicSpatialEncoder(
                self.spatial_grid_size, self.d_model
            )
        else:
            self.state_projection = nn.Linear(self.input_dim, self.d_model)
        self.decoder_attention = nn.MultiheadAttention(
            self.d_model,
            int(nhead),
            dropout=float(dropout),
            batch_first=True,
        )
        self.decoder_attention_norm = nn.LayerNorm(self.d_model)
        self.decoder_cell = nn.GRUCell(self.d_model, self.d_model)
        if self.spatial_grid_size is not None:
            self.output_head = PeriodicSpatialDecoder(
                self.d_model, self.spatial_grid_size
            )
            self.local_dynamics = PeriodicLocalDynamics(self.spatial_grid_size)
        else:
            self.output_head = nn.Sequential(
                nn.LayerNorm(self.d_model),
                nn.Linear(self.d_model, self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, self.input_dim),
            )
            self.local_dynamics = None
        self.apply(self._initialize_standard_weights)
        for name, parameter in self.decoder_cell.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(parameter)
            elif "bias" in name:
                nn.init.zeros_(parameter)
        nn.init.normal_(self.horizon_embedding, mean=0.0, std=0.02)
        if self.spatial_grid_size is not None:
            self.output_head.initialize_small_output()
            self.local_dynamics.initialize_small_output()
        else:
            nn.init.normal_(self.output_head[-1].weight, mean=0.0, std=1.0e-3)
            nn.init.zeros_(self.output_head[-1].bias)

    @staticmethod
    def _initialize_standard_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
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
        previous_state = x[:, -2, :]
        predictions = []
        for step in range(self.pred_len):
            state_token = self.state_projection(current_state)
            if not self.time_homogeneous_decoder:
                state_token = state_token + self.horizon_embedding[step].unsqueeze(0)
            query = (decoder_hidden + state_token).unsqueeze(1)
            memory_context, _ = self.decoder_attention(
                query, hidden, hidden, need_weights=False
            )
            decoder_input = self.decoder_attention_norm(
                state_token + memory_context.squeeze(1)
            )
            decoder_hidden = self.decoder_cell(decoder_input, decoder_hidden)
            state_increment = self.output_head(decoder_hidden)
            if self.local_dynamics is not None:
                state_increment = state_increment + self.local_dynamics(
                    current_state, previous_state
                )
            if self.local_dynamics is not None:
                next_state = stable_periodic_field_update(
                    current_state=current_state,
                    raw_increment=state_increment,
                    grid_size=int(self.spatial_grid_size),
                    spectral_filter_fraction=float(
                        self.stability_config.get("spectral_filter_fraction", 1.0)
                    ),
                    max_normalized_step=float(
                        self.stability_config["max_normalized_step"]
                    ),
                    rms_ratio_min=float(self.stability_config["rms_ratio_min"]),
                    rms_ratio_max=float(self.stability_config["rms_ratio_max"]),
                    field_rms_min=float(self.stability_config["field_rms_min"]),
                    field_rms_max=float(self.stability_config["field_rms_max"]),
                )
            else:
                next_state = current_state + state_increment
            previous_state, current_state = current_state, next_state
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
        spatial_grid_size: int | None = None,
        stability_config: dict | None = None,
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
        super().__init__(
            input_dim,
            d_model,
            nhead,
            seq_len,
            pred_len,
            layers,
            dropout,
            spatial_grid_size,
            stability_config,
        )


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
        spatial_grid_size: int | None = None,
        stability_config: dict | None = None,
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
        super().__init__(
            input_dim,
            d_model,
            nhead,
            seq_len,
            pred_len,
            layers,
            dropout,
            spatial_grid_size,
            stability_config,
        )


class StatefulLSTM(nn.Module):
    """Stateful recurrent forecaster with periodic-flow spatial dynamics."""

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
        spatial_grid_size: int | None = None,
        stability_config: dict | None = None,
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
        self.spatial_grid_size = (
            None if spatial_grid_size is None else int(spatial_grid_size)
        )
        self.stability_config = dict(stability_config or {})
        if self.state_increment_prediction and self.output_size != self.input_size:
            raise ValueError(
                "State-increment prediction requires output_size == input_size"
            )
        if self.spatial_grid_size is not None:
            if self.input_size != self.spatial_grid_size * self.spatial_grid_size:
                raise ValueError("LSTM input_size does not match the configured spatial grid")
            self.state_encoder = PeriodicSpatialEncoder(
                self.spatial_grid_size, self.hidden_size
            )
            recurrent_input_size = self.hidden_size
        else:
            self.state_encoder = None
            recurrent_input_size = self.input_size
        self.lstm_layers = nn.ModuleList(
            [nn.LSTM(recurrent_input_size, self.hidden_size, batch_first=True)]
            + [
                nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
                for _ in range(1, self.num_layers)
            ]
        )
        if self.spatial_grid_size is not None:
            self.output_projection = PeriodicSpatialDecoder(
                self.hidden_size, self.spatial_grid_size
            )
            self.local_dynamics = PeriodicLocalDynamics(self.spatial_grid_size)
        else:
            self.output_projection = nn.Linear(self.hidden_size, self.output_size)
            self.local_dynamics = None
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for name, parameter in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(parameter)
            elif "bias" in name:
                nn.init.zeros_(parameter)
        if self.state_increment_prediction and self.spatial_grid_size is not None:
            self.output_projection.initialize_small_output()
            self.local_dynamics.initialize_small_output()
        elif self.state_increment_prediction:
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
        hidden_state: dict | list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        training: bool | None = None,
    ) -> tuple[torch.Tensor, dict]:
        if x.ndim != 3 or x.shape[2] != self.input_size:
            raise ValueError(
                f"Expected LSTM input [B,T,{self.input_size}], found {tuple(x.shape)}"
            )
        x = x.to(dtype=torch.float32)
        input_reference = x
        use_training_behavior = self.training if training is None else bool(training)
        previous_input = None
        reference_rms = None
        if hidden_state is None:
            layer_states = self.zero_state(x.shape[0], x.device)
        elif isinstance(hidden_state, dict):
            layer_states = hidden_state["layers"]
            previous_input = hidden_state.get("previous_input")
            reference_rms = hidden_state.get("reference_rms")
        else:
            # Compatibility with pre-v2 callers; new checkpoints always use the
            # explicit carry dictionary below.
            layer_states = hidden_state
        if len(layer_states) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} hidden-state pairs")
        model_input = self.state_encoder(x) if self.state_encoder is not None else x
        if use_training_behavior and self.dropout_keep_prob < 1.0:
            model_input = F.dropout(
                model_input, p=1.0 - self.dropout_keep_prob, training=True
            )

        layer_input = model_input
        next_states: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_index, layer in enumerate(self.lstm_layers):
            old_h, old_c = layer_states[layer_index]
            layer_output, (new_h, new_c) = layer(layer_input, (old_h, old_c))
            next_states.append(
                self._zoneout(new_h, new_c, old_h, old_c, use_training_behavior)
            )
            if self.residual_connections and layer_index > 0:
                layer_input = layer_output + layer_input
            else:
                layer_input = layer_output
        output = self.output_projection(layer_input)
        if self.local_dynamics is not None:
            if input_reference.shape[1] > 1:
                previous_reference = torch.cat(
                    (input_reference[:, :1, :], input_reference[:, :-1, :]), dim=1
                )
            elif previous_input is not None:
                previous_reference = previous_input
            else:
                previous_reference = input_reference
            output = output + self.local_dynamics(
                input_reference, previous_reference
            )
        if self.state_increment_prediction:
            if self.local_dynamics is not None:
                if reference_rms is None:
                    reference_window = int(
                        self.stability_config.get("rms_reference_window", 64)
                    )
                    reference_values = input_reference[
                        :, -min(reference_window, input_reference.shape[1]) :, :
                    ]
                    reference_rms = (
                        reference_values.square()
                        .mean(dim=(1, 2), keepdim=True)
                        .add(1.0e-8)
                        .sqrt()
                    )
                output = stable_periodic_field_update(
                    current_state=input_reference,
                    raw_increment=output,
                    grid_size=int(self.spatial_grid_size),
                    spectral_filter_fraction=float(
                        self.stability_config.get("spectral_filter_fraction", 1.0)
                    ),
                    max_normalized_step=float(
                        self.stability_config["max_normalized_step"]
                    ),
                    rms_ratio_min=float(self.stability_config["rms_ratio_min"]),
                    rms_ratio_max=float(self.stability_config["rms_ratio_max"]),
                    field_rms_min=float(self.stability_config["field_rms_min"]),
                    field_rms_max=float(self.stability_config["field_rms_max"]),
                    rms_relaxation_strength=float(
                        self.stability_config.get("rms_relaxation_strength", 0.0)
                    ),
                    rms_relaxation_target=reference_rms,
                )
            else:
                output = input_reference + output
        carry = {
            "layers": next_states,
            "previous_input": input_reference[:, -1:, :],
            "reference_rms": reference_rms,
        }
        return output, carry


def build_model(model_type: str, input_dim: int, config: dict, pred_len: int) -> nn.Module:
    if config["model"].get("prediction_parameterization") != "state_increment":
        raise ValueError("model.prediction_parameterization must be 'state_increment'")
    spatial_config = config.get("spatial_adapter", {})
    spatial_grid_size = (
        int(spatial_config["grid_size"])
        if bool(spatial_config.get("enabled", False))
        else None
    )
    stability_config = (
        spatial_config.get("stability") if spatial_grid_size is not None else None
    )
    if spatial_grid_size is not None:
        required_stability = {
            "max_normalized_step",
            "rms_ratio_min",
            "rms_ratio_max",
            "field_rms_min",
            "field_rms_max",
        }
        missing = required_stability.difference(stability_config or {})
        if missing:
            raise ValueError(f"Missing spatial stability settings: {sorted(missing)}")
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
            spatial_grid_size=spatial_grid_size,
            stability_config=stability_config,
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
            spatial_grid_size=spatial_grid_size,
            stability_config=stability_config,
        )
    if model_type == "lstm":
        if config["lstm"]["dtype"] != "float32":
            raise ValueError("The formal LSTM configuration requires dtype='float32'")
        lstm_stability_config = dict(stability_config or {})
        lstm_stability_config.update(
            dict(config["lstm"].get("stability_override", {}))
        )
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
            spatial_grid_size=spatial_grid_size,
            stability_config=lstm_stability_config,
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
