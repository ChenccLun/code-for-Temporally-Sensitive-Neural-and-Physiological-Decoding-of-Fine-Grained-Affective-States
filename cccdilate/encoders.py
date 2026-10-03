"""Feature encoders shared by the paper's network.

The three input modalities (EEG, GSR, eye tracking) are each encoded by the same
two-stage stack:

1. :class:`SpatialFeatureExtractor` mixes the channels of a single time step into
   a 32-dimensional embedding.
2. Two :class:`TemporalFeatureExtractor` blocks convolve along time, each halving
   the number of time steps and doubling the channel width (32 -> 64 -> 128).

A :class:`PositionalEncoding` is then added to each modality before the gated
cross-modal attention stage.
"""

import math

import torch
import torch.nn as nn

__all__ = [
    "SpatialFeatureExtractor",
    "TemporalFeatureExtractor",
    "PositionalEncoding",
]


class SpatialFeatureExtractor(nn.Module):
    """Project the channel vector of each time step to ``output_dim`` features.

    The projection is implemented as a ``Conv1d`` with ``kernel_size ==
    input_dim`` and ``in_channels == 1`` over the flattened feature axis. Because
    the kernel spans the whole channel vector and the stride is one, the
    convolution is exactly a fully connected layer shared across time.

    Shapes:
        input  ``[B, T, input_dim]``
        output ``[B, T, output_dim]``
    """

    def __init__(self, input_dim: int, output_dim: int = 32, pooling: bool = False):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=output_dim,
            kernel_size=input_dim,
            stride=1,
        )
        self.bn = nn.BatchNorm1d(output_dim)
        self.elu = nn.ELU()
        if pooling:
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.output_dim = output_dim
        self.pooling = pooling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = x.size()

        x = x.reshape(-1, 1, channels)  # [B*T, 1, input_dim]
        x = self.conv(x)                # [B*T, output_dim, 1]
        x = self.bn(x)
        x = self.elu(x)

        x = x.squeeze(-1)                                # [B*T, output_dim]
        x = x.reshape(batch_size, seq_len, self.output_dim)

        if self.pooling:
            x = x.permute(0, 2, 1)
            x = self.pool(x)
            x = x.permute(0, 2, 1)
        return x


class TemporalFeatureExtractor(nn.Module):
    """Convolve along time, then halve the sequence length by max pooling.

    Shapes:
        input  ``[B, T, in_channels]``
        output ``[B, T // 2, out_channels]``
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.elu = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # [B, in_channels, T]

        x = self.conv(x)        # [B, out_channels, T]
        x = self.bn(x)
        x = self.elu(x)
        x = self.pool(x)        # [B, out_channels, T // 2]

        return x.permute(0, 2, 1)


class PositionalEncoding(nn.Module):
    """Add fixed sinusoidal position signals to a ``[B, T, d_model]`` sequence.

    ``PE[t, 2i]   = sin(t / 10000^(2i / d_model))``
    ``PE[t, 2i+1] = cos(t / 10000^(2i / d_model))``

    The table is registered as a buffer so it follows the module across devices
    and is not part of ``state_dict``-driven training.
    """

    def __init__(self, d_model: int, dropout: float = 0.2, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :].to(x.device)
        return self.dropout(x)
