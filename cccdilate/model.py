"""The CCC-DILATE network: tri-modal encoders + gated cross-modal attention.

The released model corresponds to the ``EEGTransformer`` class of the original
research code (``predict_tf_att_crossattn_pcag_loso_dsh.py``). Only the
``forward_transformer`` branch of that class is kept, because that is the branch
used for every reported result; the unused ``forward_static`` branch and the
unused ``ModulationModule`` have been removed. All attribute names that appear in
the released checkpoint are preserved so that the weights load with
``strict=True``.
"""

import math

import torch
import torch.nn as nn

from .attention import CrossModalFusionBlock
from .encoders import (
    PositionalEncoding,
    SpatialFeatureExtractor,
    TemporalFeatureExtractor,
)

__all__ = ["CCCDilateNet"]


class CCCDilateNet(nn.Module):
    """Predict per-timestep (valence, arousal) from a window of tri-modal features.

    Input layout
    ------------
    The model consumes ``[B, S, C * 4]`` windows. The trailing factor 4 is an
    artefact of the preprocessing pipeline: physiological signals are recorded at
    128 Hz (eye tracking at 60 Hz), resampled to 120 Hz and then packed into
    non-overlapping groups of four consecutive samples, so a single "frame" of the
    stored feature matrix holds 4 sub-samples of each of the ``C`` channels
    (channel-major: ``ch0.f0, ch0.f1, ch0.f2, ch0.f3, ch1.f0, ...``).

    The first operation of :meth:`forward` unfolds that packing,
    ``[B, S, C, 4] -> [B, S * 4, C]``, recovering the native 120 Hz time series
    with ``C`` channels per step. Two :class:`TemporalFeatureExtractor` blocks then
    halve the time axis twice, so the output has exactly ``S`` time steps, one per
    stored 30 Hz frame.

    Channel layout
    --------------
    ``C = eeg_channels + gsr_channels + eye_channels`` and the slices are
    contiguous in that order. For the released MAHNOB configuration
    (``64 + 18 + 18 = 100``): EEG is 32 electrode channels plus their 32
    gamma-band (30-49 Hz) Hilbert envelopes; GSR is one channel plus its five
    band envelopes, expanded to position/velocity/acceleration (18); eye tracking
    is six gaze/pupil channels expanded to position/velocity/acceleration (18).

    Output
    ------
    ``[B, S, label_output_dim]``, i.e. one (valence, arousal) pair per stored frame.

    Note on ``ln_gsr``/``ln_eye``: they are declared because the released
    checkpoint contains them, but the forward pass does not use them (only
    ``ln_eeg`` normalises the concatenated six-way context).
    """

    def __init__(
        self,
        label_output_dim: int = 2,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        eeg_channels: int = 64,
        gsr_channels: int = 18,
        eye_channels: int = 18,
        d_modal: int = 128,
        dropout: float = 0.2,
        pe_max_len: int = 2000,
    ):
        super().__init__()
        self.d_model = d_model
        self.eeg_channels = eeg_channels
        self.gsr_channels = gsr_channels
        self.eye_channels = eye_channels
        self.d_modal = d_modal

        # --- per-modality spatial + temporal encoders (32 -> 64 -> 128) ---
        self.eeg_spatial = SpatialFeatureExtractor(eeg_channels)
        self.eeg_temporal1 = TemporalFeatureExtractor(32, 64)
        self.eeg_temporal2 = TemporalFeatureExtractor(64, d_modal)

        self.gsr_spatial = SpatialFeatureExtractor(gsr_channels)
        self.gsr_temporal1 = TemporalFeatureExtractor(32, 64)
        self.gsr_temporal2 = TemporalFeatureExtractor(64, d_modal)

        self.eye_spatial = SpatialFeatureExtractor(eye_channels)
        self.eye_temporal1 = TemporalFeatureExtractor(32, 64)
        self.eye_temporal2 = TemporalFeatureExtractor(64, d_modal)

        # --- channel attention over the six concatenated context streams ---
        self.channel_attention = nn.MultiheadAttention(
            embed_dim=d_modal * 6, num_heads=1, batch_first=True
        )

        # --- gated cross-modal attention ---
        self.positional_encoding_modal = PositionalEncoding(
            d_model=d_modal, dropout=dropout, max_len=pe_max_len
        )
        self.cross_modal_block = CrossModalFusionBlock(
            d_each=d_modal, nhead=nhead, dropout=dropout
        )

        # --- fuse to the temporal model width ---
        self.fuse_proj = nn.Linear(d_modal * 6, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.ln_eeg = nn.LayerNorm(d_modal * 6)
        self.ln_gsr = nn.LayerNorm(d_modal)
        self.ln_eye = nn.LayerNorm(d_modal)

        self.fc_out = nn.Linear(d_model, label_output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``[B, S, C * 4]`` window of tri-modal features.

        Returns:
            ``[B, S, label_output_dim]`` predictions.
        """
        batch_size, seq_len, _ = x.size()

        # Unfold the 4 sub-samples per channel: [B, S, C, 4] -> [B, S*4, C].
        x = x.reshape(batch_size, seq_len, -1, 4).permute(0, 1, 3, 2)
        x = x.reshape(batch_size, seq_len * 4, -1)

        eeg = x[:, :, : self.eeg_channels]
        gsr = x[:, :, self.eeg_channels : -self.eye_channels]
        eye = x[:, :, -self.eye_channels :]

        # Spatial + temporal encoding, [B, S*4, C] -> [B, S, d_modal].
        eeg = self.eeg_temporal2(self.eeg_temporal1(self.eeg_spatial(eeg)))
        gsr = self.gsr_temporal2(self.gsr_temporal1(self.gsr_spatial(gsr)))
        eye = self.eye_temporal2(self.eye_temporal1(self.eye_spatial(eye)))

        eeg = self.positional_encoding_modal(eeg)
        gsr = self.positional_encoding_modal(gsr)
        eye = self.positional_encoding_modal(eye)

        # Six gated cross-modal interactions + contextual-aware gating.
        contexts = self.cross_modal_block(eeg, gsr, eye, attn_mask=None)

        # Channel attention over the 6 * d_modal concatenated context streams.
        # Evaluated in float32: normalising and softmaxing 768 channels is not
        # numerically safe in half precision (the original code disabled autocast
        # around this call for the same reason). In float32 this is a no-op.
        emb = self.ln_eeg(torch.cat(contexts, dim=-1))
        emb = emb.float()
        emb, _ = self.channel_attention(emb, emb, emb)
        emb = self.fuse_proj(emb)

        emb = emb * math.sqrt(self.d_model)

        # Temporal model over the fused sequence (no additional positional encoding).
        emb = emb.permute(1, 0, 2)          # [T, B, d_model]
        emb = self.transformer_encoder(emb)
        emb = emb.permute(1, 0, 2)          # [B, T, d_model]

        return self.fc_out(emb)
