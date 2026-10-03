"""Gated cross-modal attention used for tri-modal (EEG, GSR, eye) fusion.

The block implements the paper's *pre-gating* cross-attention: the scaled
dot-product logits are multiplied element-wise by a bounded, learned gate
computed from the same query/key projections. This lets the network suppress
query/key pairs whose interaction is not informative, which is important when
one modality is dominated by noise (e.g. GSR artefacts or eye-tracker loss).
"""

import math
from typing import Optional

import torch
import torch.nn as nn

__all__ = ["GatedMultiHeadCrossAttention", "CrossModalFusionBlock"]


class GatedMultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention with pre-gating.

    For query modality ``m`` attending to modality ``n`` (both ``[B, T, d_model]``)::

        Q = W_q X_m,  K = W_k X_n,  V = W_v X_n            # [B, T, d_model]
        logits = Q K^T / sqrt(d_head)                      # scaled dot product
        G_raw  = tanh(Q) tanh(K)^T
        G      = (G_raw / d_head + 1) / 2                  # element-wise in [0, 1]
        A      = softmax(logits * G)                       # gated attention
        out    = W_o (A V)

    ``G_raw`` is a sum of ``d_head`` products of ``tanh`` values, so every entry
    of ``G_raw / d_head`` lies in ``[-1, 1]`` and every gate value lies in
    ``[0, 1]``: the gate can attenuate a logit but never invert its sign.
    """

    def __init__(self, d_model: int = 128, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(
        self,
        Qx: torch.Tensor,
        Kx: torch.Tensor,
        Vx: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            Qx: queries, ``[B, T, d_model]``, from the query modality.
            Kx: keys, ``[B, T, d_model]``, from the context modality.
            Vx: values, ``[B, T, d_model]``, from the context modality.
            attn_mask: optional boolean mask broadcastable to ``[B, nhead, T, T]``
                where ``True`` marks a visible position.

        Returns:
            ``[B, T, d_model]``
        """
        B, T, _ = Qx.shape

        Q = self.q_proj(Qx)
        K = self.k_proj(Kx)
        V = self.v_proj(Vx)

        Q = Q.view(B, T, self.nhead, self.d_head).transpose(1, 2)  # [B, h, T, d_head]
        K = K.view(B, T, self.nhead, self.d_head).transpose(1, 2)
        V = V.view(B, T, self.nhead, self.d_head).transpose(1, 2)

        logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)

        gate_raw = torch.matmul(torch.tanh(Q), torch.tanh(K).transpose(-2, -1))
        gate = (gate_raw / self.d_head + 1.0) * 0.5  # [B, h, T, T], values in [0, 1]

        gated_logits = logits * gate

        if attn_mask is not None:
            gated_logits = gated_logits.masked_fill(~attn_mask, float("-inf"))

        attn = torch.softmax(gated_logits, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, V)                                  # [B, h, T, d_head]
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.out_proj(out)
        return self.proj_drop(out)


class CrossModalFusionBlock(nn.Module):
    """All six ordered cross-modal interactions plus a contextual-aware gate.

    Six *independent* gated cross-attentions cover every ordered modality pair
    (no weight sharing between directions). Each attention output is then passed
    through a contextual-aware gate (CAG) that fuses the attended context with
    the query modality's own representation::

        G   = ReLU(W_h^G Qhat + W_q^G Q_m)      # how much context to admit
        E   = ReLU(W_E Qhat)                    # context evidence
        C   = LayerNorm(E) * LayerNorm(G)       # gated context
        out = ReLU(fc_C(C)) + ReLU(fc_Q(Q_m))   # context + residual query

    The CAG parameters are shared by the two directions that share a query
    modality (EEG receives from GSR and from eye, etc.), because the gate is a
    property of the query modality.

    Shapes:
        input  ``Heeg, Hgsr, Heye``: ``[B, T, d_each]``
        output six tensors ``[B, T, d_each]`` in the order
        ``(eeg_from_gsr, eeg_from_eye, gsr_from_eeg, gsr_from_eye,
        eye_from_eeg, eye_from_gsr)``
    """

    def __init__(
        self,
        d_each: int = 128,
        nhead: int = 8,
        dropout: float = 0.2,
        d_gate: Optional[int] = None,
    ):
        super().__init__()
        self.d_each = d_each
        self.d_gate = d_gate or d_each

        self.attn_eeg_from_gsr = GatedMultiHeadCrossAttention(d_each, nhead, dropout)
        self.attn_eeg_from_eye = GatedMultiHeadCrossAttention(d_each, nhead, dropout)
        self.attn_gsr_from_eeg = GatedMultiHeadCrossAttention(d_each, nhead, dropout)
        self.attn_gsr_from_eye = GatedMultiHeadCrossAttention(d_each, nhead, dropout)
        self.attn_eye_from_eeg = GatedMultiHeadCrossAttention(d_each, nhead, dropout)
        self.attn_eye_from_gsr = GatedMultiHeadCrossAttention(d_each, nhead, dropout)

        for name in ("eeg", "gsr", "eye"):
            setattr(self, f"W_h_G_{name}", nn.Linear(d_each, self.d_gate, bias=True))
            setattr(self, f"W_q_G_{name}", nn.Linear(d_each, self.d_gate, bias=True))
            setattr(self, f"W_E_{name}", nn.Linear(d_each, self.d_gate, bias=True))
            setattr(self, f"fc_C_{name}", nn.Linear(self.d_gate, d_each))
            setattr(self, f"fc_Q_{name}", nn.Linear(d_each, d_each))

        self.normE = nn.LayerNorm(self.d_gate)
        self.normG = nn.LayerNorm(self.d_gate)
        self.relu = nn.ReLU()

    @staticmethod
    def _cag_once(
        Qm: torch.Tensor,
        Qhat_mn: torch.Tensor,
        W_h_G: nn.Linear,
        W_q_G: nn.Linear,
        W_E: nn.Linear,
        fc_C: nn.Linear,
        fc_Q: nn.Linear,
        normE: nn.LayerNorm,
        normG: nn.LayerNorm,
        relu: nn.ReLU,
    ) -> torch.Tensor:
        """Apply the contextual-aware gate for one direction ``m <- n``.

        Args:
            Qm: the query modality representation ``[B, T, d_each]``.
            Qhat_mn: the gated cross-attention output for ``m <- n``.
        """
        gate = relu(W_h_G(Qhat_mn) + W_q_G(Qm))
        evidence = relu(W_E(Qhat_mn))
        context = normE(evidence) * normG(gate)
        return relu(fc_C(context)) + relu(fc_Q(Qm))

    def forward(
        self,
        Heeg: torch.Tensor,
        Hgsr: torch.Tensor,
        Heye: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        # Gated cross-attention outputs, one per ordered modality pair.
        qhat_eeg_from_gsr = self.attn_eeg_from_gsr(Heeg, Hgsr, Hgsr, attn_mask)
        qhat_eeg_from_eye = self.attn_eeg_from_eye(Heeg, Heye, Heye, attn_mask)
        qhat_gsr_from_eeg = self.attn_gsr_from_eeg(Hgsr, Heeg, Heeg, attn_mask)
        qhat_gsr_from_eye = self.attn_gsr_from_eye(Hgsr, Heye, Heye, attn_mask)
        qhat_eye_from_eeg = self.attn_eye_from_eeg(Heye, Heeg, Heeg, attn_mask)
        qhat_eye_from_gsr = self.attn_eye_from_gsr(Heye, Hgsr, Hgsr, attn_mask)

        cag = self._cag_once
        cag_args = (self.normE, self.normG, self.relu)

        eeg_from_gsr = cag(
            Heeg, qhat_eeg_from_gsr,
            self.W_h_G_eeg, self.W_q_G_eeg, self.W_E_eeg,
            self.fc_C_eeg, self.fc_Q_eeg, *cag_args,
        )
        eeg_from_eye = cag(
            Heeg, qhat_eeg_from_eye,
            self.W_h_G_eeg, self.W_q_G_eeg, self.W_E_eeg,
            self.fc_C_eeg, self.fc_Q_eeg, *cag_args,
        )
        gsr_from_eeg = cag(
            Hgsr, qhat_gsr_from_eeg,
            self.W_h_G_gsr, self.W_q_G_gsr, self.W_E_gsr,
            self.fc_C_gsr, self.fc_Q_gsr, *cag_args,
        )
        gsr_from_eye = cag(
            Hgsr, qhat_gsr_from_eye,
            self.W_h_G_gsr, self.W_q_G_gsr, self.W_E_gsr,
            self.fc_C_gsr, self.fc_Q_gsr, *cag_args,
        )
        eye_from_eeg = cag(
            Heye, qhat_eye_from_eeg,
            self.W_h_G_eye, self.W_q_G_eye, self.W_E_eye,
            self.fc_C_eye, self.fc_Q_eye, *cag_args,
        )
        eye_from_gsr = cag(
            Heye, qhat_eye_from_gsr,
            self.W_h_G_eye, self.W_q_G_eye, self.W_E_eye,
            self.fc_C_eye, self.fc_Q_eye, *cag_args,
        )

        return (
            eeg_from_gsr,
            eeg_from_eye,
            gsr_from_eeg,
            gsr_from_eye,
            eye_from_eeg,
            eye_from_gsr,
        )
