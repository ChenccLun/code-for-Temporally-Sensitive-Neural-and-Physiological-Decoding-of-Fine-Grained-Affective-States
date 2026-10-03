"""CCC-DILATE: inference-only release of the tri-modal affect regression network.

Public API
----------
:class:`~cccdilate.model.CCCDilateNet`
    The network: modality-specific encoders, gated cross-modal attention with
    contextual-aware gating, and a transformer temporal model.
:class:`~cccdilate.attention.GatedMultiHeadCrossAttention`,
:class:`~cccdilate.attention.CrossModalFusionBlock`
    The pre-gated cross-modal attention building blocks.
:class:`~cccdilate.encoders.SpatialFeatureExtractor`,
:class:`~cccdilate.encoders.TemporalFeatureExtractor`,
:class:`~cccdilate.encoders.PositionalEncoding`
    Feature encoders.
:class:`~cccdilate.losses.CCCLoss`, :class:`~cccdilate.losses.MAECCCLoss`,
:class:`~cccdilate.losses.DILATE`, :class:`~cccdilate.losses.CCCDILATELoss`
    The concordance, DILATE and combined CCC-DILATE objectives.
"""

from .attention import CrossModalFusionBlock, GatedMultiHeadCrossAttention
from .encoders import (
    PositionalEncoding,
    SpatialFeatureExtractor,
    TemporalFeatureExtractor,
)
from .losses import CCCDILATELoss, CCCLoss, DILATE, MAECCCLoss, ccc, pcc
from .model import CCCDilateNet

__version__ = "1.0.0"

__all__ = [
    "CCCDilateNet",
    "GatedMultiHeadCrossAttention",
    "CrossModalFusionBlock",
    "SpatialFeatureExtractor",
    "TemporalFeatureExtractor",
    "PositionalEncoding",
    "CCCLoss",
    "MAECCCLoss",
    "DILATE",
    "CCCDILATELoss",
    "ccc",
    "pcc",
    "__version__",
]
