"""Concordance (CCC) and DILATE objectives for the CCC-DILATE model.

Two families of losses live here.

**Concordance losses.** :class:`CCCLoss` optimises ``1 - CCC`` and
:class:`MAECCCLoss` adds a smooth-L1 term. Both use the batch-global CCC of
Lin (1989)::

    CCC = 2 * cov(y, y_hat) / (var(y) + var(y_hat) + (mean(y) - mean(y_hat))^2)

which penalises a decorrelated prediction, a mis-scaled prediction *and* a biased
prediction at once.

**DILATE.** :class:`DILATE` is the batch-wise extension of the DILATE loss
(Le Guen & Thome, 2019). It splits the distortion between prediction and target
into a *shape* term and a *temporal* term::

    L_DILATE = alpha * L_shape + (1 - alpha) * L_temporal
    L_shape  = softDTW(gamma)               # soft-min over alignments
    L_temporal = <A*_gamma, Omega> / T^2    # expected temporal displacement

``Omega[i, j] = (i - j)^2`` is the temporal distortion of an alignment path and
``A*_gamma`` is the soft-DTW alignment matrix, obtained as the gradient of the
soft-DTW value with respect to the pairwise cost matrix.

**CCC-DILATE.** :class:`CCCDILATELoss` is the paper's combined objective::

    L_CCC-DILATE = L_CCC + lambda * L_DILATE

The soft-DTW dynamic programs are executed in NumPy float32 on the CPU, exactly
as in the original implementation (which JIT-compiled the same code with numba);
the autograd wrappers move the results back to the input device.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function

__all__ = [
    "ccc",
    "pcc",
    "CCCLoss",
    "MAECCCLoss",
    "DILATE",
    "CCCDILATELoss",
]

# Sentinel cost for the padded borders of the dynamic-programming tables. It is
# added to (never multiplied by) real costs, and exp(-1e8 / gamma) underflows to
# zero for every gamma used here.
_SENTINEL = 1e8


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def ccc(y_true, y_pred) -> float:
    """Concordance correlation coefficient between two 1-D arrays.

    ``CCC = 2 * cov / (var(y) + var(y_hat) + (mean(y) - mean(y_hat))^2)``
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    true_mean = y_true.mean()
    pred_mean = y_pred.mean()
    covariance = np.mean((y_true - true_mean) * (y_pred - pred_mean))
    denominator = y_true.var() + y_pred.var() + (true_mean - pred_mean) ** 2
    return float(2.0 * covariance / denominator)


def pcc(y_true, y_pred) -> float:
    """Pearson correlation coefficient between two 1-D arrays."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return float(np.corrcoef(y_true, y_pred)[0, 1])


# --------------------------------------------------------------------------- #
# concordance losses
# --------------------------------------------------------------------------- #
class CCCLoss(nn.Module):
    """``1 - CCC`` over the whole batch (a differentiable surrogate of the metric).

    The statistics are computed over every element of ``y_pred``/``y_true`` at
    once, so the loss couples all samples and both output dimensions.
    """

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        true_mean = torch.mean(y_true)
        pred_mean = torch.mean(y_pred)

        covariance = torch.mean((y_true - true_mean) * (y_pred - pred_mean))
        true_var = torch.var(y_true)
        pred_var = torch.var(y_pred)

        ccc_value = 2 * covariance / (
            true_var + pred_var + (true_mean - pred_mean) ** 2 + 1e-8
        )
        return -ccc_value


class MAECCCLoss(nn.Module):
    """``SmoothL1(y_pred, y_true) - CCC``."""

    def __init__(self):
        super().__init__()
        self.mae_loss = nn.SmoothL1Loss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        mae = self.mae_loss(y_pred, y_true)

        true_mean = torch.mean(y_true)
        pred_mean = torch.mean(y_pred)

        covariance = torch.mean((y_true - true_mean) * (y_pred - pred_mean))
        true_var = torch.var(y_true)
        pred_var = torch.var(y_pred)

        ccc_value = 2 * covariance / (
            true_var + pred_var + (true_mean - pred_mean) ** 2 + 1e-8
        )
        return mae - ccc_value


# --------------------------------------------------------------------------- #
# soft-DTW / DILATE core (NumPy, float32)
# --------------------------------------------------------------------------- #
def _softmin(x: np.ndarray, gamma: float):
    """Soft-minimum over the last axis.

    ``min_gamma(x) = -gamma * log(sum_i exp(-x_i / gamma))``

    Args:
        x: array of shape ``[..., 3]`` holding the three candidate costs.
        gamma: smoothing temperature.

    Returns:
        ``(values, weights)`` with values of shape ``[...]`` and weights of shape
        ``[..., 3]`` summing to one (the soft-argmin distribution).
    """
    negated = -x
    max_negated = negated.max(axis=1, keepdims=True)
    exp_negated = np.exp((negated - max_negated) / gamma)
    normaliser = exp_negated.sum(axis=1, keepdims=True)
    values = -(gamma * np.log(normaliser[:, 0]) + max_negated[:, 0])
    return values, exp_negated / normaliser


def _softmax_hessian_product(p: np.ndarray, z: np.ndarray, gamma: float) -> np.ndarray:
    """Hessian-vector product of the softmax with respect to its input.

    With ``p`` the softmax output, the Hessian is ``(diag(p) - p pᵀ) / gamma``;
    this evaluates it against ``z`` without materialising the matrix. The soft-min
    used by the DTW recurrence is ``-softmax(-x)``, so its Hessian is the
    negative of this quantity.
    """
    b, c, _ = p.shape
    interm_sum = (p * z).reshape(-1, 3).sum(axis=1)
    return (p * z - p * interm_sum.reshape(b, c, 1)) / gamma


def _softdtw_forward(D: np.ndarray, gamma: float) -> np.ndarray:
    """Soft-DTW recursion over a batch of pairwise cost matrices.

    Args:
        D: ``[B, C, N, M]`` pairwise squared costs.
        gamma: smoothing temperature.

    Returns:
        ``[B, C, N + 2, M + 2]`` accumulated cost table ``R``; the soft-DTW value
        for each ``(b, c)`` is ``R[b, c, -2, -2]``.
    """
    batch_size, num_channels, n, m = D.shape
    R = np.zeros((batch_size, num_channels, n + 2, m + 2), dtype=np.float32) + _SENTINEL
    R[:, :, 0, 0] = 0
    for j in range(1, m + 1):
        for i in range(1, n + 1):
            r0 = -R[:, :, i - 1, j - 1] / gamma
            r1 = -R[:, :, i - 1, j] / gamma
            r2 = -R[:, :, i, j - 1] / gamma
            rmax = np.maximum(np.maximum(r0, r1), r2)
            rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
            softmin = -gamma * (np.log(rsum) + rmax)
            R[:, :, i, j] = D[:, :, i - 1, j - 1] + softmin
    return R


def _softdtw_backward(D_in: np.ndarray, R: np.ndarray, gamma: float) -> np.ndarray:
    """Gradient of the soft-DTW value with respect to the cost matrix.

    Returns the soft alignment matrix ``A*_gamma`` of shape ``[B, C, N, M]``.
    """
    b, c, n, m = D_in.shape
    D = np.zeros((b, c, n + 2, m + 2), dtype=np.float32)
    E = np.zeros((b, c, n + 2, m + 2), dtype=np.float32)

    D[:, :, 1 : n + 1, 1 : m + 1] = D_in
    E[:, :, -1, -1] = 1
    R[:, :, :, -1] = -_SENTINEL
    R[:, :, -1, :] = -_SENTINEL
    R[:, :, -1, -1] = R[:, :, -2, -2]

    R2 = R[:, :, ::-1, ::-1]
    D2 = D[:, :, ::-1, ::-1]
    E2 = E[:, :, ::-1, ::-1]
    for j in range(1, m + 1):
        for i in range(1, n + 1):
            a0 = (R2[:, :, i - 1, j] - R2[:, :, i, j] - D2[:, :, i - 1, j]) / gamma
            b0 = (R2[:, :, i, j - 1] - R2[:, :, i, j] - D2[:, :, i, j - 1]) / gamma
            c0 = (R2[:, :, i - 1, j - 1] - R2[:, :, i, j] - D2[:, :, i - 1, j - 1]) / gamma
            E2[:, :, i, j] = (
                E2[:, :, i - 1, j] * np.exp(a0)
                + E2[:, :, i, j - 1] * np.exp(b0)
                + E2[:, :, i - 1, j - 1] * np.exp(c0)
            )

    E3 = E2[:, :, ::-1, ::-1]
    return E3[:, :, 1 : n + 1, 1 : m + 1]


def _dtw_path_grad(theta: np.ndarray, gamma: float):
    """Forward pass of the soft-DTW alignment-path gradient."""
    b, c, m, n = theta.shape

    V = np.zeros((b, c, m + 1, n + 1), dtype=np.float32)
    V[:, :, :, 0] = _SENTINEL
    V[:, :, 0, :] = _SENTINEL
    V[:, :, 0, 0] = 0

    Q = np.zeros((b, c, m + 2, n + 2, 3), dtype=np.float32)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            interm = np.zeros((b, c, 3), dtype=np.float32)
            interm[:, :, 0] = V[:, :, i, j - 1]
            interm[:, :, 1] = V[:, :, i - 1, j - 1]
            interm[:, :, 2] = V[:, :, i - 1, j]

            value, weights = _softmin(interm.reshape((b * c, 3)), gamma)
            Q[:, :, i, j] = weights.reshape((b, c, 3))
            V[:, :, i, j] = theta[:, :, i - 1, j - 1] + value.reshape((b, c))

    E2 = np.zeros((b, c, m + 2, n + 2), dtype=np.float32)
    E2[:, :, m + 1, :] = 0
    E2[:, :, :, n + 1] = 0
    E2[:, :, m + 1, n + 1] = 1

    Q[:, :, m + 1, n + 1] = 1
    Q2 = Q[:, :, ::-1, ::-1, :]
    E3 = E2[:, :, ::-1, ::-1]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            E3[:, :, i, j] = (
                Q2[:, :, i, j - 1, 0] * E3[:, :, i, j - 1]
                + Q2[:, :, i - 1, j - 1, 1] * E3[:, :, i - 1, j - 1]
                + Q2[:, :, i - 1, j, 2] * E3[:, :, i - 1, j]
            )

    E4 = E3[:, :, ::-1, ::-1]
    return V[:, :, m, n], E4[:, :, 1 : m + 1, 1 : n + 1], Q, E4


def _dtw_path_hessian(theta: np.ndarray, Z: np.ndarray, Q: np.ndarray, E: np.ndarray, gamma: float):
    """Backward pass of the soft-DTW alignment-path gradient."""
    b, num_ch, m, n = Z.shape

    V_dot = np.zeros((b, num_ch, m + 1, n + 1), dtype=np.float32)
    V_dot[:, :, 0, 0] = 0

    Q_dot = np.zeros((b, num_ch, m + 2, n + 2, 3), dtype=np.float32)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            V_dot[:, :, i, j] = (
                Z[:, :, i - 1, j - 1]
                + Q[:, :, i, j, 0] * V_dot[:, :, i, j - 1]
                + Q[:, :, i, j, 1] * V_dot[:, :, i - 1, j - 1]
                + Q[:, :, i, j, 2] * V_dot[:, :, i - 1, j]
            )

            v = np.zeros((b, num_ch, 3))
            v[:, :, 0] = V_dot[:, :, i, j - 1]
            v[:, :, 1] = V_dot[:, :, i - 1, j - 1]
            v[:, :, 2] = V_dot[:, :, i - 1, j]

            Q_dot[:, :, i, j] = -_softmax_hessian_product(Q[:, :, i, j], v, gamma)

    E2_dot = np.zeros((b, num_ch, m + 2, n + 2), dtype=np.float32)

    Q2 = Q[:, :, ::-1, ::-1, :]
    E2 = E[:, :, ::-1, ::-1]
    Q2_dot = Q_dot[:, :, ::-1, ::-1, :]
    E3_dot = E2_dot[:, :, ::-1, ::-1]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            E3_dot[:, :, i, j] = (
                Q2_dot[:, :, i, j - 1, 0] * E2[:, :, i, j - 1]
                + Q2[:, :, i, j - 1, 0] * E3_dot[:, :, i, j - 1]
                + Q2_dot[:, :, i - 1, j - 1, 1] * E2[:, :, i - 1, j - 1]
                + Q2[:, :, i - 1, j - 1, 1] * E3_dot[:, :, i - 1, j - 1]
                + Q2_dot[:, :, i - 1, j, 2] * E2[:, :, i - 1, j]
                + Q2[:, :, i - 1, j, 2] * E3_dot[:, :, i - 1, j]
            )
    E4_dot = E3_dot[:, :, ::-1, ::-1]

    return V_dot[:, :, m, n], E4_dot[:, :, 1 : m + 1, 1 : n + 1]


# --------------------------------------------------------------------------- #
# autograd wrappers
# --------------------------------------------------------------------------- #
class SoftDTWBatch(Function):
    """Soft-DTW value for a batch of cost matrices, with a custom backward pass."""

    @staticmethod
    def forward(ctx, D: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
        device = D.device
        D_np = D.detach().cpu().numpy()
        R = _softdtw_forward(D_np.astype(np.float32), float(gamma))
        R_t = torch.as_tensor(R, dtype=torch.float32).to(device)
        ctx.save_for_backward(D, R_t, torch.as_tensor([float(gamma)], device=device))
        return R_t[:, :, -2, -2]

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        device = grad_output.device
        D, R, gamma = ctx.saved_tensors
        E = _softdtw_backward(
            D.detach().cpu().numpy().astype(np.float32),
            R.detach().cpu().numpy().astype(np.float32),
            float(gamma.item()),
        )
        E_t = torch.as_tensor(E, dtype=torch.float32).to(device)
        return torch.einsum("bc, bcxy-> bcxy", grad_output, E_t), None


class PathDTWBatch(Function):
    """Soft-DTW alignment matrix for a batch of cost matrices."""

    @staticmethod
    def forward(ctx, D: torch.Tensor, gamma: float) -> torch.Tensor:
        device = D.device
        batch_size, num_channels, n, _ = D.shape

        grad = torch.zeros((batch_size, num_channels, n, n), device=device)
        Q_gpu = torch.zeros((batch_size, num_channels, n + 2, n + 2, 3), device=device)
        E_gpu = torch.zeros((batch_size, num_channels, n + 2, n + 2), device=device)

        _, grad_cpu, Q_cpu, E_cpu = _dtw_path_grad(
            D.detach().cpu().numpy().astype(np.float32), float(gamma)
        )

        grad[:, :, :] = torch.as_tensor(grad_cpu, dtype=torch.float32).to(device)
        Q_gpu[:, :, :, :] = torch.as_tensor(Q_cpu, dtype=torch.float32).to(device)
        E_gpu[:, :, :] = torch.as_tensor(E_cpu, dtype=torch.float32).to(device)
        ctx.save_for_backward(grad, D, Q_gpu, E_gpu, torch.as_tensor([float(gamma)], device=device))

        return grad

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        device = grad_output.device
        _, D, Q, E, gamma = ctx.saved_tensors

        _, hessian = _dtw_path_hessian(
            D.detach().cpu().numpy().astype(np.float32),
            grad_output.detach().cpu().numpy().astype(np.float32),
            Q.detach().cpu().numpy().astype(np.float32),
            E.detach().cpu().numpy().astype(np.float32),
            float(gamma.item()),
        )
        hessian_t = torch.as_tensor(hessian, dtype=torch.float32).to(device)
        return hessian_t, None


# --------------------------------------------------------------------------- #
# DILATE and CCC-DILATE
# --------------------------------------------------------------------------- #
def _pairwise_distances(x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Squared Euclidean distances between the rows of ``x`` and ``y``."""
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    return torch.clamp(x_norm + y_norm - 2.0 * torch.mm(x, y_t), 0.0, float("inf"))


def _pairwise_distances_batched(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Squared distances between two ``[N, T, 1]`` batches of 1-D sequences."""
    batch_size, n_output = x.shape[0], x.shape[1]
    x_norm = (x ** 2).sum(2).view(batch_size, -1, 1)
    y_t = torch.transpose(y, 1, 2)
    y_norm = (y ** 2).sum(2).view(batch_size, 1, -1)
    dist = x_norm + y_norm - 2.0 * torch.bmm(
        x.view(batch_size, n_output, 1),
        y_t.contiguous().view(batch_size, 1, n_output),
    ).view(batch_size, n_output, n_output)
    return torch.clamp(dist, 0.0, float("inf"))


class DILATE(nn.Module):
    """Batch-wise DILATE loss: ``alpha * shape + (1 - alpha) * temporal``.

    Args:
        alpha: weight of the shape term versus the temporal term.
        gamma: soft-DTW smoothing temperature. Small values approach hard DTW.
        reduction: ``"mean"`` or ``"sum"`` over batch and channel.
    """

    def __init__(self, alpha: float = 0.5, gamma: float = 0.01, reduction: str = "mean"):
        super().__init__()
        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if not 0 <= gamma <= 1:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        Args:
            input: prediction ``[B, C, T]``.
            target: ground truth ``[B, C, T]``.

        Returns:
            ``(total, shape, temporal)``
        """
        if input.device != target.device:
            raise ValueError(
                f"input and target must be on the same device, got {input.device} and {target.device}"
            )
        if input.shape != target.shape:
            raise ValueError(
                f"input shape {tuple(input.shape)} and target shape {tuple(target.shape)} must match"
            )

        batch_size, n_channel, n_output = input.shape

        D = _pairwise_distances_batched(
            target.reshape(batch_size * n_channel, n_output, 1).double(),
            input.reshape(batch_size * n_channel, n_output, 1).double(),
        ).reshape(batch_size, n_channel, n_output, n_output)

        loss_shape = SoftDTWBatch.apply(D, self.gamma)

        path = PathDTWBatch.apply(D, self.gamma)
        # Temporal distortion of an alignment: squared distance to the diagonal.
        idx = torch.arange(1, n_output + 1, dtype=torch.float32, device=target.device)
        omega = _pairwise_distances(idx.view(n_output, 1)).repeat(n_channel, 1, 1)
        loss_temporal = torch.sum(path * omega, dim=(2, 3)) / (n_output * n_output)

        if self.reduction == "mean":
            loss_shape = torch.mean(loss_shape, dim=(0, 1))
            loss_temporal = torch.mean(loss_temporal, dim=(0, 1))
        elif self.reduction == "sum":
            loss_shape = torch.sum(loss_shape, dim=(0, 1))
            loss_temporal = torch.sum(loss_temporal, dim=(0, 1))
        else:
            raise ValueError(f"reduction must be 'mean' or 'sum', got {self.reduction}")

        total = self.alpha * loss_shape + (1 - self.alpha) * loss_temporal
        return total, loss_shape, loss_temporal


class CCCDILATELoss(nn.Module):
    """The paper's objective: ``L = ccc_weight * L_CCC + dilate_weight * L_DILATE``.

    DILATE operates on ``[B, C, T]``, so the ``[B, T, C]`` predictions are
    transposed before the temporal term is evaluated.

    Args:
        ccc_weight: weight of the concordance term (``1`` in the paper).
        dilate_weight: ``lambda`` in the paper, the strength of the temporal-shape
            regulariser.
        dilate_alpha: shape/temporal split inside DILATE (``0.5`` in the paper).
        dilate_gamma: soft-DTW smoothing temperature.
    """

    def __init__(
        self,
        ccc_weight: float = 1.0,
        dilate_weight: float = 1.0,
        dilate_alpha: float = 0.5,
        dilate_gamma: float = 0.01,
    ):
        super().__init__()
        self.ccc_loss = CCCLoss()
        self.dilate_loss = DILATE(alpha=dilate_alpha, gamma=dilate_gamma, reduction="mean")
        self.ccc_weight = ccc_weight
        self.dilate_weight = dilate_weight

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
        concordance = self.ccc_loss(outputs, targets)
        dilate, _, _ = self.dilate_loss(outputs.permute(0, 2, 1), targets.permute(0, 2, 1))
        return self.ccc_weight * concordance + self.dilate_weight * dilate
