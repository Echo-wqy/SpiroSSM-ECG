"""Selected morphology and diagnostic components, without training utilities."""

from typing import Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LeadRecalibration(nn.Module):
    """Residual lead gate for input shaped (batch, leads, time)."""

    def __init__(self, input_channels: int = 12, gamma: float = 0.10, hidden_ratio: float = 1.0):
        super().__init__()
        if input_channels < 1 or hidden_ratio <= 0 or gamma < 0:
            raise ValueError("Invalid lead-gate configuration")
        hidden = max(8, int(input_channels * hidden_ratio))
        self.input_channels = input_channels
        self.gamma = gamma
        self.net = nn.Sequential(
            nn.Linear(input_channels * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, input_channels),
        )
        self.reset_identity()

    def reset_identity(self) -> None:
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[1] != self.input_channels or x.shape[2] < 1:
            raise ValueError("Expected (batch, configured leads, nonempty time)")
        mean = x.mean(dim=-1)
        std = x.std(dim=-1, unbiased=False)
        gate = torch.tanh(self.net(torch.cat([mean, std], dim=1))).unsqueeze(-1)
        return x * (1.0 + self.gamma * gate)


def split_lead_groups(x: Tensor) -> Tuple[Tensor, Tensor]:
    """Split standard 12-lead input into limb and precordial lead groups."""
    if x.ndim != 3 or x.shape[1] != 12 or x.shape[2] < 1:
        raise ValueError("Expected (batch, 12, nonempty time)")
    return x[:, :6, :], x[:, 6:12, :]


class LeadGroupEncoder(nn.Module):
    """Multi-scale temporal encoding for one lead group."""

    def __init__(self, in_channels: int = 6, out_channels: int = 64):
        super().__init__()
        if in_channels < 1 or out_channels < 3:
            raise ValueError("Expected positive inputs and at least three output channels")
        small = out_channels // 3
        remainder = out_channels - 2 * small
        self.in_channels = in_channels
        self.conv3 = nn.Conv1d(in_channels, small, 3, padding=1, bias=False)
        self.conv7 = nn.Conv1d(in_channels, small, 7, padding=3, bias=False)
        self.conv15 = nn.Conv1d(in_channels, remainder, 15, padding=7, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[1] != self.in_channels or x.shape[2] < 1:
            raise ValueError("Expected (batch, configured channels, nonempty time)")
        features = torch.cat([self.conv3(x), self.conv7(x), self.conv15(x)], dim=1)
        return F.gelu(self.bn(features))


def ternary_events(membrane: Tensor, threshold: Union[Tensor, float]) -> Tensor:
    """Forward-only threshold rule; no adaptive state or surrogate backward."""
    if not membrane.is_floating_point():
        raise ValueError("Expected a floating-point membrane tensor")
    threshold = torch.as_tensor(threshold, dtype=membrane.dtype, device=membrane.device)
    if not torch.isfinite(threshold).all() or not (threshold > 0).all():
        raise ValueError("Thresholds must be finite and positive")
    try:
        torch.broadcast_shapes(membrane.shape, threshold.shape)
    except RuntimeError as exc:
        raise ValueError("Thresholds must broadcast to the membrane tensor") from exc
    if torch.broadcast_shapes(membrane.shape, threshold.shape) != membrane.shape:
        raise ValueError("Thresholds must not expand the membrane shape")
    one = torch.ones_like(membrane)
    zero = torch.zeros_like(membrane)
    return torch.where(membrane > threshold, one, torch.where(membrane < -threshold, -one, zero))


class TemporalAttention(nn.Module):
    """Return normalized temporal weights for (batch, time, features)."""

    def __init__(self, d_model: int = 128, hidden_ratio: float = 0.5):
        super().__init__()
        if d_model < 1 or hidden_ratio <= 0:
            raise ValueError("Invalid temporal-attention configuration")
        self.d_model = d_model
        hidden = max(16, int(d_model * hidden_ratio))
        self.attn = nn.Sequential(nn.Linear(d_model, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.d_model or x.shape[1] < 1:
            raise ValueError("Expected (batch, nonempty time, configured features)")
        return torch.softmax(self.attn(x), dim=1)


class LowRankLabelCorrection(nn.Module):
    """Residual label correction applied to existing class logits."""

    def __init__(self, num_classes: int = 5, rank: int = 4, initial_scale: float = 0.10):
        super().__init__()
        if not 0 < rank < num_classes or not 0 <= initial_scale <= 0.5:
            raise ValueError("Expected 0 < rank < classes and a scale in [0, 0.5]")
        self.num_classes = num_classes
        self.corr_down = nn.Linear(num_classes, rank, bias=False)
        self.corr_up = nn.Linear(rank, num_classes, bias=False)
        self.corr_scale = nn.Parameter(torch.tensor(initial_scale))
        nn.init.zeros_(self.corr_up.weight)

    def forward(self, logits: Tensor) -> Tensor:
        if logits.ndim != 2 or logits.shape[-1] != self.num_classes:
            raise ValueError("Expected (batch, configured classes)")
        correction = self.corr_up(F.gelu(self.corr_down(torch.tanh(logits))))
        return logits + self.corr_scale.clamp(0.0, 0.5) * correction
