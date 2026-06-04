"""
Vectorized discretization for spectroscopic scalars (intensity, FWHM, etc.).

Maps continuous targets to bin indices for cross-entropy MIP and back to
mid-bin continuous values for validation metrics (MAE, R²).
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn

Strategy = Literal["log", "uniform"]

_LOG_EPS = 1e-4


class SpectroscopicDiscretizer(nn.Module):
    """
    Discretize continuous spectroscopic scalars into classification bins.

    Args:
        num_bins: Number of discrete classes.
        min_val: Lower bound of the physical range (values are clamped here).
        max_val: Upper bound of the physical range.
        strategy: ``log`` for power-scaling quantities (e.g. intensity);
            ``uniform`` for symmetric quantities (e.g. FWHM).
    """

    def __init__(
        self,
        num_bins: int = 256,
        min_val: float = 0.0,
        max_val: float = 1.0,
        strategy: Strategy = "log",
    ):
        super().__init__()
        if num_bins < 1:
            raise ValueError(f"num_bins must be >= 1, got {num_bins}")
        if max_val <= min_val:
            raise ValueError(f"max_val ({max_val}) must be > min_val ({min_val})")

        self.num_bins = int(num_bins)
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.strategy = strategy

        boundaries = self._build_boundaries()
        self.register_buffer("boundaries", boundaries, persistent=True)

    def _build_boundaries(self) -> torch.Tensor:
        n = self.num_bins
        lo, hi = self.min_val, self.max_val
        if self.strategy == "log":
            lo_safe = lo + _LOG_EPS
            hi_safe = hi + _LOG_EPS
            boundaries = torch.logspace(
                math.log10(lo_safe),
                math.log10(hi_safe),
                n + 1,
                dtype=torch.float32,
            )
        elif self.strategy == "uniform":
            boundaries = torch.linspace(lo, hi, n + 1, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown strategy {self.strategy!r}; use 'log' or 'uniform'")
        return boundaries

    def to_bins(self, x: torch.Tensor) -> torch.Tensor:
        """Map continuous values to bin indices in ``[0, num_bins - 1]``."""
        x = x.to(dtype=self.boundaries.dtype, device=self.boundaries.device)
        x = x.clamp(self.min_val, self.max_val)
        # Internal edges: 255 edges for 256 bins → indices 0..255
        edges = self.boundaries[1:-1]
        return torch.bucketize(x.contiguous(), edges.contiguous()).long()

    def to_continuous(self, bins: torch.Tensor) -> torch.Tensor:
        """Map bin indices to mid-bin continuous reconstructions."""
        bins = bins.long().clamp(0, self.num_bins - 1)
        lo = self.boundaries[bins]
        hi = self.boundaries[bins + 1]
        return (lo + hi) * 0.5

    def to(self, *args, **kwargs):
        out = super().to(*args, **kwargs)
        return out


def build_discretizer_from_config(cfg: dict) -> SpectroscopicDiscretizer:
    """Construct a discretizer from a YAML-style dict."""
    return SpectroscopicDiscretizer(
        num_bins=int(cfg.get("num_bins", 256)),
        min_val=float(cfg.get("min_val", 0.0)),
        max_val=float(cfg.get("max_val", 1.0)),
        strategy=str(cfg.get("strategy", "log")),
    )
