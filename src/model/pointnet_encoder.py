

"""PointNet-style encoder for GT point clouds.

This module encodes a set of 3D points (e.g., sampled from a ground-truth
mesh) into a single fixed-size embedding vector. It is intentionally minimal:

    - Per-point MLP (shared across points)
    - Global max or mean pooling

Shape conventions:

    points: [B, N, 3]
    output: [B, out_dim]

This encoder is used to produce `gt_embed` for conditioning the Transformer
(backbone) and the parameter MLP.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


class PointNetEncoder(nn.Module):
    """Minimal PointNet-style point cloud encoder.

    Parameters
    ----------
    in_dim : int
        Dimensionality of each point (3 for xyz).
    hidden_dim : int
        Hidden size of per-point MLP layers.
    out_dim : int
        Output embedding dimensionality.
    num_layers : int
        Number of linear layers in the per-point MLP (>= 2).
    activation : callable
        Nonlinearity to use between layers (default: ReLU).
    pool : {"max", "mean"}
        Global pooling type over the point dimension.
    """

    def __init__(
        self,
        in_dim: int = 3,
        hidden_dim: int = 128,
        out_dim: int = 256,
        num_layers: int = 3,
        activation: nn.Module | None = None,
        pool: Literal["max", "mean"] = "max",
    ) -> None:
        super().__init__()

        if num_layers < 2:
            raise ValueError("PointNetEncoder requires num_layers >= 2")

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.pool = pool

        if activation is None:
            activation = nn.ReLU()
        self.activation = activation

        layers = []
        # First layer: in_dim -> hidden_dim
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(self.activation)

        # Middle layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self.activation)

        # Final layer: hidden_dim -> out_dim
        layers.append(nn.Linear(hidden_dim, out_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Encode a batch of point clouds.

        Parameters
        ----------
        points : FloatTensor of shape [B, N, in_dim]
            Batch of point sets.

        Returns
        -------
        gt_embed : FloatTensor of shape [B, out_dim]
            Global embedding for each point cloud.
        """

        if points.dim() != 3:
            raise ValueError(f"points must have shape [B, N, C], got {points.shape}")
        B, N, C = points.shape
        if C != self.in_dim:
            raise ValueError(
                f"Expected last dim in points to be {self.in_dim}, got {C}"
            )

        # Flatten points to [B * N, C] for shared MLP
        pts_flat = points.view(B * N, C)
        feats = self.mlp(pts_flat)  # [B * N, out_dim]
        feats = feats.view(B, N, self.out_dim)

        if self.pool == "max":
            gt_embed, _ = feats.max(dim=1)  # [B, out_dim]
        elif self.pool == "mean":
            gt_embed = feats.mean(dim=1)
        else:
            raise ValueError(f"Unknown pool type: {self.pool!r}")

        return gt_embed


if __name__ == "__main__":  # pragma: no cover - simple smoke test
    encoder = PointNetEncoder(in_dim=3, hidden_dim=64, out_dim=128, num_layers=3)
    B, N = 4, 256
    pts = torch.randn(B, N, 3)
    emb = encoder(pts)
    print("points shape:", pts.shape)
    print("embed shape:", emb.shape)