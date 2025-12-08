"""PointNet-style encoder for 3D point clouds.

This module provides a simple PointNet-like encoder that maps an input
point cloud of shape [B, N, 3] to a global embedding of shape [B, out_dim].

The encoder is intentionally lightweight: a stack of pointwise MLPs
implemented as 1x1 convolutions followed by a symmetric max-pool over
points. This embedding is then used as conditioning for the CAD
Transformer backbone.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PointNetEncoder(nn.Module):
    """Minimal PointNet-style encoder.

    Parameters
    ----------
    in_dim : int
        Input feature dimension per point (3 for xyz).
    hidden_dim : int
        Hidden size for intermediate layers.
    out_dim : int
        Output embedding dimension.
    """

    def __init__(
        self,
        in_dim: int = 3,
        hidden_dim: int = 64,
        out_dim: int = 256,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        # Pointwise MLP via 1x1 convolutions: [B, in_dim, N] -> [B, out_dim, N]
        self.mlp = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim * 2, out_dim, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Encode a batch of point clouds.

        Parameters
        ----------
        points : FloatTensor [B, N, in_dim]
            Input point clouds.

        Returns
        -------
        embed : FloatTensor [B, out_dim]
            Global embedding per point cloud.
        """
        if points.dim() != 3:
            raise ValueError(f"points must have shape [B, N, C], got {points.shape}")
        B, N, C = points.shape
        if C != self.in_dim:
            raise ValueError(f"Expected in_dim={self.in_dim}, got {C}")

        # [B, N, C] -> [B, C, N]
        x = points.transpose(1, 2)
        x = self.mlp(x)             # [B, out_dim, N]
        x = torch.max(x, dim=2)[0]  # [B, out_dim] global max-pool
        return x


if __name__ == "__main__":  # pragma: no cover - simple smoke test
    B, N = 4, 256
    encoder = PointNetEncoder(in_dim=3, hidden_dim=64, out_dim=128)
    pts = torch.randn(B, N, 3)
    emb = encoder(pts)
    print("points shape:", pts.shape)
    print("embed shape:", emb.shape)