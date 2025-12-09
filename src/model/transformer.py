"""Transformer backbone for the CAD DSL sequence model.

`CADTransformer` is a lightweight Transformer encoder that operates over
DSL token sequences and is conditioned on:

    - A global GT shape embedding (from PointNet).
    - A global current-state embedding (from PointNet).
    - A global error embedding (from geometry feedback).

The Transformer itself only produces hidden states for each token
position. Separate heads (e.g. TokenHead, ParamHead) consume these
hidden states to predict next tokens and primitive parameters.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class CADTransformer(nn.Module):
    """Transformer encoder over token sequences with optional conditioning.

    Parameters
    ----------
    vocab_size : int
        Size of the DSL token vocabulary.
    d_model : int
        Hidden size of the Transformer.
    n_heads : int
        Number of attention heads.
    num_layers : int
        Number of Transformer encoder layers.
    dim_feedforward : int
        Hidden size of the feedforward layers inside Transformer blocks.
    max_seq_len : int
        Maximum supported sequence length.
    dropout : float
        Dropout probability.
    gt_cond_dim : int, optional
        Dimensionality of GT embedding used for conditioning (None to disable).
    cur_cond_dim : int, optional
        Dimensionality of current-state embedding for conditioning.
    err_cond_dim : int, optional
        Dimensionality of error embedding for conditioning.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        max_seq_len: int = 64,
        dropout: float = 0.1,
        gt_cond_dim: Optional[int] = None,
        cur_cond_dim: Optional[int] = None,
        err_cond_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token + positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # [B, T, D]
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Conditioning projections
        self.gt_proj = nn.Linear(gt_cond_dim, d_model) if gt_cond_dim is not None else None
        self.cur_proj = nn.Linear(cur_cond_dim, d_model) if cur_cond_dim is not None else None
        self.err_proj = (
            nn.Linear(err_cond_dim, d_model)
            if (err_cond_dim is not None and err_cond_dim > 0)
            else None
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        gt_embed: Optional[torch.Tensor] = None,
        cur_embed: Optional[torch.Tensor] = None,
        err_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the Transformer.

        Parameters
        ----------
        tokens : LongTensor [B, T]
            Input DSL token ids (teacher-forced history).
        gt_embed : FloatTensor [B, H_gt], optional
            Global GT shape embedding.
        cur_embed : FloatTensor [B, H_cur], optional
            Global current-state embedding.
        err_embed : FloatTensor [B, H_err], optional
            Global error embedding.

        Returns
        -------
        hidden_states : FloatTensor [B, T, d_model]
            Transformer hidden states for each position.
        """
        if tokens.dim() != 2:
            raise ValueError(f"tokens must have shape [B, T], got {tokens.shape}")
        B, T = tokens.shape
        if T > self.max_seq_len:
            raise ValueError(f"Sequence length T={T} exceeds max_seq_len={self.max_seq_len}")

        device = tokens.device
        pos_ids = torch.arange(T, device=device).unsqueeze(0)  # [1, T]

        x = self.token_embedding(tokens) + self.pos_embedding(pos_ids)  # [B, T, d_model]

        # Conditioning: add broadcast projections
        if self.gt_proj is not None and gt_embed is not None:
            if gt_embed.dim() != 2 or gt_embed.shape[0] != B:
                raise ValueError(
                    f"gt_embed must be [B, H_gt] with B={B}, got {gt_embed.shape}"
                )
            gcond = self.gt_proj(gt_embed).unsqueeze(1)  # [B, 1, d_model]
            x = x + gcond

        if self.cur_proj is not None and cur_embed is not None:
            if cur_embed.dim() != 2 or cur_embed.shape[0] != B:
                raise ValueError(
                    f"cur_embed must be [B, H_cur] with B={B}, got {cur_embed.shape}"
                )
            ccond = self.cur_proj(cur_embed).unsqueeze(1)  # [B, 1, d_model]
            x = x + ccond

        if self.err_proj is not None and err_embed is not None:
            if err_embed.dim() != 2 or err_embed.shape[0] != B:
                raise ValueError(
                    f"err_embed must be [B, H_err] with B={B}, got {err_embed.shape}"
                )
            in_features = self.err_proj.in_features
            if err_embed.shape[1] != in_features:
                raise ValueError(
                    f"err_embed last dim {err_embed.shape[1]} does not match "
                    f"err_proj in_features={in_features}"
                )
            econd = self.err_proj(err_embed).unsqueeze(1)  # [B, 1, d_model]
            x = x + econd

        x = self.dropout(x)

        # With batch_first=True, TransformerEncoder expects [B, T, D]
        hidden_states = self.encoder(x)  # [B, T, d_model]
        return hidden_states


if __name__ == "__main__":  # pragma: no cover - simple smoke test
    vocab_size = 16
    d_model = 64
    B, T = 2, 5

    model = CADTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=4,
        num_layers=2,
        dim_feedforward=128,
        max_seq_len=16,
        gt_cond_dim=32,
        cur_cond_dim=32,
        err_cond_dim=8,
    )

    tokens = torch.randint(0, vocab_size, (B, T))
    gt_embed = torch.randn(B, 32)
    cur_embed = torch.randn(B, 32)
    err_embed = torch.randn(B, 8)

    hidden = model(tokens, gt_embed=gt_embed, cur_embed=cur_embed, err_embed=err_embed)
    print("hidden shape:", hidden.shape)
