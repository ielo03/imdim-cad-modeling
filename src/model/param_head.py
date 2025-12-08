

"""Parameter head for continuous primitive arguments.

This module maps Transformer hidden state + GT-shape embedding + token
identity (and optionally an error embedding) to continuous primitive
parameters.

The intended usage per decoding step t is:

    - `h_t`      : hidden state from CADTransformer, shape [B, d_model]
    - `gt_embed` : PointNetEncoder output,        shape [B, gt_dim]
    - `token_t`  : DSL token ids,                 shape [B]
    - `err_emb`  : optional error embedding,      shape [B, err_dim]

We embed `token_t`, concatenate everything, and pass through an MLP to
produce a fixed-size parameter vector. The caller interprets the output
according to the token type (e.g., first 3 dims = center, next 3 = size
for a box, etc.).

A typical use case in this project is out_dim=10, where:
    - First 9 components are geometry parameters,
    - 10th component is a sign scalar used for positive/negative role.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ParamHead(nn.Module):
    """MLP for predicting continuous primitive parameters.

    Parameters
    ----------
    d_model : int
        Dimensionality of the Transformer hidden state `h_t`.
    gt_dim : int
        Dimensionality of the GT-shape embedding from PointNetEncoder.
    vocab_size : int
        Size of the DSL token vocabulary (for token embeddings).
    tok_emb_dim : int, optional
        Dimensionality of the internal token embedding. If None, defaults
        to `d_model`.
    out_dim : int
        Dimensionality of the parameter vector output. The caller is
        responsible for interpreting slices of this vector per token type.
        In this project, out_dim is often 10 (9 geom + 1 sign), but is configurable.
    hidden_dim : int
        Hidden size for the MLP.
    err_dim : int
        Dimensionality of an optional error embedding. Set to 0 to ignore.
    """

    def __init__(
        self,
        d_model: int,
        gt_dim: int,
        vocab_size: int,
        tok_emb_dim: Optional[int] = None,
        out_dim: int = 8,
        hidden_dim: int = 256,
        err_dim: int = 0,
    ) -> None:
        super().__init__()

        if tok_emb_dim is None:
            tok_emb_dim = d_model

        self.d_model = d_model
        self.gt_dim = gt_dim
        self.vocab_size = vocab_size
        self.tok_emb_dim = tok_emb_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.err_dim = err_dim

        # Token embedding for the current DSL token.
        self.token_embedding = nn.Embedding(vocab_size, tok_emb_dim)

        in_dim = d_model + gt_dim + tok_emb_dim
        if err_dim > 0:
            in_dim += err_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        h_t: torch.Tensor,
        gt_embed: torch.Tensor,
        token_t: torch.Tensor,
        err_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict primitive parameters for a single decoding step.

        Parameters
        ----------
        h_t : FloatTensor [B, d_model]
            Hidden state from CADTransformer at step t.
        gt_embed : FloatTensor [B, gt_dim]
            Global GT-shape embedding.
        token_t : LongTensor [B]
            Token ids for the DSL action chosen/emitted at step t.
        err_emb : FloatTensor [B, err_dim], optional
            Optional error embedding summarizing current pred-vs-GT
            geometry. If provided, must have the configured err_dim.

        Returns
        -------
        params_pred : FloatTensor [B, out_dim]
            Predicted continuous parameter vector for this step.
        """

        if h_t.dim() != 2:
            raise ValueError(f"h_t must have shape [B, d_model], got {h_t.shape}")
        if gt_embed.dim() != 2:
            raise ValueError(
                f"gt_embed must have shape [B, gt_dim], got {gt_embed.shape}"
            )
        if token_t.dim() != 1:
            raise ValueError(f"token_t must have shape [B], got {token_t.shape}")

        B, d_model = h_t.shape
        if d_model != self.d_model:
            raise ValueError(
                f"Expected h_t dim {self.d_model}, got {d_model}"
            )
        if gt_embed.shape[0] != B:
            raise ValueError(
                f"Batch size mismatch between h_t (B={B}) and gt_embed "
                f"(B={gt_embed.shape[0]})"
            )
        if token_t.shape[0] != B:
            raise ValueError(
                f"Batch size mismatch between h_t (B={B}) and token_t "
                f"(B={token_t.shape[0]})"
            )

        tok_emb = self.token_embedding(token_t)  # [B, tok_emb_dim]

        pieces = [h_t, gt_embed, tok_emb]

        if self.err_dim > 0:
            if err_emb is None:
                raise ValueError("err_emb is required but None was provided")
            if err_emb.dim() != 2 or err_emb.shape[1] != self.err_dim:
                raise ValueError(
                    f"err_emb must have shape [B, {self.err_dim}], got {err_emb.shape}"
                )
            if err_emb.shape[0] != B:
                raise ValueError(
                    f"Batch size mismatch between h_t (B={B}) and err_emb "
                    f"(B={err_emb.shape[0]})"
                )
            pieces.append(err_emb)

        x = torch.cat(pieces, dim=-1)  # [B, in_dim]
        params_pred = self.mlp(x)      # [B, out_dim]
        return params_pred


if __name__ == "__main__":  # pragma: no cover - simple smoke test
    B = 4
    d_model = 64
    gt_dim = 128
    vocab_size = 16
    err_dim = 0

    head = ParamHead(d_model=d_model, gt_dim=gt_dim, vocab_size=vocab_size,
                     tok_emb_dim=32, out_dim=10, hidden_dim=128, err_dim=err_dim)

    h_t = torch.randn(B, d_model)
    gt_embed = torch.randn(B, gt_dim)
    token_t = torch.randint(0, vocab_size, (B,))

    params = head(h_t, gt_embed, token_t)
    print("params shape:", params.shape)