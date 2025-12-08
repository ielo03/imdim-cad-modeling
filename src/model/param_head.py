"""Parameter head for continuous primitive arguments.

This module maps a Transformer hidden state + token identity to
continuous primitive parameters. All other conditioning (GT-shape
embedding, current-state geometry, error embedding, etc.) is assumed to
have been folded into the hidden state `h_t` by the backbone.

The intended usage per decoding step t is:

    - `h_t`      : hidden state from CADTransformer, shape [B, d_model]
    - `token_t`  : DSL token ids for the action at step t, shape [B]

We embed `token_t`, concatenate with `h_t`, and pass through an MLP to
produce a fixed-size parameter vector. The caller interprets the output
according to the token type (e.g., first 3 dims = center, next 3 = size
for a box, etc.).
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
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        tok_emb_dim: Optional[int] = None,
        out_dim: int = 8,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        if tok_emb_dim is None:
            tok_emb_dim = d_model

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.tok_emb_dim = tok_emb_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        # Token embedding for the current DSL token.
        self.token_embedding = nn.Embedding(vocab_size, tok_emb_dim)

        in_dim = d_model + tok_emb_dim

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
        token_t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict primitive parameters for a single decoding step.

        Parameters
        ----------
        h_t : FloatTensor [B, d_model]
            Hidden state from CADTransformer at step t.
        token_t : LongTensor [B]
            Token ids for the DSL action chosen/emitted at step t.

        Returns
        -------
        params_pred : FloatTensor [B, out_dim]
            Predicted continuous parameter vector for this step.
        """

        if h_t.dim() != 2:
            raise ValueError(f"h_t must have shape [B, d_model], got {h_t.shape}")
        if token_t.dim() != 1:
            raise ValueError(f"token_t must have shape [B], got {token_t.shape}")

        B, d_model = h_t.shape
        if d_model != self.d_model:
            raise ValueError(
                f"Expected h_t dim {self.d_model}, got {d_model}"
            )
        if token_t.shape[0] != B:
            raise ValueError(
                f"Batch size mismatch between h_t (B={B}) and token_t "
                f"(B={token_t.shape[0]})"
            )

        tok_emb = self.token_embedding(token_t)  # [B, tok_emb_dim]

        pieces = [h_t, tok_emb]

        x = torch.cat(pieces, dim=-1)  # [B, in_dim]
        params_pred = self.mlp(x)      # [B, out_dim]
        return params_pred


if __name__ == "__main__":  # pragma: no cover - simple smoke test
    B = 4
    d_model = 64
    vocab_size = 16

    head = ParamHead(d_model=d_model, vocab_size=vocab_size,
                     tok_emb_dim=32, out_dim=10, hidden_dim=128)

    h_t = torch.randn(B, d_model)
    token_t = torch.randint(0, vocab_size, (B,))

    params = head(h_t, token_t)
    print("params shape:", params.shape)