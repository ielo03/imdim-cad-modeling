"""Transformer backbone for the CAD DSL policy.

This module implements a small, decoder-style Transformer that operates over
sequences of DSL tokens. It provides:

    - Token embeddings + learned positional embeddings
    - Optional conditioning on a global GT-geometry embedding (`gt_embed`)
    - A stack of Transformer encoder layers used in causal (autoregressive) mode
    - A token head that predicts the next token logits at each position

The key thing we care about for parameter pretraining and RL is the hidden
state `h_t` at each step, which is returned as `hidden_states`.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def _build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Return an [seq_len, seq_len] causal mask for autoregressive attention.

    mask[i, j] = 0   if j <= i (can attend to self and past)
    mask[i, j] = -inf if j > i (cannot attend to future)
    """

    # PyTorch Transformer expects additive mask with -inf for disallowed.
    # We construct an upper-triangular matrix with -inf above the diagonal.
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask


class CADTransformer(nn.Module):
    """Autoregressive Transformer over DSL token sequences.

    This is a GPT-style encoder-only transformer:

        - Input: token ids [B, T]
        - Optional conditioning on GT geometry via `gt_embed` [B, H_gt]
        - Output:
            hidden_states: [B, T, d_model]
            token_logits:  [B, T, vocab_size]

    The hidden states are consumed by the parameter head. The token logits are
    used for the token policy (RL) or teacher forcing during pretraining.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        max_seq_len: int = 64,
        dropout: float = 0.1,
        gt_cond_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token + positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Optional linear projection for GT conditioning
        self.gt_cond_dim = gt_cond_dim
        if gt_cond_dim is not None:
            self.gt_proj = nn.Linear(gt_cond_dim, d_model)
        else:
            self.gt_proj = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.token_head = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        tokens: torch.Tensor,
        gt_embed: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        tokens : LongTensor of shape [B, T]
            Sequence of DSL token ids.
        gt_embed : FloatTensor of shape [B, H_gt], optional
            Global GT-geometry embedding (e.g., from PointNet). If provided
            and `gt_cond_dim` is set, this is projected and added to all
            token embeddings.

        Returns
        -------
        hidden_states : FloatTensor [B, T, d_model]
        token_logits  : FloatTensor [B, T, vocab_size]
        """

        if tokens.dim() != 2:
            raise ValueError(f"tokens must have shape [B, T], got {tokens.shape}")

        B, T = tokens.shape
        device = tokens.device

        if T > self.max_seq_len:
            raise ValueError(
                f"Sequence length {T} exceeds max_seq_len={self.max_seq_len}. "
                f"Increase max_seq_len when constructing CADTransformer."
            )

        # Token + positional embeddings
        tok_emb = self.token_embedding(tokens)  # [B, T, d_model]
        pos_ids = torch.arange(T, device=device).unsqueeze(0)  # [1, T]
        pos_emb = self.pos_embedding(pos_ids)  # [1, T, d_model]
        x = tok_emb + pos_emb

        # Optional GT conditioning: add a broadcasted projection
        if self.gt_proj is not None and gt_embed is not None:
            if gt_embed.dim() != 2:
                raise ValueError(
                    f"gt_embed must have shape [B, H_gt], got {gt_embed.shape}"
                )
            if gt_embed.shape[0] != B:
                raise ValueError(
                    f"Batch size mismatch between tokens (B={B}) and gt_embed "
                    f"(B={gt_embed.shape[0]})"
                )
            cond = self.gt_proj(gt_embed)  # [B, d_model]
            cond = cond.unsqueeze(1)  # [B, 1, d_model]
            x = x + cond

        # Causal mask so position t cannot attend to > t
        causal_mask = _build_causal_mask(T, device=device)

        hidden_states = self.transformer(x, mask=causal_mask)  # [B, T, d_model]
        token_logits = self.token_head(hidden_states)  # [B, T, vocab_size]

        return hidden_states, token_logits


if __name__ == "__main__":  # pragma: no cover - simple smoke test
    # Minimal smoke test: random tokens through the transformer
    vocab_size = 16
    model = CADTransformer(vocab_size=vocab_size, d_model=64, n_heads=4, num_layers=2)

    B, T = 2, 5
    tokens = torch.randint(0, vocab_size, (B, T))
    gt_embed = torch.randn(B, 32)

    hidden, logits = model(tokens, gt_embed=None)
    print("hidden shape:", hidden.shape)
    print("logits shape:", logits.shape)
