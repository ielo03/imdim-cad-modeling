from __future__ import annotations

import torch
import torch.nn as nn


class TokenHead(nn.Module):
    """Simple token prediction head.

    Parameters
    ----------
    d_model : int
        Dimensionality of the Transformer hidden states.
    vocab_size : int
        Size of the DSL token vocabulary.
    hidden_dim : int, optional
        Hidden size for the internal MLP. Defaults to 256.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.proj = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute token logits from hidden states.

        Parameters
        ----------
        hidden_states : FloatTensor [B, T, d_model]
            Transformer hidden states.

        Returns
        -------
        logits : FloatTensor [B, T, vocab_size]
            Logits over the DSL vocabulary for each position.
        """
        if hidden_states.dim() != 3:
            raise ValueError(
                f"hidden_states must have shape [B, T, D], got {hidden_states.shape}"
            )
        B, T, D = hidden_states.shape
        if D != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {D}")

        x = hidden_states.reshape(B * T, D)      # [B*T, D]
        logits = self.proj(x)                    # [B*T, V]
        logits = logits.view(B, T, self.vocab_size)
        return logits


if __name__ == "__main__":  # pragma: no cover - simple smoke test
    B, T, D = 2, 5, 64
    V = 16
    head = TokenHead(d_model=D, vocab_size=V, hidden_dim=32)
    h = torch.randn(B, T, D)
    logits = head(h)
    print("logits shape:", logits.shape)