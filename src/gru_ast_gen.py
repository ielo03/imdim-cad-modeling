"""GRU-based policy network for AST generation.

This module defines a simple GRU-based policy/value network that can be used
in an RL loop to incrementally build an AST using `ASTBuilder` and `Action`
from `ast_gen_utils`.

The model:
  - Encodes the target image with a small CNN.
  - Encodes the action history as a sequence of integer action IDs with an
    embedding + GRU.
  - Combines image and history features and outputs:
      * logits over the next action (discrete Action enum)
      * a scalar state-value estimate (for PPO/actor-critic)

This file intentionally does NOT implement a full RL loop; it just gives you
an architecture you can plug into PPO/REINFORCE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports
try:
    from .ast_gen_utils import Action
except ImportError:  # fallback for direct invocation
    from ast_gen_utils import Action


@dataclass
class PolicyOutput:
    """Container for a single forward pass output.

    Attributes
    ----------
    logits : torch.Tensor
        Unnormalized log-probabilities over actions, shape [B, num_actions].
    value : torch.Tensor
        State-value estimate, shape [B].
    hidden : torch.Tensor
        GRU hidden state, shape [1, B, hidden_dim]. Can be fed back in for
        step-wise rollout if you don't want to re-encode the whole sequence.
    """

    logits: torch.Tensor
    value: torch.Tensor
    hidden: torch.Tensor


class GRUPolicyNet(nn.Module):
    """Image-conditioned GRU policy for discrete Action selection.

    Parameters
    ----------
    num_actions : int
        Number of discrete actions (should be `len(Action)`).
    img_channels : int, default 1
        Number of channels in the input image (1=grayscale, 3=RGB).
    img_size : int, default 64
        Assumed square input size (H=W=img_size). Adjust if needed.
    hidden_dim : int, default 256
        Hidden dimension for GRU and fully-connected layers.
    action_embed_dim : int, default 64
        Embedding dimension for discrete action IDs.
    """

    def __init__(
        self,
        num_actions: int,
        img_channels: int = 1,
        img_size: int = 64,
        hidden_dim: int = 256,
        action_embed_dim: int = 64,
    ) -> None:
        super().__init__()

        self.num_actions = num_actions
        self.hidden_dim = hidden_dim

        # ---------------------
        # Image encoder (simple CNN)
        # ---------------------
        self.img_conv = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/4
        )

        # Compute flattened feature size assuming square images of img_size
        with torch.no_grad():
            dummy = torch.zeros(1, img_channels, img_size, img_size)
            feat = self.img_conv(dummy)
            conv_flat_dim = feat.view(1, -1).size(1)

        self.img_fc = nn.Linear(conv_flat_dim, hidden_dim)

        # ---------------------
        # Action history encoder (Embedding + GRU)
        # ---------------------
        self.action_emb = nn.Embedding(num_actions + 1, action_embed_dim)
        # +1 in case you want to reserve 0 for a PAD/BOS token

        self.gru = nn.GRU(
            input_size=action_embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # ---------------------
        # Fusion + heads
        # ---------------------
        self.fuse_fc = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        img: torch.Tensor,
        action_seq: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> PolicyOutput:
        """Forward pass.

        Parameters
        ----------
        img : torch.Tensor
            Batch of target images, shape [B, C, H, W]. H,W should match the
            size passed at construction time (or be downsampled accordingly).
        action_seq : torch.Tensor
            Integer tensor of shape [B, T] containing action IDs. These should
            correspond to the `Action` enum values (or a remapped contiguous
            1..N index).
        hidden : torch.Tensor, optional
            Initial hidden state for the GRU, shape [1, B, hidden_dim]. If
            None, the GRU is initialized with zeros.

        Returns
        -------
        PolicyOutput
            logits, value, new_hidden.
        """

        # Image branch
        # -------------
        x_img = self.img_conv(img)
        x_img = x_img.view(x_img.size(0), -1)
        x_img = F.relu(self.img_fc(x_img))  # [B, hidden_dim]

        # Action sequence branch
        # ----------------------
        # Expect action_seq as ints in [0, num_actions], where 0 can be PAD/BOS.
        x_act = self.action_emb(action_seq)  # [B, T, action_embed_dim]
        _, h = self.gru(x_act, hidden)       # h: [1, B, hidden_dim]
        x_hist = h.squeeze(0)                # [B, hidden_dim]

        # Fuse
        # ----
        x = torch.cat([x_img, x_hist], dim=1)  # [B, 2*hidden_dim]
        x = F.relu(self.fuse_fc(x))            # [B, hidden_dim]

        logits = self.policy_head(x)           # [B, num_actions]
        value = self.value_head(x).squeeze(-1) # [B]

        return PolicyOutput(logits=logits, value=value, hidden=h)


def num_actions_from_enum() -> int:
    """Utility to get `num_actions` from the Action enum.

    Assumes Action values are contiguous or at least that max(Action) is safe
    to use as the number of actions when you map IDs appropriately.
    """
    return len(Action)


if __name__ == "__main__":  # pragma: no cover - simple manual test
    """Minimal smoke test.

    Runs a forward pass with dummy image + action sequence and prints shapes.
    """

    # Example: grayscale 64x64 image, batch size 2, action sequence length 5
    B, C, H, W = 2, 1, 64, 64
    T = 5

    num_actions = num_actions_from_enum()
    net = GRUPolicyNet(num_actions=num_actions, img_channels=C, img_size=H)

    dummy_img = torch.randn(B, C, H, W)
    # Dummy action history: just use 1..num_actions range wrapped around
    action_ids = torch.randint(low=0, high=num_actions + 1, size=(B, T))

    out = net(dummy_img, action_ids)

    print("logits shape:", out.logits.shape)  # [B, num_actions]
    print("value shape:", out.value.shape)    # [B]
    print("hidden shape:", out.hidden.shape)  # [1, B, hidden_dim]
