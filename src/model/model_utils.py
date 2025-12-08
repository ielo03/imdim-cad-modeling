"""Shared utilities for the CAD sequence model.

This module provides small helper functions that glue together the
PointNet encoder, Transformer backbone, parameter head, and the
`ShapeState` from the DSL/state machine.

The goals are:

    - Centralize how we build a consistent model stack (dims, vocab).
    - Provide a clean way to run a forward pass for param pretraining.
    - Provide a helper to apply a single model step to a ShapeState.

The core convention for parameters is a fixed 10D vector:

    params_vec = [cx, cy, cz, p0, p1, p2, rx, ry, rz, sign_raw]

`ShapeState.apply` is responsible for interpreting (p0, p1, p2)
according to the current token. The 10th entry, sign_raw, is used to decide the primitive sign.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import random

# Ensure the project src root is on sys.path so this module can be run
# both as part of the `model` package and as a standalone script.
_HERE = Path(__file__).resolve()
_SRC_ROOT = _HERE.parents[1]  # .../src
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from typing import Optional, Sequence, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn

from model.pointnet_encoder import PointNetEncoder
from model.transformer import CADTransformer
from model.param_head import ParamHead
from state_machine import ShapeState, Token


@dataclass
class ModelComponents:
    """Bundle of core model components for the CAD policy.

    Attributes
    ----------
    pointnet : PointNetEncoder
        Encodes GT point clouds into `gt_embed`.
    transformer : CADTransformer
        Autoregressive backbone over DSL tokens.
    param_head : ParamHead
        Predicts the 10D parameter vector at each step (9 geom + sign).
    """

    pointnet: PointNetEncoder
    transformer: CADTransformer
    param_head: ParamHead

    def to(self, device: torch.device | str) -> "ModelComponents":
        """Move all submodules to a given device and return self."""

        dev = torch.device(device)
        self.pointnet.to(dev)
        self.transformer.to(dev)
        self.param_head.to(dev)
        return self


def build_cad_model(
    vocab_size: int,
    gt_dim: int = 256,
    d_model: int = 256,
    n_heads: int = 4,
    num_layers: int = 4,
    max_seq_len: int = 64,
    tok_emb_dim: Optional[int] = None,
    hidden_dim: int = 256,
    err_dim: int = 0,
    device: Optional[torch.device | str] = None,
) -> ModelComponents:
    """Construct a consistent set of model components for the CAD task.

    Parameters
    ----------
    vocab_size : int
        Number of DSL tokens in the vocabulary.
    gt_dim : int
        Output embedding size of PointNetEncoder and conditioning dim
        for the Transformer.
    d_model : int
        Transformer hidden size.
    n_heads : int
        Number of attention heads.
    num_layers : int
        Number of Transformer encoder layers.
    max_seq_len : int
        Maximum token sequence length.
    tok_emb_dim : int, optional
        Internal token embedding size for ParamHead. Defaults to d_model.
    hidden_dim : int
        Hidden size of the ParamHead MLP.
    err_dim : int
        Size of optional error embedding for ParamHead (0 to disable).
    device : torch.device or str, optional
        If provided, all modules are moved to this device.

    Returns
    -------
    ModelComponents
        A simple dataclass bundling pointnet, transformer, and param_head.
    """

    pointnet = PointNetEncoder(in_dim=3, hidden_dim=gt_dim // 2, out_dim=gt_dim)

    transformer = CADTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        dim_feedforward=2 * d_model,
        max_seq_len=max_seq_len,
        dropout=0.1,
        gt_cond_dim=gt_dim,
    )

    param_head = ParamHead(
        d_model=d_model,
        gt_dim=gt_dim,
        vocab_size=vocab_size,
        tok_emb_dim=tok_emb_dim or d_model,
        out_dim=10,           # [cx, cy, cz, p0, p1, p2, rx, ry, rz, sign_raw]
        hidden_dim=hidden_dim,
        err_dim=err_dim,
    )

    components = ModelComponents(pointnet=pointnet, transformer=transformer, param_head=param_head)
    if device is not None:
        components.to(device)
    return components


def forward_params_only(
    components: ModelComponents,
    gt_points: torch.Tensor,
    tokens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a forward pass that predicts per-step parameters.

    This is intended for param pretraining and analysis. It:

        1. Encodes GT points with PointNet → gt_embed
        2. Runs the Transformer with teacher-forced tokens → hidden_states
        3. Applies the ParamHead at each step → params_pred

    Parameters
    ----------
    components : ModelComponents
        The model bundle from `build_cad_model`.
    gt_points : FloatTensor [B, N, 3]
        Ground-truth point clouds.
    tokens : LongTensor [B, T]
        Teacher-forced DSL token ids.

    Returns
    -------
    params_pred : FloatTensor [B, T, 10]
        Predicted parameter vectors per step.
    hidden_states : FloatTensor [B, T, d_model]
        Transformer hidden states.
    token_logits : FloatTensor [B, T, vocab_size]
        Token logits from the Transformer (can be ignored for pure
        param pretraining, or used for auxiliary losses).
    """

    pointnet, transformer, param_head = (
        components.pointnet,
        components.transformer,
        components.param_head,
    )

    if gt_points.dim() != 3:
        raise ValueError(f"gt_points must have shape [B, N, 3], got {gt_points.shape}")
    if tokens.dim() != 2:
        raise ValueError(f"tokens must have shape [B, T], got {tokens.shape}")

    gt_embed = pointnet(gt_points)  # [B, gt_dim]
    hidden_states, token_logits = transformer(tokens, gt_embed=gt_embed)

    B, T, d_model = hidden_states.shape
    params_list = []

    for t in range(T):
        h_t = hidden_states[:, t, :]        # [B, d_model]
        tok_t = tokens[:, t]               # [B]
        params_t = param_head(h_t, gt_embed, tok_t)  # [B, 10]
        params_list.append(params_t.unsqueeze(1))

    params_pred = torch.cat(params_list, dim=1)  # [B, T, 10]
    return params_pred, hidden_states, token_logits


def apply_model_step(
    state: ShapeState,
    token_id: int | Token,
    params_vec: torch.Tensor | Sequence[float],
) -> None:
    """Apply a single model decision to a ShapeState.

    Parameters
    ----------
    state : ShapeState
        The mutable program state.
    token_id : int or Token
        DSL token to apply. If an int is provided, it is cast to `Token`.
    params_vec : Tensor or sequence of length 10
        Predicted parameter vector for this step; interpreted as
        [cx, cy, cz, p0, p1, p2, rx, ry, rz, sign_raw]. `ShapeState.apply` will
        handle the token-specific interpretation of p0/p1/p2 and sign_raw.
    """

    if isinstance(token_id, int):
        token = Token(token_id)
    else:
        token = token_id

    if isinstance(params_vec, torch.Tensor):
        if params_vec.ndim == 2 and params_vec.shape[0] == 1:
            params_vec = params_vec[0]
        if params_vec.ndim != 1 or params_vec.shape[0] != 10:
            raise ValueError(
                f"params_vec tensor must have shape [10] or [1,10], got {tuple(params_vec.shape)}"
            )
        params = params_vec.detach().cpu().tolist()
    else:
        params = list(params_vec)
        if len(params) != 10:
            raise ValueError(f"params_vec sequence must have length 10, got {len(params)}")

    state.apply(token, params)


if __name__ == "__main__":  # pragma: no cover - simple smoke test
    # Tiny smoke test wiring everything together with fake data.
    vocab_size = len(Token)
    components = build_cad_model(vocab_size=vocab_size, gt_dim=32, d_model=32, max_seq_len=8)

    B, N, T = 2, 64, 5
    gt_points = torch.randn(B, N, 3)
    tokens = torch.randint(0, vocab_size, (B, T))

    params_pred, hidden_states, token_logits = forward_params_only(components, gt_points, tokens)
    print("params_pred shape:", params_pred.shape)

    # Single-step application demo: randomly choose a primitive-adding token.
    s = ShapeState()
    primitive_tokens = [Token.ADD_BOX, Token.ADD_SPHERE, Token.ADD_CYLINDER]
    token0 = random.choice(primitive_tokens)
    params0 = params_pred[0, 0, :]
    apply_model_step(s, token0, params0)
    print("State after one step, primitives:", len(s.primitives))
