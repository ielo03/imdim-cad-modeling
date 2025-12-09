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

from geometry_backend import chamfer_distance

from model.pointnet_encoder import PointNetEncoder
from model.transformer import CADTransformer
from model.param_head import ParamHead
from model.token_head import TokenHead
from state_machine import ShapeState, Token


ERR_EMBED_DIM = 8


@dataclass
class ModelComponents:
    """Bundle of core model components for the CAD policy.

    Attributes
    ----------
    pointnet : PointNetEncoder
        Encodes point clouds (GT and current) into global embeddings.
    transformer : CADTransformer
        Autoregressive backbone over DSL tokens, conditioned on
        gt_embed, cur_embed, and an error embedding.
    param_head : ParamHead
        Predicts the 10D parameter vector at each step (9 geom + sign).
    token_head : TokenHead
        Predicts logits over the DSL token vocabulary at each step.
    """

    pointnet: PointNetEncoder
    transformer: CADTransformer
    param_head: ParamHead
    token_head: TokenHead

    def to(self, device: torch.device | str) -> "ModelComponents":
        """Move all submodules to a given device and return self."""

        dev = torch.device(device)
        self.pointnet.to(dev)
        self.transformer.to(dev)
        self.param_head.to(dev)
        self.token_head.to(dev)
        return self


def compute_error_embedding(
    gt_points: torch.Tensor,
    cur_points: torch.Tensor,
) -> torch.Tensor:
    """Compute a richer error embedding from GT and current point clouds.

    The embedding has ERR_EMBED_DIM=8 scalars per batch element:

        0: Symmetric Chamfer distance between cur and GT clouds.
        1: L2 distance between centroids of cur and GT.
        2: Mean per-point L2 error (aligned subset).
        3: Max per-point L2 error (aligned subset).
        4: Std per-point L2 error (aligned subset).
        5: Scale ratio (cur / GT) based on average radius.
        6: GT average radius.
        7: CUR average radius.

    Inputs are expected to be [B, N, 3] and [B, M, 3]. If either cloud has
    zero points, a zero embedding is returned for that batch.
    """

    if gt_points.dim() != 3 or cur_points.dim() != 3:
        raise ValueError(
            f"gt_points and cur_points must be [B, N, 3] / [B, M, 3], got "
            f"{gt_points.shape} and {cur_points.shape}"
        )

    B = gt_points.shape[0]

    # If either cloud has zero points, return a zero error embedding.
    if gt_points.shape[1] == 0 or cur_points.shape[1] == 0:
        return torch.zeros(
            B,
            ERR_EMBED_DIM,
            device=gt_points.device,
            dtype=gt_points.dtype,
        )

    # 1) Chamfer distance (no grad through geometry backend).
    # chamfer_distance expects rank-2 [N,3] tensors, so compute it per batch element.
    with torch.no_grad():
        chamf_list = []
        for b in range(B):
            chamf_b = chamfer_distance(cur_points[b], gt_points[b])
            if not isinstance(chamf_b, torch.Tensor):
                chamf_b = torch.as_tensor(
                    chamf_b,
                    device=gt_points.device,
                    dtype=gt_points.dtype,
                )
            chamf_list.append(chamf_b.reshape(1))
        chamf = torch.cat(chamf_list, dim=0)  # [B]

    # 2) Centroid distance
    centroid_gt = gt_points.mean(dim=1)   # [B, 3]
    centroid_cur = cur_points.mean(dim=1) # [B, 3]
    centroid_dist = torch.linalg.norm(centroid_gt - centroid_cur, dim=-1)  # [B]

    # 3) Per-point errors on aligned subset (min(N_gt, N_cur))
    Ng = gt_points.shape[1]
    Nc = cur_points.shape[1]
    N = min(Ng, Nc)
    gt_sub = gt_points[:, :N, :]   # [B, N, 3]
    cur_sub = cur_points[:, :N, :] # [B, N, 3]
    point_dists = torch.linalg.norm(cur_sub - gt_sub, dim=-1)  # [B, N]

    mean_point_err = point_dists.mean(dim=1)      # [B]
    max_point_err = point_dists.max(dim=1).values # [B]
    std_point_err = point_dists.std(dim=1)        # [B]

    # 4) Scale via average radius from centroid for each cloud
    gt_centered = gt_points - centroid_gt.unsqueeze(1)   # [B, N_gt, 3]
    cur_centered = cur_points - centroid_cur.unsqueeze(1) # [B, N_cur, 3]

    gt_radius = torch.sqrt((gt_centered ** 2).sum(dim=-1)).mean(dim=1)  # [B]
    cur_radius = torch.sqrt((cur_centered ** 2).sum(dim=-1)).mean(dim=1)  # [B]

    eps = 1e-6
    scale_ratio = cur_radius / (gt_radius + eps)  # [B]

    # Stack into [B, ERR_EMBED_DIM]
    err_embed = torch.stack(
        [
            chamf,
            centroid_dist,
            mean_point_err,
            max_point_err,
            std_point_err,
            scale_ratio,
            gt_radius,
            cur_radius,
        ],
        dim=-1,
    )
    return err_embed


def build_cad_model(
    vocab_size: int,
    gt_dim: int = 256,
    d_model: int = 256,
    n_heads: int = 4,
    num_layers: int = 6,
    max_seq_len: int = 64,
    tok_emb_dim: Optional[int] = None,
    hidden_dim: int = 256,
    err_dim: int = ERR_EMBED_DIM,
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
        Size of optional error embedding for the Transformer conditioning.
        Must match the dimensionality returned by `compute_error_embedding`
        (currently 3).
    device : torch.device or str, optional
        If provided, all modules are moved to this device.

    Returns
    -------
    ModelComponents
        A simple dataclass bundling pointnet, transformer, param_head, and token_head.
    """

    pointnet = PointNetEncoder(in_dim=3, hidden_dim=gt_dim // 2, out_dim=gt_dim)

    transformer = CADTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        dim_feedforward=4 * d_model,
        max_seq_len=max_seq_len,
        dropout=0.1,
        gt_cond_dim=gt_dim,
        cur_cond_dim=gt_dim,
        err_cond_dim=err_dim if err_dim > 0 else None,
    )

    param_head = ParamHead(
        d_model=d_model,
        vocab_size=vocab_size,
        tok_emb_dim=tok_emb_dim or d_model,
        out_dim=10,           # [cx, cy, cz, p0, p1, p2, rx, ry, rz, sign_raw]
        hidden_dim=hidden_dim,
    )

    token_head = TokenHead(
        d_model=d_model,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
    )

    components = ModelComponents(
        pointnet=pointnet,
        transformer=transformer,
        param_head=param_head,
        token_head=token_head,
    )
    if device is not None:
        components.to(device)
    return components


def forward_params_only(
    components: ModelComponents,
    gt_points: torch.Tensor,
    cur_points: torch.Tensor,
    tokens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a forward pass that predicts per-step parameters.

    This is intended for param pretraining and analysis. It:

        1. Encodes GT points with PointNet → gt_embed
        2. Encodes current points with the same PointNet → cur_embed
        3. Computes a richer error embedding from current and GT points.
        4. Runs the Transformer with teacher-forced tokens and conditioning
           → hidden_states
        5. Applies the ParamHead at each step → params_pred
        6. Applies the TokenHead to all hidden_states → token_logits
    """

    pointnet, transformer, param_head, token_head = (
        components.pointnet,
        components.transformer,
        components.param_head,
        components.token_head,
    )

    if gt_points.dim() != 3:
        raise ValueError(f"gt_points must have shape [B, N, 3], got {gt_points.shape}")
    if cur_points is not None and cur_points.dim() != 3:
        raise ValueError(f"cur_points must have shape [B, M, 3], got {cur_points.shape}")
    if tokens.dim() != 2:
        raise ValueError(f"tokens must have shape [B, T], got {tokens.shape}")

    # Global embeddings
    gt_embed = pointnet(gt_points)  # [B, gt_dim]

    B = gt_points.shape[0]

    # Handle possibly empty current point clouds
    if cur_points is not None and cur_points.shape[1] > 0:
        # Normal case: encode current mesh and compute error embedding
        cur_embed = pointnet(cur_points)  # [B, gt_dim]
        err_embed = compute_error_embedding(gt_points, cur_points)  # [B, ERR_EMBED_DIM]
    else:
        # Empty or missing current mesh: fall back to zeros
        cur_embed = torch.zeros_like(gt_embed)
        err_embed = torch.zeros(
            B,
            ERR_EMBED_DIM,
            device=gt_points.device,
            dtype=gt_points.dtype,
        )

    # Transformer backbone
    hidden_states = transformer(
        tokens,
        gt_embed=gt_embed,
        cur_embed=cur_embed,
        err_embed=err_embed,
    )  # [B, T, d_model]

    # Token logits via TokenHead
    token_logits = token_head(hidden_states)  # [B, T, vocab_size]

    # Per-step parameter predictions
    B, T, d_model = hidden_states.shape
    params_list = []

    for t in range(T):
        h_t = hidden_states[:, t, :]        # [B, d_model]
        tok_t = tokens[:, t]               # [B]
        params_t = param_head(h_t, tok_t)  # [B, 10]
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
    cur_points = gt_points.clone()
    tokens = torch.randint(0, vocab_size, (B, T))

    params_pred, hidden_states, token_logits = forward_params_only(components, gt_points, cur_points, tokens)
    print("params_pred shape:", params_pred.shape)

    # Single-step application demo: randomly choose a primitive-adding token.
    s = ShapeState()
    primitive_tokens = [Token.ADD_BOX, Token.ADD_SPHERE, Token.ADD_CYLINDER]
    token0 = random.choice(primitive_tokens)
    params0 = params_pred[0, 0, :]
    apply_model_step(s, token0, params0)
    print("State after one step, primitives:", len(s.primitives))
