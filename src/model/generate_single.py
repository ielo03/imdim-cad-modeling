r"""Single-sample generation script for the CAD model.

This script:
    - Loads the best pre-trained model from models/best.pt
    - Loads a single NPZ sample containing:
        gt_mesh     : [N, 3] float16/float32
        cur_mesh    : [M, 3] float16/float32
        hist_params : [H, 10]
        hist_tokens : [H]
        next_params : [K, 10]
        next_tokens : [K]
    - Treats `hist_tokens` as the current program history.
    - Runs the transformer over `hist_tokens` conditioned on:
        * gt_points (GT point cloud)
        * cur_points (current-state point cloud)
        * error embedding (Chamfer + simple stats)
    - Uses the last hidden state to:
        * predict the next token with the TokenHead,
        * feed that predicted token into the ParamHead to predict
          the corresponding 10D parameter vector.

Usage:
    python -m model.generate_single --sample path/to/sample_0000.npz --device mps
"""

from __future__ import annotations

import sys
from pathlib import Path
import argparse
from typing import Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve()
_SRC_ROOT = _HERE.parents[1]  # .../src
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from state_machine import Token  # type: ignore
from model.model_utils import (
    build_cad_model,
    compute_error_embedding,
)

# ---------------------------------------------------------------------------
# Utility: load sample NPZ
# ---------------------------------------------------------------------------


def load_sample_npz(path: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load GT / current point clouds and history tokens from a sample NPZ.

    Returns
    -------
    gt_points : FloatTensor [1, N, 3]
    cur_points: FloatTensor [1, M, 3]
    hist_tokens : LongTensor [1, H]
    """
    data = np.load(path)

    # GT mesh -> gt_points
    gt_mesh = data["gt_mesh"].astype(np.float32)  # [N,3]
    if gt_mesh.ndim != 2 or gt_mesh.shape[1] != 3:
        raise ValueError(f"gt_mesh must be [N,3], got {gt_mesh.shape}")
    gt_points = torch.from_numpy(gt_mesh).unsqueeze(0)  # [1,N,3]

    # Current-state mesh -> cur_points
    cur_mesh = data["cur_mesh"].astype(np.float32)  # [M,3]
    if cur_mesh.ndim != 2 or cur_mesh.shape[1] != 3:
        raise ValueError(f"cur_mesh must be [M,3], got {cur_mesh.shape}")
    cur_points = torch.from_numpy(cur_mesh).unsqueeze(0)  # [1,M,3]

    # History tokens
    hist_tokens_np = data["hist_tokens"].astype(np.int64)  # [H] or [0]
    if hist_tokens_np.ndim == 0:
        hist_tokens_np = hist_tokens_np.reshape(1)
    hist_tokens = torch.from_numpy(hist_tokens_np).long().unsqueeze(0)  # [1,H]

    # If there is no history (H==0), create a single dummy token to bootstrap
    if hist_tokens.shape[1] == 0:
        # Use a simple placeholder token; this depends on your DSL.
        hist_tokens = torch.full((1, 1), Token.ADD_BOX.value, dtype=torch.long)

    return gt_points, cur_points, hist_tokens


# ---------------------------------------------------------------------------
# Single-step prediction
# ---------------------------------------------------------------------------


def predict_next_step(
    components,
    gt_points: torch.Tensor,   # [1, N, 3]
    cur_points: torch.Tensor,  # [1, M, 3]
    hist_tokens: torch.Tensor, # [1, H]
    device: torch.device,
):
    """Run a single next-step prediction.

    Pipeline:
        1) Encode gt_points and cur_points with PointNet → gt_embed, cur_embed
        2) Compute error embedding from (gt_points, cur_points)
        3) Run Transformer over hist_tokens with conditioning
        4) Use last hidden state to:
            - get next-token logits from TokenHead
            - pick argmax token id
        5) Feed that predicted token into ParamHead to predict params.
    """
    components.pointnet.eval()
    components.transformer.eval()
    components.param_head.eval()
    components.token_head.eval()

    gt_points = gt_points.to(device)
    cur_points = cur_points.to(device)
    hist_tokens = hist_tokens.to(device)

    with torch.no_grad():
        # 1) PointNet encodings
        gt_embed = components.pointnet(gt_points)   # [1, D_gt]

        # Handle possibly empty current point cloud
        if cur_points is not None and cur_points.shape[1] > 0:
            cur_embed = components.pointnet(cur_points)  # [1, D_gt]
            # 2) Error embedding (Chamfer, centroid dist, scale ratio)
            err_embed = compute_error_embedding(gt_points, cur_points)  # [1, err_dim]
        else:
            # No current geometry: fall back to zeros, same convention as training
            cur_embed = torch.zeros_like(gt_embed)
            err_embed = torch.zeros(
                gt_points.shape[0],
                3,
                device=gt_points.device,
                dtype=gt_points.dtype,
            )

        # 3) Transformer over history tokens
        #    Assumes CADTransformer.forward(tokens, gt_embed, cur_embed, err_embed)
        hidden_states = components.transformer(hist_tokens, gt_embed, cur_embed, err_embed)
        # hidden_states: [1, H, d_model]
        last_hidden = hidden_states[:, -1, :]  # [1, d_model]

        # 4) Next-token logits from TokenHead
        token_logits_full = components.token_head(hidden_states)  # [1, H, V]
        next_token_logits = token_logits_full[:, -1, :]          # [1, V]
        next_token_id = next_token_logits.argmax(dim=-1)         # [1]

        # 5) Feed predicted token into ParamHead.
        #    ParamHead in this codebase expects (h_t, token_t) → [B, 10].
        next_params = components.param_head(last_hidden, next_token_id)  # [1, 10]

    return next_token_id.cpu(), next_params.cpu()


# ---------------------------------------------------------------------------
# Helper: run range of samples
# ---------------------------------------------------------------------------

def run_sample_range(
    components,
    device: torch.device,
    base_sample_path: Path,
    count: int = 11,
) -> None:
    """Run prediction on `base_sample_path` and the next `count-1` samples.

    Assumes file names are integer stems, e.g. 9900.npz, 9901.npz, ...
    Prints only the next-token id per sample.
    """
    parent = base_sample_path.parent
    stem = base_sample_path.stem

    try:
        base_idx = int(stem)
    except ValueError:
        raise ValueError(f"Cannot interpret sample stem '{stem}' as an integer index")

    width = len(stem)  # preserve zero-padding width if any

    for offset in range(count):
        idx = base_idx + offset
        fname = f"{idx:0{width}d}.npz"
        sample_path = parent / fname
        if not sample_path.exists():
            print(f"{fname}: MISSING")
            continue

        gt_points, cur_points, hist_tokens = load_sample_npz(sample_path)
        next_token_id, _ = predict_next_step(
            components, gt_points, cur_points, hist_tokens, device
        )
        token_int = int(next_token_id.item())

        print(f"{fname}: {token_int}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate single-step prediction from best CAD model")
    parser.add_argument(
        "--sample",
        type=str,
        required=True,
        help="Path to a single sample_XXXXX.npz file (with gt_mesh/cur_mesh/hist_tokens/...)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (e.g., cpu, cuda, mps)",
    )
    parser.add_argument(
        "--range",
        action="store_true",
        help="If set, run this sample and the next 10 sequential samples, printing only next token ids",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    sample_path = Path(args.sample)
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample NPZ not found: {sample_path}")

    # Build model with same hyperparams as in pretrain.py
    vocab_size = len(Token)
    components = build_cad_model(
        vocab_size=vocab_size,
        gt_dim=256,
        d_model=256,
        n_heads=4,
        num_layers=4,
        max_seq_len=32,
        hidden_dim=256,
        err_dim=3,
        device=device,
    )

    # Load best checkpoint
    models_dir = _SRC_ROOT.parent / "models"
    best_path = models_dir / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"No best model checkpoint found at {best_path}")

    checkpoint = torch.load(best_path, map_location=device)
    components.pointnet.load_state_dict(checkpoint["pointnet"])
    components.transformer.load_state_dict(checkpoint["transformer"])
    components.param_head.load_state_dict(checkpoint["param_head"])
    components.token_head.load_state_dict(checkpoint["token_head"])
    print(f"Loaded best model from {best_path} (epoch={checkpoint.get('epoch')}, val_loss={checkpoint.get('val_loss')})")

    if args.range:
        # Run this sample and the next 10, only printing next-token ids
        run_sample_range(components, device, sample_path, count=11)
    else:
        # Single-sample mode (original behavior)
        gt_points, cur_points, hist_tokens = load_sample_npz(sample_path)

        # Load ground-truth next_tokens (if present) for comparison
        gt_next_tokens = None
        try:
            data = np.load(sample_path)
            if "next_tokens" in data.files:
                next_tokens_np = data["next_tokens"].astype(np.int64)
                if next_tokens_np.ndim == 0:
                    next_tokens_np = next_tokens_np.reshape(1)
                gt_next_tokens = next_tokens_np
        except Exception:
            gt_next_tokens = None

        next_token_id, next_params = predict_next_step(
            components, gt_points, cur_points, hist_tokens, device
        )

        token_int = int(next_token_id.item())
        try:
            token_enum = Token(token_int)
            token_name = token_enum.name
        except ValueError:
            token_name = f"<unknown:{token_int}>"

        print("\n=== Prediction ===")
        print(f"Next token id   : {token_int}")
        print(f"Next token name : {token_name}")
        print(f"Next params [10]: {next_params.numpy().reshape(-1)}")

        # Ground-truth next token(s), if available
        if gt_next_tokens is not None:
            gt_ids = gt_next_tokens.tolist()
            print(f"GT next token ids   : {gt_ids}")
            # Decode the first GT token name if possible
            try:
                gt_first = int(gt_ids[0])
                gt_enum = Token(gt_first)
                gt_name = gt_enum.name
            except Exception:
                gt_name = f"<unknown:{gt_ids[0]}>"
            print(f"GT first token name : {gt_name}")
        else:
            print("GT next_tokens not found in sample.")


if __name__ == "__main__":  # pragma: no cover
    main()