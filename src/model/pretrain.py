r"""Parameter + token pretraining script for the CAD model.

This script trains the PointNet encoder + Transformer + ParamHead + TokenHead
stack on:

    - Mean-squared-error loss over the 10D parameter vectors:
        params_vec = [cx, cy, cz, p0, p1, p2, rx, ry, rz, sign_raw]
    - Cross-entropy loss over the DSL tokens (auxiliary token head).

There is still deliberately:
    - NO learning rate scheduler
    - NO validation loop

The goal is to get a minimal, debuggable training loop that you can run
and iterate on quickly.

You must provide a dataset that yields:

    {
        "gt_points": FloatTensor[B, N, 3],   # GT point cloud
        "cur_points": FloatTensor[B, M, 3],  # current-state point cloud
        "gt_tokens": LongTensor[B, T],       # full token sequence
        "gt_params": FloatTensor[B, T, 10],  # full param sequence
    }

By default, this file includes a tiny DummyCADDataset for smoke testing.
Replace it with your real dataset when ready.
"""

from __future__ import annotations

import sys
from pathlib import Path
import argparse
import random
import time
from typing import Dict, Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Path setup so this script works when run directly
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve()
_SRC_ROOT = _HERE.parents[1]  # .../src
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from state_machine import Token  # type: ignore
from model.model_utils import build_cad_model, forward_params_only


# ---------------------------------------------------------------------------
# NPZ dataset (replace DummyCADDataset)
# ---------------------------------------------------------------------------


class NPZCADDataset(Dataset):
    """Dataset that reads CAD samples from sample.npz files.

    Each sample.npz must contain:
        gt_mesh     : [N, 3] float16 or float32 (GT point cloud)
        cur_mesh    : [M, 3] float16 or float32 (current state point cloud)
        hist_params : [H, 10]
        hist_tokens : [H]
        next_params : [K, 10]
        next_tokens : [K]

    We return:
        gt_points : [N, 3] float32
        cur_points: [M, 3] float32 (not yet used in training loop, but required)
        gt_tokens : [T]    long, where T = H + K
        gt_params : [T,10] float32, matching gt_tokens
    """

    def __init__(self, root_dir: str):
        super().__init__()
        self.root = Path(root_dir)

        # Either a single root/sample.npz, or many root/XXXXX.npz
        single_path = self.root / "sample.npz"
        if single_path.exists():
            self.sample_paths = [single_path]
        else:
            # New flat layout: dataset/train/XXXXX.npz, dataset/val/pyXXXXX.npz
            self.sample_paths = sorted(self.root.glob("*.npz"))

        if not self.sample_paths:
            raise RuntimeError(f"No sample.npz or *.npz files found under {self.root}")

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:  # type: ignore
        path = self.sample_paths[idx]
        data = np.load(path)

        # --- GT point cloud ---
        gt_mesh = data["gt_mesh"].astype(np.float32)  # [N,3]
        if gt_mesh.ndim != 2 or gt_mesh.shape[1] != 3:
            raise ValueError(f"gt_mesh must be [N,3], got {gt_mesh.shape}")
        gt_points = torch.from_numpy(gt_mesh)  # [N,3]

        # --- current-state point cloud (required, even if not yet used) ---
        cur_mesh = data["cur_mesh"].astype(np.float32)  # [M,3]
        if cur_mesh.ndim != 2 or cur_mesh.shape[1] != 3:
            raise ValueError(f"cur_mesh must be [M,3], got {cur_mesh.shape}")
        cur_points = torch.from_numpy(cur_mesh)  # [M,3]

        # --- history & next ---
        hist_params = data["hist_params"].astype(np.float32)  # [H,10] or [0,10]
        hist_tokens = data["hist_tokens"].astype(np.int64)    # [H]

        next_params = data["next_params"].astype(np.float32)  # [K,10] or [1,10]
        next_tokens = data["next_tokens"].astype(np.int64)    # [K]

        # Safety reshapes for empty or 1D cases
        if hist_params.ndim == 1:
            hist_params = hist_params.reshape(-1, 10)
        if next_params.ndim == 1:
            next_params = next_params.reshape(-1, 10)

        # concat history + next into full program
        gt_tokens = np.concatenate([hist_tokens, next_tokens], axis=0)      # [T]
        gt_params = np.concatenate([hist_params, next_params], axis=0)      # [T,10]

        gt_tokens_t = torch.from_numpy(gt_tokens).long()   # [T]
        gt_params_t = torch.from_numpy(gt_params).float()  # [T,10]

        return {
            "gt_points": gt_points,      # [N,3]
            "cur_points": cur_points,    # [M,3]
            "gt_tokens": gt_tokens_t,    # [T]
            "gt_params": gt_params_t,    # [T,10]
        }



# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Collate & point-cloud resampling for batching
# ---------------------------------------------------------------------------


def _resample_points(points: torch.Tensor, target_n: int) -> torch.Tensor:
    """Resample a point cloud to a fixed number of points.

    This makes it possible to batch samples with different N by enforcing
    a common size [target_n, 3] for all `gt_points` and `cur_points`.

    Strategy:
        - If N == target_n: return as-is.
        - If N > target_n: random downsample without replacement (if possible).
        - If 0 < N < target_n: sample with replacement until target_n.
        - If N == 0: return all zeros.
    """

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape [N, 3], got {tuple(points.shape)}")

    N = points.shape[0]
    device = points.device
    dtype = points.dtype

    if N == 0:
        # No points: just return zeros as a placeholder cloud.
        return torch.zeros(target_n, 3, device=device, dtype=dtype)

    if N == target_n:
        return points

    if N > target_n:
        # Downsample without replacement if possible.
        if N >= target_n:
            idx = torch.randperm(N, device=device)[:target_n]
        else:
            # Fallback, though logically N > target_n here, so we shouldn't hit this.
            idx = torch.randint(0, N, (target_n,), device=device)
        return points[idx]

    # 0 < N < target_n: sample with replacement.
    idx = torch.randint(0, N, (target_n,), device=device)
    return points[idx]


def cad_collate(
    batch,
    max_gt_points: int = 2048,
    max_cur_points: int = 2048,
):
    """Custom collate_fn that makes shapes batchable.

    - Resamples `gt_points` and `cur_points` in each sample to fixed
      sizes [max_gt_points, 3] and [max_cur_points, 3].
    - Stacks tokens and params assuming they already share a common
      sequence length T across the batch.

    If token/param sequence lengths differ across samples, this will
    raise a ValueError so we don't silently mis-align supervision.
    """

    gt_points_list = []
    cur_points_list = []
    gt_tokens_list = []
    gt_params_list = []

    for item in batch:
        gp = item["gt_points"]  # [N,3]
        cp = item["cur_points"]  # [M,3]
        gt_tokens = item["gt_tokens"]  # [T]
        gt_params = item["gt_params"]  # [T,10]

        # Ensure tensors are on CPU here; they will be moved to device later.
        if gp.device.type != "cpu":
            gp = gp.cpu()
        if cp.device.type != "cpu":
            cp = cp.cpu()

        gp_fixed = _resample_points(gp, max_gt_points)   # [max_gt_points, 3]
        cp_fixed = _resample_points(cp, max_cur_points)  # [max_cur_points, 3]

        gt_points_list.append(gp_fixed)
        cur_points_list.append(cp_fixed)
        gt_tokens_list.append(gt_tokens)
        gt_params_list.append(gt_params)

    # Stack point clouds: [B, N_fixed, 3]
    gt_points = torch.stack(gt_points_list, dim=0)
    cur_points = torch.stack(cur_points_list, dim=0)

    # Check and stack tokens/params. We assume equal T across the batch.
    T = gt_tokens_list[0].shape[0]
    for t in gt_tokens_list:
        if t.shape[0] != T:
            raise ValueError(
                "Variable token sequence lengths in batch; "
                "add padding + masking logic before increasing batch_size."
            )
    for p in gt_params_list:
        if p.shape[0] != T:
            raise ValueError(
                "gt_params length does not match gt_tokens length in batch."
            )

    gt_tokens = torch.stack(gt_tokens_list, dim=0)   # [B, T]
    gt_params = torch.stack(gt_params_list, dim=0)   # [B, T, 10]

    return {
        "gt_points": gt_points,
        "cur_points": cur_points,
        "gt_tokens": gt_tokens,
        "gt_params": gt_params,
    }


def build_dataloader(
    batch_size: int,
    data_root: str,
) -> DataLoader:
    """Build a DataLoader over NPZCADDataset.

    We enable batching over samples with different point counts by
    resampling `gt_points` and `cur_points` inside a custom collate_fn
    (`cad_collate`).

    Assumptions:
        - All samples in a batch share the same token/param sequence
          length T. If they do not, cad_collate will raise an error
          so we don't silently mis-align supervision.
    """

    dataset = NPZCADDataset(data_root)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=cad_collate,
    )


def compute_param_mse_loss(
    params_pred: torch.Tensor,
    gt_params: torch.Tensor,
    tokens: torch.Tensor,
) -> torch.Tensor:
    """Compute MSE over params, masking to primitive-adding tokens only.

    Parameters
    ----------
    params_pred : FloatTensor [B, T, 10]
        Predicted parameter vectors.
    gt_params : FloatTensor [B, T, 10]
        Ground-truth parameter vectors.
    tokens : LongTensor [B, T]
        Token ids per step.

    Returns
    -------
    loss : scalar Tensor
        Mean squared error over primitive steps. If no primitives are
        present in the batch, returns zero.
    """

    if params_pred.shape != gt_params.shape:
        raise ValueError(
            f"params_pred and gt_params must have the same shape, got "
            f"{params_pred.shape} vs {gt_params.shape}"
        )

    B, T, D = params_pred.shape
    if D != 10:
        raise ValueError(f"Expected last dim=10, got {D}")

    # Mask to primitive-adding tokens only
    primitive_ids = torch.tensor(
        [
            Token.ADD_BOX.value,
            Token.ADD_SPHERE.value,
            Token.ADD_CYLINDER.value,
        ],
        device=tokens.device,
    )

    # tokens: [B, T]
    # primitive_mask: [B, T]
    primitive_mask = (tokens.unsqueeze(-1) == primitive_ids.unsqueeze(0).unsqueeze(0)).any(dim=-1)

    if not primitive_mask.any():
        # No primitives in this batch; return zero to avoid NaNs
        return torch.zeros((), device=params_pred.device, dtype=params_pred.dtype)

    # Select only primitive steps
    pred_prim = params_pred[primitive_mask]  # [K, 10]
    gt_prim = gt_params[primitive_mask]      # [K, 10]

    loss = nn.functional.mse_loss(pred_prim, gt_prim)
    return loss



# ---------------------------------------------------------------------------
# Token cross-entropy loss helper
# ---------------------------------------------------------------------------

def compute_token_ce_loss(
    token_logits: torch.Tensor,
    tokens: torch.Tensor,
) -> torch.Tensor:
    """Compute cross-entropy loss over tokens.

    Parameters
    ----------
    token_logits : FloatTensor [B, T, V]
        Predicted logits over the DSL vocabulary.
    tokens : LongTensor [B, T]
        Ground-truth token ids.

    Returns
    -------
    loss : scalar Tensor
        Mean cross-entropy over all positions in the batch.
    """

    if token_logits.dim() != 3:
        raise ValueError(
            f"token_logits must have shape [B, T, V], got {token_logits.shape}"
        )
    if tokens.dim() != 2:
        raise ValueError(f"tokens must have shape [B, T], got {tokens.shape}")

    B, T, V = token_logits.shape
    if tokens.shape[0] != B or tokens.shape[1] != T:
        raise ValueError(
            f"Shape mismatch between token_logits {token_logits.shape} and tokens {tokens.shape}"
        )

    # Flatten batch + time
    logits_flat = token_logits.reshape(B * T, V)   # [B*T, V]
    targets_flat = tokens.reshape(B * T)           # [B*T]

    loss = nn.functional.cross_entropy(logits_flat, targets_flat)
    return loss


def train_epoch(
    components,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """Run a single training epoch and return average loss."""

    components.pointnet.train()
    components.transformer.train()
    components.param_head.train()
    components.token_head.train()

    total_loss = 0.0
    total_param_loss = 0.0
    total_token_loss = 0.0
    num_batches = 0
    total_samples = 0

    epoch_start = time.time()

    for batch_idx, batch in enumerate(dataloader, start=1):
        gt_points = batch["gt_points"].to(device)   # [B, N, 3]
        cur_points = batch["cur_points"].to(device) # [B, M, 3] (unused for now)
        gt_tokens = batch["gt_tokens"].to(device)   # [B, T]
        gt_params = batch["gt_params"].to(device)   # [B, T, 10]

        params_pred, hidden_states, token_logits = forward_params_only(
            components, gt_points, cur_points, gt_tokens
        )

        # Parameter regression loss (masked to primitive-adding tokens)
        param_loss = compute_param_mse_loss(params_pred, gt_params, gt_tokens)

        # Token prediction loss (auxiliary)
        token_loss = compute_token_ce_loss(token_logits, gt_tokens)

        # Total loss: simple sum for now (can reweight later)
        loss = param_loss + token_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = gt_points.shape[0]
        total_samples += batch_size
        total_loss += loss.item()
        total_param_loss += param_loss.item()
        total_token_loss += token_loss.item()
        num_batches += 1

    epoch_time = time.time() - epoch_start
    avg_loss = total_loss / max(num_batches, 1)
    avg_param_loss = total_param_loss / max(num_batches, 1)
    avg_token_loss = total_token_loss / max(num_batches, 1)
    avg_time_per_batch = epoch_time / max(num_batches, 1)
    avg_time_per_sample = epoch_time / max(total_samples, 1)

    print(
        f"Epoch {epoch}: "
        f"avg total loss = {avg_loss:.6f}, "
        f"avg param loss = {avg_param_loss:.6f}, "
        f"avg token loss = {avg_token_loss:.6f}, "
        f"time = {epoch_time:.4f}s, "
        f"{avg_time_per_batch:.4f}s/batch, "
        f"{avg_time_per_sample:.6f}s/sample"
    )

    return avg_loss


def eval_epoch(
    components,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
) -> float:
    """Run a validation epoch and return average loss.

    Uses the same loss definition as training (param MSE + token CE),
    but without gradient updates.
    """

    components.pointnet.eval()
    components.transformer.eval()
    components.param_head.eval()
    components.token_head.eval()

    total_loss = 0.0
    total_param_loss = 0.0
    total_token_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            gt_points = batch["gt_points"].to(device)   # [B, N, 3]
            cur_points = batch["cur_points"].to(device) # [B, M, 3]
            gt_tokens = batch["gt_tokens"].to(device)   # [B, T]
            gt_params = batch["gt_params"].to(device)   # [B, T, 10]

            params_pred, hidden_states, token_logits = forward_params_only(
                components, gt_points, cur_points, gt_tokens
            )

            param_loss = compute_param_mse_loss(params_pred, gt_params, gt_tokens)
            token_loss = compute_token_ce_loss(token_logits, gt_tokens)
            loss = param_loss + token_loss

            total_loss += loss.item()
            total_param_loss += param_loss.item()
            total_token_loss += token_loss.item()
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_param_loss = total_param_loss / max(num_batches, 1)
    avg_token_loss = total_token_loss / max(num_batches, 1)
    print(
        f"Epoch {epoch}: VAL avg total loss = {avg_loss:.6f}, "
        f"param = {avg_param_loss:.6f}, token = {avg_token_loss:.6f}"
    )
    return avg_loss


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Param pretraining for CAD model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (use 1 if variable lengths)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory with TRAIN sample_XXXXX.npz files",
    )
    parser.add_argument(
        "--val_root",
        type=str,
        default=None,
        help="Root directory with VAL sample_XXXXX.npz files (optional)",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=32,
        help="Maximum token sequence length for the transformer",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    vocab_size = len(Token)
    components = build_cad_model(
        vocab_size=vocab_size,
        gt_dim=256,
        d_model=256,
        n_heads=4,
        num_layers=4,
        max_seq_len=args.max_seq_len,
        hidden_dim=256,
        err_dim=3,  # 3D error embedding (chamfer, centroid_dist, log_scale_ratio)
        device=device,
    )

    optimizer = torch.optim.Adam(
        list(components.pointnet.parameters())
        + list(components.transformer.parameters())
        + list(components.param_head.parameters())
        + list(components.token_head.parameters()),
        lr=args.lr,
    )

    dataloader = build_dataloader(
        batch_size=args.batch_size,
        data_root=args.data_root,
    )

    # Optional validation loader
    val_dataloader = None
    if args.val_root is not None:
        print(f"Using validation data from: {args.val_root}")
        val_dataloader = build_dataloader(
            batch_size=args.batch_size,
            data_root=args.val_root,
        )

    # Best-model tracking
    best_val_loss = float("inf")
    models_dir = _SRC_ROOT.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    best_path = models_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(components, dataloader, optimizer, device, epoch)

        val_loss = None
        if val_dataloader is not None:
            val_loss = eval_epoch(components, val_dataloader, device, epoch)

            # Update and save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                state = {
                    "pointnet": components.pointnet.state_dict(),
                    "transformer": components.transformer.state_dict(),
                    "param_head": components.param_head.state_dict(),
                    "token_head": components.token_head.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                }
                torch.save(state, best_path)
                print(f"  --> New best model saved to {best_path} (val_loss={best_val_loss:.6f})")

        if val_loss is None:
            print(f"Epoch {epoch} completed. Train avg loss = {train_loss:.6f}")


if __name__ == "__main__":  # pragma: no cover
    main()
