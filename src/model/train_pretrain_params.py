r"""Parameter pretraining script for the CAD model.

This script trains the PointNet encoder + Transformer + ParamHead stack
purely on a mean-squared-error loss over the 10D parameter vectors:

    params_vec = [cx, cy, cz, p0, p1, p2, rx, ry, rz, sign_raw]

There is deliberately:
    - NO token auxiliary loss
    - NO learning rate scheduler
    - NO validation loop

The goal is to get a minimal, debuggable training loop that you can run
and iterate on quickly.

You must provide a dataset that yields:

    {
        "gt_points": FloatTensor[B, N, 3],
        "gt_tokens": LongTensor[B, T],
        "gt_params": FloatTensor[B, T, 10],
    }

By default, this file includes a tiny DummyCADDataset for smoke testing.
Replace it with your real dataset when ready.
"""

from __future__ import annotations

import sys
from pathlib import Path
import argparse
import random
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

        # Either a single root/sample.npz, or many root/sample_XXXXX.npz
        single_path = self.root / "sample.npz"
        if single_path.exists():
            self.sample_paths = [single_path]
        else:
            # New flat layout: dataset/train/sample_XXXXX.npz, dataset/val/sample_XXXXX.npz
            self.sample_paths = sorted(self.root.glob("sample_*.npz"))

        if not self.sample_paths:
            raise RuntimeError(f"No sample.npz or sample_*.npz files found under {self.root}")

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


def build_dataloader(
    batch_size: int,
    data_root: str,
) -> DataLoader:
    """Build a DataLoader over NPZCADDataset.

    Note: if samples have varying N/T, use batch_size=1 or add a custom
    collate_fn to pad sequences.
    """

    dataset = NPZCADDataset(data_root)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


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

    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        gt_points = batch["gt_points"].to(device)   # [B, N, 3]
        cur_points = batch["cur_points"].to(device) # [B, M, 3] (unused for now)
        gt_tokens = batch["gt_tokens"].to(device)   # [B, T]
        gt_params = batch["gt_params"].to(device)   # [B, T, 10]

        params_pred, hidden_states, token_logits = forward_params_only(components, gt_points, gt_tokens)

        loss = compute_param_mse_loss(params_pred, gt_params, gt_tokens)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    print(f"Epoch {epoch}: avg loss = {avg_loss:.6f}")
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
    parser.add_argument("--data_root", type=str, required=True, help="Root directory with sample.npz files")
    parser.add_argument("--max_seq_len", type=int, default=32, help="Maximum token sequence length for the transformer")
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
        err_dim=0,
        device=device,
    )

    optimizer = torch.optim.Adam(
        list(components.pointnet.parameters())
        + list(components.transformer.parameters())
        + list(components.param_head.parameters()),
        lr=args.lr,
    )

    dataloader = build_dataloader(
        batch_size=args.batch_size,
        data_root=args.data_root,
    )

    for epoch in range(1, args.epochs + 1):
        train_epoch(components, dataloader, optimizer, device, epoch)


if __name__ == "__main__":  # pragma: no cover
    main()
