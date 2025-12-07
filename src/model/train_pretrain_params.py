"""Parameter pretraining script for the CAD model.

This script trains the PointNet encoder + Transformer + ParamHead stack
purely on a mean-squared-error loss over the 9D parameter vectors:

    params_vec = [cx, cy, cz, p0, p1, p2, rx, ry, rz]

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
        "gt_params": FloatTensor[B, T, 9],
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
# Dummy dataset (replace with real one)
# ---------------------------------------------------------------------------


class DummyCADDataset(Dataset):
    """Tiny random dataset for smoke testing the training loop.

    This is NOT meant for real training. It just ensures the plumbing
    works end-to-end before you plug in your real dataset.
    """

    def __init__(self, num_samples: int = 32, num_points: int = 128, seq_len: int = 6):
        super().__init__()
        self.num_samples = num_samples
        self.num_points = num_points
        self.seq_len = seq_len

        self._primitive_token_ids = [
            Token.ADD_BOX.value,
            Token.ADD_SPHERE.value,
            Token.ADD_CYLINDER.value,
        ]

        self._all_token_ids = [t.value for t in Token]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:  # type: ignore
        # Random GT point cloud
        gt_points = torch.randn(self.num_points, 3)

        # Random token sequence (mix of primitives + others)
        tokens = []
        for _ in range(self.seq_len):
            tokens.append(random.choice(self._all_token_ids))
        gt_tokens = torch.tensor(tokens, dtype=torch.long)

        # Random GT params per step: [T, 9]
        gt_params = torch.randn(self.seq_len, 9)

        return {
            "gt_points": gt_points,
            "gt_tokens": gt_tokens,
            "gt_params": gt_params,
        }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def build_dataloader(
    batch_size: int,
    num_samples: int = 128,
    num_points: int = 256,
    seq_len: int = 8,
) -> DataLoader:
    """Build a DataLoader.

    For now, this uses DummyCADDataset. Replace with your real Dataset
    when it's ready.
    """

    dataset = DummyCADDataset(num_samples=num_samples, num_points=num_points, seq_len=seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


def compute_param_mse_loss(
    params_pred: torch.Tensor,
    gt_params: torch.Tensor,
    tokens: torch.Tensor,
) -> torch.Tensor:
    """Compute MSE over params, masking to primitive-adding tokens only.

    Parameters
    ----------
    params_pred : FloatTensor [B, T, 9]
        Predicted parameter vectors.
    gt_params : FloatTensor [B, T, 9]
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
    if D != 9:
        raise ValueError(f"Expected last dim=9, got {D}")

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
    pred_prim = params_pred[primitive_mask]  # [K, 9]
    gt_prim = gt_params[primitive_mask]      # [K, 9]

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
        gt_tokens = batch["gt_tokens"].to(device)   # [B, T]
        gt_params = batch["gt_params"].to(device)   # [B, T, 9]

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
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_samples", type=int, default=128, help="Number of dummy samples")
    parser.add_argument("--num_points", type=int, default=256, help="Points per GT shape")
    parser.add_argument("--seq_len", type=int, default=8, help="Token sequence length")
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
        max_seq_len=args.seq_len,
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
        num_samples=args.num_samples,
        num_points=args.num_points,
        seq_len=args.seq_len,
    )

    for epoch in range(1, args.epochs + 1):
        train_epoch(components, dataloader, optimizer, device, epoch)


if __name__ == "__main__":  # pragma: no cover
    main()
