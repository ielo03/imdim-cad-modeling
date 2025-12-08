"""Command-line utility to generate GT point clouds from sample DSL.

Preferred (new) sample format: a single `dsl.npz` file in each sample dir:

    dsl.npz containing:
        hist_tokens : int64, shape [T_hist]
        hist_params : float32, shape [T_hist, 10]
        next_tokens : int64, shape [K]
        next_params : float32, shape [K, 10]

Legacy format (still supported as a fallback):

    hist_tokens.npy   # int64, shape [T_hist]
    hist_params.npy   # float32, shape [T_hist, 9]
    next_tokens.npy   # int64, shape [K] or scalar (K>=1)
    next_params.npy   # float32, shape [K, 9] or [9]

We reconstruct a `ShapeState` by applying all history steps and all
next steps, then use the geometry backend to sample a point cloud for
the resulting shape and save it as `gt_points.npy` in the same directory.

Usage examples:

    # Process a single sample directory
    python -m dataset.sample_to_mesh --sample-dir path/to/sample_0000

    # Process all subdirectories under a root (e.g. dataset/train)
    python -m dataset.sample_to_mesh --root dataset/train

By default it will not overwrite an existing `gt_points.npy` unless
`--overwrite` is passed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup so we can import state_machine and geometry_backend
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve()
_SRC_ROOT = _HERE.parents[1]  # .../src
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from state_machine import ShapeState, Token  # type: ignore
import geometry_backend  # type: ignore


def _state_to_point_cloud(state: ShapeState, n_points: int, device: torch.device) -> torch.Tensor:
    """Adapter to call the appropriate function in geometry_backend.

    Tries a few likely function names so this script does not depend on
    a single exact symbol being exported.
    """

    if hasattr(geometry_backend, "state_to_point_cloud"):
        return geometry_backend.state_to_point_cloud(state, n_points=n_points, device=device)
    if hasattr(geometry_backend, "state_to_points"):
        return geometry_backend.state_to_points(state, n_points=n_points, device=device)
    if hasattr(geometry_backend, "sample_state_point_cloud"):
        return geometry_backend.sample_state_point_cloud(state, n_points=n_points, device=device)

    raise RuntimeError(
        "geometry_backend must define one of: state_to_point_cloud, "
        "state_to_points, sample_state_point_cloud"
    )



def _load_from_dsl_npz(sample_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load hist/next tokens and params from dsl.npz (10D params)."""

    dsl_path = sample_dir / "dsl.npz"
    if not dsl_path.exists():
        raise FileNotFoundError(f"Missing dsl.npz in {sample_dir}")

    data = np.load(dsl_path)
    required_keys = ["hist_tokens", "hist_params", "next_tokens", "next_params"]
    for k in required_keys:
        if k not in data:
            raise KeyError(f"dsl.npz in {sample_dir} missing key {k!r}")

    hist_tokens = np.asarray(data["hist_tokens"], dtype=np.int64)
    hist_params = np.asarray(data["hist_params"], dtype=np.float32)
    next_tokens = np.asarray(data["next_tokens"], dtype=np.int64)
    next_params = np.asarray(data["next_params"], dtype=np.float32)

    # Normalize shapes
    if hist_tokens.ndim != 1:
        raise ValueError(f"dsl.npz: hist_tokens must be 1D, got shape {hist_tokens.shape}")
    if hist_params.ndim != 2 or hist_params.shape[1] != 10:
        raise ValueError(f"dsl.npz: hist_params must be [T_hist, 10], got shape {hist_params.shape}")
    if hist_tokens.shape[0] != hist_params.shape[0]:
        raise ValueError(
            f"dsl.npz: hist_tokens length {hist_tokens.shape[0]} does not match "
            f"hist_params length {hist_params.shape[0]}"
        )

    if next_tokens.ndim == 0:
        next_tokens = np.array([int(next_tokens)], dtype=np.int64)
    elif next_tokens.ndim == 1:
        next_tokens = next_tokens.astype(np.int64)
    else:
        raise ValueError(f"dsl.npz: next_tokens must be scalar or 1D, got shape {next_tokens.shape}")

    if next_params.ndim == 1:
        if next_params.shape[0] != 10:
            raise ValueError(
                f"dsl.npz: next_params 1D must have length 10, got {next_params.shape[0]}"
            )
        next_params = next_params.reshape(1, 10)
    elif next_params.ndim == 2 and next_params.shape[1] == 10:
        pass
    else:
        raise ValueError(
            f"dsl.npz: next_params must be [10] or [K, 10], got shape {next_params.shape}"
        )

    if next_tokens.shape[0] != next_params.shape[0]:
        raise ValueError(
            f"dsl.npz: next_tokens length {next_tokens.shape[0]} does not match "
            f"next_params length {next_params.shape[0]}"
        )

    return hist_tokens, hist_params, next_tokens, next_params


def _load_from_legacy_npy(sample_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Legacy loader: separate npy files with 9D params."""

    hist_tokens_path = sample_dir / "hist_tokens.npy"
    hist_params_path = sample_dir / "hist_params.npy"
    next_tokens_path = sample_dir / "next_tokens.npy"
    next_params_path = sample_dir / "next_params.npy"

    if not hist_tokens_path.exists() or not hist_params_path.exists():
        raise FileNotFoundError(f"Missing hist_* files in {sample_dir}")
    if not next_tokens_path.exists() or not next_params_path.exists():
        raise FileNotFoundError(f"Missing next_* files in {sample_dir}")

    hist_tokens = np.load(hist_tokens_path, allow_pickle=True)
    hist_tokens = np.asarray(hist_tokens, dtype=np.int64)  # [T_hist]

    hist_params = np.load(hist_params_path, allow_pickle=True)
    hist_params = np.asarray(hist_params, dtype=np.float32)  # [T_hist, 9]

    next_tokens = np.load(next_tokens_path, allow_pickle=True)
    next_tokens = np.asarray(next_tokens, dtype=np.int64)  # [K] or scalar

    next_params = np.load(next_params_path, allow_pickle=True)
    next_params = np.asarray(next_params, dtype=np.float32)  # [K, 9] or [9]

    # Normalize shapes (legacy: 9D)
    if hist_tokens.ndim != 1:
        raise ValueError(f"hist_tokens.npy must be 1D, got shape {hist_tokens.shape}")
    if hist_params.ndim != 2 or hist_params.shape[1] != 9:
        raise ValueError(f"hist_params.npy must be [T_hist, 9], got shape {hist_params.shape}")
    if hist_tokens.shape[0] != hist_params.shape[0]:
        raise ValueError(
            f"hist_tokens length {hist_tokens.shape[0]} does not match "
            f"hist_params length {hist_params.shape[0]}"
        )

    if next_tokens.ndim == 0:
        next_tokens = np.array([int(next_tokens)], dtype=np.int64)
    elif next_tokens.ndim == 1:
        next_tokens = next_tokens.astype(np.int64)
    else:
        raise ValueError(f"next_tokens.npy must be scalar or 1D, got shape {next_tokens.shape}")

    if next_params.ndim == 1:
        if next_params.shape[0] != 9:
            raise ValueError(
                f"next_params.npy 1D must have length 9, got {next_params.shape[0]}"
            )
        next_params = next_params.reshape(1, 9)
    elif next_params.ndim == 2 and next_params.shape[1] == 9:
        pass
    else:
        raise ValueError(f"next_params.npy must be [9] or [K, 9], got shape {next_params.shape}")

    if next_tokens.shape[0] != next_params.shape[0]:
        raise ValueError(
            f"next_tokens length {next_tokens.shape[0]} does not match "
            f"next_params length {next_params.shape[0]}"
        )

    return hist_tokens, hist_params, next_tokens, next_params


def build_state_from_sample_dir(sample_dir: Path) -> ShapeState:
    """Reconstruct a ShapeState from a sample directory.

    Preferred: load from dsl.npz with 10D params; fallback: legacy 9D npy files.
    """

    dsl_path = sample_dir / "dsl.npz"
    if dsl_path.exists():
        hist_tokens, hist_params, next_tokens, next_params = _load_from_dsl_npz(sample_dir)
    else:
        # Legacy path (9D params); still supported
        hist_tokens, hist_params, next_tokens, next_params = _load_from_legacy_npy(sample_dir)

    state = ShapeState()

    # Apply history
    for tok_id, params in zip(hist_tokens, hist_params):
        token = Token(int(tok_id))
        state.apply(token, params.astype(float).tolist())

    # Apply all next steps (could be one or more)
    for tok_id, params in zip(next_tokens, next_params):
        token = Token(int(tok_id))
        state.apply(token, params.astype(float).tolist())

    return state


def generate_gt_points_for_sample(
    sample_dir: Path,
    n_points: int = 2048,
    overwrite: bool = False,
    device: Optional[str] = None,
) -> Path:
    """Generate gt_points.npy for a single sample directory.

    Returns the path to the written `gt_points.npy`.
    """

    out_path = sample_dir / "gt_points.npy"
    if out_path.exists() and not overwrite:
        print(f"[SKIP] {out_path} already exists (use --overwrite to regenerate)")
        return out_path

    state = build_state_from_sample_dir(sample_dir)

    # Use geometry backend to sample a point cloud
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    pts = _state_to_point_cloud(state, n_points=n_points, device=dev)  # expected [N, 3] tensor
    if not isinstance(pts, torch.Tensor):
        raise TypeError("state_to_point_cloud must return a torch.Tensor")

    pts = pts.detach().cpu().numpy().astype(np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"state_to_point_cloud returned invalid shape {pts.shape}, expected [N, 3]")

    np.save(out_path, pts)
    print(f"[OK] Wrote {out_path} with shape {pts.shape}")
    return out_path


def iter_sample_dirs(root: Path):
    """Yield all subdirectories of `root` that look like sample dirs.

    A directory is considered a sample dir if it contains either
    `dsl.npz` (preferred) or `hist_tokens.npy` (legacy).
    """

    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if (p / "dsl.npz").exists() or (p / "hist_tokens.npy").exists():
            yield p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GT point clouds from sample DSL state")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample-dir", type=str, help="Path to a single sample directory")
    group.add_argument("--root", type=str, help="Root directory containing multiple sample dirs")

    parser.add_argument("--n-points", type=int, default=2048, help="Number of points to sample per shape")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing gt_points.npy")
    parser.add_argument("--device", type=str, default=None, help="Torch device (e.g. cuda, cpu)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.sample_dir is not None:
        sample_dir = Path(args.sample_dir).resolve()
        if not sample_dir.is_dir():
            raise SystemExit(f"Not a directory: {sample_dir}")
        generate_gt_points_for_sample(sample_dir, n_points=args.n_points, overwrite=args.overwrite, device=args.device)
        return

    root = Path(args.root).resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    print(f"Scanning for sample dirs under {root}...")
    for sample_dir in iter_sample_dirs(root):
        generate_gt_points_for_sample(sample_dir, n_points=args.n_points, overwrite=args.overwrite, device=args.device)


if __name__ == "__main__":  # pragma: no cover
    main()
