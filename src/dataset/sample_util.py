

"""Utility to create synthetic training samples for the CAD modeling project.

This script generates a single sample directory under a specified root.
Each sample consists of:
  - dsl.npz  (hist_tokens, hist_params, next_tokens, next_params)

Usage (example):
    python -m dataset.sample_util --out-dir dataset/train --idx 0

You can customize the synthetic example in `make_synthetic_sample`.
"""

import argparse
from pathlib import Path
import numpy as np

# Token IDs must match Token enum in state_machine
# (simplified here; adjust as needed)
ADD_BOX = 0
ADD_SPHERE = 1
ADD_CYLINDER = 2
MAKE_LAST_NEGATIVE = 3

def make_synthetic_sample():
    """Return a synthetic sample: (hist_tokens, hist_params, next_tokens, next_params).

    hist_tokens: [T_hist]
    hist_params: [T_hist, 9]
    next_tokens: [K]
    next_params: [K, 9]

    Modify this as needed to generate random or more complex samples.
    """
    # Example: history of two primitives
    hist_tokens = np.array([ADD_BOX, ADD_SPHERE], dtype=np.int64)

    # params: [cx,cy,cz,  p0,p1,p2,  rx,ry,rz]
    # For box, p0..p2 = size
    hist_params = np.array([
        [0.0, 0.0, 0.0,   2.0, 2.0, 2.0,   0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5,   1.0, 0.0, 0.0,   0.0, 0.0, 0.0],  # sphere: p0=radius
    ], dtype=np.float32)

    # Next action: add cylinder and then make negative
    next_tokens = np.array([ADD_CYLINDER, MAKE_LAST_NEGATIVE], dtype=np.int64)

    next_params = np.array([
        [0.5, 0.0, 0.5,   0.5, 2.0, 0.0,   0.0, 45.0, 0.0],  # cylinder
        [0.0, 0.0, 0.0,   0.0, 0.0, 0.0,   0.0, 0.0, 0.0],    # make negative
    ], dtype=np.float32)

    return hist_tokens, hist_params, next_tokens, next_params


def save_sample(out_dir: Path, idx: int):
    """Create a sample directory and write dsl.npz."""
    sample_dir = out_dir / f"sample_{idx:04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    hist_tokens, hist_params, next_tokens, next_params = make_synthetic_sample()

    np.savez(
        sample_dir / "dsl.npz",
        hist_tokens=hist_tokens,
        hist_params=hist_params,
        next_tokens=next_tokens,
        next_params=next_params,
    )
    print(f"[OK] Wrote {sample_dir}/dsl.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, required=True, help="Root dataset directory")
    parser.add_argument("--idx", type=int, default=0, help="Sample index to write")
    args = parser.parse_args()

    save_sample(Path(args.out_dir), args.idx)