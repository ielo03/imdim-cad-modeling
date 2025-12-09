#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np

def main(data_root: str, out_path: str):
    root = Path(data_root)
    paths = sorted(root.glob("*.npz"))
    if not paths:
        raise SystemExit(f"No .npz files found under {root}")

    all_next = []

    for p in paths:
        data = np.load(p)
        if "next_params" not in data:
            continue
        next_params = data["next_params"].astype(np.float32).reshape(-1, 10)
        if next_params.shape[0] == 0:
            continue
        # we only supervise the first "next" step
        all_next.append(next_params[0:1])  # [1,10]

    if not all_next:
        raise SystemExit("No next_params found in dataset")

    all_next = np.concatenate(all_next, axis=0)  # [S,10]
    scales = np.max(np.abs(all_next), axis=0)    # [10]
    scales[scales < 1e-6] = 1.0                  # avoid zeros

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, scales)
    print(f"Saved param scales to {out_path}")
    print("scales:", scales)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: compute_param_scales.py DATA_ROOT OUT_PATH")
        raise SystemExit(1)
    main(sys.argv[1], sys.argv[2])