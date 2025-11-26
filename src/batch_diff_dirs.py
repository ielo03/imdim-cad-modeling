#!/usr/bin/env python3
"""Batch-compare renders in two directories.

Expect structure:
    DIR_A/
      model_00001/
        view_front.png
        view_back.png
        ...
      model_00002/
        ...
    DIR_B/
      model_00001/
        view_front.png
        view_back.png
        ...
      ...

The script:
- Discovers the set of image filenames to compare by inspecting a local "renders/" directory
  (project-root) if present, otherwise uses a default set:
    ["view_front.png", "view_back.png", "view_left.png", "view_right.png", "view_top.png", "view_bottom.png"]
- For each model subdirectory present in DIR_A, finds the same subdirectory name in DIR_B and
  compares each image using dataset.train.diff_png.diff_png().
- Emits a CSV with per-view diffs and a mean diff per model. Missing files are noted.

Usage:
    python3 dataset/train/batch_diff_dirs.py /abs/path/to/dir_a /abs/path/to/dir_b --out diffs.csv

Notes:
- The script imports and uses dataset.train.diff_png.diff_png, which prefers Pillow+NumPy
  and falls back to ImageMagick compare if available.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import List, Sequence, Set, Tuple

# Ensure project root is on path so we can import the diff util.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from src.diff_png import diff_png
except Exception:
    # Try alternate import path if running as a module
    try:
        from diff_png import diff_png  # type: ignore
    except Exception as exc:
        raise ImportError(
            "Could not import diff_png. Ensure dataset/train/diff_png.py is present "
            "and the project root is on PYTHONPATH."
        ) from exc


DEFAULT_VIEW_NAMES = [
    "view_front.png",
    "view_back.png",
    "view_left.png",
    "view_right.png",
    "view_top.png",
    "view_bottom.png",
]


def discover_view_names_from_renders(renders_dir: str) -> List[str]:
    """Scan `renders_dir` for png filenames and return a sorted list of unique basenames."""
    names: Set[str] = set()
    if not os.path.isdir(renders_dir):
        return []
    for root, _, files in os.walk(renders_dir):
        for fn in files:
            if fn.lower().endswith(".png"):
                names.add(fn)
    return sorted(names)


def list_model_subdirs(d: str) -> List[str]:
    """Force flat-mode: treat dir d as containing the view PNGs directly."""
    return ["."]


def compare_model(dir_a: str, dir_b: str, model: str, view_names: Sequence[str]) -> Tuple[str, dict]:
    """Compare all view images for a single model; return (model, results dict).

    Supports two layout modes:
      - per-model subdirectories: dir_a/model_name/<views...>
      - flat renders directory mode: when `model == '.'` we treat dir_a and dir_b
        themselves as the directories holding the view PNGs.
    """
    results = {}

    # Support flat mode marker '.' which means compare files directly in dir_a/dir_b
    if model == ".":
        base_a = dir_a
        base_b = dir_b
    else:
        base_a = os.path.join(dir_a, model)
        base_b = os.path.join(dir_b, model)

    if not os.path.isdir(base_b):
        results["_missing_model_in_b"] = True
        # mark all views as missing
        for v in view_names:
            results[v] = None
        results["_mean"] = None
        # return a human-friendly model name when in flat mode
        model_name = os.path.basename(os.path.abspath(dir_a)) if model == "." else model
        return model_name, results

    diffs = []
    for v in view_names:
        path_a = os.path.join(base_a, v)
        path_b = os.path.join(base_b, v)
        if not os.path.exists(path_a):
            results[v] = None
            continue
        if not os.path.exists(path_b):
            results[v] = None
            continue
        try:
            val = diff_png(path_a, path_b)
        except Exception as e:
            # capture error string for later debugging
            results[v] = f"ERROR: {e}"
            continue
        results[v] = float(val)
        diffs.append(float(val))

    if diffs:
        results["_mean"] = sum(diffs) / len(diffs)
    else:
        results["_mean"] = None

    model_name = os.path.basename(os.path.abspath(dir_a)) if model == "." else model
    return model_name, results

# ---------------------------------------------------------------------------
# New API requested:
#  - compare_model_scores(...) -> (model_name, scores_dict)
#      where scores_dict maps integer indexes 0..N-1 to per-view score (float) or None
#  - compute_mse_from_scores(scores_dict) -> float | None
#      computes mean squared error across available numeric scores
# ---------------------------------------------------------------------------


def compare_model_scores(dir_a: str, dir_b: str, model: str, view_names: Sequence[str]) -> Tuple[str, dict]:
    """Return per-view scores mapped to integer indices (0..N-1).

    The returned tuple is (model_name, scores_dict) where scores_dict is a dict:
        {0: score_for_view_names[0], 1: score_for_view_names[1], ...}
    A score is a float (normalized RMSE) when available, or None when the file
    was missing or an error occurred.
    """
    # Reuse compare_model's layout logic but collect scores by index.
    # Support the same flat-mode marker '.' as compare_model.
    if model == ".":
        base_a = dir_a
        base_b = dir_b
    else:
        base_a = os.path.join(dir_a, model)
        base_b = os.path.join(dir_b, model)

    model_name = os.path.basename(os.path.abspath(dir_a)) if model == "." else model

    scores = {}
    if not os.path.isdir(base_b):
        # mark all views as missing
        for idx in range(len(view_names)):
            scores[idx] = None
        return model_name, scores

    for idx, v in enumerate(view_names):
        path_a = os.path.join(base_a, v)
        path_b = os.path.join(base_b, v)
        if not os.path.exists(path_a) or not os.path.exists(path_b):
            scores[idx] = None
            continue
        try:
            val = diff_png(path_a, path_b)
        except Exception:
            scores[idx] = None
            continue
        scores[idx] = float(val)

    return model_name, scores


def compute_rms_from_scores(scores: dict) -> Optional[float]:
    """Compute RMS (root-mean-square) of normalized RMSE scores from a scores dict.

    The RMS is computed as sqrt(mean(score**2)) over numeric entries and is in
    the same [0,1] range as the per-view normalized RMSE values returned by
    diff_png(). Returns None if there are no numeric entries.
    """
    numeric = [float(v) for v in scores.values() if isinstance(v, (float, int))]
    if not numeric:
        return None
    mse = sum(x * x for x in numeric) / len(numeric)
    rms = mse ** 0.5
    return float(rms)


def rms_to_reward(rms: Optional[float], mode: str = "linear", **kwargs) -> Optional[float]:
    """Map an RMS error to a reward in [0,1].

    Parameters
    ----------
    rms : Optional[float]
        RMS error in [0,1] or None. If None, the function returns None.
    mode : str
        One of:
          - "linear": reward = 1 - rms
          - "exp":    reward = exp(-k * rms)   (k provided via kwargs, default 5.0)
          - "power":  reward = (1 - rms) ** gamma (gamma via kwargs, default 1.0)
    Returns
    -------
    Optional[float]
        Reward in [0,1] or None.
    """
    import math

    if rms is None:
        return None

    rms = float(rms)
    if mode == "linear":
        return float(max(0.0, min(1.0, 1.0 - rms)))
    if mode == "exp":
        k = float(kwargs.get("k", 5.0))
        return float(math.exp(-k * rms))
    if mode == "power":
        gamma = float(kwargs.get("gamma", 1.0))
        return float(max(0.0, (1.0 - rms)) ** gamma)

    raise ValueError(f"Unknown mode: {mode!r}")


def write_csv(out_path: str, view_names: Sequence[str], rows: Sequence[Tuple[str, dict]]) -> None:
    """Write results to CSV. Columns: model, <views...>, mean, notes"""
    headers = ["model"] + list(view_names) + ["mean", "notes"]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        for model, res in rows:
            notes = []
            row = [model]
            for v in view_names:
                val = res.get(v)
                if val is None:
                    row.append("")
                elif isinstance(val, (float, int)):
                    row.append(f"{val:.6f}")
                else:
                    # error string
                    row.append("")
                    notes.append(f"{v}:{val}")
            mean = res.get("_mean")
            row.append(f"{mean:.6f}" if isinstance(mean, (float, int)) else "")
            row.append(";".join(notes))
            writer.writerow(row)


def main(argv: Sequence[str]) -> int:
    p = argparse.ArgumentParser(description="Batch-diff renders between two directories (model subdirs with render PNGs).")
    p.add_argument("dir_a", help="Path to directory A (contains subdirs per model)")
    p.add_argument("dir_b", help="Path to directory B (contains subdirs per model)")
    p.add_argument("--renders-dir", default="renders", help="Optional renders/ example dir to discover view filenames")
    p.add_argument("--out", default="diffs.csv", help="CSV output file (created in cwd or absolute path)")
    p.add_argument("--models", nargs="*", help="Optional explicit list of model subdirs to compare (by name).")
    p.add_argument("--threshold", type=float, default=None, help="Optional mean-diff threshold to mark failures.")
    args = p.parse_args(list(argv))

    dir_a = os.path.abspath(args.dir_a)
    dir_b = os.path.abspath(args.dir_b)

    # Discover view names
    discovered = discover_view_names_from_renders(args.renders_dir)
    view_names = discovered if discovered else DEFAULT_VIEW_NAMES
    if not view_names:
        print("No view names discovered and no defaults available. Exiting.")
        return 2

    # Determine models to iterate
    if args.models:
        models = sorted(args.models)
    else:
        models = list_model_subdirs(dir_a)
        # If no subdirectories found, allow a flat renders directory case where
        # dir_a itself contains PNGs (e.g. "renders/" with view_front.png, ...).
        if not models:
            pngs = [f for f in os.listdir(dir_a) if f.lower().endswith(".png")]
            if pngs:
                # Use the special marker '.' to indicate flat-mode (compare files
                # directly in dir_a vs dir_b).
                models = ["."]
            else:
                print(f"No model subdirectories found in {dir_a}. Nothing to do.")
                return 1

    rows = []
    for model in models:
        model_path_a = os.path.join(dir_a, model)
        if not os.path.isdir(model_path_a):
            # skip non-subdir entries
            continue
        model_name, res = compare_model(dir_a, dir_b, model, view_names)
        rows.append((model_name, res))

    write_csv(args.out, view_names, rows)

    # Print summary (counts and failures if threshold provided)
    n = len(rows)
    print(f"Wrote {args.out} with {n} models compared.")
    if args.threshold is not None:
        fails = [m for m, r in rows if isinstance(r.get("_mean"), (float, int)) and r.get("_mean") > args.threshold]
        print(f"{len(fails)} models exceed threshold {args.threshold} (examples: {fails[:10]})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))