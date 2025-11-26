#!/usr/bin/env python3
"""Compute a diff value between two PNG images.

Primary strategy:
- Use Pillow + NumPy to compute the per-pixel Root-Mean-Square (RMS)
  difference normalized to [0.0, 1.0].

Fallback:
- If PIL/NumPy are not available, attempt to call ImageMagick's `compare`
  with the `-metric RMSE` option and parse the output.

Function
--------
diff_png(png1: str, png2: str) -> float
    Accepts absolute paths to two PNG files and returns a float diff value.
    Lower is more similar. The returned RMSE value is normalized to [0.0, 1.0]
    (0.0: identical; 1.0: maximal per-channel difference).

Notes
-----
- The function requires that both images have the same dimensions. If sizes
  differ, a ValueError is raised. This preserves exact pixel-wise comparison.
- It raises FileNotFoundError if files are missing, and RuntimeError if no
  supported diff backend is available.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from typing import Optional

# Try to import PIL/NumPy but keep imports local inside function to avoid hard
# dependency at import time for callers that only want CLI fallback behavior.


def _is_abs_path(p: str) -> bool:
    return os.path.isabs(p)


def _ensure_abs(p: str) -> str:
    if not _is_abs_path(p):
        return os.path.abspath(p)
    return p


def diff_png(png1: str, png2: str) -> float:
    """Compute normalized RMS difference between two PNG files.

    Parameters
    ----------
    png1, png2 : str
        Absolute or relative paths to PNG files. Relative paths will be resolved.

    Returns
    -------
    float
        Normalized RMSE in range [0.0, 1.0]. Smaller => more similar.

    Raises
    ------
    FileNotFoundError
        If either input path does not exist.
    ValueError
        If images have different sizes.
    RuntimeError
        If no diff backend (Pillow+NumPy or ImageMagick) is available or parsing fails.
    """
    p1 = _ensure_abs(png1)
    p2 = _ensure_abs(png2)

    if not os.path.exists(p1):
        raise FileNotFoundError(f"File not found: {p1}")
    if not os.path.exists(p2):
        raise FileNotFoundError(f"File not found: {p2}")

    # Try PIL + NumPy first (native, fast, no external CLI)
    try:
        from PIL import Image
        import numpy as np  # type: ignore
    except Exception:
        Image = None  # type: ignore
        np = None  # type: ignore

    if Image is not None and np is not None:
        img1 = Image.open(p1).convert("RGBA")
        img2 = Image.open(p2).convert("RGBA")

        if img1.size != img2.size:
            raise ValueError(f"Image sizes differ: {img1.size} vs {img2.size}")

        a1 = np.asarray(img1).astype(np.float32)
        a2 = np.asarray(img2).astype(np.float32)

        # Compute per-channel squared difference, mean over all channels & pixels
        diff = a1 - a2
        mse = float((diff * diff).mean())
        rmse = float(mse ** 0.5)

        # Max possible per-channel diff is 255. We normalize by 255 to give a
        # value in [0, 1].
        rmse_norm = rmse / 255.0
        return rmse_norm

    # Fallback: use ImageMagick's `compare -metric RMSE`
    # Example output to stderr: "18345 (0.12345)"
    compare_cmd = ["compare", "-metric", "RMSE", p1, p2, "null:"]

    try:
        proc = subprocess.run(compare_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except FileNotFoundError:
        raise RuntimeError(
            "No supported image diff backend available. Install Pillow+NumPy "
            "or ImageMagick (`compare` command)."
        )

    # ImageMagick writes metric output to stderr for compare -metric RMSE
    stderr = proc.stderr.decode("utf-8", errors="ignore").strip()
    stdout = proc.stdout.decode("utf-8", errors="ignore").strip()

    text = stderr if stderr else stdout

    # Try to parse a parenthesized normalized value: e.g. "18345 (0.12345)"
    m = re.search(r"\(([-+eE0-9.]+)\)", text)
    if m:
        try:
            val = float(m.group(1))
            # ImageMagick's normalized RMSE is already in [0, 1] (per-pixel),
            # so we can directly return it.
            return val
        except Exception:
            pass

    # As a fallback, try to parse the first floating number in the output
    m2 = re.search(r"([-+eE0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)", text)
    if m2:
        try:
            val = float(m2.group(1))
            # If this looks larger than 1.0 it's probably the raw pixel RMSE;
            # attempt to normalize if reasonable (>1 and <=255).
            if val > 1.0 and val <= 255.0:
                return val / 255.0
            return val
        except Exception:
            pass

    raise RuntimeError(f"Unable to parse compare output: {text!r}")


# Optional CLI for quick checks:
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute normalized RMSE diff between two PNGs.")
    parser.add_argument("png1", help="Path to first PNG")
    parser.add_argument("png2", help="Path to second PNG")
    parser.add_argument("--raw", action="store_true", help="Print raw (un-normalized) RMSE * 255 value when using PIL+NumPy backend")
    args = parser.parse_args()

    try:
        val = diff_png(args.png1, args.png2)
    except Exception as e:
        print("ERROR:", e)
        raise SystemExit(2)

    print(val)