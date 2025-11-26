# Train dataset SCAD programs

This directory contains generated OpenSCAD programs for training.

- `all_openscad` (100 files): 100 varied, valid OpenSCAD programs created by the generator.
- `ast_only` (100 files): 100 programs restricted to the AST subset defined in
  [`test_files/ast_codegen/ast.md`](test_files/ast_codegen/ast.md:1) and serializable by
  [`src/scad_codegen.py`](src/scad_codegen.py:1).

Generation script: [`dataset/train/generate_scads.py`](dataset/train/generate_scads.py:1)

Reproducibility

- Random seed: 12345 (set in the generator).
- To regenerate the files, run:
  python3 dataset/train/generate_scads.py

Notes

- Files are named `scad_XXXXX.scad` and include a header comment with the generator name.
- The `ast_only` set only uses primitives, transforms, and CSG nodes from the AST spec.

Contact

- If you need changes to the generation rules (ranges, bins, or variety), edit
  [`dataset/train/generate_scads.py`](dataset/train/generate_scads.py:1).

## Diffing images with `diff_png`

- Script: [`dataset/train/diff_png.py`](dataset/train/diff_png.py:1)
- Function: [`diff_png()`](dataset/train/diff_png.py:49)

Quick CLI usage

- Run from project root:
  python3 dataset/train/diff_png.py /absolute/path/to/img1.png /absolute/path/to/img2.png
  (prints a normalized RMSE value in [0.0, 1.0]; 0.0 = identical)

Python usage

- Example (from project root):
  import sys
  sys.path.insert(0, ".") # ensure project root on PYTHONPATH
  from dataset.train.diff_png import diff_png
  val = diff_png("/absolute/path/to/img1.png", "/absolute/path/to/img2.png")
  print(val)

Dependencies

- Preferred: Pillow + NumPy (pip install pillow numpy)
- Fallback: ImageMagick `compare` command (e.g. brew install imagemagick)
- The script will use Pillow+NumPy if available, otherwise it will call ImageMagick.

Notes

- The function requires both images to have the same dimensions (raises ValueError otherwise).
- The returned value is normalized RMSE (0..1). Use a small threshold (e.g. 0.01) to consider images identical for most cases.

## RL reward helpers (RMS + reward mapping)

I implemented an RMS-based aggregator and reward mapper in the batch comparator file. Use these when you want a single scalar reward returned to your RL agent.

- Per-view diff implementation: [`src/diff_png.py`](src/diff_png.py:49)
- Per-model per-view scores: [`src/batch_diff_dirs.py`](src/batch_diff_dirs.py:157) (function: `compare_model_scores`)
- RMS aggregator + reward mapper: [`src/batch_diff_dirs.py`](src/batch_diff_dirs.py:199) (functions: `compute_rms_from_scores`, `rms_to_reward`)

Quick example (Python)

- From project root:
  import sys
  sys.path.insert(0, ".")
  from src.batch_diff_dirs import compare_model_scores, compute_rms_from_scores, rms_to_reward, discover_view_names_from_renders

  views = discover_view_names_from_renders("renders") or [
  "view_front.png","view_back.png","view_left.png",
  "view_right.png","view_top.png","view_bottom.png"
  ]

  # Compare renders/ vs renders_ref/ in flat mode (model=".")

  model_name, scores = compare_model_scores("renders", "renders_ref", ".", views)

  # scores is a dict {0:score0, 1:score1, ..., 5:score5} where scores are normalized RMSE in [0,1] or None

  rms = compute_rms_from_scores(scores)

  # rms is a float in [0,1] (or None if no numeric entries)

  # Convert RMS -> reward in [0,1] (linear baseline)

  reward = rms_to_reward(rms, mode="linear") # reward = 1 - rms

  # Or exponential shaping (sharpen near-perfect)

  reward_exp = rms_to_reward(rms, mode="exp", k=5.0)

Notes

- The per-view values returned by [`src/diff_png.py`](src/diff_png.py:49) are normalized RMSE in [0,1].
- I recommend using RMS (sqrt of mean squared per-view RMSE) as the aggregator for RL reward shaping — it preserves units and penalizes large per-view errors.
- Use `rms_to_reward(..., mode="linear")` as a stable default (reward = 1.0 - rms). For stronger shaping try the exponential mode.

- The per-view values returned by [`src/diff_png.py`](src/diff_png.py:49) are normalized RMSE in [0,1].
- I recommend using RMS (sqrt of mean squared per-view RMSE) as the aggregator for RL reward shaping — it preserves units and penalizes large per-view errors.
- Use `rms_to_reward(..., mode="linear")` as a stable default (reward = 1.0 - rms). For stronger shaping try the exponential mode.
