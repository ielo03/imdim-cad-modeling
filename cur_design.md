# Current Design and Dataflow (up-to-date)

This document describes the current repository structure, dataflow, model components, training processes, and issues discovered so far. It has been updated after re-reading the current codebase to ensure accuracy.

Important file references (click to open at the referenced line):

- Core training entry: [`src/model/pretrain.py`](src/model/pretrain.py:1)
- Model utilities and forward used by training: [`forward_params_only()`](src/model/model_utils.py:269)
- Build model helper: [`build_cad_model()`](src/model/model_utils.py:183)
- Param scale computation utility: [`src/model/compute_param_scales.py`](src/model/compute_param_scales.py:1)
- ParamHead implementation: [`src/model/param_head.py`](src/model/param_head.py:1)
- TokenHead implementation: [`src/model/token_head.py`](src/model/token_head.py:1)
- PointNet encoder: [`src/model/pointnet_encoder.py`](src/model/pointnet_encoder.py:1)
- Transformer backbone: [`src/model/transformer.py`](src/model/transformer.py:1)

Summary of what I re-read and verified

- I re-read the current training entry at [`src/model/pretrain.py`](src/model/pretrain.py:1) (dataset class, collate, loss helpers, and the `train_epoch` implementation).
- I re-read utilities in [`src/model/model_utils.py`](src/model/model_utils.py:1), including the model builder [`build_cad_model()`](src/model/model_utils.py:183) and the forward used by training [`forward_params_only()`](src/model/model_utils.py:269).
- I re-read the dataset-scale helper script at [`src/model/compute_param_scales.py`](src/model/compute_param_scales.py:1) which computes global per-dimension scales and saves them as a .npy file.
- I inspected the model-level modules (encoder, transformer, heads) referenced above to confirm interfaces and how predictions are produced.

Updated, accurate dataflow (step-by-step)

1. Data loading

   - The dataset class [`NPZCADDataset`](src/model/pretrain.py:62) reads .npz files and returns:
     - `gt_points` [B, N, 3], `cur_points` [B, M, 3], `gt_tokens` [B, T], `gt_params` [B, T, 10], and `hist_len` [B].
   - Collation: [`cad_collate()`](src/model/pretrain.py:194) resamples point clouds to fixed sizes and stacks sequences. It currently requires identical sequence length T across a batch (no padding/masking).

2. Encoding & conditioning

   - Point clouds are encoded using [`PointNetEncoder`](src/model/pointnet_encoder.py:1) and used to produce `gt_embed` and `cur_embed`.
   - An error embedding is computed by [`compute_error_embedding()`](src/model/model_utils.py:86) (chamfer distance, centroid distance, statistics).

3. Sequence model

   - The Transformer (built with [`build_cad_model()`](src/model/model_utils.py:183)) receives token sequences (teacher-forced during pretraining), plus `gt_embed`, `cur_embed`, and `err_embed` conditioning, and outputs `hidden_states` [B, T, d_model]. See [`CADTransformer`](src/model/transformer.py:1).

4. Heads and predictions

   - Token logits: [`TokenHead`](src/model/token_head.py:1) applied to `hidden_states` → `token_logits` [B, T, V].
   - Param predictions: the training forward currently uses [`forward_params_only()`](src/model/model_utils.py:269) which calls `param_head` per timestep (Python loop), producing `params_pred` [B, T, 10]. The per-step call pattern is:
     - h_t = hidden_states[:, t, :]
     - tok_t = tokens[:, t]
     - params_t = param_head(h_t, tok_t)

5. Supervision & losses (immediate-next)
   - Training uses "immediate-next" supervision: for each sample, the supervised timestep index is `hist_len[b]` (first element of the `next` sequence). This is implemented in [`train_epoch()`](src/model/pretrain.py:411).
   - Token loss: cross-entropy on `token_logits` at the supervised index (helper [`compute_token_ce_loss()`](src/model/pretrain.py:347)).
   - Param loss: masked MSE on the 10D parameter vector at the supervised index (helper [`compute_param_mse_loss()`](src/model/pretrain.py:296)). When `scales` are provided, the MSE is computed on (pred - gt)/scales.

Key code design choices and why they matter

- Immediate-next supervision (first element of next) matches dataset convention where `next_params` often contains the actual primitive in index 0 and an END token afterward. This avoids supervising on trailing END tokens that lack params.
- Teacher-forcing for tokens in `forward_params_only()` simplifies stabilizing gradient flow to ParamHead but creates exposure bias at inference where ParamHead sees predicted tokens instead of ground-truth tokens.
- Per-batch collation currently enforces equal T across the batch — no padding or mask logic exists. This keeps implementation simple but prevents flexible batching and requires all samples in a batch to have identical sequence length.

Issues found and verification (current, after re-read)

- Parameter normalization and zero-variance handling:
  - The `compute_param_mse_loss()` helper clamps scales with `torch.clamp(s, min=1e-6)` then divides by `s` (see [`compute_param_mse_loss()`](src/model/pretrain.py:330)). This can still cause huge losses if upstream logic produces tiny scales or if per-batch scales were used; however `compute_param_scales.py` (the dataset-level utility) already computes dataset-level scales and sets very small values to 1.0 before saving (see [`src/model/compute_param_scales.py`](src/model/compute_param_scales.py:27..31)). Use of the dataset-level scales avoids the exploding-loss scenario.
- Presence of `compute_param_scales.py`:
  - There is a helper script at [`src/model/compute_param_scales.py`](src/model/compute_param_scales.py:1) that:
    - Collects the first `next_params[0]` across all .npz files,
    - Computes per-dimension max-abs,
    - Replaces very small scales (< 1e-6) with 1.0,
    - Saves the scales as a .npy file for training to consume.
- Per-batch vs dataset-level scaling:
  - Earlier diagnostics used per-batch max-abs for normalization which was noisy and included tiny clamps; the stable approach is to run the provided `compute_param_scales.py` once and load the resulting `models/param_scales.npy` in training (`load_param_scales()` exists in [`src/model/pretrain.py`](src/model/pretrain.py:384)).
- ParamHead calling convention:
  - `ParamHead` expects (hidden, token) per-step; this is implemented in a Python loop in [`forward_params_only()`](src/model/model_utils.py:337..343). This is functional but slow; vectorizing `ParamHead` to accept [B, T, d_model] + [B, T] tokens and return [B, T, 10] is recommended for performance.
- Exposure bias / teacher forcing:
  - Confirmed: `forward_params_only()` uses ground-truth `tokens` to compute params predictions during training. At inference, `ParamHead` will be fed token predictions from `TokenHead`. Consider scheduled sampling or occasional feeding of predicted tokens during training to reduce exposure bias.

Concrete, reproducible next steps I recommend (priority)

1. Ensure `compute_param_scales.py` is run and training loads `models/param_scales.npy`, so `train_epoch()` receives stable `param_scales` (callable via `load_param_scales()` in [`src/model/pretrain.py`](src/model/pretrain.py:384)). This avoids dividing by tiny clamps and exploding param loss.
2. Update `compute_param_mse_loss()` to explicitly treat zero-variance dims by setting scales <= 0 to 1.0 before division, and _document_ the convention.
3. Add padding + masking support in `cad_collate()` (currently at [`src/model/pretrain.py`](src/model/pretrain.py:194)) so variable-length sequences can be batched and masked during loss computation.
4. Vectorize `ParamHead` to accept full sequences, remove Python-level loop in [`forward_params_only()`](src/model/model_utils.py:269).
5. Introduce a lightweight scheduled-sampling strategy for `ParamHead` training to reduce exposure bias (alternating ground-truth vs predicted tokens).
6. Add a single-sample overfit unit test using [`forward_params_only()`](src/model/model_utils.py:269) + `train_epoch()` to verify the model can memorize a toy example (this guards against regressions).

What I changed earlier vs what exists now

- Earlier debugging added per-batch debug runners; after re-reading the current workspace, the authoritative pieces present now are:
  - The primary training loop and helpers in [`src/model/pretrain.py`](src/model/pretrain.py:1).
  - The global scale calculator in [`src/model/compute_param_scales.py`](src/model/compute_param_scales.py:1).
  - The `model_utils` module with `forward_params_only()` which still uses a per-timestep Python loop.
- If you have local edits not yet present in the repository (you indicated you made updates), I have re-read the current files and updated this design doc accordingly. If you want me to include newer local changes, please ensure they are saved in the repository and I will re-read and re-sync `cur_design.md`.

References inside this doc (clickable)

- Training loop and helpers: [`src/model/pretrain.py`](src/model/pretrain.py:1)
- Forward and model builder: [`forward_params_only()`](src/model/model_utils.py:269), [`build_cad_model()`](src/model/model_utils.py:183)
- Global scale computation: [`src/model/compute_param_scales.py`](src/model/compute_param_scales.py:1)
- ParamHead (candidate for vectorization): [`src/model/param_head.py`](src/model/param_head.py:1)

If you'd like, I can:

- Run [`src/model/compute_param_scales.py`](src/model/compute_param_scales.py:1) on your dataset and save `models/param_scales.npy`, then update training invocation to load it and re-run a short debug epoch; or
- Implement the minimal safety change to `compute_param_mse_loss()` so zero-variance dims are always handled as scale = 1.0; or
- Vectorize `ParamHead` (larger code change).

---

History of error formulas used (evolution and current implementation)

Background

- Early iterations of the project used a single scalar geometry error (Chamfer distance) between the current-state and ground-truth point clouds as the conditioning signal for the sequence model.
- Over time the error embedding was expanded to give the Transformer richer, multi-scale geometric summary statistics to help it predict tokens and parameters conditioned on how the current reconstruction deviates from GT.

Timeline / evolution

1. Initial: Chamfer-only

   - The earliest design used only chamfer distance (symmetric nearest-neighbor distance between two point clouds) as the error metric. This provided a single scalar describing global shape difference but lacked information about centroid shift, scale, or per-point error distribution.

2. Expanded embedding (current)
   - The error representation was expanded to an 8-dimensional embedding computed in [`compute_error_embedding()`](src/model/model_utils.py:86). The components are: 0) Chamfer distance (symmetric)
     - Implementation: `chamfer_distance(cur, gt)` (no grad inside embedding computation).
     - Intuition: captures overall shape mismatch between current and ground-truth clouds.
     1. Centroid distance
        - Formula: L2 norm between centroids: || centroid_gt - centroid_cur ||\_2.
        - Intuition: captures gross translation/offset of the current reconstruction.
     2. Mean per-point L2 error (aligned subset)
        - Computed by aligning the first N = min(N_gt, N_cur) points and taking mean of per-point L2 distances.
        - Intuition: average local discrepancy.
     3. Max per-point L2 error (aligned subset)
        - The maximum per-point error in the aligned subset; highlights outliers or missed features.
     4. Std per-point L2 error (aligned subset)
        - Standard deviation of per-point L2 errors; indicates spread/consistency of errors.
     5. Scale ratio
        - Formula: cur_radius / (gt_radius + eps), where radius is mean distance of points to centroid.
        - Intuition: captures global size/scale differences between clouds.
     6. GT average radius
        - Mean distance from GT centroid to GT points (a rough size indicator).
     7. CUR average radius
        - Mean distance from current centroid to current points.

Implementation notes

- Chamfer is computed via the geometry backend and is called under `torch.no_grad()` inside the embedding calculation to avoid propagating gradients through the expensive geometry computation.
- Per-point errors are computed on the aligned prefix (first N points) rather than using a nearest-neighbor pairing; this is pragmatic and cheap given that point clouds are already resampled deterministically during collate. If higher fidelity is required, a proper NN-based per-point matching could be used but would be more expensive.
- Scale ratio uses a small epsilon (1e-6) to avoid divide-by-zero for degenerate clouds.

Why expand the embedding?

- A single scalar (chamfer) conflates translation, scale, and local structure errors. The richer embedding disentangles these effects so the Transformer can condition differently on, e.g., translation offset versus local missing geometry when predicting the next primitive and its parameters.

Where this embedding is used

- The error embedding (`err_embed`) is passed into the Transformer as additional conditioning along with `gt_embed` and `cur_embed` inside [`forward_params_only()`](src/model/model_utils.py:269) and the model builder created by [`build_cad_model()`](src/model/model_utils.py:183).

If you want, I can:

- Add explicit formula snippets (math-like pseudocode) for each entry to `cur_design.md`.
- Replace the aligned-prefix per-point error with a nearest-neighbor-based mean/max/std or augment the embedding with a Hausdorff metric if you want more robust geometric signals.
- Expose a flag to compute chamfer with gradients (if needed) though that may be expensive and complicate training.

Pick an option and I'll implement it next.
