# IMDIM CAD Modeling — Full Design

> **Audience:** professor + teammates  
> **Stack:** Python + PyTorch for learning, Rust + MicroCAD/ucad for high‑fidelity geometry & point clouds

Goal: learn to convert a **target 3D shape** (given as a point cloud) into a **short program of parametric primitives** (BOX, SPHERE, CYLINDER with pose + sign) that approximates that shape. The program is represented in a small DSL; a transformer generates the token sequence; a geometry backend turns the program into a point cloud; losses/rewards compare that to the target.

This doc describes the DSL, state machine, geometry backend, model architecture, dataset format, and training flow (supervised pretraining now, RL later as future work).

---

## 1. High‑Level Architecture

Pipeline:

1. **Input:** ground‑truth point cloud `gt_points` sampled from a target mesh (STL/PLY).

   - For synthetic training data we can use either:
     - Python analytic primitives/CSG, or
     - Rust + MicroCAD/ucad to generate more realistic meshes/fusion shapes and sample point clouds.

2. **State machine (`ShapeState`):** holds a list of primitives with parameters + sign (positive/negative). It applies a sequence of DSL tokens and parameter vectors to maintain an internal representation of the current CAD program.

3. **Geometry backend (`geometry_backend.py`):**

   - Converts a `ShapeState` into an approximate point cloud `cur_points` using simple per‑primitive sampling and union/difference logic in Python.
   - Used online during training and RL because it’s fast and differentiable enough to support Chamfer‑based feedback.

4. **Model (PyTorch):**

   - PointNet encoder for point clouds.
   - Transformer decoder over DSL token histories.
   - Token head (predicts next DSL token).
   - Parameter head (predicts 10‑D primitive parameters including sign).

5. **Training loop (`pretrain.py`):**

   - Supervised pretraining with ground‑truth token sequences + parameters and point clouds.
   - Loss = parameter MSE + sign BCE + token cross‑entropy (+ optional error embedding terms).
   - Validation loop and checkpointing of best model.

6. **Future RL:** treat the token head as a policy over DSL tokens, with reward from geometry backend or from MicroCAD‑based point clouds.

---

## 2. DSL and Program Representation

We use a **simple, append‑only DSL** that only manipulates a flat list of primitives. No nested trees, no arbitrary CSG nodes.

### 2.1 Primitive Representation

Each primitive is a dict/record with:

- `kind`: `"box" | "sphere" | "cylinder"`
- `center`: `(cx, cy, cz)`
- `size_params`: `(p0, p1, p2)` (interpreted per primitive)
- `rotation`: `(rx, ry, rz)` in degrees (Euler angles)
- `sign`: `+1` for **positive** solid, `-1` for **negative** cutout

These are derived from a **10‑D parameter vector** produced by the model when a primitive token is emitted:

```text
params[0:3] = (cx, cy, cz)       # position / center
params[3:6] = (p0, p1, p2)       # size/radius/height (primitive‑specific)
params[6:9] = (rx, ry, rz)       # rotation angles in degrees
params[9]   = sign_raw           # real‑valued sign logit
```

Sign decoding:

```text
sign_prob = sigmoid(sign_raw)
role = POSITIVE if sign_prob >= 0.5 else NEGATIVE
```

**Interpretation of size params**:

- BOX: `p0, p1, p2` → size `(sx, sy, sz)`.
- SPHERE: `p0` = radius, `(p1, p2)` ignored or used for optional ellipsoid scaling.
- CYLINDER: `p0` = radius, `p1` = height, `p2` optional.

We store the raw 10‑D vector in the state so we can regenerate exact geometry later.

### 2.2 Token Vocabulary

We use a small, RL‑friendly token set:

- `ADD_BOX`
- `ADD_SPHERE`
- `ADD_CYLINDER`
- `UNDO_LAST`
- `END`

**Semantics:**

- `ADD_*`:

  - Model chooses this token at step `t`.
  - Parameter head outputs 10‑D vector for this step.
  - State machine decodes it into a primitive and appends to the list.

- `UNDO_LAST`:

  - If there is at least one primitive, pop the last one.
  - If there is none, this token is masked (cannot be chosen).

- `END`:
  - Terminate sequence/program.

### 2.3 Masking Rules

To avoid obvious invalid actions:

- Mask `UNDO_LAST` if there are **no primitives** in the current state.
- Optionally mask `END` on the **first step** so you always add at least one primitive.

Everything else is valid by construction; we do not allow arbitrary structural tokens, so the policy always produces syntactically valid programs.

---

## 3. State Machine (`state_machine.py`)

The state machine is a pure Python class that applies token + parameter actions to build an internal list of primitives.

### 3.1 Core Data Structure

```python
class Primitive(NamedTuple):
    kind: str              # "box" | "sphere" | "cylinder"
    center: np.ndarray     # (3,)
    size: np.ndarray       # (3,)
    rotation: np.ndarray   # (3,) degrees
    sign: int              # +1 (positive) or -1 (negative)
    raw_params: np.ndarray # (10,) original param vector

class ShapeState:
    primitives: List[Primitive]
    done: bool
```

### 3.2 Applying Tokens

At step `t`, we have:

- a token `tok_t` (int ID),
- a 10‑D parameter vector `params_t` (for primitive tokens only).

`ShapeState.apply_action(tok_t, params_t)` implements:

- If `tok_t` is `ADD_BOX` / `ADD_SPHERE` / `ADD_CYLINDER`:

  - Decode `params_t` into `center`, `size`, `rotation`, `sign` as above.
  - Append a new `Primitive` of the corresponding kind.

- If `tok_t` is `UNDO_LAST`:

  - If `primitives` not empty, pop the last one.

- If `tok_t` is `END`:
  - Set `done = True`.

### 3.3 State Export for Geometry

The geometry backend consumes the state as a list of primitives:

```python
state.to_primitives() -> List[Primitive]
```

No geometry computation happens in the state machine itself; that’s delegated to `geometry_backend.py`.

---

## 4. Geometry Backend (`geometry_backend.py`)

This is the **Python approximation** used during training.

### 4.1 Primitive Sampling & Membership

For each primitive we implement:

- `sample_points(primitive, n_points)` → `[n_points, 3]` in world coords.
- `is_inside(primitive, points)` → boolean mask `[N]` for a batch of points.

This uses analytic formulas for box/sphere/cylinder in local coordinates (after applying rotation + translation).

### 4.2 Sequential Union/Difference Over Point Clouds

We maintain a global point cloud representing the current predicted shape.

```python
def state_to_point_cloud(state: ShapeState, n_points: int, device) -> torch.Tensor:
    """Approximate predicted shape as a point cloud.

    Returns points [N, 3] in torch.Tensor.
    """
    pts = empty
    for prim in state.primitives_in_order():
        if prim.sign > 0:  # POSITIVE
            pts_prim = sample_points(prim, n_pos_per_prim)
            pts = concat(pts, pts_prim)
        else:  # NEGATIVE
            # Remove points of pts that fall inside the negative primitive
            mask = ~is_inside(prim, pts)
            pts = pts[mask]

    # Optional: resample / downsample pts to exactly n_points for stability
    return resample_to_fixed_size(pts, n_points)
```

We also handle empty cases (no primitives → return zeros or random placeholder points) so the model utils never see length‑0 point clouds when that would break downstream ops.

### 4.3 Error Metrics and Embedding

We compute a **richer error embedding** between GT and current point clouds, used as conditioning for the transformer:

- Chamfer distance (scalar)
- Centroid distance (L2 distance between point cloud centroids)
- log scale ratio (log of ratio of bounding box diagonals)

`compute_error_embedding(gt_points, cur_points) -> [B, 3]` returns:

```text
[ chamfer, centroid_dist, log_scale_ratio ]
```

If either set is empty, we return zeros.

This embedding is concatenated to other conditioning features and fed into the transformer at every step.

---

## 5. Model Architecture

All model code lives under `src/model/`.

### 5.1 PointNet Encoder (`pointnet_encoder.py`)

- Input: `points : [B, N, 3]`.
- Transpose to `[B, 3, N]` and apply a small MLP with 1D conv layers:
  - `Conv1d(3 → 64) → ReLU → Conv1d(64 → 128) → ReLU → Conv1d(128 → out_dim)`.
- Max‑pool over `N` to get `[B, out_dim]`.

We instantiate one PointNet encoder and reuse it for:

- `gt_points` → `gt_embed` (global embedding of target shape).
- `cur_points` → `cur_embed` (global embedding of current prediction).

### 5.2 Transformer Decoder (`transformer.py`)

- Input tokens: `gt_tokens : [B, T]` (history + next tokens if teacher forcing).
- Token embedding: `token_emb : [B, T, d_model]`.
- Add learned positional encodings.
- Optionally condition on:
  - `gt_embed : [B, d_model]`
  - `cur_embed : [B, d_model]`
  - `err_embed : [B, 3]` (after projecting to `d_model`).

The transformer runs over the sequence of tokens and returns:

```text
hidden_states : [B, T, d_model]
```

We use **batch_first=True** so indexing is natural.

### 5.3 Token Head (`token_head.py`)

- Input: `hidden_states : [B, T, d_model]`.
- Output: `token_logits : [B, T, V]` where `V` is vocab size (number of DSL tokens).

Implementation: single linear layer:

```python
self.fc = nn.Linear(d_model, vocab_size)
```

During pretraining we always use **teacher forcing**: the transformer input tokens are the GT tokens, and we train the head to predict the same GT tokens at each position.

### 5.4 Parameter Head (`param_head.py`)

The parameter head is conditioned on:

- hidden state `h_t : [B, d_model]` for step `t`,
- `gt_embed : [B, gt_dim]`,
- `cur_embed : [B, gt_dim]`,
- error embedding `err_embed : [B, 3]`,
- token embedding `tok_emb_t : [B, tok_dim]` for the **current** step.

We build per‑step input:

```text
x_t = concat(h_t, gt_embed, cur_embed, tok_emb_t, err_embed)  # [B, D_total]
```

The head is a small MLP:

```python
self.mlp = nn.Sequential(
    nn.Linear(D_total, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, 10),   # 10D geometry + sign
)
```

Output interpretation:

- `out[..., 0:9]` → geometry params.
- `out[..., 9]` → sign logit.

### 5.5 Model Wrapper / Utilities (`model_utils.py`)

`build_cad_model(...)` constructs:

- `pointnet`
- `transformer`
- `param_head`
- `token_head`

and returns them inside a `ModelComponents` dataclass.

`forward_params_only(...)` implements the **supervised pretraining forward path**:

1. Encode `gt_points` → `gt_embed`.
2. Encode `cur_points` (or zeros if empty) → `cur_embed`.
3. Compute `err_embed` (or zeros if empty).
4. Run `transformer(gt_tokens, gt_embed, cur_embed, err_embed)` → `hidden_states`.
5. Run `token_head(hidden_states)` → `token_logits`.
6. For each step `t`, build `x_t` and run `param_head(x_t)` → `params_pred_t`.
7. Stack all `params_pred_t` to `[B, T, 10]`.

---

## 6. Dataset Format

All samples are stored as `.npz` files:

- Train: `dataset/train/sample_XXXXX.npz`
- Val: `dataset/val/sample_XXXXX.npz`

Each sample contains:

- `gt_points : [N_gt, 3]` — target point cloud.
- `cur_points : [N_cur, 3]` — point cloud obtained by applying the **history** DSL program (can be empty for early steps).
- `gt_tokens : [T]` — integer sequence of DSL tokens (history + next steps).
- `gt_params : [T, 10]` — 10‑D parameter vectors aligned with `gt_tokens`.

The **history vs next** is encoded simply by where in the sequence you supervise the model; pretraining uses full sequences.

A DataLoader (`NPZCADDataset` in `pretrain.py` or a separate module) batches these into:

- `gt_points : [B, N_gt, 3]`
- `cur_points : [B, N_cur, 3]`
- `gt_tokens : [B, T]`
- `gt_params : [B, T, 10]`

Padding/variable lengths are handled at the dataset/dataloader level or by keeping batch_size small.

---

## 7. Training Flow (`pretrain.py`)

### 7.1 Supervised Pretraining Objective

For a batch:

1. Forward pass via `forward_params_only`:

```python
params_pred, hidden_states, token_logits = forward_params_only(
    components, gt_points, cur_points, gt_tokens
)
```

2. Parameter regression loss (only on primitive tokens):

```python
L_params = MSE(params_pred[primitive_steps, :9], gt_params[primitive_steps, :9])
L_sign   = BCEWithLogits(params_pred[primitive_steps, 9], sign_gt[primitive_steps])
```

3. Token auxiliary loss (all steps):

```python
L_token = CrossEntropy(token_logits, gt_tokens)
```

4. Total loss (current implementation):

```python
loss = L_params + L_sign_weight * L_sign + L_token
```

### 7.2 Training & Validation Loops

- `train_epoch(...)`:

  - Sets all modules to `train()`.
  - Iterates batches from train dataloader.
  - Computes losses, backprop, optimizes.
  - Logs average train loss.

- `eval_epoch(...)`:

  - Sets all modules to `eval()`.
  - Uses `torch.no_grad()`.
  - Computes the same loss definition on the val dataloader.
  - Returns **avg val loss**.

- In `main()`:
  - Build train dataloader from `--data_root`.
  - Optionally build val dataloader from `--val_root`.
  - Track best val loss.
  - Save best model state dict to `models/best.pt` whenever avg val loss improves.

### 7.3 CLI

`pretrain.py` supports:

- `--data_root dataset/train`
- `--val_root dataset/val`
- `--epochs N`
- `--batch_size B`
- `--lr LR`
- `--max_seq_len T`
- `--device cpu|cuda`

Example:

```bash
python src/model/pretrain.py \
  --data_root dataset/train \
  --val_root dataset/val \
  --epochs 20 \
  --batch_size 1
```

---

## 8. MicroCAD / Rust Integration (Future Work)

Right now, point clouds for GT and current predictions are generated in **Python** using analytic primitives and a simple CSG over point sets. This keeps training fast and debuggable.

Longer‑term, we can:

1. Swap out or augment the GT point clouds with ones generated by **Rust + MicroCAD/ucad** DSL:

   - Rust library generates complex CAD geometry.
   - MicroCAD/ucad can evaluate the program and expose a fast method to sample point clouds.
   - Rust side writes `.npy` or `.npz` point clouds that the Python side loads directly.

2. Potentially use the same DSL (or a compatible subset) for both RL and offline generation.

The design keeps the DSL + state machine in Python **compatible in spirit** with a richer MicroCAD DSL so that we can move the evaluation backend to Rust later without retraining everything from scratch.

---

## 9. Future RL Phase (Outline)

Once supervised pretraining is stable:

- Treat the token head as a policy over tokens.
- Use the state machine + geometry backend as the environment.
- Reward:
  - `-Chamfer(pred, gt)`
  - penalties on number of tokens / primitives.
- Generate full episodes by sampling tokens until `END` or max length.
- Use policy gradient / PPO to improve token policy.
- Optionally finetune parameter head with RL as well or keep it under gradient‑based losses from Chamfer.

---

## 10. What’s Done vs What’s Left

**Implemented / in progress:**

- DSL design and state machine concept.
- Geometry backend with primitive sampling and sequential union/difference.
- PointNet encoder.
- Transformer + token head.
- Param head (10D output including sign).
- Model wiring + forward path (`forward_params_only`).
- Dataset format using `.npz` samples.
- Supervised pretraining loop with train/val and best‑model checkpointing.

**Future / possible extensions:**

- Swap Python geometry backend for Rust + MicroCAD/ucad in the GT pipeline.
- Introduce RL for token structure optimization.
- Extend primitive set (e.g., ellipsoids, tori) if time permits.
- Better normalization of coordinates, sizes, and rotations.

This design is scoped so that **supervised pretraining is realistic to complete now**, with a clear path to plug in MicroCAD and RL later if time and compute allow.
