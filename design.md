# Design (Simplified and Project-Ready)

Goal: Generate a sequence of parametric primitives that approximate a target mesh. Use a transformer to pick primitive types and parameters. Use a simple geometry interpreter so every token always produces a valid shape.

---

## Core DSL (append-only, simplest viable)

State: a list of primitives, each with:

- type: BOX / SPHERE / CYLINDER (keep 2–3 types)
- params: position + size (predict by MLP)
- role: positive or negative

Tokens:

- ADD_BOX
- ADD_SPHERE
- ADD_CYLINDER
- MAKE_LAST_NEGATIVE
- END

Evaluation rule:

```
shape = empty
for prim in primitives:
    if prim.role == positive:
        shape = shape ∪ prim
    else:
        shape = shape \ prim
final_shape = shape
```

This removes the need for UNION/SUBTRACT tokens or stack-based CSG. Far fewer invalid programs and easier training.

This sequential add/subtract evaluation allows the policy to recover from overly large negative primitives by adding new positives afterward, improving iterative feedback.

---

## Token Semantics

ADD\_\*: append a primitive with default role "positive".
MAKE_LAST_NEGATIVE: flip the role of the last primitive (mask this token if no primitive exists).
END: stop program generation early.

Mask illegal tokens:

- If no primitive exists, mask MAKE_LAST_NEGATIVE.
- Optionally mask END on the first step.

All other invalid actions are impossible by construction.

---

## Model

Transformer (autoregressive decoder):

- Input: previous tokens (+ optional learned embedding of partial shape state)
- Output each step:
  - primitive_type (categorical)
  - parameters via MLP head

### Hidden State and Parameter Inputs

Each decoding step t produces a transformer hidden state `h_t`. The parameter MLP receives a concatenation of:

- `h_t`: transformer context of the partial program
- `gt_embed`: global embedding of the ground-truth point cloud (PointNet-style per-point MLP + max/mean pool)
- `tok_emb`: learned embedding of the current DSL token
- `err_emb` (optional later): embedding capturing current predicted-vs-GT error features

The parameter MLP outputs continuous primitive parameters (e.g., position and scale). Primitive tokens consume these parameters; non-primitive tokens ignore them.

---

## Geometry

Primitives -> analytic mesh or sampled points.
Sample a few hundred points per shape (uniform by area). Cache ground-truth samples.

Final predicted mesh → Chamfer distance + optional normal loss.

---

## Training (Hybrid)

Do both:

1. Gradient-based loss (Chamfer + optional normals) to refine continuous parameters.
2. RL reward for structure:
   - negative Chamfer
   - penalty for too many primitives
   - small penalty per token (encourages shorter programs)

The transformer learns discrete tokens with RL. The parameter MLP learns continuous values with gradients.

### Parameter Pretraining

To pretrain parameters with known ground-truth tokens and parameters:

1. Implement a minimal transformer decoder first.
2. Teacher-force ground-truth token sequences to obtain hidden states `h_t`.
3. Encode the GT mesh once per example via a small PointNet module to obtain `gt_embed`.
4. Input to the parameter MLP at step t is `concat(h_t, gt_embed, tok_emb)` (skip `err_emb` initially).
5. Supervise parameters via MSE:

```
L_t = MSE(params_pred_t, params_gt_t)
```

6. Sum/average `L_t` over steps and backprop to train the encoder, transformer backbone, and parameter MLP.
7. After pretraining, add the token RL head and switch to hybrid training.

This ensures a good initialization for the parameter head before RL fine-tuning.

---

## Reward / Loss

Reward:

```
R = -Chamfer(final_mesh, GT) - lambda * (#primitives)
```

Loss:

```
L = Chamfer + normal_loss(optional)
```

Backprop L through parameter heads. Use RL (policy gradient or PPO) for token selection.

---

## Implementation Steps

1. Define DSL tokens and masking rules.
2. Write apply_token that appends primitives or flips role.
3. Write evaluate_shape that unions positives and subtracts negatives.
4. Sample GT mesh points once and cache.
5. Train with Chamfer loss first (warm start, maybe teacher-forcing simple sequences).
6. Add RL after the model produces roughly valid shapes.

---

## Scope

Do NOT implement full CSG trees, complex boolean logic or many primitive types. Stick to the minimal DSL above. This is the most stable approach you can finish in a shorter time and still demonstrate hybrid training, transformer decoding, and differentiable geometric feedback.

# Design (Simplified and Project-Ready)

Goal: Generate a sequence of parametric primitives that approximate a target mesh. Use a transformer to pick primitive types and a parameter head to choose continuous parameters **including a sign flag** (positive/negative). Use a simple geometry interpreter so every token always produces a valid shape.

---

## Core DSL (append-only, simple, RL-friendly)

State: a list of primitives, each with:

- `type`: `BOX` / `SPHERE` / `CYLINDER`
- `params`: 9D geometry vector (position, size, rotation)
- `role`: `POSITIVE` or `NEGATIVE` (derived from the 10th parameter output by the model)

Tokens (discrete actions):

- `ADD_BOX`
- `ADD_SPHERE`
- `ADD_CYLINDER`
- `UNDO_LAST` (remove last primitive if any)
- `END` (terminate sequence)

Parameter vector (continuous output per primitive step):

```text
params[0:3] = (cx, cy, cz)       # center
params[3:6] = (p0, p1, p2)       # size / radius / height (interpreted per primitive)
params[6:9] = (rx, ry, rz)       # rotation angles
params[9]   = sign_raw           # real-valued sign logit for POSITIVE vs NEGATIVE
```

The **sign** is not a separate token. Instead, the 10th parameter `sign_raw` is interpreted as:

```text
role = POSITIVE if sigmoid(sign_raw) >= 0.5 else NEGATIVE
```

This means each primitive is born positive or negative in a single step. There is no intermediate "wrong" positive state for cutouts; the geometry and reward always see the intended sign.

---

## Evaluation Rule (Geometry Semantics)

We use simple sequential CSG semantics over primitives:

```text
shape = empty
for prim in primitives_in_order:
    if prim.role == POSITIVE:
        shape = shape ∪ prim
    else:  # NEGATIVE
        shape = shape \ prim
final_shape = shape
```

The geometry backend **does not build full CSG trees**. Instead, it maintains a point cloud approximation by:

- sampling points for each POSITIVE primitive and concatenating them,
- removing any existing points that fall inside NEGATIVE primitives.

This gives an approximate but consistent notion of add vs subtract that is cheap enough for RL.

---

## Token Semantics and Masking

- `ADD_*` (primitive tokens):

  - The transformer chooses a primitive token.
  - The parameter head outputs a 10D vector.
  - The state machine:
    - decodes center, size, rotation from the first 9 dims,
    - interprets `sign_raw` as POSITIVE/NEGATIVE,
    - appends a new primitive with that role.

- `UNDO_LAST`:

  - If there is at least one primitive in the list, remove the last primitive.
  - If there is none, the token is masked out.

- `END`:
  - Terminate program generation.

Mask illegal tokens dynamically:

- If no primitive exists, mask `UNDO_LAST`.
- Optionally mask `END` on the very first step.

All other invalid actions are impossible by construction.

This sequential add/subtract/undo evaluation allows the policy to recover from overly large negative primitives by adding new positives afterward or undoing mistakes, improving iterative feedback.

---

## Model

### Overview

We use an autoregressive Transformer decoder as the backbone, plus separate heads for:

- **token prediction** (which discrete action next),
- **parameter + sign prediction** (continuous geometry and sign flag).

At each decoding step \(t\):

- Input: previous tokens (and optionally their parameters) and global context.
- Transformer produces hidden state `h_t` for the last step.
- From `h_t` we derive:
  - token logits (for next token),
  - param+sign vector (10D).

### Hidden State and Parameter Inputs

Each decoding step `t` uses a hidden state `h_t` from the transformer. The parameter head receives a concatenation of:

- `h_t`: transformer context of the partial program (token/history embedding),
- `gt_embed`: global embedding of the ground-truth point cloud (PointNet-style per-point MLP + max/mean pool),
- `tok_emb`: learned embedding of the **chosen** DSL token at step `t` (e.g., `ADD_BOX`, `ADD_SPHERE`, etc.),
- `err_emb` (optional later): embedding of current predicted-vs-GT error features.

The parameter head outputs a 10D vector:

```text
out_t = [cx, cy, cz, p0, p1, p2, rx, ry, rz, sign_raw]
```

- Geometry parameters: `params_geom = out_t[0:9]`.
- Sign logit: `sign_raw = out_t[9]`.
- Sign probability: `sign_prob = sigmoid(sign_raw)`.
- Sign label used by the state machine: `POSITIVE` if `sign_prob >= 0.5`, else `NEGATIVE`.

Primitive tokens consume these parameters; non-primitive tokens ignore the continuous outputs for that step.

---

## Geometry Backend

- Primitive types: BOX / SPHERE / CYLINDER.
- Roles: POSITIVE / NEGATIVE (driven entirely by the 10th parameter).
- Each primitive is converted into an analytic shape from which we can:
  - sample interior or surface points for positives,
  - test membership for negatives.

Approximate point cloud evaluation:

1. Maintain a global point cloud `pts`.
2. For each primitive in order:
   - If POSITIVE: sample `n_pos` points from that primitive and concatenate to `pts`.
   - If NEGATIVE: remove any points in `pts` that fall inside the negative primitive.

Chamfer distance is computed between:

- GT point cloud (sampled once per GT mesh/state), and
- predicted point cloud from this backend.

The same sampler is used for both training GT (for synthetic DSL samples) and predicted shapes, ensuring consistent reward.

---

## Training (Hybrid)

We use a hybrid setup:

1. **Supervised pretraining** for:

   - token head (next-token prediction from ground-truth sequences),
   - parameter head (MSE on geometry and BCE on sign).

2. **RL fine-tuning** for token structure:
   - reward from geometry backend (Chamfer + penalties),
   - policy gradient or PPO on token head,
   - parameter head can continue to receive gradient from Chamfer (as a continuous loss) or be partially frozen.

### Parameter + Sign Pretraining

To pretrain the parameter head with known ground-truth tokens and parameters:

1. Implement a minimal Transformer decoder and token head.
2. Teacher-force ground-truth token sequences to obtain hidden states `h_t`.
3. Encode the GT mesh once per example via PointNet to obtain `gt_embed`.
4. At step `t`, build the param head input as:

   ```text
   x_t = concat(h_t, gt_embed, tok_emb[, err_emb])
   ```

5. Param head outputs:

   ```text
   out_t = param_head(x_t)  # 10D
   params_pred_t = out_t[0:9]
   sign_logit_t  = out_t[9]
   ```

6. Supervise geometry via MSE and sign via BCE-with-logits:

   ```text
   L_params_t = MSE(params_pred_t, params_gt_t)
   L_sign_t   = BCEWithLogits(sign_logit_t, sign_gt_t)
   ```

   where `sign_gt_t ∈ {0,1}` indicates POSITIVE/NEGATIVE from the dataset.

7. Sum/average `L_params_t + λ_sign * L_sign_t` over primitive steps and backprop to train:
   - PointNet encoder,
   - transformer backbone,
   - parameter head.

### Token Pretraining

In parallel, we pretrain the token head with cross-entropy loss on the ground-truth next token:

```text
L_token_t = CrossEntropy(token_logits_t, token_gt_t)
```

Total supervised pretraining loss (per example):

```text
L_supervised = Σ_t L_token_t
             + λ_params Σ_t L_params_t (for primitive steps)
             + λ_sign Σ_t L_sign_t    (for primitive steps)
```

### RL Reward

After supervised pretraining, we can switch to a hybrid RL setup where:

- the token head is treated as a policy over tokens,
- the parameter head continues to be trained via Chamfer-based gradients (optional),
- the reward encourages accurate and compact shapes.

Reward for an episode (final shape):

```text
R = -Chamfer(pred_points, gt_points)
    - λ_prim * (#primitives)
    - λ_len  * (sequence_length)
```

Optionally, intermediate rewards can be given after each step using the current predicted point cloud.

---

## Implementation Steps (updated)

1. **DSL + State Machine**

   - Define token enum: `ADD_BOX`, `ADD_SPHERE`, `ADD_CYLINDER`, `UNDO_LAST`, `END`.
   - Implement `ShapeState.apply(token, params)` where:
     - primitive tokens interpret a 10D param vector and append a POSITIVE/NEGATIVE primitive based on `sign_raw`,
     - `UNDO_LAST` pops the last primitive if any,
     - `END` flags termination.

2. **Geometry Backend**

   - Implement analytic membership + sampling for BOX/SPHERE/CYLINDER.
   - Implement sequential union/subtraction over a point cloud as described.
   - Add a function `state_to_point_cloud(state, n_points, device)` used for both training and evaluation.

3. **PointNet Encoder**

   - Implement `PointNetEncoder(points) -> gt_embed` for GT point clouds.

4. **Transformer + Heads**

   - Transformer decoder over token histories.
   - Token head: linear layer from `h_t` to logits over tokens.
   - Param head: MLP from `concat(h_t, gt_embed, tok_emb[, err_emb])` to a 10D vector.

5. **Supervised Pretraining**

   - Build datasets of (GT mesh / point cloud, token sequences, param sequences including sign).
   - Train with `L_supervised` as above.

6. **RL Fine-Tuning (Optional)**
   - Wrap the state machine + geometry backend + Chamfer reward into an RL environment.
   - Treat the token head as the policy; optionally treat params as part of the policy or keep them under gradient-based refinement.
   - Use policy gradient / PPO to improve token selection under the geometric reward.

---

## Scope

Do **not** implement full CSG trees, complex boolean logic, or many primitive types. Stick to:

- a minimal DSL with a few primitives,
- sign handled as the 10th continuous parameter,
- a simple, consistent point-cloud-based geometry backend.

This is the most stable approach you can finish in a short time and still demonstrate:

- hybrid training (supervised + RL),
- transformer-based program generation,
- differentiable geometric feedback for shape approximation.
