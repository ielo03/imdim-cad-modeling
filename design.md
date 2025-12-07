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
