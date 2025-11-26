"""End-to-end RL training loop for GRU-based AST generation.

This script wires together:
  - GRUPolicyNet (CNN + GRU policy/value net)
  - ASTBuilder / Action / apply_action (to build OpenSCAD ASTs)
  - scad_codegen.emit_program (to emit .scad source)
  - an external renderer (render_views.sh) to create 6-view PNG renders
  - batch_diff_dirs.compare_model_scores / compute_rms_from_scores / rms_to_reward
    to compute a scalar reward from PNG diffs

It assumes a dataset layout like:

    dataset/train/ast_only/
      scad_00001/
        view_back.png
        view_bottom.png
        view_front.png
        view_left.png
        view_right.png
        view_top.png
      scad_00002/
        ...
      ...

During training, for each sample ID we:
  1. Load the 6 GT views and feed them as a multi-view tensor to the policy.
  2. Roll out a sequence of discrete `Action` IDs to build an AST.
  3. Emit SCAD, save it, and render 6 predicted views to a temp directory.
  4. Compare predicted views vs GT views using existing diff utilities.
  5. Map RMS diff → reward and apply a REINFORCE-style update.

NOTE: This is a simple on-policy REINFORCE trainer with value baseline.
      It is intentionally minimal and single-sample (batch size 1) per update.
"""

from __future__ import annotations

import math
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

# Local imports
try:
    from .ast_gen_utils import ASTBuilder, Action, apply_action
    from .gru_ast_gen import GRUPolicyNet, num_actions_from_enum
    from .scad_codegen import emit_program
    from .batch_diff_dirs import (
        compare_model_scores,
        compute_rms_from_scores,
        rms_to_reward,
    )
except ImportError:
    # Fallback for running as a script from project root
    from src.ast_gen_utils import ASTBuilder, Action, apply_action  # type: ignore
    from src.gru_ast_gen import GRUPolicyNet, num_actions_from_enum  # type: ignore
    from src.scad_codegen import emit_program  # type: ignore
    from src.batch_diff_dirs import (  # type: ignore
        compare_model_scores,
        compute_rms_from_scores,
        rms_to_reward,
    )

try:
    from PIL import Image
except ImportError as exc:
    raise RuntimeError("Pillow is required for train_model.py") from exc

try:
    import torchvision.transforms as T
except ImportError as exc:
    raise RuntimeError("torchvision is required for train_model.py") from exc


# -----------------------------------------------------------------------------
# Paths and dataset layout
# -----------------------------------------------------------------------------


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset" / "train" / "ast_only"

# Sample directories are expected to be named scad_00001 .. scad_00100, etc.
MODEL_NAME_PATTERN = "scad_{:05d}"

# Views present per model directory in DATASET_ROOT.
VIEW_NAMES: List[str] = [
    "view_back.png",
    "view_bottom.png",
    "view_front.png",
    "view_left.png",
    "view_right.png",
    "view_top.png",
]

# Where to write temporary renders and SCAD files during training.
WORK_ROOT = PROJECT_ROOT / "work"
PRED_RENDER_ROOT = WORK_ROOT / "train_renders"
SCAD_OUT_ROOT = WORK_ROOT / "train_scad"

PRED_RENDER_ROOT.mkdir(parents=True, exist_ok=True)
SCAD_OUT_ROOT.mkdir(parents=True, exist_ok=True)


# External renderer command. You MUST adapt this to your project.
# It should:
#   - take an input .scad file
#   - output the standard 6 view_*.png files into the given output directory
#
# Example shell script (adjust path):
#   ./render_views.sh input.scad output_dir
RENDER_CMD: Sequence[str] = [
    "./render_views.sh",  # TODO: adjust if your script lives elsewhere
]


# -----------------------------------------------------------------------------
# Image loading for multi-view input
# -----------------------------------------------------------------------------


@dataclass
class ImageLoaderConfig:
    img_size: int = 64
    channels: int = 1  # 1=grayscale, 3=RGB


def make_transform(cfg: ImageLoaderConfig) -> T.Compose:
    ops: List[torch.nn.Module] = []
    if cfg.channels == 1:
        ops.append(T.Grayscale(num_output_channels=1))
    elif cfg.channels == 3:
        ops.append(T.ConvertImageDtype(torch.float))  # noop-ish placeholder
    else:
        raise ValueError(f"Unsupported channels: {cfg.channels}")
    ops.extend(
        [
            T.Resize((cfg.img_size, cfg.img_size)),
            T.ToTensor(),  # [C, H, W], values in [0,1]
        ]
    )
    return T.Compose(ops)


def load_gt_views(sample_id: int, cfg: ImageLoaderConfig, device: torch.device) -> torch.Tensor:
    """Load 6 GT view PNGs into a tensor of shape [1, V, C, H, W]."""

    dirname = MODEL_NAME_PATTERN.format(sample_id)
    model_dir = DATASET_ROOT / dirname
    if not model_dir.is_dir():
        raise FileNotFoundError(f"GT dir not found: {model_dir}")

    transform = make_transform(cfg)

    view_tensors: List[torch.Tensor] = []
    for name in VIEW_NAMES:
        img_path = model_dir / name
        if not img_path.is_file():
            raise FileNotFoundError(f"GT view missing: {img_path}")
        img = Image.open(img_path)
        t = transform(img)  # [C, H, W]
        view_tensors.append(t)  # [C, H, W]

    # stack to [V, C, H, W] then add batch: [1, V, C, H, W]
    stacked = torch.stack(view_tensors, dim=0)  # [V, C, H, W]
    batch = stacked.unsqueeze(0).to(device)     # [1, V, C, H, W]
    return batch


# -----------------------------------------------------------------------------
# Rendering and reward computation
# -----------------------------------------------------------------------------


def render_scad_for_sample(sample_id: int, scad_path: Path) -> None:
    """Render a .scad program to the 6 standard views under PRED_RENDER_ROOT.

    This will create:
        PRED_RENDER_ROOT / scad_00001 / view_*.png

    You must ensure that RENDER_CMD points to a working script.
    """

    dirname = MODEL_NAME_PATTERN.format(sample_id)
    out_dir = PRED_RENDER_ROOT / dirname
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = list(RENDER_CMD) + [str(scad_path), str(out_dir)]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Render command failed: {cmd}") from exc


def compute_reward_for_sample(sample_id: int) -> Optional[float]:
    """Use batch_diff_dirs utilities to compute a scalar reward for one sample.

    We compare:
        GT:   DATASET_ROOT / scad_xxxxx
        PRED: PRED_RENDER_ROOT / scad_xxxxx

    and use RMS over the standard VIEW_NAMES, then map to reward via rms_to_reward.
    """

    dirname = MODEL_NAME_PATTERN.format(sample_id)
    model_name, scores = compare_model_scores(
        dir_a=str(DATASET_ROOT),
        dir_b=str(PRED_RENDER_ROOT),
        model=dirname,
        view_names=VIEW_NAMES,
    )
    # model_name is not used; we care about scores.
    rms = compute_rms_from_scores(scores)
    reward = rms_to_reward(rms, mode="linear")
    return reward


# -----------------------------------------------------------------------------
# RL rollout and loss
# -----------------------------------------------------------------------------


@dataclass
class EpisodeResult:
    log_probs: List[torch.Tensor]
    values: List[torch.Tensor]
    reward: float
    entropy_terms: List[torch.Tensor]


def rollout_episode(
    net: GRUPolicyNet,
    sample_id: int,
    img_cfg: ImageLoaderConfig,
    max_steps: int,
    device: torch.device,
) -> Optional[EpisodeResult]:
    """Generate a program for one sample_id and compute reward.

    This:
      - loads GT views for the sample,
      - rolls out a sequence of Action IDs using the policy,
      - builds an AST with ASTBuilder,
      - emits SCAD, renders it, and computes reward.

    Returns None if AST build fails or rendering/reward fails.
    """

    # 1) Load GT image views (observation)
    img_batch = load_gt_views(sample_id, img_cfg, device=device)  # [1, V, C, H, W]

    # 2) Prepare AST builder and action history
    builder = ASTBuilder()
    num_actions = num_actions_from_enum()

    # We reserve 0 as BOS/PAD token for the action sequence.
    action_history: List[int] = [0]

    log_probs: List[torch.Tensor] = []
    values: List[torch.Tensor] = []
    entropy_terms: List[torch.Tensor] = []

    done = False

    for step in range(max_steps):
        action_seq_tensor = torch.tensor(
            [action_history],
            dtype=torch.long,
            device=device,
        )  # [1, T]

        out = net(img_batch, action_seq_tensor)
        logits = out.logits  # [1, num_actions]
        value = out.value    # [1]

        # Mask out the PAD/BOS index 0 so the policy can't keep emitting it.
        logits = logits.clone()
        logits[:, 0] = -1e9

        dist = Categorical(logits=logits)
        action_id = dist.sample()           # [1]
        log_prob = dist.log_prob(action_id) # [1]
        entropy = dist.entropy()            # [1]

        a_int = int(action_id.item())
        action_history.append(a_int)

        log_probs.append(log_prob)
        values.append(value)
        entropy_terms.append(entropy)

        try:
            act_enum = Action(a_int)
        except ValueError:
            # Unknown/invalid action id; treat as terminal and abort episode.
            done = True
            break

        if act_enum == Action.END_PROGRAM:
            done = True
            break

        # Structural actions do not require a numeric parameter.
        # Parameter-setting actions (SET_*) expect a float; for now we **skip**
        # them to keep this trainer simple and avoid passing bogus values.
        #
        # You can extend this by sampling parameter values separately and
        # calling apply_action(builder, act_enum, value=sampled_value).
        if act_enum in {
            Action.ADD_CUBE,
            Action.ADD_SPHERE,
            Action.ADD_CYLINDER,
            Action.START_UNION,
            Action.START_DIFF,
            Action.START_INTER,
            Action.WRAP_TRANSLATE,
            Action.WRAP_ROTATE,
            Action.WRAP_SCALE,
            Action.CLOSE_NODE,
        }:
            from src.ast_gen_utils import ASTBuildError  # local import to avoid cycle issues

            try:
                apply_action(builder, act_enum)
            except ASTBuildError:
                # Invalid structural sequence (e.g., CLOSE_NODE on empty stack).
                # Treat as terminal with no reward (or strong negative if desired).
                done = True
                break
        else:
            # Skip param actions for now (no-op). AST will use default params.
            continue

    if not done:
        # If we hit max_steps without END_PROGRAM or failure, treat as invalid episode.
        return None

    # 3) Finalize AST and emit SCAD
    try:
        program = builder.to_program()
    except Exception:
        return None

    dirname = MODEL_NAME_PATTERN.format(sample_id)
    scad_path = SCAD_OUT_ROOT / f"{dirname}.scad"
    try:
        scad_src = emit_program(program)
    except Exception:
        # Invalid AST for codegen (e.g., transform with missing child); skip episode.
        return None
    scad_path.write_text(scad_src, encoding="utf-8")

    # 4) Render SCAD to PNG views
    try:
        render_scad_for_sample(sample_id, scad_path)
    except Exception:
        return None

    # 5) Compute reward from renders
    reward = compute_reward_for_sample(sample_id)
    if reward is None:
        return None

    return EpisodeResult(
        log_probs=log_probs,
        values=values,
        reward=float(reward),
        entropy_terms=entropy_terms,
    )


def compute_loss_from_episode(
    episode: EpisodeResult,
    gamma: float = 0.99,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> torch.Tensor:
    """Compute a simple REINFORCE + value baseline loss for one episode.

    We treat reward as terminal and use the same scalar for all steps, but we
    still allow discounting via `gamma` if desired.
    """

    R = episode.reward
    # Same scalar return for all steps (no intermediate rewards in this setup).
    returns = [R * (gamma ** i) for i in range(len(episode.values))]
    returns_t = torch.tensor(returns, dtype=torch.float32, device=episode.values[0].device)

    values_t = torch.cat(episode.values, dim=0)        # [T]
    log_probs_t = torch.cat(episode.log_probs, dim=0)  # [T]
    entropy_t = torch.cat(episode.entropy_terms, dim=0)  # [T]

    # Advantage = R - V(s)
    advantages = returns_t - values_t.detach()

    policy_loss = -(log_probs_t * advantages).mean()
    value_loss = F.mse_loss(values_t, returns_t)
    entropy_loss = -entropy_t.mean()

    loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
    return loss


# -----------------------------------------------------------------------------
# Top-level training loop
# -----------------------------------------------------------------------------


def train(
    num_epochs: int = 10,
    episodes_per_epoch: int = 100,
    max_steps_per_episode: int = 64,
    img_size: int = 64,
    lr: float = 1e-4,
    device: Optional[torch.device] = None,
) -> None:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_cfg = ImageLoaderConfig(img_size=img_size, channels=1)

    num_actions = num_actions_from_enum()
    net = GRUPolicyNet(
        num_actions=num_actions,
        img_channels=1,
        img_size=img_size,
        hidden_dim=256,
        action_embed_dim=64,
    ).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # Infer sample IDs by scanning DATASET_ROOT for scad_xxxxx directories.
    sample_ids: List[int] = []
    for entry in sorted(DATASET_ROOT.iterdir()):
        if entry.is_dir() and entry.name.startswith("scad_"):
            try:
                sid = int(entry.name.split("_")[1])
                sample_ids.append(sid)
            except Exception:
                continue

    if not sample_ids:
        raise RuntimeError(f"No scad_xxxxx dirs found under {DATASET_ROOT}")

    print(f"Found {len(sample_ids)} samples: {sample_ids[:5]} ...")

    global_step = 0

    for epoch in range(num_epochs):
        epoch_losses: List[float] = []
        epoch_rewards: List[float] = []

        for ep in range(episodes_per_epoch):
            global_step += 1
            sample_id = random.choice(sample_ids)

            net.train()
            optimizer.zero_grad()

            episode_result = rollout_episode(
                net=net,
                sample_id=sample_id,
                img_cfg=img_cfg,
                max_steps=max_steps_per_episode,
                device=device,
            )

            if episode_result is None:
                # Could not build/program/render/compute reward; skip this episode.
                continue

            loss = compute_loss_from_episode(episode_result)
            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.item()))
            epoch_rewards.append(float(episode_result.reward))

            if global_step % 10 == 0:
                mean_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("nan")
                mean_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else float("nan")
                print(
                    f"[epoch {epoch+1}/{num_epochs} step {global_step}] "
                    f"loss={mean_loss:.4f} reward={mean_reward:.4f}"
                )

        # Save checkpoint per epoch
        ckpt_path = WORK_ROOT / f"gru_policy_epoch_{epoch+1:03d}.pt"
        torch.save(
            {
                "epoch": epoch + 1,
                "global_step": global_step,
                "model_state": net.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
            ckpt_path,
        )
        print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    # Simple CLI entry point. You can tweak hyperparameters here or expose argparse.
    train()
