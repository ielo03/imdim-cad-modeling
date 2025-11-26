#!/usr/bin/env python3
"""Run a trained GRU model to produce a SCAD for a single test case and collect artifacts.

Usage (example)
---------------
python3 src/run_model_on_case.py \
  --model-ckpt checkpoints/ckpt_epoch001_step000001.pt \
  --test-dir path/to/test_case_dir \
  --results-dir out/run_001 \
  --device cpu

Behavior
--------
- Expects `--test-dir` to contain a .scad file (the test case) and/or a set of rendered PNGs (truth images).
- Copies the test-case .scad and truth images into results_dir/truth_imgs/.
- Loads the GRU policy from the provided checkpoint and uses a simple action-decoder loop to produce an AST,
  serializes it to SCAD and writes it to results_dir/generated.scad.
- Renders the generated.scad using the repository's `render_views.sh` script into results_dir/model_imgs/.
- Returns exit code 0 on success.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import torch

# Ensure project root on path for local imports
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.gru_ast_gen import GRUPolicyNet, num_actions_from_enum
from src.ast_gen_utils import ASTBuilder, apply_action, Action
from src.scad_codegen import emit_program, SCADCodegenError
from src.batch_diff_dirs import discover_view_names_from_renders
from src.diff_png import diff_png

from PIL import Image
import numpy as np

# Numeric bins (same as dataset/train/generate_scads.py)
SIZE_BINS = [5, 10, 15, 20, 25, 30]
OFFSET_BINS = [-30, -20, -10, 0, 10, 20, 30]
ANGLE_BINS = [0, 15, 30, 45, 60, 90, 120, 180]
RADIUS_BINS = [2, 4, 6, 8, 10, 12]
HEIGHT_BINS = [5, 10, 15, 20, 25, 30]
SCALE_BINS = [0.5, 1.0, 1.5, 2.0]


def find_scad_in_dir(d: str) -> Optional[str]:
    for fn in os.listdir(d):
        if fn.lower().endswith(".scad"):
            return os.path.join(d, fn)
    return None


def find_pngs_in_dir(d: str):
    return sorted([os.path.join(d, fn) for fn in os.listdir(d) if fn.lower().endswith(".png")])


def load_image_for_model(img_path: str, img_size: int = 64, device: torch.device = torch.device("cpu")):
    im = Image.open(img_path).convert("L").resize((img_size, img_size))
    arr = np.asarray(im).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    return t.to(device)


def load_model_from_ckpt(ckpt_path: str, img_size: int = 64, device: str = "cpu"):
    """Load GRUPolicyNet from a checkpoint, handling multiple checkpoint formats.

    Supports:
      - New RL trainer checkpoints with key "model_state"
      - Older checkpoints with key "model"
      - Raw state_dict files
    """
    device_t = torch.device(device)
    num_actions = num_actions_from_enum()
    net = GRUPolicyNet(num_actions=num_actions, img_channels=1, img_size=img_size)
    ck = torch.load(ckpt_path, map_location=device_t)

    if isinstance(ck, dict) and "model_state" in ck:
        net.load_state_dict(ck["model_state"])
    elif isinstance(ck, dict) and "model" in ck:
        net.load_state_dict(ck["model"])
    else:
        # assume whole-file is a plain state_dict
        net.load_state_dict(ck)

    net.to(device_t)
    net.eval()
    return net


def sample_value_for_action(action: Action):
    """Map parameter action enum to a sampled numeric value from bins."""
    # Cube size params
    if action == Action.SET_SIZE_X or action == Action.SET_SIZE_Y or action == Action.SET_SIZE_Z:
        return float(np.random.choice(SIZE_BINS))
    # Radius/height
    if action == Action.SET_RADIUS:
        return float(np.random.choice(RADIUS_BINS))
    if action == Action.SET_HEIGHT:
        return float(np.random.choice(HEIGHT_BINS))
    # Translate params
    if action in {Action.SET_TRANSLATE_X, Action.SET_TRANSLATE_Y, Action.SET_TRANSLATE_Z}:
        return float(np.random.choice(OFFSET_BINS))
    # Rotate params
    if action in {Action.SET_ROTATE_X, Action.SET_ROTATE_Y, Action.SET_ROTATE_Z}:
        return float(np.random.choice(ANGLE_BINS))
    # Scale params
    if action in {Action.SET_SCALE_X, Action.SET_SCALE_Y, Action.SET_SCALE_Z}:
        return float(np.random.choice(SCALE_BINS))
    # Default fallback
    return 0.0


def decode_ast_from_model(net: GRUPolicyNet, img_tensor: torch.Tensor, max_steps: int = 64, device: str = "cpu"):
    """Decode a program AST by sampling actions from the policy network and applying to ASTBuilder.

    This is a simple, stochastic decoder:
      - maintain an action_seq (history) tensor of fixed length (filled with zeros initially)
      - at each step, forward the model with the image and action_seq, sample the next discrete action
      - for parameter actions, sample a numeric value from discretized bins and call apply_action(builder, action, value)
      - stop when END_PROGRAM is sampled or max_steps reached
    """
    from torch.nn.functional import one_hot

    device_t = torch.device(device)
    builder = ASTBuilder()
    action_seq_len = 8
    num_actions = num_actions_from_enum()

    action_seq = torch.zeros((1, action_seq_len), dtype=torch.long, device=device_t)  # zeros allowed as PAD/BOS
    hidden = None

    for step in range(max_steps):
        with torch.no_grad():
            out = net(img_tensor.to(device_t), action_seq)
        logits = out.logits  # [1, num_actions]
        # sample action
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        act_id = int(dist.sample().cpu().item())
        try:
            act = Action(act_id)
        except ValueError:
            # unknown action id -> skip
            continue

        # Parameter actions require numeric value
        if act in {
            Action.SET_SIZE_X,
            Action.SET_SIZE_Y,
            Action.SET_SIZE_Z,
            Action.SET_RADIUS,
            Action.SET_HEIGHT,
            Action.SET_TRANSLATE_X,
            Action.SET_TRANSLATE_Y,
            Action.SET_TRANSLATE_Z,
            Action.SET_ROTATE_X,
            Action.SET_ROTATE_Y,
            Action.SET_ROTATE_Z,
            Action.SET_SCALE_X,
            Action.SET_SCALE_Y,
            Action.SET_SCALE_Z,
        }:
            val = sample_value_for_action(act)
            try:
                apply_action(builder, act, float(val))
            except Exception:
                # ignore invalid parameter contexts
                pass
        else:
            # structural actions
            try:
                res = apply_action(builder, act)
                # apply_action returns program on END_PROGRAM
                if act == Action.END_PROGRAM:
                    return builder.to_program()
            except Exception:
                # ignore structural errors (invalid sequences)
                pass

        # update action_seq (shift left and append sampled id)
        action_seq = torch.roll(action_seq, shifts=-1, dims=1)
        action_seq[0, -1] = act_id

    # If we exit without END_PROGRAM, try to finalize whatever we have
    try:
        program = builder.to_program()
        return program
    except Exception:
        # fallback: simple cube program
        return {"root": {"kind": "cube", "params": {"size": [10, 10, 10], "center": True}}}


def run_and_collect(ckpt_path: str, test_dir: str, results_dir: str, img_size: int = 64, device: str = "cpu"):
    results_dir = os.path.abspath(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    # copy test case files into results_dir/truth_imgs and copy scad
    truth_imgs_dir = os.path.join(results_dir, "truth_imgs")
    model_imgs_dir = os.path.join(results_dir, "model_imgs")
    os.makedirs(truth_imgs_dir, exist_ok=True)
    os.makedirs(model_imgs_dir, exist_ok=True)

    test_dir = os.path.abspath(test_dir)

    # Support either:
    #  - a single .scad file path (e.g. dataset/train/ast_only/scad_00002.scad)
    #  - a directory containing a .scad and/or pngs
    scad_file = None
    pngs = []

    if os.path.isfile(test_dir) and test_dir.lower().endswith(".scad"):
        # test_dir is a single scad file path -> render truth views from it
        scad_file = test_dir
        dst_scad_root = os.path.join(results_dir, os.path.basename(scad_file))
        shutil.copy(scad_file, dst_scad_root)
        # render truth views from the scad into truth_imgs_dir
        script = os.path.join(ROOT, "render_views.sh")
        if not os.path.exists(script):
            raise RuntimeError(f"render_views.sh not found at {script}")
        subprocess.check_call([script, scad_file, truth_imgs_dir])
        pngs = find_pngs_in_dir(truth_imgs_dir)
    else:
        # treat test_dir as a directory
        scad_file = find_scad_in_dir(test_dir)
        if scad_file:
            dst_scad_root = os.path.join(results_dir, os.path.basename(scad_file))
            shutil.copy(scad_file, dst_scad_root)
        pngs = find_pngs_in_dir(test_dir)
        # if no pngs present but scad exists, render it
        if not pngs and scad_file:
            script = os.path.join(ROOT, "render_views.sh")
            if not os.path.exists(script):
                raise RuntimeError(f"No PNG truth images found in {test_dir} and render_views.sh not available to produce them")
            subprocess.check_call([script, scad_file, truth_imgs_dir])
            pngs = find_pngs_in_dir(truth_imgs_dir)

    if not pngs:
        raise RuntimeError(f"No PNG truth images found or produced for test case: {test_dir}")

    # ensure truth PNGs are present in truth_imgs_dir
    for p in pngs:
        dst = os.path.join(truth_imgs_dir, os.path.basename(p))
        if os.path.abspath(p) != os.path.abspath(dst):
            try:
                shutil.copy(p, truth_imgs_dir)
            except Exception:
                pass

    # choose first PNG from truth_imgs as model input
    truth_png_list = sorted([f for f in os.listdir(truth_imgs_dir) if f.lower().endswith(".png")])
    if not truth_png_list:
        raise RuntimeError("No truth PNGs available after rendering/copy")
    img0 = os.path.join(truth_imgs_dir, truth_png_list[0])

    # load model
    net = load_model_from_ckpt(ckpt_path, img_size=img_size, device=device)

    img_tensor = load_image_for_model(img0, img_size=img_size, device=torch.device(device))

    program = decode_ast_from_model(net, img_tensor, max_steps=64, device=device)

    # emit scad (attempt to handle occasionally-invalid ASTs produced by stochastic decoder)
    try:
        scad_src = emit_program(program)
    except Exception:
        # Try a conservative repair pass for common AST issues (empty CSG children, missing transform child)
        def repair_node(node):
            if not isinstance(node, dict):
                return node
            kind = node.get("kind")
            if kind in {"union", "intersection", "difference"}:
                children = node.get("children") or []
                # keep only dict children
                children = [c for c in children if isinstance(c, dict)]
                # if difference has <2 children, convert to union or fill with a default cube
                if kind == "difference" and len(children) < 2:
                    if len(children) == 1:
                        node["kind"] = "union"
                    elif len(children) == 0:
                        # replace entirely with a cube
                        node.clear()
                        node.update({"kind": "cube", "params": {"size": [10, 10, 10], "center": True}})
                        return node
                node["children"] = children
                for c in node.get("children", []):
                    repair_node(c)
            elif kind in {"translate", "rotate", "scale"}:
                child = node.get("child")
                if not isinstance(child, dict):
                    # replace transform with a default cube
                    node.clear()
                    node.update({"kind": "cube", "params": {"size": [10, 10, 10], "center": True}})
                    return node
                repair_node(child)
            return node

        try:
            if isinstance(program, dict) and "root" in program:
                repair_node(program["root"])
            scad_src = emit_program(program)
        except Exception:
            # last-resort fallback program
            fallback = {"root": {"kind": "cube", "params": {"size": [10, 10, 10], "center": True}}}
            scad_src = emit_program(fallback)

    gen_scad_path = os.path.join(results_dir, "generated.scad")
    with open(gen_scad_path, "w", encoding="utf-8") as fh:
        fh.write("// Generated by run_model_on_case.py\n")
        fh.write(scad_src)

    # render the generated scad into model_imgs_dir using render_views.sh
    script = os.path.join(ROOT, "render_views.sh")
    if not os.path.exists(script):
        raise RuntimeError(f"render_views.sh not found at {script}")

    # render into a temp dir then move files
    with tempfile.TemporaryDirectory() as tmpd:
        cmd = [script, gen_scad_path, tmpd]
        subprocess.check_call(cmd)
        # move pngs to model_imgs_dir
        for fn in os.listdir(tmpd):
            if fn.lower().endswith(".png"):
                shutil.move(os.path.join(tmpd, fn), os.path.join(model_imgs_dir, fn))

    return {
        "results_dir": results_dir,
        "generated_scad": gen_scad_path,
        "truth_imgs_dir": truth_imgs_dir,
        "model_imgs_dir": model_imgs_dir,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Use a trained model to generate SCAD for a test case and collect artifacts.")
    p.add_argument("--model-ckpt", required=True, help="Path to model checkpoint (.pt)")
    p.add_argument("--test-dir", required=True, help="Directory containing the test case (.scad) and truth PNGs")
    p.add_argument("--results-dir", required=True, help="Directory where results/artifacts will be written")
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    out = run_and_collect(args.model_ckpt, args.test_dir, args.results_dir, img_size=args.img_size, device=args.device)
    print("Wrote results to:", out["results_dir"])
    print("Generated SCAD:", out["generated_scad"])
    print("Truth images:", out["truth_imgs_dir"])
    print("Model images:", out["model_imgs_dir"])


if __name__ == "__main__":
    main()