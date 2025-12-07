r"""Geometry backend: from ShapeState to point clouds and feedback.

This module is the numeric backend for the minimal parametric-shape DSL.
It does **not** depend on MicroCAD; it works directly on the symbolic
`ShapeState` defined in `state_machine.py` and exposes:

    - `sample_state_points(state, n_samples, device)`
    - `chamfer_distance(pred_points, gt_points)`
    - `compute_feedback(state, gt_points, ...)`

We use simple analytic sampling for primitives and an approximate sequential CSG:

    - Iterate primitives in order.
    - For each positive primitive: sample points and add them to the cloud.
    - For each negative primitive: remove any existing points that fall inside it.

This matches the sequential semantics:

    shape = empty
    for prim in primitives:
        if prim.role == positive: shape = shape ∪ prim
        else:                     shape = shape \\ prim

All heavy lifting (pairwise distances, sampling, etc.) is done in PyTorch
so it can run on CPU or GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

try:
    # Prefer relative import inside the package
    from .state_machine import ShapeState, Primitive, Role
except ImportError:  # pragma: no cover - direct execution fallback
    from state_machine import ShapeState, Primitive, Role  # type: ignore


# ---------------------------------------------------------------------------
# Primitive sampling utilities
# ---------------------------------------------------------------------------


def _ensure_tensor_device(device: Optional[torch.device] = None) -> torch.device:
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_tensor(xs, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(xs, dtype=torch.float32, device=device)


def _rotation_matrix_from_euler_deg(rx: float, ry: float, rz: float, device: torch.device) -> torch.Tensor:
    """Build a 3x3 rotation matrix from Euler angles in degrees.

    We use Z-Y-X convention: R = Rz @ Ry @ Rx. Angles are applied about the
    primitive center. For spheres, rotation has no geometric effect but is
    still well-defined.
    """

    angles = torch.deg2rad(torch.tensor([rx, ry, rz], dtype=torch.float32, device=device))
    rx_r, ry_r, rz_r = angles[0], angles[1], angles[2]

    cx, sx = torch.cos(rx_r), torch.sin(rx_r)
    cy, sy = torch.cos(ry_r), torch.sin(ry_r)
    cz, sz = torch.cos(rz_r), torch.sin(rz_r)

    Rx = torch.tensor(
        [[1.0, 0.0, 0.0],
         [0.0, cx, -sx],
         [0.0, sx, cx]],
        dtype=torch.float32,
        device=device,
    )
    Ry = torch.tensor(
        [[cy, 0.0, sy],
         [0.0, 1.0, 0.0],
         [-sy, 0.0, cy]],
        dtype=torch.float32,
        device=device,
    )
    Rz = torch.tensor(
        [[cz, -sz, 0.0],
         [sz, cz, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )

    return Rz @ Ry @ Rx


def _get_rotation_matrix(prim: Primitive, device: torch.device) -> torch.Tensor:
    """Extract a rotation matrix from a Primitive's params.

    Params expected in prim.params (optional):
        - rotation: [rx, ry, rz] in degrees

    If missing, returns identity.
    """

    p = prim.params or {}
    rot = p.get("rotation", [0.0, 0.0, 0.0])
    if not isinstance(rot, (list, tuple)) or len(rot) != 3:
        rot = [0.0, 0.0, 0.0]
    rx, ry, rz = float(rot[0]), float(rot[1]), float(rot[2])

    if rx == 0.0 and ry == 0.0 and rz == 0.0:
        return torch.eye(3, dtype=torch.float32, device=device)

    return _rotation_matrix_from_euler_deg(rx, ry, rz, device)


def sample_box_points(prim: Primitive, n_samples: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Sample points inside an axis-aligned box primitive.

    Params expected in prim.params:
        - center:   [cx, cy, cz]
        - size:     [sx, sy, sz]
        - rotation: [rx, ry, rz] in degrees (optional)
    """

    device = _ensure_tensor_device(device)
    p = prim.params or {}
    center = p.get("center", [0.0, 0.0, 0.0])
    size = p.get("size", [1.0, 1.0, 1.0])

    center_t = _to_tensor(center, device)
    size_t = _to_tensor(size, device)
    R = _get_rotation_matrix(prim, device)  # [3, 3]

    # Uniform in [-0.5, 0.5]^3 scaled by size and rotated and shifted by center
    u = torch.rand(n_samples, 3, device=device) - 0.5
    pts_local = u * size_t  # axis-aligned local box
    pts = center_t + pts_local @ R.T
    return pts


def sample_sphere_points(prim: Primitive, n_samples: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Sample points on the surface of a sphere primitive.

    Params expected in prim.params:
        - center:   [cx, cy, cz]
        - radius:   float (or `r`)
        - rotation: [rx, ry, rz] in degrees (optional; no effect on a perfect sphere)
    """

    device = _ensure_tensor_device(device)
    p = prim.params or {}
    center = p.get("center", [0.0, 0.0, 0.0])
    radius = float(p.get("radius", p.get("r", 1.0)))

    center_t = _to_tensor(center, device)
    r = torch.tensor(float(radius), dtype=torch.float32, device=device)

    # Sample directions from normal distribution and normalize
    dirs = torch.randn(n_samples, 3, device=device)
    dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True).clamp_min(1e-8)
    pts = center_t + r * dirs
    return pts


def sample_cylinder_points(prim: Primitive, n_samples: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Sample points inside an axis-aligned cylinder primitive.

    Cylinder is assumed to be aligned with the z-axis, centered at `center`.

    Params expected in prim.params:
        - center:   [cx, cy, cz]
        - radius:   float (or `r`)
        - height:   float (or `h`)
        - rotation: [rx, ry, rz] in degrees (optional)
    """

    device = _ensure_tensor_device(device)
    p = prim.params or {}
    center = p.get("center", [0.0, 0.0, 0.0])
    radius = float(p.get("radius", p.get("r", 1.0)))
    height = float(p.get("height", p.get("h", 1.0)))

    center_t = _to_tensor(center, device)
    r = float(radius)
    h = float(height)

    # Sample radius with sqrt for uniform area in disc, angle uniform, z uniform
    u = torch.rand(n_samples, 3, device=device)
    radial = torch.sqrt(u[:, 0:1]) * r
    theta = 2.0 * torch.pi * u[:, 1:2]
    z = (u[:, 2:3] - 0.5) * h

    x = radial * torch.cos(theta)
    y = radial * torch.sin(theta)
    pts_local = torch.cat([x, y, z], dim=-1)
    R = _get_rotation_matrix(prim, device)  # [3, 3]
    pts = center_t + pts_local @ R.T
    return pts


# ---------------------------------------------------------------------------
# Point-in-primitive tests (for subtractive filtering)
# ---------------------------------------------------------------------------


def points_in_box(points: torch.Tensor, prim: Primitive) -> torch.Tensor:
    """Return a boolean mask of points inside the (possibly rotated) box primitive."""

    p = prim.params or {}
    center = _to_tensor(p.get("center", [0.0, 0.0, 0.0]), points.device)
    size = _to_tensor(p.get("size", [1.0, 1.0, 1.0]), points.device)
    R = _get_rotation_matrix(prim, points.device)

    # Transform points into the box's local frame: local = (points - center) @ R
    rel = points - center
    local = rel @ R
    half = size / 2.0
    diff = torch.abs(local)
    inside = (diff <= half).all(dim=-1)
    return inside


def points_in_sphere(points: torch.Tensor, prim: Primitive) -> torch.Tensor:
    """Return a boolean mask of points inside the sphere primitive."""

    p = prim.params or {}
    center = _to_tensor(p.get("center", [0.0, 0.0, 0.0]), points.device)
    radius = float(p.get("radius", p.get("r", 1.0)))

    diff = points - center
    dist2 = (diff * diff).sum(dim=-1)
    inside = dist2 <= radius * radius
    return inside


def points_in_cylinder(points: torch.Tensor, prim: Primitive) -> torch.Tensor:
    """Return a boolean mask of points inside a (possibly rotated) cylinder."""

    p = prim.params or {}
    center = _to_tensor(p.get("center", [0.0, 0.0, 0.0]), points.device)
    radius = float(p.get("radius", p.get("r", 1.0)))
    height = float(p.get("height", p.get("h", 1.0)))
    R = _get_rotation_matrix(prim, points.device)

    rel = points - center
    # Transform into local cylinder frame: axis aligned with local z
    local = rel @ R
    xy2 = local[:, 0] ** 2 + local[:, 1] ** 2
    inside_rad = xy2 <= radius * radius
    inside_z = local[:, 2].abs() <= (height / 2.0)
    return inside_rad & inside_z


def points_in_primitive(points: torch.Tensor, prim: Primitive) -> torch.Tensor:
    """Dispatch point-inside test based on primitive kind."""

    if prim.kind == "box":
        return points_in_box(points, prim)
    if prim.kind == "sphere":
        return points_in_sphere(points, prim)
    if prim.kind == "cylinder":
        return points_in_cylinder(points, prim)
    # Unknown kind: treat as nothing
    return torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)


# ---------------------------------------------------------------------------
# State -> point cloud
# ---------------------------------------------------------------------------


def sample_state_points(
    state: ShapeState,
    n_samples: int = 1024,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Sample points representing the shape encoded by `state`.

    We:
      1. Iterate primitives in order.
      2. For each positive primitive: sample points and add them to the cloud.
      3. For each negative primitive: remove any existing points that fall inside it.

    Returns a tensor of shape [M, 3] (M <= n_samples). If no positive
    primitives exist, returns a single degenerate point at the origin.
    """

    device = _ensure_tensor_device(device)
    prims = state.primitives

    num_pos = sum(1 for p in prims if p.role == Role.POSITIVE)
    if num_pos == 0:
        # Degenerate: represent empty shape as a single point at origin.
        return torch.zeros(1, 3, device=device)

    per = max(1, n_samples // num_pos)
    pts = None

    for p in prims:
        if p.role == Role.POSITIVE:
            if p.kind == "box":
                new_pts = sample_box_points(p, per, device=device)
            elif p.kind == "sphere":
                new_pts = sample_sphere_points(p, per, device=device)
            elif p.kind == "cylinder":
                new_pts = sample_cylinder_points(p, per, device=device)
            else:
                # Unknown kind: skip
                continue
            if pts is None:
                pts = new_pts
            else:
                pts = torch.cat([pts, new_pts], dim=0)
        elif p.role == Role.NEGATIVE and pts is not None:
            inside = points_in_primitive(pts, p)
            pts = pts[~inside]
            if pts.shape[0] == 0:
                pts = None

    if pts is None:
        return torch.zeros(1, 3, device=device)

    if pts.shape[0] > n_samples:
        pts = pts[:n_samples]

    return pts

# ---------------------------------------------------------------------------
# GT mesh loading / preprocessing
# ---------------------------------------------------------------------------

import os

try:
    import trimesh  # type: ignore
except ImportError:
    trimesh = None


def load_ply_points(path: str, device: Optional[torch.device] = None) -> torch.Tensor:
    """Load a PLY file and return its vertices as a [N,3] float32 tensor on `device`.

    Requires `trimesh`. If not available, raises ImportError.
    Normalizes nothing; caller can do centering/scaling if desired.
    """
    if trimesh is None:
        raise ImportError("trimesh must be installed to load PLY meshes")

    if not os.path.isfile(path):
        raise FileNotFoundError(f"PLY file not found: {path}")

    mesh = trimesh.load(path, file_type='ply', force='mesh')
    if not hasattr(mesh, 'vertices'):
        raise ValueError(f"PLY does not contain vertices: {path}")

    pts = torch.tensor(mesh.vertices, dtype=torch.float32)
    if device is not None:
        pts = pts.to(device)
    return pts

# ---------------------------------------------------------------------------
# Chamfer distance and feedback
# ---------------------------------------------------------------------------

def chamfer_distance(pred_points: torch.Tensor, gt_points: torch.Tensor) -> torch.Tensor:
    """Compute a simple symmetric Chamfer distance between two point clouds.

    Both inputs should be [N, 3] and [M, 3]. Returns a scalar tensor.
    """

    if pred_points.ndim != 2 or gt_points.ndim != 2:
        raise ValueError("pred_points and gt_points must be rank-2 [N,3] tensors")
    if pred_points.shape[1] != 3 or gt_points.shape[1] != 3:
        raise ValueError("pred_points and gt_points must have shape [*, 3]")

    # [N, M, 3]
    diff = pred_points.unsqueeze(1) - gt_points.unsqueeze(0)
    dist2 = (diff * diff).sum(dim=-1)

    # For each predicted point, nearest GT
    min_pred2gt, _ = dist2.min(dim=1)
    # For each GT point, nearest predicted
    min_gt2pred, _ = dist2.min(dim=0)

    cd = min_pred2gt.mean() + min_gt2pred.mean()
    return cd


@dataclass
class Feedback:
    chamfer: float
    reward: float
    n_pred_points: int
    n_primitives: int


def compute_feedback(
    state: ShapeState,
    gt_points: torch.Tensor,
    *,
    n_samples: int = 1024,
    lambda_prim_count: float = 0.0,
    device: Optional[torch.device] = None,
) -> Feedback:
    """Sample a shape from `state`, compare to GT, and compute reward.

    Parameters
    ----------
    state : ShapeState
        Current program state.
    gt_points : torch.Tensor
        Ground-truth point cloud [M, 3], already on the correct device.
    n_samples : int
        Number of samples to draw from the predicted shape.
    lambda_prim_count : float
        Penalty per primitive used in the reward.
    device : torch.device, optional
        Device for sampling. If None, infer from gt_points or default.
    """

    if device is None:
        device = gt_points.device if gt_points.is_cuda or gt_points.device.type != "cpu" else _ensure_tensor_device()

    # Ensure GT is on the same device as predicted points.
    gt_points = gt_points.to(device=device, dtype=torch.float32)

    pred_points = sample_state_points(state, n_samples=n_samples, device=device)
    cd = chamfer_distance(pred_points, gt_points)

    # Reward: negative Chamfer minus primitive-count penalty.
    prim_penalty = lambda_prim_count * len(state.primitives)
    reward = -cd.item() - prim_penalty

    return Feedback(
        chamfer=cd.item(),
        reward=reward,
        n_pred_points=int(pred_points.shape[0]),
        n_primitives=len(state.primitives),
    )


if __name__ == "__main__":  # pragma: no cover - simple smoke test
    # Tiny manual test: compare a single box state to a GT box-like cloud.
    from state_machine import ShapeState, Primitive, Role  # type: ignore

    device = _ensure_tensor_device()

    state = ShapeState()
    state.primitives.append(
        Primitive(
            kind="box",
            role=Role.POSITIVE,
            params={"center": [0.0, 0.0, 0.0], "size": [1.0, 1.0, 1.0]},
        )
    )

    # GT as another cube, slightly shifted
    gt_pts = torch.rand(2048, 3, device=device) - 0.5

    fb = compute_feedback(state, gt_pts, n_samples=1024, lambda_prim_count=0.01, device=device)

    # Additional smoke test: load a PLY and compute Chamfer
    try:
        # Attempt to load a real mesh file and compare against a trivial state
        ply_path = "test_files/airplane.ply"
        ply_points = load_ply_points(ply_path, device=device)

        # Example predicted state (just a small sphere)
        state2 = ShapeState()
        state2.primitives.append(
            Primitive(
                kind="sphere",
                role=Role.POSITIVE,
                params={"center": [0.0, 0.0, 0.0], "radius": 0.2},
            )
        )

        fb2 = compute_feedback(state2, ply_points, n_samples=2048, device=device)
        print("PLY smoke test:", fb2)
    except Exception as e:
        print("PLY smoke test skipped (", e, ")")

    print(fb)
