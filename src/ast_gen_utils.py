"""Utilities for incrementally building a minimal OpenSCAD-style AST.

This module is designed to be used by an RL environment / policy that emits
high-level *actions* like:

    - Structural:
        ADD_CUBE, ADD_SPHERE, ADD_CYLINDER
        START_UNION, START_DIFF, START_INTER
        WRAP_TRANSLATE, WRAP_ROTATE, WRAP_SCALE
        CLOSE_NODE, END_PROGRAM

    - Parameter actions (examples):
        SET_SIZE_X, SET_SIZE_Y, SET_SIZE_Z
        SET_RADIUS, SET_HEIGHT
        SET_TRANSLATE_X/Y/Z
        SET_ROTATE_X/Y/Z
        SET_SCALE_X/Y/Z

We represent the AST as plain Python dicts, compatible with `scad_codegen.py`:

    Program = {"root": Node}

    Node = {
        "kind": "cube" | "sphere" | "cylinder" |
                 "translate" | "rotate" | "scale" |
                 "union" | "difference" | "intersection",
        ...
    }

The core entry point is `ASTBuilder`, which maintains internal mutable state
(root node + a node stack) and exposes methods like `add_cube()`,
`start_csg("union")`, `wrap_translate()`, etc. At `END_PROGRAM`, you call
`to_program()` to obtain a `Program` dict that can be passed into
`scad_codegen.emit_program`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from enum import IntEnum

class Action(IntEnum):
    # Primitive additions
    ADD_CUBE = 1
    ADD_SPHERE = 2
    ADD_CYLINDER = 3

    # CSG starts
    START_UNION = 4
    START_DIFF = 5
    START_INTER = 6

    # Transforms
    WRAP_TRANSLATE = 7
    WRAP_ROTATE = 8
    WRAP_SCALE = 9

    # Structural control
    CLOSE_NODE = 10
    END_PROGRAM = 11

    # Cube size params
    SET_SIZE_X = 12
    SET_SIZE_Y = 13
    SET_SIZE_Z = 14

    # Sphere/cylinder params
    SET_RADIUS = 15
    SET_HEIGHT = 16

    # Translate params
    SET_TRANSLATE_X = 17
    SET_TRANSLATE_Y = 18
    SET_TRANSLATE_Z = 19

    # Rotate params
    SET_ROTATE_X = 20
    SET_ROTATE_Y = 21
    SET_ROTATE_Z = 22

    # Scale params
    SET_SCALE_X = 23
    SET_SCALE_Y = 24
    SET_SCALE_Z = 25


Node = Dict[str, Any]
Program = Dict[str, Any]


class ASTBuildError(Exception):
    """Error raised when an invalid edit/action is applied to the AST."""


@dataclass
class ASTBuilder:
    """Incremental AST builder for the OpenSCAD-subset language.

    State:
        - `root`: the root node (or None until first structural action)
        - `stack`: nodes that can still accept children (CSG or transform)
        - `current_param_target`: node whose parameters are being edited

    Typical usage:

        b = ASTBuilder()
        b.start_csg("union")
        b.add_cube()
        b.set_size_xyz(20, 10, 5)
        b.wrap_translate()
        b.set_translate_xyz(10, 0, 0)
        b.add_sphere()
        b.set_radius(8)
        b.close_node()        # close translate
        b.close_node()        # close union
        program = b.to_program()
    """

    root: Optional[Node] = None
    stack: List[Node] = field(default_factory=list)
    current_param_target: Optional[Node] = None

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _attach_to_parent(self, node: Node) -> None:
        """Attach `node` under the current parent on the stack, or set root.

        - If there is no parent and root is None → this becomes root.
        - If there is no parent and root already set → error (multiple roots).
        - If parent is a CSG node → append to `children`.
        - If parent is a transform node → set its `child` (must be empty).
        """

        if not self.stack:
            if self.root is None:
                self.root = node
                return
            raise ASTBuildError("Cannot attach new root: program root already set")

        parent = self.stack[-1]
        pkind = parent.get("kind")

        if pkind in {"union", "difference", "intersection"}:
            parent.setdefault("children", []).append(node)
        elif pkind in {"translate", "rotate", "scale"}:
            if "child" in parent and parent["child"] is not None:
                raise ASTBuildError(f"Transform {pkind} already has a child")
            parent["child"] = node
        else:
            raise ASTBuildError(f"Parent of kind={pkind!r} cannot accept children")

    def _push(self, node: Node) -> None:
        self.stack.append(node)
        self.current_param_target = node

    def _pop(self) -> Node:
        if not self.stack:
            raise ASTBuildError("Cannot CLOSE_NODE: stack is empty")
        node = self.stack.pop()
        # After closing, parameters likely switch to parent (if any)
        self.current_param_target = self.stack[-1] if self.stack else None
        return node

    # ------------------------------------------------------------------
    # Structural actions
    # ------------------------------------------------------------------

    def start_csg(self, kind: str) -> None:
        """Start a CSG node (union, difference, intersection) and push it.

        This corresponds to actions:
            START_UNION, START_DIFF, START_INTER
        """

        if kind not in {"union", "difference", "intersection"}:
            raise ValueError(f"Invalid CSG kind: {kind}")

        node: Node = {"kind": kind, "children": []}
        self._attach_to_parent(node)
        self._push(node)

    def start_union(self) -> None:
        self.start_csg("union")

    def start_difference(self) -> None:
        self.start_csg("difference")

    def start_intersection(self) -> None:
        self.start_csg("intersection")

    def add_cube(self) -> None:
        node: Node = {
            "kind": "cube",
            "params": {
                # Defaults; can be overridden via SET_SIZE_* actions
                "size": [10.0, 10.0, 10.0],
                "center": True,
            },
        }
        self._attach_to_parent(node)
        self.current_param_target = node

    def add_sphere(self) -> None:
        node: Node = {
            "kind": "sphere",
            "params": {
                # Default radius; overridden by SET_RADIUS
                "r": 5.0,
            },
        }
        self._attach_to_parent(node)
        self.current_param_target = node

    def add_cylinder(self) -> None:
        node: Node = {
            "kind": "cylinder",
            "params": {
                # Default height + radius
                "h": 10.0,
                "r": 3.0,
                "center": True,
            },
        }
        self._attach_to_parent(node)
        self.current_param_target = node

    def wrap_transform(self, kind: str) -> None:
        """Wrap the next child in a transform (translate, rotate, scale).

        Semantics: we create a transform node and push it on the stack so that
        the *next* structural node (primitive or CSG) becomes its `child`.

        This corresponds to actions:
            WRAP_TRANSLATE, WRAP_ROTATE, WRAP_SCALE
        """

        if kind not in {"translate", "rotate", "scale"}:
            raise ValueError(f"Invalid transform kind: {kind}")

        base_params: Dict[str, Any]
        if kind == "translate":
            base_params = {"offset": [0.0, 0.0, 0.0]}
        elif kind == "rotate":
            base_params = {"angles": [0.0, 0.0, 0.0]}
        else:  # scale
            base_params = {"factors": [1.0, 1.0, 1.0]}

        node: Node = {"kind": kind, "params": base_params, "child": None}
        self._attach_to_parent(node)
        self._push(node)

    def wrap_translate(self) -> None:
        self.wrap_transform("translate")

    def wrap_rotate(self) -> None:
        self.wrap_transform("rotate")

    def wrap_scale(self) -> None:
        self.wrap_transform("scale")

    def close_node(self) -> None:
        """Close the current node (transform or CSG) and pop from stack.

        This corresponds to the CLOSE_NODE action.
        """

        self._pop()

    # ------------------------------------------------------------------
    # Parameter editing helpers
    # ------------------------------------------------------------------

    def _ensure_param_target(self) -> Node:
        if self.current_param_target is None:
            raise ASTBuildError("No current_param_target to set parameters on")
        return self.current_param_target

    # --- cube size ---

    def set_size_x(self, value: float) -> None:
        node = self._ensure_param_target()
        if node.get("kind") != "cube":
            raise ASTBuildError("SET_SIZE_X only valid for cube nodes")
        size = node["params"].setdefault("size", [10.0, 10.0, 10.0])
        size[0] = float(value)

    def set_size_y(self, value: float) -> None:
        node = self._ensure_param_target()
        if node.get("kind") != "cube":
            raise ASTBuildError("SET_SIZE_Y only valid for cube nodes")
        size = node["params"].setdefault("size", [10.0, 10.0, 10.0])
        size[1] = float(value)

    def set_size_z(self, value: float) -> None:
        node = self._ensure_param_target()
        if node.get("kind") != "cube":
            raise ASTBuildError("SET_SIZE_Z only valid for cube nodes")
        size = node["params"].setdefault("size", [10.0, 10.0, 10.0])
        size[2] = float(value)

    def set_size_xyz(self, sx: float, sy: float, sz: float) -> None:
        self.set_size_x(sx)
        self.set_size_y(sy)
        self.set_size_z(sz)

    # --- sphere/cylinder radius ---

    def set_radius(self, value: float) -> None:
        node = self._ensure_param_target()
        kind = node.get("kind")
        if kind not in {"sphere", "cylinder"}:
            raise ASTBuildError("SET_RADIUS only valid for sphere/cylinder nodes")
        node["params"]["r"] = float(value)

    # --- cylinder height ---

    def set_height(self, value: float) -> None:
        node = self._ensure_param_target()
        if node.get("kind") != "cylinder":
            raise ASTBuildError("SET_HEIGHT only valid for cylinder nodes")
        node["params"]["h"] = float(value)

    # --- translate ---

    def set_translate_x(self, value: float) -> None:
        node = self._ensure_param_target()
        if node.get("kind") != "translate":
            raise ASTBuildError("SET_TRANSLATE_X only valid for translate nodes")
        off = node["params"].setdefault("offset", [0.0, 0.0, 0.0])
        off[0] = float(value)

    def set_translate_y(self, value: float) -> None:
        node = self._ensure_param_target()
        if node.get("kind") != "translate":
            raise ASTBuildError("SET_TRANSLATE_Y only valid for translate nodes")
        off = node["params"].setdefault("offset", [0.0, 0.0, 0.0])
        off[1] = float(value)

    def set_translate_z(self, value: float) -> None:
        node = self._ensure_param_target()
        if node.get("kind") != "translate":
            raise ASTBuildError("SET_TRANSLATE_Z only valid for translate nodes")
        off = node["params"].setdefault("offset", [0.0, 0.0, 0.0])
        off[2] = float(value)

    def set_translate_xyz(self, tx: float, ty: float, tz: float) -> None:
        self.set_translate_x(tx)
        self.set_translate_y(ty)
        self.set_translate_z(tz)

    # --- rotate ---

    def set_rotate_x(self, value: float) -> None:
        node = self._ensure_param_target()
        if node.get("kind") != "rotate":
            raise ASTBuildError("SET_ROTATE_X only valid for rotate nodes")
        ang = node["params"].setdefault("angles", [0.0, 0.0, 0.0])
        ang[0] = float(value)

    def set_rotate_y(self, value: float) -> None:
        node = self._ensure_param_target()
        if node.get("kind") != "rotate":
            raise ASTBuildError("SET_ROTATE_Y only valid for rotate nodes")
        ang = node["params"].setdefault("angles", [0.0, 0.0, 0.0])
        ang[1] = float(value)

    def set_rotate_z(self, value: float) -> None:
        node = self._ensure_param_target()
        if node.get("kind") != "rotate":
            raise ASTBuildError("SET_ROTATE_Z only valid for rotate nodes")
        ang = node["params"].setdefault("angles", [0.0, 0.0, 0.0])
        ang[2] = float(value)

    def set_rotate_xyz(self, rx: float, ry: float, rz: float) -> None:
        self.set_rotate_x(rx)
        self.set_rotate_y(ry)
        self.set_rotate_z(rz)

    # --- scale ---

    def set_scale_x(self, value: float) -> None:
        node = self._ensure_param_target()
        if node.get("kind") != "scale":
            raise ASTBuildError("SET_SCALE_X only valid for scale nodes")
        fac = node["params"].setdefault("factors", [1.0, 1.0, 1.0])
        fac[0] = float(value)

    def set_scale_y(self, value: float) -> None:
        node = self._ensure_param_target()
        if node.get("kind") != "scale":
            raise ASTBuildError("SET_SCALE_Y only valid for scale nodes")
        fac = node["params"].setdefault("factors", [1.0, 1.0, 1.0])
        fac[1] = float(value)

    def set_scale_z(self, value: float) -> None:
        node = self._ensure_param_target()
        if node.get("kind") != "scale":
            raise ASTBuildError("SET_SCALE_Z only valid for scale nodes")
        fac = node["params"].setdefault("factors", [1.0, 1.0, 1.0])
        fac[2] = float(value)

    def set_scale_xyz(self, sx: float, sy: float, sz: float) -> None:
        self.set_scale_x(sx)
        self.set_scale_y(sy)
        self.set_scale_z(sz)

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def to_program(self) -> Program:
        """Return the finished Program dict.

        Typically called when the policy emits END_PROGRAM. This does *not*
        mutate internal state; you can keep building afterwards if desired.
        """

        if self.root is None:
            raise ASTBuildError("Cannot build program: root is None")
        return {"root": self.root}


# ----------------------------------------------------------------------
# Optional: a simple action dispatcher for string-based actions
# ----------------------------------------------------------------------


def apply_action(builder: ASTBuilder, action: int | Action, value: Optional[float] = None):
    if isinstance(action, int):
        try:
            action = Action(action)
        except ValueError:
            raise ASTBuildError(f"Unknown action id: {action}")

    if action == Action.ADD_CUBE:
        builder.add_cube(); return
    if action == Action.ADD_SPHERE:
        builder.add_sphere(); return
    if action == Action.ADD_CYLINDER:
        builder.add_cylinder(); return

    if action == Action.START_UNION:
        builder.start_union(); return
    if action == Action.START_DIFF:
        builder.start_difference(); return
    if action == Action.START_INTER:
        builder.start_intersection(); return

    if action == Action.WRAP_TRANSLATE:
        builder.wrap_translate(); return
    if action == Action.WRAP_ROTATE:
        builder.wrap_rotate(); return
    if action == Action.WRAP_SCALE:
        builder.wrap_scale(); return

    if action == Action.CLOSE_NODE:
        builder.close_node(); return

    if action == Action.END_PROGRAM:
        return builder.to_program()

    # param actions:
    if value is None:
        raise ASTBuildError(f"Action {action} requires a numeric value")

    if action == Action.SET_SIZE_X:
        builder.set_size_x(value); return
    if action == Action.SET_SIZE_Y:
        builder.set_size_y(value); return
    if action == Action.SET_SIZE_Z:
        builder.set_size_z(value); return

    if action == Action.SET_RADIUS:
        builder.set_radius(value); return
    if action == Action.SET_HEIGHT:
        builder.set_height(value); return

    if action == Action.SET_TRANSLATE_X:
        builder.set_translate_x(value); return
    if action == Action.SET_TRANSLATE_Y:
        builder.set_translate_y(value); return
    if action == Action.SET_TRANSLATE_Z:
        builder.set_translate_z(value); return

    if action == Action.SET_ROTATE_X:
        builder.set_rotate_x(value); return
    if action == Action.SET_ROTATE_Y:
        builder.set_rotate_y(value); return
    if action == Action.SET_ROTATE_Z:
        builder.set_rotate_z(value); return

    if action == Action.SET_SCALE_X:
        builder.set_scale_x(value); return
    if action == Action.SET_SCALE_Y:
        builder.set_scale_y(value); return
    if action == Action.SET_SCALE_Z:
        builder.set_scale_z(value); return

    raise ASTBuildError(f"Unhandled action: {action}")


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    """Simple demo: simulate a sequence of actions and print the AST + SCAD.

    This builds roughly the same structure as the scad_codegen smoke test:

        union() {
          cube(size=[20,10,5], center=true);
          translate([10,0,0]) {
            sphere(r=8);
          }
        }
    """

    from pprint import pprint

    try:
        from .scad_codegen import emit_program  # type: ignore
    except ImportError:
        # Fallback for direct invocation if relative import fails
        from scad_codegen import emit_program  # type: ignore

    builder = ASTBuilder()

    # Simulated action stream from a policy
    action_stream = [
        (Action.START_UNION, None),

        (Action.ADD_CUBE, None),
        (Action.SET_SIZE_X, 20.0),
        (Action.SET_SIZE_Y, 10.0),
        (Action.SET_SIZE_Z, 5.0),

        (Action.WRAP_TRANSLATE, None),
        (Action.SET_TRANSLATE_X, 10.0),
        (Action.SET_TRANSLATE_Y, 0.0),
        (Action.SET_TRANSLATE_Z, 0.0),

        (Action.ADD_SPHERE, None),
        (Action.SET_RADIUS, 8.0),

        (Action.CLOSE_NODE, None),  # close translate
        (Action.CLOSE_NODE, None),  # close union

        (Action.END_PROGRAM, None),
    ]

    program = None
    for act, val in action_stream:
        result = apply_action(builder, act, val)
        if act == Action.END_PROGRAM:
            program = result

    if program is None:
        program = builder.to_program()

    print("\n=== Program AST ===")
    pprint(program)

    print("\n=== SCAD Output ===")
    scad_src = emit_program(program)
    print(scad_src)
