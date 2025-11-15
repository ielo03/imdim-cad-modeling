"""SCAD code generation from a minimal OpenSCAD-style AST.

This module assumes an in-memory AST structure consistent with the design in
`ast.md`. The AST is represented as plain Python dictionaries, typically
originating from JSON/YAML or being constructed programmatically.

Expected top-level shape:

    program = {
        "root": <node>,
    }

Where a node is one of:

    Primitive node:
        {
            "kind": "cube" | "sphere" | "cylinder",
            "params": { ... }
        }

    Transform node:
        {
            "kind": "translate" | "rotate" | "scale",
            "params": { ... },
            "child": <node>
        }

    CSG node:
        {
            "kind": "union" | "difference" | "intersection",
            "children": [<node>, ...]
        }

The main entry point is `emit_program(program: dict) -> str`, which returns a
string containing valid OpenSCAD code.
"""

from __future__ import annotations

from typing import Dict, List, Any


Node = Dict[str, Any]
Program = Dict[str, Any]


class SCADCodegenError(Exception):
    """Domain-specific error for SCAD code generation."""


def emit_program(program: Program) -> str:
    """Emit full SCAD source for a `Program` AST.

    Parameters
    ----------
    program:
        Dictionary with at least a `"root"` key containing a node.

    Returns
    -------
    str
        OpenSCAD source code as a string.
    """

    if "root" not in program:
        raise SCADCodegenError("Program dict must contain a 'root' node")

    root = program["root"]
    body = emit_node(root)
    # Ensure trailing newline for nicer files.
    if not body.endswith("\n"):
        body += "\n"
    return body


def emit_node(node: Node) -> str:
    """Dispatch based on node["kind"]."""

    kind = node.get("kind")
    if kind is None:
        raise SCADCodegenError(f"Node is missing 'kind': {node!r}")

    if kind in {"cube", "sphere", "cylinder"}:
        return emit_primitive(node)
    if kind in {"translate", "rotate", "scale"}:
        return emit_transform(node)
    if kind in {"union", "difference", "intersection"}:
        return emit_csg(node)

    raise SCADCodegenError(f"Unknown node kind: {kind!r}")


# ---------------------------------------------------------------------------
# Primitive nodes
# ---------------------------------------------------------------------------


def emit_primitive(node: Node) -> str:
    kind = node["kind"]
    params = node.get("params", {})

    if kind == "cube":
        try:
            size = params["size"]  # [sx, sy, sz]
        except KeyError as exc:
            raise SCADCodegenError("cube primitive missing 'size' param") from exc
        center = bool(params.get("center", True))
        sx, sy, sz = size
        return f"cube(size=[{sx},{sy},{sz}], center={str(center).lower()});"

    if kind == "sphere":
        try:
            r = params["r"]
        except KeyError as exc:
            raise SCADCodegenError("sphere primitive missing 'r' param") from exc
        return f"sphere(r={r});"

    if kind == "cylinder":
        try:
            h = params["h"]
            r = params["r"]
        except KeyError as exc:
            raise SCADCodegenError(
                "cylinder primitive missing 'h' or 'r' param"
            ) from exc
        center = bool(params.get("center", True))
        return f"cylinder(h={h}, r={r}, center={str(center).lower()});"

    raise SCADCodegenError(f"emit_primitive cannot handle kind={kind!r}")


# ---------------------------------------------------------------------------
# Transform nodes
# ---------------------------------------------------------------------------


def emit_transform(node: Node) -> str:
    kind = node["kind"]
    params = node.get("params", {})

    try:
        child = node["child"]
    except KeyError as exc:
        raise SCADCodegenError(f"Transform node {kind!r} missing 'child'") from exc

    inner = emit_node(child)
    inner_indented = indent(inner)

    if kind == "translate":
        try:
            tx, ty, tz = params["offset"]
        except KeyError as exc:
            raise SCADCodegenError(
                "translate transform missing 'offset' param"
            ) from exc
        return f"translate([{tx},{ty},{tz}]) {{\n{inner_indented}\n}}"

    if kind == "rotate":
        try:
            rx, ry, rz = params["angles"]
        except KeyError as exc:
            raise SCADCodegenError(
                "rotate transform missing 'angles' param"
            ) from exc
        return f"rotate([{rx},{ry},{rz}]) {{\n{inner_indented}\n}}"

    if kind == "scale":
        try:
            sx, sy, sz = params["factors"]
        except KeyError as exc:
            raise SCADCodegenError(
                "scale transform missing 'factors' param"
            ) from exc
        return f"scale([{sx},{sy},{sz}]) {{\n{inner_indented}\n}}"

    raise SCADCodegenError(f"emit_transform cannot handle kind={kind!r}")


# ---------------------------------------------------------------------------
# CSG nodes
# ---------------------------------------------------------------------------


def emit_csg(node: Node) -> str:
    kind = node["kind"]
    children: List[Node] = node.get("children", [])

    if not children:
        raise SCADCodegenError(f"CSG node {kind!r} must have at least one child")

    # difference specifically requires at least two children
    if kind == "difference" and len(children) < 2:
        raise SCADCodegenError("difference CSG node must have >= 2 children")

    body_parts = [emit_node(ch) for ch in children]
    body = "\n".join(body_parts)
    body_indented = indent(body)

    if kind not in {"union", "difference", "intersection"}:
        raise SCADCodegenError(f"emit_csg cannot handle kind={kind!r}")

    return f"{kind}() {{\n{body_indented}\n}}"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def indent(text: str, spaces: int = 2) -> str:
    """Indent all lines in `text` by `spaces` spaces.

    Handles multi-line strings. Empty input returns an empty string.
    """

    if not text:
        return ""

    pad = " " * spaces
    # Ensure we don't add trailing spaces to a final empty line.
    lines = text.splitlines()
    return "\n".join(pad + line for line in lines)


if __name__ == "__main__":  # pragma: no cover - simple manual test hook
    # A small suite of example programs to sanity-check the AST and codegen.
    # Run:
    #   python -m src.scad_codegen
    # and inspect the emitted OpenSCAD.

    examples: List[Program] = []

    # 1. Original union of cube and translated sphere
    examples.append(
        {
            "root": {
                "kind": "union",
                "children": [
                    {
                        "kind": "cube",
                        "params": {"size": [20, 10, 5], "center": True},
                    },
                    {
                        "kind": "translate",
                        "params": {"offset": [10, 0, 0]},
                        "child": {
                            "kind": "sphere",
                            "params": {"r": 8},
                        },
                    },
                ],
            }
        }
    )

    # 2. Single centered cube primitive
    examples.append(
        {
            "root": {
                "kind": "cube",
                "params": {"size": [10, 10, 10], "center": True},
            }
        }
    )

    # 3. Single sphere primitive
    examples.append(
        {
            "root": {
                "kind": "sphere",
                "params": {"r": 5},
            }
        }
    )

    # 4. Single non-centered cylinder primitive
    examples.append(
        {
            "root": {
                "kind": "cylinder",
                "params": {"h": 20, "r": 4, "center": False},
            }
        }
    )

    # 5. Translated cube
    examples.append(
        {
            "root": {
                "kind": "translate",
                "params": {"offset": [15, -5, 3]},
                "child": {
                    "kind": "cube",
                    "params": {"size": [8, 6, 4], "center": True},
                },
            }
        }
    )

    # 6. Rotated cylinder
    examples.append(
        {
            "root": {
                "kind": "rotate",
                "params": {"angles": [90, 0, 45]},
                "child": {
                    "kind": "cylinder",
                    "params": {"h": 30, "r": 3, "center": True},
                },
            }
        }
    )

    # 7. Scaled sphere
    examples.append(
        {
            "root": {
                "kind": "scale",
                "params": {"factors": [1.5, 0.5, 2.0]},
                "child": {
                    "kind": "sphere",
                    "params": {"r": 6},
                },
            }
        }
    )

    # 8. Union of three cubes in a row
    examples.append(
        {
            "root": {
                "kind": "union",
                "children": [
                    {
                        "kind": "cube",
                        "params": {"size": [5, 5, 5], "center": True},
                    },
                    {
                        "kind": "translate",
                        "params": {"offset": [6, 0, 0]},
                        "child": {
                            "kind": "cube",
                            "params": {"size": [5, 5, 5], "center": True},
                        },
                    },
                    {
                        "kind": "translate",
                        "params": {"offset": [12, 0, 0]},
                        "child": {
                            "kind": "cube",
                            "params": {"size": [5, 5, 5], "center": True},
                        },
                    },
                ],
            }
        }
    )

    # 9. Difference: cylinder drilled through a cube
    examples.append(
        {
            "root": {
                "kind": "difference",
                "children": [
                    {
                        "kind": "cube",
                        "params": {"size": [20, 20, 20], "center": True},
                    },
                    {
                        "kind": "cylinder",
                        "params": {"h": 40, "r": 5, "center": True},
                    },
                ],
            }
        }
    )

    # 10. Intersection: overlapping sphere and cube
    examples.append(
        {
            "root": {
                "kind": "intersection",
                "children": [
                    {
                        "kind": "sphere",
                        "params": {"r": 12},
                    },
                    {
                        "kind": "translate",
                        "params": {"offset": [5, 0, 0]},
                        "child": {
                            "kind": "cube",
                            "params": {"size": [15, 15, 15], "center": True},
                        },
                    },
                ],
            }
        }
    )

    # Emit all examples, separated by comments
    for idx, program in enumerate(examples, start=1):
        print(f"// --- Example {idx} ---")
        print(emit_program(program))
