"""MicroCAD code generation for the minimal parametric-shape DSL.

This module turns a `ShapeState` (see `state_machine.py`) into a textual
MicroCAD-style program. The goal is not to use every MicroCAD feature, but to
emit something syntactically simple and readable that mirrors our sequential
add/subtract semantics:

    shape = first_primitive
    for each subsequent primitive:
        if role == positive:  shape = shape + prim
        if role == negative:  shape = shape - prim

The output is a `part` definition that can be saved as a `.mcad`-style file or
used purely as a human-readable representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

try:
    # Prefer relative import if this is used as a package
    from .state_machine import ShapeState, Primitive, Role
except ImportError:  # pragma: no cover - fallback for direct execution
    from state_machine import ShapeState, Primitive, Role  # type: ignore


# ---------------------------------------------------------------------------
# Primitive -> MicroCAD expression helpers
# ---------------------------------------------------------------------------


def _fmt_float(x: float) -> str:
    """Format floats compactly for code emission."""

    # Keep a few decimals, strip trailing zeros and dot.
    s = f"{x:.6f}"
    s = s.rstrip("0").rstrip(".")
    if s == "":
        s = "0"
    return s


def primitive_expr(prim: Primitive, name: str) -> str:
    """Return a single-line MicroCAD expression for a primitive.

    We stick to a very small subset of a MicroCAD-like API:

        Box(center = (cx, cy, cz), size = (sx, sy, sz), rotation = (rx, ry, rz))
        Sphere(center = (cx, cy, cz), radius = r, rotation = (rx, ry, rz))
        Cylinder(center = (cx, cy, cz), radius = r, height = h,
                 rotation = (rx, ry, rz))

    `name` is a suggested identifier (e.g., "p0"), but we only use it for
    comments in this version.
    """

    k = prim.kind
    p = prim.params or {}

    rotation = p.get("rotation", [0.0, 0.0, 0.0])
    if not isinstance(rotation, (list, tuple)) or len(rotation) != 3:
        rotation = [0.0, 0.0, 0.0]
    rx_s, ry_s, rz_s = map(_fmt_float, rotation)

    if k == "box":
        center = p.get("center", [0.0, 0.0, 0.0])
        size = p.get("size", [1.0, 1.0, 1.0])
        cx, cy, cz = map(_fmt_float, center)
        sx, sy, sz = map(_fmt_float, size)
        return (
            "Box(center = ({cx}, {cy}, {cz}), size = ({sx}, {sy}, {sz}), "
            "rotation = ({rx}, {ry}, {rz}))"
        ).format(cx=cx, cy=cy, cz=cz, sx=sx, sy=sy, sz=sz, rx=rx_s, ry=ry_s, rz=rz_s)

    if k == "sphere":
        center = p.get("center", [0.0, 0.0, 0.0])
        cx, cy, cz = map(_fmt_float, center)
        r = _fmt_float(p.get("radius", p.get("r", 1.0)))
        return (
            "Sphere(center = ({cx}, {cy}, {cz}), radius = {r}, "
            "rotation = ({rx}, {ry}, {rz}))"
        ).format(cx=cx, cy=cy, cz=cz, r=r, rx=rx_s, ry=ry_s, rz=rz_s)

    if k == "cylinder":
        center = p.get("center", [0.0, 0.0, 0.0])
        cx, cy, cz = map(_fmt_float, center)
        r = _fmt_float(p.get("radius", p.get("r", 1.0)))
        h = _fmt_float(p.get("height", p.get("h", 1.0)))
        return (
            "Cylinder(center = ({cx}, {cy}, {cz}), radius = {r}, height = {h}, "
            "rotation = ({rx}, {ry}, {rz}))"
        ).format(cx=cx, cy=cy, cz=cz, r=r, h=h, rx=rx_s, ry=ry_s, rz=rz_s)

    # Fallback: emit a comment-like placeholder so we don't crash.
    return f"/* unsupported primitive kind={k!r} */ Box(center = (0, 0, 0), size = (0, 0, 0))"


# ---------------------------------------------------------------------------
# Top-level emission
# ---------------------------------------------------------------------------


def emit_microcad(state: ShapeState, part_name: str = "GeneratedShape") -> str:
    """Emit a MicroCAD-style program for a given ShapeState.

    Parameters
    ----------
    state : ShapeState
        The program state produced by the DSL state machine.
    part_name : str, optional
        Name of the generated `part`, by default "GeneratedShape".

    Returns
    -------
    str
        MicroCAD-like source code as a single string.
    """

    prims = state.primitives

    lines: List[str] = []
    lines.append("// Auto-generated from ShapeState")
    lines.append("// Sequential add/subtract semantics")
    lines.append("")
    lines.append(f"part {part_name}() {{")

    if not prims:
        # Degenerate case: no primitives, emit an empty stub.
        lines.append("    // Empty shape; no primitives in state")
        lines.append("    // Return a degenerate sphere as placeholder")
        lines.append("    Sphere(center = (0, 0, 0), radius = 0.0);")
        lines.append("}")
        return "\n".join(lines)

    # Emit each primitive as a local binding p0, p1, ... for readability.
    for idx, prim in enumerate(prims):
        var = f"p{idx}"
        expr = primitive_expr(prim, var)
        role_comment = f"// {prim.role.value}"
        lines.append(f"    let {var} = {expr}; {role_comment}")

    lines.append("")

    # Build the combined shape with sequential union/difference.
    lines.append("    // Sequential constructive solid geometry")
    lines.append("    let shape = p0;")
    for idx, prim in enumerate(prims[1:], start=1):
        var = f"p{idx}"
        if prim.role == Role.POSITIVE:
            op = "+"  # union
        else:
            op = "-"  # subtraction
        lines.append(f"    shape = shape {op} {var};")

    lines.append("    shape;")
    lines.append("}")

    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    # Minimal manual test: construct a tiny state and print the MicroCAD code.
    from state_machine import ShapeState, Primitive, Role  # type: ignore

    s = ShapeState()
    s.primitives.append(
        Primitive(
            kind="box",
            role=Role.POSITIVE,
            params={"center": [0.0, 0.0, 0.0], "size": [2.0, 2.0, 2.0]},
        )
    )
    s.primitives.append(
        Primitive(
            kind="sphere",
            role=Role.NEGATIVE,
            params={"center": [0.5, 0.5, 0.5], "radius": 0.75},
        )
    )

    print(emit_microcad(s))
