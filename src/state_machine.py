"""
State machine for the minimal parametric-shape DSL.

This implements the semantics described in design.md:

  - State: append-only (with UNDO) list of primitives.
  - Tokens:
        ADD_BOX
        ADD_SPHERE
        ADD_CYLINDER
        MAKE_LAST_NEGATIVE
        UNDO_LAST
        END
  - Each decoding step from the model yields:
        (token, params)
    where `params` is a flat float vector interpreted according to the token.

    - Primitive params always include center + size/radius/height + rotation:
        * center:   (cx, cy, cz)
        * rotation: (rx, ry, rz) in degrees, applied about the primitive center

  - Evaluation rule (at a higher level):
        shape = empty
        for prim in primitives:
            if prim.role == positive:
                shape = shape ∪ prim
            else:
                shape = shape \ prim
        final_shape = shape

This module does NOT know about meshes or Chamfer loss. It just maintains a
clean, minimal, serializable representation of the "program" that other
components (e.g., meshing / renderer) can consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum, IntEnum, auto
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# DSL definitions
# ---------------------------------------------------------------------------


class Role(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class Token(IntEnum):
    """Discrete DSL tokens.

    These should line up with whatever the transformer policy outputs.
    """

    ADD_BOX = 0
    ADD_SPHERE = 1
    ADD_CYLINDER = 2
    MAKE_LAST_NEGATIVE = 3
    UNDO_LAST = 4
    END = 5


# ---------------------------------------------------------------------------
# Primitive + state representation
# ---------------------------------------------------------------------------


@dataclass
class Primitive:
    """Single parametric primitive in the DSL state."""

    kind: str  # "box" | "sphere" | "cylinder"
    role: Role = Role.POSITIVE
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Store role as simple string for easier JSON/serialization.
        d["role"] = self.role.value
        return d


@dataclass
class ShapeState:
    """Mutable state for a single program (episode).

    This is what the "state machine" operates on: each (token, params) pair
    mutates this state until END is reached.
    """

    primitives: List[Primitive] = field(default_factory=list)
    ended: bool = False

    # ------------------------------------------------------------------
    # Core mutation API
    # ------------------------------------------------------------------

    def apply(
        self,
        token: Token | int,
        params: Optional[Sequence[float]] = None,
    ) -> None:
        """Apply a token + params to the current state.

        - `token` is one of the Token enum values (or its int id).
        - `params` is a flat float vector. For non-primitive tokens it may be
          None and is ignored.

        Param layout (by convention; keep in sync with model):

            The model always emits a fixed-length vector of 9 floats:

                params_vec = [cx, cy, cz, p0, p1, p2, rx, ry, rz]

            where:
                - (cx, cy, cz) is the primitive center
                - (rx, ry, rz) are Euler rotation angles in degrees (Z-Y-X order),
                  applied about the primitive center
                - (p0, p1, p2) are interpreted per token as:

                    ADD_BOX:
                        size = (sx, sy, sz) = (p0, p1, p2)

                    ADD_SPHERE:
                        radius = r = p0 (p1, p2 are ignored)

                    ADD_CYLINDER:
                        radius = r = p0
                        height = h = p1 (p2 is ignored)

            `ShapeState.apply` is responsible for mapping this unified vector
            into the token-specific parameter dicts stored on each Primitive.
        """
        if isinstance(token, int) and not isinstance(token, Token):
            try:
                token = Token(token)
            except ValueError as exc:
                raise ValueError(f"Unknown token id: {token}") from exc

        if self.ended:
            # For safety: do not allow further mutation after END.
            # Caller can decide whether to silently ignore instead.
            raise RuntimeError("Cannot apply token after END has been emitted")

        if token == Token.ADD_BOX:
            self._add_box(params)
        elif token == Token.ADD_SPHERE:
            self._add_sphere(params)
        elif token == Token.ADD_CYLINDER:
            self._add_cylinder(params)
        elif token == Token.MAKE_LAST_NEGATIVE:
            self._make_last_negative()
        elif token == Token.UNDO_LAST:
            self._undo_last()
        elif token == Token.END:
            self.ended = True
        else:
            raise ValueError(f"Unhandled token: {token!r}")

    # ------------------------------------------------------------------
    # Primitive creation helpers
    # ------------------------------------------------------------------

    def _require_params(
        self, token: Token, params: Optional[Sequence[float]], expected_len: int
    ) -> Sequence[float]:
        if params is None:
            raise ValueError(f"{token.name} requires {expected_len} params, got None")
        if len(params) < expected_len:
            raise ValueError(
                f"{token.name} requires {expected_len} params, got {len(params)}"
            )
        return params

    def _add_box(self, params: Optional[Sequence[float]]) -> None:
        p = self._require_params(Token.ADD_BOX, params, expected_len=9)
        cx, cy, cz, p0, p1, p2, rx, ry, rz = p[:9]
        sx, sy, sz = p0, p1, p2
        prim = Primitive(
            kind="box",
            role=Role.POSITIVE,
            params={
                "center": [float(cx), float(cy), float(cz)],
                "size": [float(sx), float(sy), float(sz)],
                "rotation": [float(rx), float(ry), float(rz)],
            },
        )
        self.primitives.append(prim)

    def _add_sphere(self, params: Optional[Sequence[float]]) -> None:
        p = self._require_params(Token.ADD_SPHERE, params, expected_len=9)
        cx, cy, cz, p0, p1, p2, rx, ry, rz = p[:9]
        r = p0
        prim = Primitive(
            kind="sphere",
            role=Role.POSITIVE,
            params={
                "center": [float(cx), float(cy), float(cz)],
                "radius": float(r),
                "rotation": [float(rx), float(ry), float(rz)],
            },
        )
        self.primitives.append(prim)

    def _add_cylinder(self, params: Optional[Sequence[float]]) -> None:
        p = self._require_params(Token.ADD_CYLINDER, params, expected_len=9)
        cx, cy, cz, p0, p1, p2, rx, ry, rz = p[:9]
        r, h = p0, p1
        prim = Primitive(
            kind="cylinder",
            role=Role.POSITIVE,
            params={
                "center": [float(cx), float(cy), float(cz)],
                "radius": float(r),
                "height": float(h),
                "rotation": [float(rx), float(ry), float(rz)],
                # We can add an "axis" field later if needed.
            },
        )
        self.primitives.append(prim)

    # ------------------------------------------------------------------
    # Non-primitive token helpers
    # ------------------------------------------------------------------

    def _make_last_negative(self) -> None:
        if not self.primitives:
            # Should typically be masked by the environment / policy.
            # We still guard here to avoid crashes.
            return
        self.primitives[-1].role = Role.NEGATIVE

    def _undo_last(self) -> None:
        if not self.primitives:
            # Again, this should be masked, but we make it a no-op if hit.
            return
        self.primitives.pop()

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable representation of the current program."""
        return {
            "ended": self.ended,
            "primitives": [p.to_dict() for p in self.primitives],
        }

    def clone(self) -> "ShapeState":
        """Deep-ish copy of the state (sufficient for our simple fields)."""
        new = ShapeState()
        new.ended = self.ended
        new.primitives = [
            Primitive(kind=p.kind, role=p.role, params=dict(p.params))
            for p in self.primitives
        ]
        return new


# ---------------------------------------------------------------------------
# Simple smoke test (can be run manually)
# ---------------------------------------------------------------------------


if __name__ == "__main__":  # pragma: no cover - manual sanity check
    # Example: build a simple shape with a box and a subtractive sphere.
    s = ShapeState()
    s.apply(Token.ADD_BOX, [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0])
    s.apply(Token.ADD_SPHERE, [0.5, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    s.apply(Token.MAKE_LAST_NEGATIVE)
    s.apply(Token.END)
    import json

    print(json.dumps(s.to_dict(), indent=2))
