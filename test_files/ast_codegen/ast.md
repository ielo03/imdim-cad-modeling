# OpenSCAD-Subset AST & Action Space Design

This document defines a **clean, stable AST** and **RL action space** for generating OpenSCAD-style models from 2D images (or other inputs). The goal is:

- Every sequence of actions corresponds to a **syntactically valid** program.
- The AST is **simple**, **serializable**, and easy to render to `.scad`.
- The action space is **discrete** and compatible with RL (PPO / REINFORCE).

We intentionally define a **subset** of OpenSCAD that covers the core geometric modeling primitives we care about.

---

## 1. Language Subset Overview

Supported constructs:

- **Primitives**

  - `cube(size = [sx, sy, sz], center = true)`
  - `sphere(r = r)`
  - `cylinder(h = h, r = r, center = true)`

- **Transforms** (unary)

  - `translate([tx, ty, tz])`
  - `rotate([rx, ry, rz])`
  - (optional later) `scale([sx, sy, sz])`

- **Boolean CSG operations** (n-ary)

  - `union() { ... }`
  - `difference() { ... }`
  - `intersection() { ... }`

- **Program structure**
  - A `Program` is a **single root node** (usually `union` or `difference`) with one or more children.

No loops, no modules, no variables, no conditionals. This keeps the AST compact and RL-friendly.

---

## 2. AST Node Types

We define a small set of node types. This is language-agnostic (you can implement it in Python, Rust, etc.).

### 2.1. Primitive Nodes

```text
PrimitiveNode:
  kind: "cube" | "sphere" | "cylinder"
  params:
    # cube
    size: [sx, sy, sz]   # floats or discretized bins
    center: bool = true

    # sphere
    r: float

    # cylinder
    h: float
    r: float
    center: bool = true
```

### 2.2. Transform Node

A transform wraps exactly **one child**.

```text
TransformNode:
  kind: "translate" | "rotate" | "scale" (optional)
  params:
    # translate
    offset: [tx, ty, tz]

    # rotate
    angles: [rx, ry, rz]  # degrees

    # scale (optional)
    factors: [sx, sy, sz]

  child: Node
```

### 2.3. Boolean CSG Node

CSG nodes combine **multiple children**:

```text
CSGNode:
  kind: "union" | "difference" | "intersection"
  children: List[Node]  # length >= 1 for union/intersection, >= 2 for difference
```

### 2.4. Root Program Node

```text
Program:
  root: Node  # usually a CSGNode, but can be a PrimitiveNode or TransformNode
```

Where `Node` is the sum type:

```text
Node = PrimitiveNode | TransformNode | CSGNode
```

---

## 3. Grammar (Informal EBNF)

This is the **generated** OpenSCAD-like syntax corresponding to the AST.

```ebnf
program        = node ;

node           = primitive
               | transform
               | csg ;

primitive      = cube | sphere | cylinder ;

cube           = "cube" "(" "[" float "," float "," float "]" "," "center" "=" bool ")" ";" ;

sphere         = "sphere" "(" "r" "=" float ")" ";" ;

cylinder       = "cylinder" "(" "h" "=" float "," "r" "=" float "," "center" "=" bool ")" ";" ;

transform      = translate | rotate | scale ;

translate      = "translate" "(" "[" float "," float "," float "]" ")" "{" node "}" ;

rotate         = "rotate" "(" "[" float "," float "," float "]" ")" "{" node "}" ;

scale          = "scale" "(" "[" float "," float "," float "]" ")" "{" node "}" ;

csg            = union | difference | intersection ;

union          = "union" "()" "{" { node } "}" ;

difference     = "difference" "()" "{" node node { node } "}" ;

intersection   = "intersection" "()" "{" { node } "}" ;

bool           = "true" | "false" ;

float          = /* numeric literal, may be from discretized bins */ ;
```

This grammar is **derived from the AST**, not the other way around. The AST is the ground truth.

---

## 4. RL Action Space Design

We want an action space where **every sequence of actions builds a well-formed AST**. We do this by:

1. Representing the partially built program as a tree with a **cursor**.
2. Allowing only context-valid actions at each cursor location.
3. Decoding the entire AST to `.scad` only at the end of an episode.

### 4.1. High-Level Action Types

We define a discrete set of **structural actions** plus **parameter actions**.

#### Structural actions

- `ADD_PRIMITIVE(kind)`
  - `kind ∈ { cube, sphere, cylinder }`
- `WRAP_TRANSFORM(kind)`
  - `kind ∈ { translate, rotate, scale }`
- `START_CSG(kind)`
  - `kind ∈ { union, difference, intersection }`
- `ADD_CHILD`
  - attach a new child node under the current CSG node
- `CLOSE_NODE`
  - finish the current node and move cursor up
- `END_PROGRAM`
  - finish the program

#### Parameter actions

To avoid invalid syntax, parameters are **semantic, not textual**. Examples:

- `SET_SIZE_X(bin_id)`
- `SET_SIZE_Y(bin_id)`
- `SET_SIZE_Z(bin_id)`
- `SET_RADIUS(bin_id)`
- `SET_HEIGHT(bin_id)`
- `SET_TRANSLATE_X(bin_id)` etc.
- `SET_ROTATE_X(bin_id)` etc.

Where `bin_id` indexes into a predefined set of numeric values, e.g.:

```text
SIZE_BINS     = [5, 10, 15, 20, 25, 30]
OFFSET_BINS   = [-30, -20, -10, 0, 10, 20, 30]
ANGLE_BINS    = [0, 15, 30, 45, 60, 90, 120, 180]
RADIUS_BINS   = [2, 4, 6, 8, 10, 12]
HEIGHT_BINS   = [5, 10, 15, 20, 25, 30]
```

### 4.2. State Representation (for the Policy)

A state at time `t` might include:

- Embedding of the **target image**.
- Embedding of the **partial render** (optional but useful).
- Serialized view of the **partial AST** (e.g., via a tree encoder or flattened token sequence).
- The **current cursor location** in the tree.

The policy `π(a | state)` decides the next valid action.

### 4.3. Ensuring Syntactic Validity

The environment guarantees syntactic validity by:

- Keeping an explicit **stack of open nodes** (like a parser stack).
- At each step, only allowing actions that are valid in the current context.

Example constraints:

- You can only call `ADD_CHILD` if the cursor is on a `CSGNode`.
- You can only call `SET_RADIUS` right after `ADD_PRIMITIVE(sphere)` or `ADD_PRIMITIVE(cylinder)` before closing the primitive.
- `difference` must have at least 2 children before `CLOSE_NODE` is allowed.
- `END_PROGRAM` is only allowed when:
  - the cursor is at the root, and
  - all nodes are syntactically complete.

This way, **every finished trajectory corresponds to a valid AST**.

---

## 5. Serialization to OpenSCAD

Given a completed `Program` AST, serialization to `.scad` is straightforward.

### 5.1. Pseudocode

```python
def emit_program(program: Program) -> str:
    return emit_node(program.root) + "\n"


def emit_node(node: Node) -> str:
    if node.kind in ["cube", "sphere", "cylinder"]:
        return emit_primitive(node)
    elif node.kind in ["translate", "rotate", "scale"]:
        return emit_transform(node)
    elif node.kind in ["union", "difference", "intersection"]:
        return emit_csg(node)
    else:
        raise ValueError(f"Unknown node kind: {node.kind}")


def emit_primitive(p: PrimitiveNode) -> str:
    if p.kind == "cube":
        sx, sy, sz = p.size
        return f"cube(size=[{sx},{sy},{sz}], center={str(p.center).lower()});"
    elif p.kind == "sphere":
        return f"sphere(r={p.r});"
    elif p.kind == "cylinder":
        return f"cylinder(h={p.h}, r={p.r}, center={str(p.center).lower()});"


def emit_transform(t: TransformNode) -> str:
    if t.kind == "translate":
        tx, ty, tz = t.offset
        inner = emit_node(t.child)
        return f"translate([{tx},{ty},{tz}]) {{\n{indent(inner)}\n}}"
    elif t.kind == "rotate":
        rx, ry, rz = t.angles
        inner = emit_node(t.child)
        return f"rotate([{rx},{ry},{rz}]) {{\n{indent(inner)}\n}}"
    elif t.kind == "scale":
        sx, sy, sz = t.factors
        inner = emit_node(t.child)
        return f"scale([{sx},{sy},{sz}]) {{\n{indent(inner)}\n}}"


def emit_csg(c: CSGNode) -> str:
    kind_str = c.kind  # "union" / "difference" / "intersection"
    body = "\n".join(emit_node(ch) for ch in c.children)
    return f"{kind_str}() {{\n{indent(body)}\n}}"


def indent(s: str, spaces: int = 2) -> str:
    pad = " " * spaces
    return "\n".join(pad + line for line in s.splitlines())
```

This gives you deterministic, readable OpenSCAD code from a clean AST.

---

## 6. Extensions (Future Work)

Possible extensions once the core system works:

- **More primitives**: `polyhedron`, `surface`, `text`, etc.
- **Mirroring / scaling** transforms.
- **Symmetry constraints** as higher-level actions.
- **2D primitives** + `linear_extrude` / `rotate_extrude`.
- **Multi-view consistency**: separate reward terms for front/top/side images.
- **Hierarchical policies**: high-level agent plans CSG structure, low-level agent fills parameters.

---

## 7. Summary

- We define a **minimal, clean AST** for OpenSCAD-style CSG modeling.
- We design an **RL action space** that guarantees **syntactic validity** via structured nodes and context-aware actions.
- We specify a **serialization path** from AST → `.scad` text.
- This setup is intended for use with **image-conditioned RL** for CAD model synthesis.
