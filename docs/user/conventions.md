# Conventions

## Indexing & coordinates

- Indices are **0-based** internally; `.re2` element ids are written 1-based.
- No `Point` class — a single point is a `(3,)` array. Coordinates live on one
  `(P,3)` array, stored by `PointMesh` at the bottom of the ladder and reached
  from every rung above as a **live, writable view**: `mesh.points[:] = X`
  propagates through the whole B-rep for free.
- 3-D is required: a `(N,2)` array is rejected, not padded to `z=0`. The refusal
  is raised by `PointMesh`, which every rung promotes a bare array through.
- `LineMesh` takes a **required** `lines` connectivity argument — nothing implies a
  default chain or wrap. {func}`linemesh.loft <nekmeshpy.linemesh.assemble.loft>`
  authors one explicitly via `loop=False`/`loop=True`, and every line-rung *shape*
  factory funnels through it.

## Immutability & storage

Every ladder container is immutable by construction: build one with a factory, a
combinator, or the array constructor. Topology is fixed; coordinates aren't.

Stored state is the B-rep — each rung holds the rung below plus what it privately
owns. `HexMesh`: a `quad_mesh` (`QuadMesh` of the shared faces), `hexes` `(E,6)`
face incidence, `orient` `(E,6)` D4 codes, `interior` `(E,(order-1)**3,3)`.
`QuadMesh` mirrors it one rung down (`line_mesh` + `quads`/`orient`/`interior`),
`LineMesh` does the same onto a `PointMesh`, which holds only coordinates and their
tags. `points` is the shared writable view described above; `corners` `(E,8)` is a
**derived read-only** view, so corner consistency is structural, not maintained
(see {doc}`concepts` § high-order elements).

No container takes or stores `order` — it derives from `interior.shape[1] + 1` at
the bottom and rides up, so a mesh cannot disagree with the nodes it stores.

## Joining: two welds, told different things

Both `merge` and `attach` coordinate-weld, at all three rungs. The difference is
what the caller states.

- `merge(meshes, *, tol=1e-7)` infers every seam in the assembly from coordinates.
  **`tol` is a fraction, not a distance** — of `conform.bbox_scale`, the largest of
  the x/y/z ranges over every point handed in, so the radius is `tol * bbox_scale`
  and the default means the same thing at any model size. A caller who knows a real
  distance divides by `bbox_scale`. Anything at or above
  `conform.MAX_WELD_FRACTION` (`0.1`) is refused, since 10% of the model is far
  likelier to be a distance passed by mistake than a coincidence tolerance.
  Coincidence is a radius and only a radius: two points fuse when they are strictly
  closer than the radius, transitively.
- `attach(meshes, seams)` is **n-ary** and takes **no tolerance**. Each `Seam` names
  which group of which block meets which — point groups at the line rung, edge
  groups at quad, face groups at hex — by tag name or by an explicit id array.
  Inside the two named groups the pairing is nearest-neighbour, proved by
  *bijectivity* rather than by a distance. `own=` picks whose coordinates the seam
  keeps (a byte copy, not an average); `attach_tag=` names the welded-shut entities,
  and the default `None` **clears** them, because a buried entity that keeps its name
  makes the exporter write one boundary row from each side of it.

`select`/`remove`/`components` are the inverse at every rung.

## Tags

There is one table type, `ElementTags` (`core/tags.py`): a sparse `ids` + `tags`
pair, so an untagged mesh stores nothing and `len()` is the *tagged* count.

**A rung's side tags *are* the rung below's `element_tags`**, read through a named
property: `HexMesh.face_tags` is `quad_mesh.element_tags`, `QuadMesh.edge_tags` is
`line_mesh.element_tags`, `LineMesh.point_tags` is `point_mesh.element_tags`. A tag
is addressed by **entity id**, never by `(element, side)` — a face is one stored
object both its hexes reference, so it carries one name and the two sides cannot
disagree.

A side-tag table is **not** "the boundary": it is a named subset of entities, and it
may also name interior planes, while `boundary_faces()` derives the topological
boundary from connectivity. `tag_report(mesh)` counts both ways they can disagree.

Only the top rung's `element_tags` names a **region** (`"fluid"`, `"solid"`). One
rung down an element is a piece of some volume's surface, so its `element_tags` is a
boundary name (`"wall"`, `"inlet"`). Not enforced, but the mechanism punishes getting
it wrong: `first_tag`/`last_tag` default to the bounding slice's own tag, so a
`"fluid"`-tagged section exports `"fluid"` caps.

`NO_TAG` is `""`. On cap arguments, **`None` means "not asked for"** (inherit) and
`NO_TAG` means an explicit override *to* untagged. `retag_element`/`retag_face`/
`retag_edge`/`retag_point` rename one vocabulary without touching the other;
`tag_edges`/`tag_faces` author tags after the fact, by `(quad, side)` rows and by
face id respectively.

## Strong typing

Enforced — `mypy` runs `disallow_untyped_defs`, `check_untyped_defs`,
`disallow_any_generics` over `files = ["nekmeshpy"]`. Everything in `nekmeshpy/` is
annotated; tests and examples are not checked, so a wrong-rung call there surfaces as
a pytest `AttributeError` rather than a type error.

- Geometry parameters take the concrete type (`LineMesh`, etc.), no
  `| np.ndarray` fallback. Open-vs-closed isn't a type or stored flag — read
  it off `lines`. Vector *literals* (axis/origin/center) use
  `Sequence[float] | FloatArray`.
- Use the dtype aliases in `nekmeshpy._typing` — `FloatArray`, `IntArray`,
  `BoolArray`, `StrArray` — never a bare `np.ndarray` (`disallow_any_generics`
  rejects it).
- `Point`/`Vec3`/`PointArray` are shape-*documentation* aliases of
  `FloatArray` (single `(3,)` location/direction, or any array whose trailing
  axis is the 3 spatial components) — the concrete shape belongs in the
  docstring. Non-position float data (fractions, grading, GLL nodes, quality
  metrics, tolerances) stays plain `FloatArray`.

The package ships a `py.typed` marker.

## Logging

Progress goes through the `nekmeshpy` logger and its module children; configure the
parent in your script and it catches all of them:

```python
logging.basicConfig(level=logging.INFO, format="%(message)s")
```
