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

## Periodic boundaries: the weld `attach` does not make

{func}`hexmesh.periodic_pairs <nekmeshpy.hexmesh.periodic.periodic_pairs>` resolves the
same *stated* correspondence `attach` does, proved the same way, and then leaves the two
sides alone: they stay two distinct boundary faces with their own names, which is what a
periodic boundary is. It returns a `(2K,4)` table of `[element, face, partner element,
partner face]` — both directions, so the two rows cannot disagree — and
{func}`writer.to_re2 <nekmeshpy.io.writer.to_re2>`'s `periodic=` writes it into the
boundary record's `bc(1)` / `bc(2)`, the fields a Nek `'P  '` row is meaningless without.

```python
from nekmeshpy.core import affine

GROUPS = {"wall": "W  ", "inlet": "P  ", "outlet": "P  "}
PERIODIC = [hexmesh.Periodic("inlet", "outlet",
                             affine.translation([0.0, 0.0, LENGTH]))]
writer.to_re2(mesh, "case.re2", groups=GROUPS, periodic=PERIODIC)
```

Unlike a `Seam`, a {class}`Periodic <nekmeshpy.hexmesh.periodic.Periodic>` is **told its
transform**. `attach` needs none because its halves are meant to end up in the same
place; a periodic pair's halves sit a lattice vector apart, where the nearest face across
the gap is not the periodic image. Stating the map is also what makes it checkable: the
worst residual after mapping is compared against `conform.entity_tol`, so a mis-typed
pitch raises quoting the number.

Two things it cannot check. A group with a rotational symmetry of its own pairs
bijectively at residual **zero** onto a cyclic shift — `attach`'s own trap. And Nek
identifies the two sides in the **global Cartesian frame**: it does not rotate a vector
across a periodic face, so a rotational pair is a periodicity for a scalar, and for a
velocity field only where that field is invariant in that frame.

The two halves must be named distinctly — one tag over both ends cannot say which end is
which. `to_re2` then enforces that a name coded `'P  '` and a name in `periodic=` are the
same set: a `'P  '` with no pairing writes partner element 0, face 0, and a pairing with
no `'P  '` exports as something else, neither visible until the solver reads the mesh.

## Conjugate meshes: `to_re2(..., fluid=, thermal=)`

A Nek conjugate heat transfer run needs its `.re2` header's two element counts to
disagree — `nelgt` (total) and `nelgv` (velocity mesh) — with the velocity-mesh elements
listed **first, contiguously**. A region tag alone does not say this: the solver reads
element order, not `element_tags`. `fluid=` names the velocity-mesh region (anything
{func}`core.tags.element_mask <nekmeshpy.core.tags.element_mask>` accepts — a tag string,
a boolean mask, or an id array) and `to_re2` reorders the **written bytes** — corners,
every boundary row, and a periodic row's partner — through the fluid-first permutation.
The `HexMesh` object returned is untouched; only the file's numbering changes.

`fluid=None` (the default) writes every element as velocity-mesh — `nelgv == nelgt` — the
right reading for a single-domain mesh, and the same bytes this writer produced before
`fluid=` existed.

The moment `nelgv < nelgt`, Nek's own reader (`core/reader_re2.f`) always reads **one
boundary block per field**, regardless of what the header declares — a file with only
the velocity block reads it fine, then dies reading the (missing) next one. `thermal=`
is that second field's table, over **every** element rather than the velocity-only
`groups`, and is required exactly when `fluid=` makes the two counts differ:

```python
GROUPS = {"interface": {"fluid": "W  ", "solid": None}, "inlet": "P  ", "outlet": "P  "}
THERMAL = {"inlet": "P  ", "outlet": "P  ", "outer": "I  ", "cut_lo": "P  ", "cut_hi": "P  "}
writer.to_re2(mesh, "case.re2", groups=GROUPS, periodic=PERIODIC,
              fluid="fluid", thermal=THERMAL)
```

A name left out of `thermal=` stays conformal (`'E  '`) — right for a genuine conjugate
interface, since temperature is solved on both sides of it and only velocity needs the
explicit wall (the solid side has no velocity unknown to fall back on). A name that *is*
coded but lands outside a field's own region raises rather than corrupting Nek's
`nel=nelv` invariant for the velocity block. A periodic pair may sit entirely inside
either region, and each field checks its own `'P  '` names against `periodic=`
independently, restricted to that field's elements.

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
