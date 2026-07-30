# Concepts

The ideas the toolkit is built on: the dimensional ladder of mesh containers,
the two tag systems, the factories, per-section smoothing, and physical groups.
Companion to the {doc}`getting-started` tutorial and {doc}`../reference/index`.

## The line → quad → hex ladder

Geometry is modeled with one mesh container per dimension, each with 2 / 4 / 8
vertices per element:

| container | element | role |
|---|---|---|
| {class}`~nekmeshpy.linemesh.LineMesh` | line (2 pts) | 1-D boundary — a ring, an edge, a spine |
| {class}`~nekmeshpy.quadmesh.QuadMesh` | quad (4 pts) | 2-D cross-section / surface |
| {class}`~nekmeshpy.hexmesh.HexMesh` | hex (8 pts)  | 3-D all-hex volume |

Each container stores coordinates as a **bare `(P,3)` NumPy array** on `.points`
(mutate in place with `mesh.points[:] = X`). There is **no `Point` class** — a
single point is just a `(3,)` array. Boundaries live in 3-D: a `(N,2)` array is
*rejected*, never padded to `z=0`.

`LineMesh` holds a shared `(N,3)` point array plus `(L,2)` `lines` connectivity
that **can branch** (a mesh, not a single ordered path). Open vs closed is a
**topological property** (`is_open` / `is_closed`), not a subclass; factories set
it:

- `LineMesh.open` — consecutive chain (default).
- `LineMesh.loop` — chain that wraps to the start.
- `LineMesh.line(start, end, fractions, …)` — straight edge sampled at the given
  fractions (a direct lerp, meshed exactly).
- `LineMesh.circle(radius, n, center=…, normal=…, start_theta=0.0)` — closed ring
  in the plane with the given `normal` (default `+z`); `start_theta` rotates the
  first point off `+e1`.
- `LineMesh.rectangle(width, height, n, center=…, normal=…, side_tags=…)` — closed
  far-field loop in the given plane, discretized into `n` line elements (`n` a
  multiple of 4): `n // 4` evenly spaced per side, CCW from the lower-left corner
  (bottom / right / top / left), corners always landing on a point. Pass `n` equal
  to the inner loop's point count and rotate the inner `circle` with `start_theta`
  so index 0 meets the lower-left corner, and the two loops pair index-for-index in
  `annulus` (the radial spokes need not be straight).
- `LineMesh.from_segments` — chain unordered segments into the largest closed
  loop (or `None`).
- `LineMesh.merge` — weld coincident **topological end points** (degree-1 chain
  ends; never interior points), the 1-D sibling of `QuadMesh.merge`/`HexMesh.merge`.
  The result is `closed` iff no degree-1 end survives, so two shared-endpoint
  `A1->A2` arcs (reverse one) weld at `A1`/`A2` into a single loop — the clean way
  to close a seam ring from two half-arcs.

Every factory meshes its points **exactly** — there is no resampling API; the
caller hands in an exactly-sized, correctly-oriented curve. The ordered ops treat
points in index order as a path/loop (`.length`).

{class}`~nekmeshpy.trimesh.TriMesh` is the **input surface** for the vessel
pipeline; its algorithms (cotan Laplacian, Dirichlet solve, boundary loops) live
in {mod}`nekmeshpy.trimesh.ops` (reached as `nekmeshpy.trimesh.ops`).

## The two tag systems

Both **propagate up the ladder** (line → quad → hex) on `extrude` / `loft`, and
both are no-ops when untagged.

### `element_tags` — dense, per-element (region / material)

One tag per line / quad / hex (`""` = untagged). Set at construction on the
`LineMesh`, copied by the section factories onto the section edges/quads and
thence onto the hex faces/hexes. `element_group_tags` is the sorted unique
non-empty set.

### `boundary_tags` — sparse, parallel with `boundaries`

A sparse string array parallel with `boundaries` `(Nbc,2)`. The second column is
the "side" at each level:

- `LineMesh`: `[elem id, side ∈ {1,2}]` → end **point** `s-1`.
- `QuadMesh`: `[quad id, side ∈ {1..4}]` → **edge** `EDGE_POINTS[s-1]`.
- `HexMesh`: `[elem id, face ∈ {1..6}]` → **face**.

On `extrude`, line end-point tags become quad boundary **edges**, then hex
boundary **faces**. `boundary_group_tags` is the sorted unique set. See
`examples/flow_past_cylinder.py`.

### Tag at the lowest level; upper overrides lower

Every section-wall tag can originate on the `LineMesh` input, which each section
factory reads:

- `ogrid` / `annulus` — the loop's per-line tags,
- `half_ogrid` — the arc's per-segment tags,
- `structured` — each edge's uniform tag.

The factory args (`wall_tag`, `inner_tag`, `outer_tag`, `boundary_tags[side]`)
are **overrides**: a non-empty arg replaces the line-level tag; an empty/absent
one falls through (a present-but-empty `boundary_tags[side]` / `NO_BOUNDARY`
suppresses the side). Sweep end caps (`first_tag` / `last_tag`) are named at the
hex level — no lower level exists for them.

## Section factories (`QuadMesh` classmethods)

Sections fill a boundary with quads. All build **natively in 3-D** — nothing is
projected to a plane, so a boundary in any plane, or a curvy / non-planar one, is
filled in place with its true shape.

| factory | fills |
|---|---|
| {meth}`~nekmeshpy.quadmesh.QuadMesh.structured` | transfinite (Coons) grid over 4 open `LineMesh` edges; resolution comes from the edges' own points (no resampling — opposite edges must match counts); each side named from its edge tag |
| {meth}`~nekmeshpy.quadmesh.QuadMesh.ogrid` | O-grid inside a closed loop (no collapsed centre); outer ring named from the loop's per-line tags |
| {meth}`~nekmeshpy.quadmesh.QuadMesh.half_ogrid` | half-disc O-grid split along a spine; wall named from the arc's per-segment tags |
| {meth}`~nekmeshpy.quadmesh.QuadMesh.annulus` | ring O-grid between inner and outer closed loops, paired **by index** (equal point counts — e.g. `LineMesh.rectangle(w, h, N)` against `circle(r, N, start_theta=…)`) |
| {meth}`~nekmeshpy.quadmesh.QuadMesh.extrude` / {meth}`~nekmeshpy.quadmesh.QuadMesh.loft` | sweep/stack a `LineMesh` one dimension down into a quad strip |
| {meth}`~nekmeshpy.quadmesh.QuadMesh.from_grid` | structured `(ni+1,nj+1)` quad grid; `element_tag` fills the per-quad tags |

`ogrid` / `annulus` build a straight-chord initial guess and rely on
`smoothing_method="conduction"` to relax the interior onto the curved boundary
ring; `structured` / `half_ogrid` blend the 3-D edge points directly. (`ogrid` /
`half_ogrid` are ICEM/Pointwise terms; the rest follow gmsh.)

## Hex-block factories (`HexMesh` classmethods)

| factory | builds |
|---|---|
| {meth}`~nekmeshpy.hexmesh.HexMesh.extrude` | sweep one section along a straight axis (gmsh Extrude + Layers + Recombine) |
| {meth}`~nekmeshpy.hexmesh.HexMesh.loft` | recombine a stack of pre-positioned conformal profiles — the general case behind `extrude` |
| {meth}`~nekmeshpy.hexmesh.HexMesh.annulus` | fill the shell between two **closed `QuadMesh` surfaces**, paired **by index** (e.g. `sphere = R*normalize(cube.points)` on `cube.quads`) |
| {meth}`~nekmeshpy.hexmesh.HexMesh.merge` | stitch blocks, welding coincident **boundary** points only |
| {meth}`~nekmeshpy.hexmesh.HexMesh.from_grid` | structured `i×j×k` block; `face_tags` maps a side `x_min`…`z_max` to a name |

`HexMesh` is **immutable by construction**. `extrude` / `loft` are shared-point
(conformal slices → index arithmetic, no weld); `merge` is the one place seam
points are coordinate-welded.

### The explicit-initial layer convention

Layer counts are set by a **normalized-position array**, not a count + grading
pair — one convention shared by every layered factory (`extrude`'s `layers`; the
`radial` of `ogrid` / `half_ogrid` / `annulus`). Values strictly increase in
`[0, 1]` with the initial position explicit: first value is the near cap / inner
ring (`0` for a full span flush with the body), last is `1`, so `array.size - 1`
layers span `array[0]..1`. Use `uniform_spacing(k)`, `geometric_spacing(k, ratio)`
(`ratio > 1` clusters toward the wall), or `numpy.linspace(a, 1, k + 1)` to start
at `a`. Both helpers live in {mod}`nekmeshpy.model.fields`.

## Per-section smoothing

Interior nodes are repositioned on a single `QuadMesh` *before* extrusion, via
{func}`nekmeshpy.quadmesh.smoothing.set_section_smoothing` (registry
`SECTION_METHODS`; extend with `@register_section_smoothing("name")`). Built-ins:

- `bilinear` / `none` — algebraic radially-graded blend (default; near no-op).
- `conduction` — harmonic (Laplace) relaxation onto a curved boundary ring.
- `winslow` — elliptic (Winslow) smoothing.

Each factory takes an optional `smoothing_method=`. There is **no** HexMesh-level
registry; the constrained volume untangle/polish is the separate
{func}`nekmeshpy.hexmesh.smoothing.smooth`.

## High-order (order-N) elements

Every factory takes an optional `order=N` (default `1`). At `order > 1` each
element carries `(N+1)` nodes per parametric direction — line `N+1`, quad
`(N+1)²`, hex `(N+1)³` — sampled at **Gauss–Lobatto–Legendre (GLL)** parameters
(the grid Nek5000's solver evaluates on; endpoints land exactly on the corners).
Each factory places its extra nodes on the **true** geometry it owns, so a
`circle`'s arc nodes lie on the exact circle and a `sphere`'s on the exact sphere
— curvature is captured by the nodes, not approximated by straight chords.

### Entity-based conformal storage

Order-N is stored **conformally, without disturbing the linear topology** — the
high-order layer mirrors the corner layer, where a shared corner is one node in
`points` referenced by every incident element:

- corner connectivity (`lines` / `quads` / `hexes`) stays authoritative — it is
  what `.re2`, quality, topology, and `merge` read;
- the non-corner nodes are decomposed by **topology** into shared entities plus
  per-element private interiors (module `nekmeshpy.model.conform`):
  - **edges** — unique undirected edges (canonical: min-corner-id first) with their
    `N−1` shared interior nodes; a per-element incidence records the edge and a *flip*
    bit when an element traverses it anti-canonically (quad 4 / hex 12 edges per cell);
  - **faces** (hex only) — unique faces with their `(N−1)²` shared interior nodes, plus
    a per-hex incidence and a **D4 orientation code** (one of the 8 square symmetries)
    mapping the hex's local face grid to the shared canonical frame;
  - **interior** — the per-element private nodes (line `N−1`, quad `(N−1)²`, hex
    `(N−1)³`) that are never shared.

Sharing is decided by **corner ids** (structural / exact conformality), not by a
coordinate search: two elements meeting on an edge or face resolve to the *same*
high-order nodes. Element-local copies of a shared entity that disagree beyond
tolerance are rejected when they are reconciled
(`conform.scatter_edge_nodes` / `scatter_face_nodes`): a non-conforming input is a loud
error, not a silent weld.

That decomposition **is** the storage — there is no per-element node block anywhere.
The containers hold the entities natively: `LineMesh.interior`;
`QuadMesh.lines` (the shared-edge `LineMesh`, whose `interior` holds the edge nodes) /
`quad` / `flip` / `interior`; `HexMesh.quads` (the shared-face `QuadMesh`) / `hex` /
`face_orient` / `interior`, with the convenience views `.edges` / `.edge_nodes`
(quad, hex) and `.faces` / `.face_nodes` (hex). Corners are single-sourced by
`points[conn]` and never duplicated, so the corner-consistency invariant is
*structural* and an in-place `mesh.points[:] = X` edit is picked up everywhere for
free. At `order == 1` every entity table is empty, so the order-1 path — `.re2`,
quality, topology, `merge`, and the order-1 VTK writers — reads only `points` / corner
connectivity and order-1 meshes stay byte-for-byte unchanged (this is what keeps the
golden regression pinned).

The conformal walks `conform.conformal_line` / `conformal_quad` / `conformal_hex`
expose the conformal model directly as `(nodes (M,3), conn_ho (E,(N+1)^d))` — every
node numbered once in one global array with dense per-element connectivity into it, the
high-order analog of `points` + `quads`. This is the single node numbering the `.vtu`
writer and the order-N quality metrics read; `nodes[conn_ho]` is the transient
per-element block when one is needed.

Both `element_tags` and `boundary_tags` propagate exactly as in the linear case;
the extra nodes are geometry only and carry no tags of their own.

### What sees the extra nodes

- **`.re2` export stays linear** — Nek's re2 format has no high-order support yet,
  so the exporter reads only the 8 corners per hex; a mesh exports byte-identically
  at any order.
- **VTK export becomes high-order** — the `.vtu` writer emits VTK Lagrange cells
  (`VTK_LAGRANGE_CURVE` / `_QUADRILATERAL` / `_HEXAHEDRON`) with `(N+1)^d` nodes per
  cell, so a viewer renders the true curved geometry. Use the XML `.vtu` writer
  ({func}`nekmeshpy.io.export.to_vtu` / `line_to_vtu` / `quad_to_vtu`) — ParaView and
  VisIt render Lagrange cells reliably from `.vtu`. See `examples/high_order_*.py`.

Order-N quality metrics and smoothing are deferred (opt-in): the defaults stay
corner-based.

## Physical groups & export

Boundaries are plain **names** during construction. Each name maps to a Nek BC
code / integer id only at **export**, via the `groups=` argument:

- a `{name: nek_code}` dict,
- a {class}`~nekmeshpy.model.physical.PhysicalGroups` registry (presets:
  `PhysicalGroups.nek_default()`, `.duct()`, `.from_tags()`), or
- `None` to auto-number the mesh's distinct names.

```python
from nekmeshpy import export
export.to_re2(mesh, "part", groups={"wall": "W  ", "inlet": "v  "})
```

`.re2` element ids are written **1-based**; all internal indices are 0-based.

## See also

- {doc}`howto` — these concepts applied to concrete geometries.
- {doc}`../developer/architecture` — why the toolkit / examples split exists.
- {doc}`../reference/index` — the full API.
