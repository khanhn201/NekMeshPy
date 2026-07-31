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
property of the `lines` array itself — a loop is a cycle of line elements with no
degree-1 end point — and is **stored nowhere**: read it off the connectivity (or
`boundary_points()`, empty for a loop). The container never *invents* connectivity
either — `lines` is a **required** constructor argument, so there is no default chain
and nothing in `LineMesh` that could imply a wrap. Factories build the wrap
explicitly:

- {meth}`~nekmeshpy.linemesh.LineMesh.loft` — the bottom rung of the uniform
  [sweep primitive](#loft-the-uniform-sweep-primitive): each "profile" is a single
  point, so the rungs joining them *are* the line elements. `loop=False` gives the
  consecutive chain, `loop=True` appends the single closing rung `[N-1, 0]`.
  It is the **only** connectivity-authoring entry point: a chain is
  `loft(points)`, a ring `loft(points, loop=True)`, and anything else comes in
  through the constructor with its `lines` spelled out.
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
- `LineMesh.merge` — weld coincident **topological end points** (degree-1 chain
  ends; never interior points), the 1-D sibling of `QuadMesh.merge`/`HexMesh.merge`.
  The welded connectivity is the answer: if no degree-1 end survives the result
  *is* a loop, so two shared-endpoint `A1->A2` arcs (reverse one) weld at
  `A1`/`A2` into a single cycle — the clean way to close a seam ring from two
  half-arcs.

Every factory meshes its points **exactly** — there is no resampling API; the
caller hands in an exactly-sized, correctly-oriented curve. The ordered ops treat
points in index order as a path/loop (`.length`).

Because closedness is not a stored flag, the section factories constrain their input
through the facts they actually need instead: `ogrid` an exact `4*n_side` point ring,
`spined_ogrid` an `8*Ntheta` ring, `annulus` and `blend` identical `lines` on both
rings, `structured` four edges that share corners. The factories that read only
`boundary.points` (`ogrid`, `half_ogrid`, `structured`) therefore **accept an open
chain and silently treat it as the equivalent closed ring** — a known, deliberate gap:
they never see the connectivity that would tell them otherwise.

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
| {meth}`~nekmeshpy.quadmesh.QuadMesh.from_grid` | structured `(ni+1,nj+1)` quad grid (itself a {meth}`~nekmeshpy.quadmesh.QuadMesh.loft` of the grid's column profiles, each itself a {meth}`~nekmeshpy.linemesh.LineMesh.loft` of that column's `i` points, whose **sweep-major, `i`-fastest** numbering it carries up unchanged — `points == P.transpose(1,0,2).reshape(-1,3)`); `element_tag` fills the per-quad tags |

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
| {meth}`~nekmeshpy.hexmesh.HexMesh.from_grid` | structured `i×j×k` block (itself a {meth}`~nekmeshpy.hexmesh.HexMesh.loft` of the grid's `k`-sections, each a {meth}`~nekmeshpy.quadmesh.QuadMesh.from_grid`, whose numbering it carries up unchanged — `i` fastest, `k` slowest, `points == P.transpose(2,1,0,3).reshape(-1,3)`); `face_tags` maps a side `x_min`…`z_max` to a name |

`HexMesh` is **immutable by construction**. `extrude` / `loft` are shared-point
(conformal slices → index arithmetic, no weld); `merge` is the one place seam
points are coordinate-welded.

## Placing a finished mesh: `translate` / `rotate` / `scale`

All three containers carry the same four affine placements, as instance methods
returning a **new** mesh:

| method | does |
|---|---|
| `mesh.translate(vector)` | shift rigidly by a `(3,)` displacement — **bit-exact** (the offset is added without a matrix), so translating by `0` returns identical coordinates |
| `mesh.rotate(angle, axis=(0,0,1), center=(0,0,0))` | rotate by `angle` **radians** about the line through `center` along `axis` (right-handed; `axis` need not be normalized) |
| `mesh.scale(factor, center=(0,0,0))` | scale about `center` by a scalar or a `(3,)` per-axis vector (every factor must be positive) |
| `mesh.transform(matrix, offset=(0,0,0))` | the general case the other three wrap: `p @ matrix.T + offset` |

Only coordinates move: connectivity, `element_tags`, `boundaries` and `boundary_tags`
ride through verbatim, so a placed mesh keeps its numbering and its BC markers. The map
reaches **every** node, private high-order `interior` tables included — a rotated
{meth}`~nekmeshpy.linemesh.LineMesh.circle` is still an exact circle, and a rigid map
leaves `scaled_jacobian` untouched.

```python
ring = LineMesh.circle(1.0, 16, center=(3.0, 0.0, 0.0), order=3)
profiles = [ring.rotate(2 * np.pi * k / 12, axis=(0, 1, 0)) for k in range(12)]
torus = QuadMesh.loft(profiles, loop=True)          # a periodic sweep of placed rings
```

`LineMesh` additionally has {meth}`~nekmeshpy.linemesh.LineMesh.reverse` — the same
curve traversed the other way (`i → N-1-i`). It moves nothing; it relabels, carrying
the high-order `interior` with it. Reach for it instead of
`LineMesh.loft(curve.points[::-1])`, which silently straight-subdivides the interior and
loses the curve at `order > 1`. Reversing one of two shared-endpoint arcs before
{meth}`~nekmeshpy.linemesh.LineMesh.merge` is how a seam ring gets closed without the
traversal crossing itself.

Use them to place something already built — a revolved profile stack, a block about to
be {meth}`~nekmeshpy.hexmesh.HexMesh.merge`d onto its neighbour. A factory that can
*construct* in position (`circle(center=…, normal=…)`, `extrude(origin=…)`) still
should; `HexMesh.extrude` is itself a stack of `translate`d slices.

(loft-the-uniform-sweep-primitive)=
## `loft`: the uniform sweep primitive

`loft` is **one primitive at three dimensions** — the same "append each profile, then
the rung entities joining it to the previous profile" assembly at every rung of the
B-rep ladder, each taking a `loop: bool = False` flag:

| rung | a "profile" is | the "rungs" are |
|---|---|---|
| {meth}`~nekmeshpy.linemesh.LineMesh.loft` | a single **point** | the **line** elements |
| {meth}`~nekmeshpy.quadmesh.QuadMesh.loft` | a `LineMesh` | rung **lines** + the quads |
| {meth}`~nekmeshpy.hexmesh.HexMesh.loft` | a `QuadMesh` | rung **faces** + the hexes |

`extrude` is the straight special case at each rung; at the bottom rung
`LineMesh.loft` *is* the constructor of a chain (`loop=False`) or a ring
(`loop=True`).

**`loop=True` makes the sweep periodic**: the last profile joins back to the
**first**, so `M` profiles give `M` layers instead of `M-1`. It is one extra
iteration of the same assembly — exactly one more rung block, appended once, with
the first profile *not* duplicated — so the seam is a genuine shared entity and the
result closes watertight in the sweep direction (a torus surface from
`QuadMesh.loft` of revolved rings; a solid torus from `HexMesh.loft` of revolved
discs). A closed sweep has no near/far cap, so at every rung `loop=True`

- **raises `ValueError`** if given `first_tag` / `last_tag` (scalar or per-element
  array) rather than silently dropping it, and
- emits **no cap boundary rows** — side walls from the profiles' own boundary
  entities are unaffected.

`QuadMesh.annulus` closes in the *ring* direction, which lives in the loops' own
connectivity rather than the loft direction, so it does **not** use `loop=True`.

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
(the grid Nek5000's solver evaluates on; the endpoint parameters are exactly `0` and
`1`, so the two extreme nodes *are* the corners and stay exact under every sweep).

### The B-rep ladder *is* the storage

There is no per-element node block anywhere — no `.curved` attribute, no
`to_conformal()` facade to ask for one. Each container stores the rung below it plus
the nodes it privately owns, and the familiar corner arrays are read off that:

| rung | stored state | derived views |
|---|---|---|
| `LineMesh` | `points (P,3)`, `lines (L,2)` (**required**), `interior (L,N-1,3)` | — |
| `QuadMesh` | `lines` — a **`LineMesh` of the shared edges**, whose `interior` holds the edge nodes — plus `quad (Q,4)` edge incidence, `flip (Q,4)`, `interior (Q,(N-1)²,3)` | `points`, `quads (Q,4)` |
| `HexMesh` | `quads` — a **`QuadMesh` of the shared faces**, whose `interior` holds the face nodes and whose `lines.interior` holds the edge nodes — plus `hex (E,6)` face incidence, `face_orient (E,6)` D4 codes, `interior (E,(N-1)³,3)` | `points`, `hexes (E,8)` |

`points` / `quads` / `hexes` are **derived, read-only** views over that storage.
A `HexMesh`'s points *are* its shared-face `QuadMesh`'s points, which *are* its
shared-edge `LineMesh`'s points: one array, single-sourced. So corner consistency is
not an invariant anyone maintains — it is structural, and an in-place
`mesh.points[:] = X` propagates to every rung for free.

The convenience readers `.edges` / `.edge_nodes` (quad, hex) and `.faces` /
`.face_nodes` (hex) surface the shared tables without digging through the ladder.
At `order == 1` the `interior` tables are empty (`(·,0,3)`) but the edge/face
*topology* is still first-class storage.

### Conformality is structural

Sharing is decided by **corner ids**, never by a coordinate search. A shared edge is
literally one row of the edge `LineMesh` referenced by every incident quad, exactly as
a shared corner is one row of `points`; a shared face is one quad of the shared-face
`QuadMesh`. Two elements meeting on an edge or face therefore *resolve to the same
nodes* — being conformal is a property of the data structure, not something checked
afterwards.

- **edges** — unique undirected edges, canonical (min-corner-id first), plus a
  per-element incidence and a *flip* bit when an element traverses one
  anti-canonically (quad 4 edges, hex 12).
- **faces** (hex only) — unique faces plus a per-hex incidence and a **D4 orientation
  code** (one of the 8 square symmetries) mapping the hex's local face grid onto the
  shared canonical frame.
- **interior** — the private per-element nodes (line `N−1`, quad `(N−1)²`, hex
  `(N−1)³`) that are never shared.

`nekmeshpy.model.conform` owns this topology / orientation / reconciliation
engine and imports no container — everything crosses as plain arrays. Where a
combinator must rebuild the shared tables against a new topology (`merge`,
`HexMesh.loft`), it reconciles with `conform.scatter_edge_nodes` /
`scatter_face_nodes`: owner-wins, with every other incident copy **verified** within
`conform.entity_tol`. A non-conforming input is a loud `ValueError`, not a silent
weld.

The conformal walks `conform.conformal_line` / `conformal_quad` / `conformal_hex`
flatten the ladder on demand into `(nodes (M,3), conn_ho (E,(N+1)^d))` — every node
numbered once in one global array with dense per-element connectivity into it, the
high-order analog of `points` + `quads`. That is the single node numbering the `.vtu`
writer and the order-N quality metrics read; `nodes[conn_ho]` is the transient
per-element block whenever one is genuinely needed.

Both `element_tags` and `boundary_tags` propagate exactly as in the linear case;
the extra nodes are geometry only and carry no tags of their own.

(true-geometry-vs-straight-subdivision)=
### True geometry vs straight subdivision

**High order in storage does not imply curved in geometry.** This is the single thing
easiest to get wrong: every factory accepts `order=N` and produces a valid order-N
mesh, but only a factory that *owns an analytic shape* can put the extra nodes
anywhere other than on the straight line between the corners.

- **Placed on the true shape.** `LineMesh.circle` and `LineMesh.arc` evaluate the exact
  arc at the interior GLL angles, so each element's nodes lie on the circle rather than
  its chord. `QuadMesh.sphere` and `QuadMesh.hemisphere` project **every** node of the
  cubed sphere / half box — corners, shared edge nodes and private quad interiors alike
  — radially, so the whole surface is exact, not just its corners. Straight-sided
  analytic shapes (`LineMesh.line`, `LineMesh.rectangle`, `QuadMesh.box` /
  `QuadMesh.half_box`) are also exact, trivially: a straight GLL blend *is* the true
  geometry of a straight side, and a flat grid cell is exact under tensor subdivision.
- **Straight GLL subdivision.** Anything built from an explicit array of points has
  nothing but those points to go on, so the nodes between them are the straight
  blend: `LineMesh.loft`, `QuadMesh.from_grid` / `HexMesh.from_grid`. Sampling a
  curve into points and calling `LineMesh.loft` thus *throws the curve away* above
  order 1 — pass the analytic `arc` / `circle` instead.
- **Region fills carry the wall's curvature inward.** `structured` owns an exact
  transfinite map, so above order 1 it evaluates that map at the GLL-refined lattice
  against each edge's own nodes: every node it emits — corner, edge and interior alike —
  is the true transfinite point, so a block bounded by an `arc` is not merely circular on
  that side but bowed all the way through. `ogrid` and `half_ogrid` have no such global
  map, so each O-ring is instead a `blend` of the block perimeter and the wall loop (the
  same mechanism `annulus` uses) and inherits its share of the wall's curvature; the
  elements are curved tangentially and straight radially, which is what a radial blend
  should give. In every region fill a quad's private interior is the transfinite patch of
  that element's own four edge curves, so a curved side bows the nodes inside it rather
  than leaving a straight fill within a curved boundary.
- **Carried through unchanged.** The pure combinators move existing nodes rather than
  inventing them: `extrude` translates a section's whole B-rep rigidly, `blend` lerps
  the two profiles' entity tables the same way it lerps their corners, and `loft`
  sweeps each column as a Coons patch that is curved along the profile (from the
  profiles' own nodes) and straight along the sweep. So curvature you put in at the
  bottom of the ladder rides all the way up — `HexMesh.annulus` between an `order=N`
  `sphere` and an `order=N` `box` keeps its inner wall exactly spherical.

If you need a curved shape that no factory owns, hand in an exactly-sampled curve at
the lowest rung — the toolkit never resamples, so what you hand in is what gets
meshed.

### What sees the extra nodes

- **`.re2` export stays linear** — Nek's re2 format has no high-order support yet,
  so the exporter reads only the 8 corners per hex; a mesh exports byte-identically
  at any order.
- **VTK export becomes high-order** — at `order > 1` the `.vtu` writer emits VTK
  Lagrange cells (`VTK_LAGRANGE_CURVE` = 68 / `_QUADRILATERAL` = 70 / `_HEXAHEDRON`
  = 72) whose `(N+1)^d` nodes per cell index the conformal (welded) node array from
  the `conform.conformal_*` walk, so a viewer renders the true curved geometry. Use
  the XML `.vtu` writer ({func}`nekmeshpy.io.export.to_vtu` / `line_to_vtu` /
  `quad_to_vtu`) — ParaView and VisIt render Lagrange cells reliably from `.vtu`.
  See `examples/high_order_*.py`.
- **Quality metrics are opt-in** — the defaults stay corner-based so pinned numbers
  hold. Pass `high_order=True` to `mesh.scaled_jacobian()` /
  `mesh.quality_summary()` to sample the scaled Jacobian at the block's `(N+1)^d` GLL
  nodes instead. At order 1 the GLL nodes *are* the corners, so the two agree exactly.
- **Smoothing is not implemented at `order > 1`** — the relaxers work on the corner
  graph and would leave mid/interior nodes behind. Rather than degrade silently, a
  *repositioning* method raises `NotImplementedError` above order 1: `conduction` and
  `winslow` in {func}`nekmeshpy.quadmesh.smoothing.set_section_smoothing`, and
  {func}`nekmeshpy.hexmesh.smoothing.smooth`. The no-op strategies (`bilinear` /
  `tfi` / `none` / `""`) stay legal at any order because they move nothing. Section
  factories elevate to order N *first* and smooth after, so the smoother sees the true
  order and the error is raised rather than swallowed.

At `order == 1` every high-order code path is a strict no-op: `.re2`, quality,
topology, `merge` and the order-1 VTK writers read only `points` and corner
connectivity, so linear meshes are byte-for-byte unchanged. That is what keeps the
golden regression pinned.

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
- {doc}`architecture` — why the toolkit / examples split exists.
- {doc}`../reference/index` — the full API.
