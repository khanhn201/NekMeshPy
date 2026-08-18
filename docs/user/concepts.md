# Concepts

The ideas the toolkit is built on: the dimensional ladder of mesh containers,
the two tag systems, the factories, per-section smoothing, and physical groups.
Companion to the {doc}`getting-started` tutorial and {doc}`../reference/index`.

## The line → quad → hex ladder

One container per dimension, 2 / 4 / 8 vertices per element:

| container | element | role |
|---|---|---|
| {class}`~nekmeshpy.linemesh.LineMesh` | line (2 pts) | 1-D boundary — a ring, an edge, a spine |
| {class}`~nekmeshpy.quadmesh.QuadMesh` | quad (4 pts) | 2-D cross-section / surface |
| {class}`~nekmeshpy.hexmesh.HexMesh` | hex (8 pts)  | 3-D all-hex volume |

Coordinates are a bare `(P,3)` array on `.points` (mutate in place with
`mesh.points[:] = X`). No `Point` class — a single point is a `(3,)` array.
Boundaries live in 3-D: a `(N,2)` array is rejected, never padded to `z=0`.

`LineMesh` holds a shared point array plus `(L,2)` `lines` connectivity that
**can branch** (a mesh, not a single path). Open vs closed is read off the
connectivity (a loop has no degree-1 end point) — never stored as a flag.
`lines` is a **required** constructor arg; nothing implies a wrap. Factories
that build one explicitly:

- {func}`linemesh.loft <nekmeshpy.linemesh.assemble.loft>` — the only
  connectivity-authoring entry point. `loop=False` chains consecutive points;
  `loop=True` adds the closing rung. Anything else goes through the
  constructor with `lines` spelled out.
- `linemesh.line` — straight edge sampled at given fractions.
- `linemesh.loft_fn(f, fractions, loop=False, order=1)` — curve meshed on its
  own parametrization; `f` is called once on the **whole** node lattice
  (corners + interior), so nothing lands on a chord. See
  [true geometry vs straight subdivision](#true-geometry-vs-straight-subdivision).
- `linemesh.loft_spline(points, ...)` — like `loft`, but interior nodes come
  from a cubic spline through the whole point chain instead of each element's
  own chord. Use when the points are all you have; use `loft_fn` when you can
  write the curve down.
- `linemesh.circle` / `linemesh.rectangle` — closed rings; `rectangle`'s
  `side_tags` mapping is keyed `bottom`/`right`/`top`/`left`.
- `linemesh.merge` — welds coincident **degree-1 end points**. If no end
  survives, the result is a loop — the way to close a seam from two half-arcs.

Every factory meshes points **exactly** — no resampling API. `quadmesh.ogrid`,
`half_ogrid`, `quadrant_ogrid`, `structured` likewise take exactly the point
count their geometry needs (`spine_fractions`/`quadrant_seam_fractions` derive
the sampling). `quadrant_ogrid`'s one non-obvious term: its core's shared
corner `M` sits at `center_scale * cos(45°) * R` (half a diagonal past the far
corner `K`, which sits at `center_scale * R`).

A quadrant face read as a triangle (core + two ring-band halves) is what
{meth}`hexmesh.tetra <nekmeshpy.hexmesh.shape.tetra>` consumes — three
quadrant faces plus a fourth fill the octant behind their common centre
(`examples/quadrant_pipe_tjunction.py`).

Since closedness isn't stored, factories that read only `boundary.points`
(`ogrid`, `half_ogrid`, `structured`) will silently treat an open chain as a
closed ring — a known gap.

{class}`~nekmeshpy.trimesh.TriMesh` is the input surface for the vessel
pipeline; its ops (cotan Laplacian, Dirichlet solve, boundary loops) live in
{mod}`nekmeshpy.trimesh.ops`.

## The two tag systems

Both propagate up the ladder on `extrude`/`loft`, and both no-op when untagged.

### `element_tags` — sparse region/material tag

`ids` + `tags`; an untagged mesh stores nothing, no `""` sentinel. Set on the
`LineMesh` at construction, copied up by section factories. `len(element_tags)`
is the **tagged** count, not element count — use `n_lines`/`n_quads`/`n_hexes`.

### `point_tags` / `edge_tags` / `face_tags` — the rung below's element tags

Not separate tables — **a rung's side tags *are* the rung below's
`element_tags`**, read through by the entity they name:

- `LineMesh.point_tags` → `point_mesh.element_tags` (point ids)
- `QuadMesh.edge_tags` → `line_mesh.element_tags` (edge ids)
- `HexMesh.face_tags` → `quad_mesh.element_tags` (face ids)

Each entity is stored once, so it can't carry two names — tag consistency is
structural, like conformality. `merge` raises on a naming conflict; a
reflection needs no side remap (re-winding is a view, not a tag concern).
Cost: a genuinely two-sided condition can't live on the entity — see
*asymmetric conditions* below.

**Not "the boundary".** *Boundary* means the topological domain boundary
(`boundary_faces`/`_edges`/`_points`, derived from connectivity). A tag table
is a *named subset* — an extruded pipe can have 192 boundary faces and 0
named ones. `hexmesh.tag_report()` counts both.

Use {func}`hexmesh.boundary_mesh <nekmeshpy.hexmesh.lower.boundary_mesh>` (or
its `quadmesh` sibling) to get either **as a mesh**: with a `tag`, that named
group; without one, the whole boundary. It carries the parent's own nodes
bit-for-bit, so a weld back onto it is exact even at `order > 1`.

### Asymmetric conditions — the two sides of one face

An interior face's two sides can differ only by reading the **region**
(`element_tags`) of the element that owns each row:

```python
GROUPS = {"wall": "W  ", "interface": {"fluid": "W  ", "solid": None}}
```

A plain string applies from every side. A mapping is keyed by region name
(`None` writes no row from that side — how a conjugate interface keeps the
fluid condition and puts nothing on the solid). An unnamed region raises.
`examples/chimera.py` is the worked case.

### Tag at the lowest level; upper overrides lower

Section factories read per-line tags from their boundary `LineMesh` and let
factory args (`wall_tag`, `inner_tag`, `outer_tag`, `side_tags[side]`)
override: non-empty replaces, empty/absent falls through. `side_tags` is a
mapping keyed `bottom`/`right`/`top`/`left` (not a positional list), used by
`structured`, `rectangle`, and both `from_grid`s. Sweep end caps
(`first_tag`/`last_tag`) exist only at the hex level.

## Section factories (`QuadMesh` classmethods)

All build **natively in 3-D** — nothing projected to a plane.

| factory | fills |
|---|---|
| {func}`quadmesh.structured <nekmeshpy.quadmesh.shape.structured>` | transfinite grid over 4 edges (mapping keyed by side name preferred over positional `[bottom, right, top, left]`) |
| {func}`quadmesh.ogrid <nekmeshpy.quadmesh.shape.ogrid>` | O-grid inside a closed loop |
| {func}`quadmesh.half_ogrid <nekmeshpy.quadmesh.shape.half_ogrid>` | half-disc O-grid split along a spine |
| {func}`quadmesh.quadrant_ogrid <nekmeshpy.quadmesh.shape.quadrant_ogrid>` | quarter-disk O-grid; four `merge` back into a conforming disk |
| {func}`quadmesh.annulus <nekmeshpy.quadmesh.lift.annulus>` | ring O-grid between two closed loops, paired **by index** |
| {func}`quadmesh.extrude <nekmeshpy.quadmesh.lift.extrude>` / {func}`loft <nekmeshpy.quadmesh.assemble.loft>` | sweep/stack a `LineMesh` into a quad strip |
| {func}`quadmesh.sweep <nekmeshpy.quadmesh.lift.sweep>` | one profile along a curved path by a moving frame |
| {func}`quadmesh.loft_fn <nekmeshpy.quadmesh.assemble.loft_fn>` | sweep with profiles **evaluated** (`f(t) -> LineMesh`) at every node level — exact at `order > 1` |
| {func}`quadmesh.loft_spline <nekmeshpy.quadmesh.assemble.loft_spline>` | sweep with intermediate profiles **fitted** by a cubic spline |
| {func}`quadmesh.from_grid <nekmeshpy.quadmesh.lift.from_grid>` | structured `(ni+1,nj+1)` grid, sweep-major `i`-fastest numbering |

`ogrid`/`annulus` start from a straight-chord guess of the interior; a section's
interior points can be repositioned onto a curved boundary afterward with
{func}`quadmesh.smoothing.set_section_smoothing <nekmeshpy.quadmesh.smoothing.set_section_smoothing>`
(`"conduction"`/`"winslow"`), which no factory calls automatically.
`structured`/`half_ogrid`/`quadrant_ogrid` blend edge points directly.

## Hex-block factories (`HexMesh` classmethods)

| factory | builds |
|---|---|
| {func}`hexmesh.extrude <nekmeshpy.hexmesh.lift.extrude>` | sweep a section along a straight axis |
| {func}`hexmesh.sweep <nekmeshpy.hexmesh.lift.sweep>` | one section along a curved path by a moving frame |
| {func}`hexmesh.loft <nekmeshpy.hexmesh.assemble.loft>` | recombine a stack of pre-positioned conformal profiles |
| {func}`hexmesh.loft_fn <nekmeshpy.hexmesh.assemble.loft_fn>` | stack **evaluated** (`f(t) -> QuadMesh`) per node level — exact at `order > 1` |
| {func}`hexmesh.loft_spline <nekmeshpy.hexmesh.assemble.loft_spline>` | stack **fitted** by a cubic spline |
| {func}`hexmesh.annulus <nekmeshpy.hexmesh.lift.annulus>` | shell between two closed `QuadMesh` surfaces, paired by index |
| {func}`hexmesh.merge <nekmeshpy.hexmesh.assemble.merge>` | stitch blocks, welding coincident boundary points |
| {func}`hexmesh.from_grid <nekmeshpy.hexmesh.lift.from_grid>` | structured `i×j×k` block, `i` fastest / `k` slowest |

`HexMesh` is immutable by construction. `extrude`/`loft` are shared-point
(index arithmetic, no weld); `merge` is the one place seams get coordinate-welded.

## Placing a finished mesh: `translate` / `rotate` / `scale`

All three containers carry the same affine placements, returning a **new** mesh —
free functions on the rung's namespace module (`linemesh.translate(mesh, ...)`,
`quadmesh.rotate(mesh, ...)`, `hexmesh.scale(mesh, ...)`, and so on):

| function | does |
|---|---|
| `translate(mesh, vector)` | rigid shift, bit-exact (offset added, no matrix) |
| `rotate(mesh, angle, axis=(0,0,1), center=(0,0,0))` | radians, right-handed |
| `scale(mesh, factor, center=(0,0,0))` | scalar or per-axis `(3,)`, factors must be positive |
| `transform(mesh, matrix, offset=(0,0,0))` | the general case: `p @ matrix.T + offset` |

Only coordinates move — connectivity and tags ride through verbatim. The map
reaches every node including private high-order `interior`.

```python
ring = linemesh.circle(1.0, 16, center=(3.0, 0.0, 0.0), order=3)
profiles = [linemesh.rotate(ring, 2 * np.pi * k / 12, axis=(0, 1, 0)) for k in range(12)]
torus = quadmesh.loft(profiles, loop=True)          # periodic sweep of placed rings
```

`LineMesh` also has {func}`linemesh.reverse <nekmeshpy.linemesh.morph.reverse>`
— same curve, opposite traversal (relabels, doesn't move). Prefer it over
`linemesh.loft(curve.points[::-1])`, which straight-subdivides the interior and
loses curvature above order 1.

## `loft`: the uniform sweep primitive

(loft-the-uniform-sweep-primitive)=

One primitive at three dimensions — append each profile, then the rung
entities joining it to the previous one — each taking `loop: bool = False`:

| rung | a "profile" is | the "rungs" are |
|---|---|---|
| {func}`linemesh.loft <nekmeshpy.linemesh.assemble.loft>` | a point | line elements |
| {func}`quadmesh.loft <nekmeshpy.quadmesh.assemble.loft>` | a `LineMesh` | lines + quads |
| {func}`hexmesh.loft <nekmeshpy.hexmesh.assemble.loft>` | a `QuadMesh` | faces + hexes |

`extrude` is the straight special case at each rung. `loop=True` makes the
sweep periodic: the last profile joins back to the first (`M` profiles → `M`
layers, not `M-1`), closing watertight. A closed sweep emits **no cap tag rows
by default** (no free side to inherit onto) but **does** place `first_tag`/
`last_tag` when given — the two caps are one seam entity seen from either side.
`quadmesh.annulus` closes in the *ring* direction instead, so it never uses
`loop=True`.

### The evaluated sweep: `loft_fn`

`loft` only sees the profiles handed to it — at `order > 1` those are corner
levels, and the sweep direction between them is subdivided **straight** (a
torus lofted from exact `circle` rings can land its interior nodes tens of
percent of the tube radius off the true surface). `loft_fn` evaluates the
profiles from a parametrization instead, at every node level:

| rung | `f` maps | evaluated |
|---|---|---|
| `linemesh.loft_fn` | `(K,) -> (K,3)` points | whole lattice, one call |
| `quadmesh.loft_fn` | param → `LineMesh` profile | once per node level |
| `hexmesh.loft_fn` | param → `QuadMesh` section | once per node level |

`fractions` are parameter values in `f`'s own units (no normalization), with
the same trailing-wrap `loop` convention (`n+1` values, last maps to first).
Quad/hex profiles must be **index-paired and conformal** — build one and
*place* it with affine ops rather than rebuilding per parameter (a rotating
`normal=` circle isn't guaranteed index-paired).

```python
ring = linemesh.circle(0.6, 8, center=(2.0, 0, 0), normal=(0, 1, 0), order=3)
torus = quadmesh.loft_fn(lambda t: linemesh.rotate(ring, t, axis=(0, 0, 1)),
                          np.linspace(0.0, 2.0 * np.pi, 7), loop=True)
```

`order` defaults to `None` at quad/hex rungs (read off `f`'s output); only
`linemesh.loft_fn` keeps an `order: int`, since `f` there returns coordinates,
not a mesh. `quadmesh.loft`/`hexmesh.loft` also take intermediate profiles
directly via `sweep_nodes=` — `loft_fn` is that argument, evaluated for you.

### The fitted sweep: `loft_spline`

For when there's no parametrization — profiles came off a scan or another
mesh. Fits a **cubic spline through the whole stack** on the sweep lattice
`loft` would otherwise subdivide straight:

| rung | fitted through |
|---|---|
| `linemesh.loft_spline` | the points given |
| `quadmesh.loft_spline` | every node block a profile stores |
| `hexmesh.loft_spline` | corners, edge interiors, face interiors |

Interpolates — every slice handed in comes back verbatim. At order 1 it's
`loft`, node for node. `loop=True` closes periodically. Quad/hex rungs still
need index-paired slices. Measured: a torus lofted from 8 exact rings at
order 3 sits 0.157 off the true surface at its worst node with plain `loft`;
`loft_spline` sits 0.0019 off, ~83× closer — it's a fit, not exact, so prefer
`loft_fn`/`sweep` when a closed form exists.

### The rigid sweep: `sweep`

For the same section carried along a path (an elbow, a U-turn, a coil):
{func}`quadmesh.sweep <nekmeshpy.quadmesh.lift.sweep>` /
{func}`hexmesh.sweep <nekmeshpy.hexmesh.lift.sweep>` place it for you — the
curved generalization of `extrude`, ending in the same `sweep_nodes` assembly.

```python
Rb = 1.0
path  = lambda t: np.column_stack([Rb * np.sin(t), Rb * (1 - np.cos(t)), 0 * t])
dpath = lambda t: np.column_stack([Rb * np.cos(t), Rb * np.sin(t), 0 * t])

disc = quadmesh.ogrid(linemesh.circle(0.1, 24, normal=(1, 0, 0), order=2), 6, 3)
bend = hexmesh.sweep(disc, path, np.linspace(0.0, 0.5 * np.pi, 11),
                     origin=(0, 0, 0), tangent=dpath,
                     orientation="fixed", up=(0, 0, 1))
```

No `order=` — the block's order is the section's own. The section is placed
**rigidly** (`p ↦ path(t) + R(t) @ p_local`), never offset point-by-point:
through a bend, the outboard wall travels `Rb + d` and inboard `Rb - d`, so
only a frame-carried rigid placement gets both right.

Worth knowing:
- `path` is vectorized `(K,) -> (K,3)`; the default frame generator integrates
  sequentially, so it can't be evaluated at one isolated parameter.
- Put a station exactly on every path junction with
  {func}`linemesh.sweep_fractions <nekmeshpy.linemesh.shape.sweep_fractions>`
  — an element straddling one is fitted across two geometries (a visible kink).
- For a turtle-walked path, use `sweep_path` instead: {func}`paths.embed
  <nekmeshpy.core.paths.embed>` lifts a 2-D `turtle_path` onto a plane, giving
  a `SpacePath` with its own tangent and junction table; `sweep_path` then
  takes a `target_length`/`layers`. The origin enters the centerline, never
  the tangent (translating a tangent tilts every frame).
- `orientation` picks the frame field: `"transport"` (default, rotation-
  minimizing, for non-planar paths), `"fixed"` with `up=` (zero-twist, planar
  paths only — fails if a tangent turns parallel to `up`), or `"frenet"`.
  Station 0 always lands the section exactly as authored.
- `tangent=` supplies the analytic derivative; without it, finite-difference
  tangents tilt end stations by ~3e-4 rad on a coarse quarter arc.
- `origin=` is **required**, no default — it's the section's reference point,
  which for an O-grid disc is *not* its centroid (a past default produced a
  quietly off-axis block).
- A bend tighter than the section is wide folds inboard elements inside out —
  rejected loudly by `loft`'s mixed-winding guard.

`examples/serpentine_pipe.py`: one O-grid disc swept along an 8-pass coil.

### The explicit-initial layer convention

Every layered factory (`extrude`'s `layers`, `radial` of `ogrid`/`half_ogrid`/
`annulus`) takes a **normalized-position array**: strictly increasing in
`[0,1]`, first value the near cap (`0` if flush), last `1`. Use
`uniform_spacing(k)`, `geometric_spacing(k, ratio)` (`ratio > 1` clusters
toward the wall), or `numpy.linspace(a, 1, k+1)` to start at `a`
({mod}`nekmeshpy.core.fields`). A plain **`int`** also works for the uniform
case (`radial=3` = `uniform_spacing(3)`) — counts cells, matching
`uniform_spacing`'s own convention. Both normalize through `validate_layers`.

## Per-section smoothing

Interior nodes on a single `QuadMesh` are repositioned *before* extrusion, via
{func}`nekmeshpy.quadmesh.smoothing.set_section_smoothing` (registry
`SECTION_METHODS`). Built-ins: `bilinear`/`none` (algebraic blend, default,
near no-op), `conduction` (harmonic relaxation onto a curved boundary),
`winslow` (elliptic). No HexMesh-level registry — volume untangle/polish is
{func}`nekmeshpy.hexmesh.smoothing.smooth`.

## High-order (order-N) elements

Order is set once at the **bottom** of the ladder and rides up. `order=N`
(default `1`) is a factory argument only where geometry is authored from
nothing (`LineMesh` shapes, `quadmesh.rectangle`/`box`/`sphere`/`half_box`/
`hemisphere`, both `from_grid`s). Everything consuming a mesh inherits its
order and rejects a mismatch across inputs.

At `order > 1` each element carries `(N+1)` nodes per parametric direction,
sampled at GLL parameters (endpoints exactly `0`/`1`, so corners stay exact
under every sweep).

### The B-rep ladder *is* the storage

No `.curved` attribute, no `to_conformal()` facade — each container stores the
rung below plus the nodes it privately owns:

| rung | stored state | derived views |
|---|---|---|
| `LineMesh` | `point_mesh` (shared points+tags) + `lines (L,2)` + `interior (L,N-1,3)` | `points`, `corners` |
| `QuadMesh` | `line_mesh` (shared edges) + `quads (Q,4)` edge incidence + `orient (Q,4)` + `interior (Q,(N-1)²,3)` | `points`, `corners (Q,4)` |
| `HexMesh` | `quad_mesh` (shared faces) + `hexes (E,6)` face incidence + `orient (E,6)` D4 codes + `interior (E,(N-1)³,3)` | `points`, `corners (E,8)` |

Three roles, three names: `<rung>_mesh` is the container one rung down; the
**plural** `lines`/`quads`/`hexes` is this rung's own stored incidence into
it; `corners` is the derived corner connectivity at every rung. `points` is
always `(N,3)` coordinates. At the line rung a point *is* its own corner, so
`lines` and `corners` are the same table under both names.

`points`/`corners` are derived, read-only — a `HexMesh`'s points *are* its
`quad_mesh`'s points, which *are* its `line_mesh`'s points: single-sourced, so
`mesh.points[:] = X` propagates for free.

`.edges`/`.edge_nodes` (quad, hex) and `.faces`/`.face_nodes` (hex) surface the
shared tables directly. At `order == 1`, `interior` is empty but edge/face
topology is still first-class.

### Conformality is structural

Sharing is decided by **corner ids**, never coordinate search. A shared edge
is one row of the edge `LineMesh` referenced by every incident quad; a shared
face is one quad of the shared-face `QuadMesh`. Being conformal is a property
of the data structure, not something checked after the fact.

- **edges** — unique undirected, canonical (min-corner-id first), plus
  per-element incidence and a flip bit.
- **faces** (hex only) — unique faces, per-hex incidence, D4 orientation code.
- **interior** — private per-element nodes, never shared.

`nekmeshpy.core.conform` owns this and imports no container. Combinators that
rebuild shared tables (`merge`, `hexmesh.loft`) reconcile via
`conform.scatter_edge_nodes`/`scatter_face_nodes`: owner wins, every other copy
verified within `conform.entity_tol` — mismatch is a loud `ValueError`.

`conform.conformal_line`/`conformal_quad`/`conformal_hex` flatten the ladder
into `(nodes (M,3), conn_ho (E,(N+1)^d))` — the numbering `.vtu` and order-N
quality metrics read.

(true-geometry-vs-straight-subdivision)=
### True geometry vs straight subdivision

**High order in storage does not imply curved in geometry.** Every factory
produces a valid order-N mesh, but only one that owns an analytic shape places
extra nodes anywhere but the chord between corners.

- **Placed on the true shape.** `linemesh.circle`/`arc` evaluate the exact arc
  at interior GLL angles. `loft_fn` generalizes this to any curve you can
  write down — one call on the whole node lattice, corners and interior
  alike. `quadmesh.sphere`/`hemisphere` project every node radially.
  Straight-sided shapes (`line`, `rectangle`, `box`, `half_box`) are also
  exact — a straight GLL blend *is* the true geometry of a straight side.
- **Straight GLL subdivision.** Anything built from a bare point array has
  nothing else to go on: `linemesh.loft`, `from_grid`. Sampling a curve into
  points and calling `loft` throws the curve away above order 1 — pass `arc`/
  `circle`, or `loft_fn`:

  ```python
  ellipse = lambda t: np.column_stack([R * np.sin(t), R * np.cos(t), R * np.sin(t)])
  fr = linemesh.arclength_fractions(ellipse, n=12, t_range=(0.0, np.pi))
  collar = linemesh.loft_fn(ellipse, fr, order=3)
  ```

  `loft_fn` meshes exactly at the fractions given — the sampling is the
  caller's to prove. `arclength_fractions` inverts a chord-length table into
  parameter values spacing nodes evenly; only *where* a node sits inherits
  that table's discretization error, never its position (still evaluated by
  the callable, to machine precision). A closed curve with no analytic form —
  e.g. a scanned loop — can be refit: `examples/carotid.py`'s `fourier_ring`
  keeps the low rFFT modes of a scanned ring (dropping STL noise) and feeds
  that series to `loft_fn`; `_arc_curve` does the same for an open arc with a
  sine series that vanishes at both endpoints, so a refit stays bit-exact at
  its ends. Otherwise resample with `trimesh.ops.resample_polyline` and
  accept the chord. Measured: a T-junction collar sampled as a 400-point
  polyline sat 2.6% of `R` off the true ellipse at order 2 with `loft`;
  `loft_fn` on the analytic form is exact at any order
  (`examples/circular_pipe_tjunction.py`).
- **Region fills carry the wall's curvature inward.** `structured` evaluates
  its exact transfinite map at the GLL-refined lattice, so a block bounded by
  an `arc` is bowed all the way through. `ogrid`/`half_ogrid`/`quadrant_ogrid`
  instead `blend` the perimeter against the wall loop, curved tangentially and
  straight radially; `half_ogrid`/`quadrant_ogrid` also stamp their seam(s)
  with the spine's own nodes rather than straight-subdividing between samples.
- **Carried through unchanged.** `extrude` rigidly translates a section's
  whole B-rep; `blend` lerps entity tables the same way it lerps corners;
  `loft` sweeps each column as a Coons patch curved along the profile but
  straight along the sweep — the gap `loft_fn`/`sweep_nodes=` closes when the
  sweep path itself is curved.

If no factory owns the shape you need, hand in an exactly-sampled curve at the
lowest rung — the toolkit never resamples.

### What sees the extra nodes

- **`.re2` stays linear** — no high-order support in the format; exports only
  the 8 corners per hex, byte-identical at any order.
- **`.vtu` becomes high-order** — VTK Lagrange cells (curve=68, quad=70,
  hex=72) indexing the conformal node array. Use {func}`nekmeshpy.io.writer.to_vtu`
  — ParaView/VisIt render Lagrange cells reliably from `.vtu`.
- **A Nek field file carries the curved geometry to the solver** —
  {func}`nekmeshpy.io.writer.to_fld` writes the full GLL block as the `X`
  field; see [Using a high-order mesh in Nek5000 / NekRS](#using-a-high-order-mesh-in-nek5000-nekrs)
  below.
- **Quality metrics always read the curved element.** `scaled_jacobian()` and
  `quality_summary()` sample the GLL nodes the mesh actually stores; there is no
  corner-only option, and at order 1 the two coincide anyway. A corner reading cannot
  see where the high-order nodes went — a node displaced clean outside its element
  scores exactly the same — so it is not something a caller should be able to ask for
  by accident.
  `quality_summary()` returns a {class}`~nekmeshpy.core.quality.QualitySummary`
  NamedTuple; its `n_poor` and `poor (<…)` line both derive from
  {data}`~nekmeshpy.core.quality.POOR_THRESHOLD` so they can't drift apart.
- **Smoothing isn't implemented above order 1** — relaxers work on the corner
  graph and would leave interior nodes behind, so `conduction`/`winslow` and
  `hexmesh.smoothing.smooth` raise `NotImplementedError` rather than degrade
  silently. No-op strategies (`bilinear`/`tfi`/`none`) stay legal at any order.

At `order == 1` every high-order code path is a strict no-op — what keeps the
golden regression pinned.

(using-a-high-order-mesh-in-nek5000-nekrs)=
### Using a high-order mesh in Nek5000 / NekRS

`.re2` alone only ever gives the solver a **linear** mesh — the curvature has to
arrive through a separate field file. Export both:

```python
writer.to_re2(mesh, "case.re2", groups=GROUPS)   # topology + BCs, always linear
writer.to_fld(mesh, "case.f00000")               # FLD file for high order nodes
```

`to_fld` writes only the `X` field — the mesh's own GLL node positions, at its
own order — nothing else. Point the solver's `.par` restart at it so the
solve reads the true curved geometry instead of `.re2`'s corners:

```
# Nek5000/NekRS .par file
[GENERAL]
startFrom = case.f00000
```

**Restarting a run with velocity while keeping the high-order geometry**: a
solver checkpoint (`c1.f00000`, say) carries velocity but not the mesh.
Combine the two files, pulling one field from each with the
`field` suffix — `v` for velocity, `x` for coordinates:

```
# Nek5000 .par file
[GENERAL]
startFrom = "c1.f00000 v,case.f00000 x"

# NekRS .par file
[GENERAL]
startFrom = "c1.f00000+v,case.f00000+x"
```

## Physical groups & export

Boundaries are plain names during construction; each maps to a Nek BC code
only at **export**, via `groups=`: a `{name: spec}` dict (a code, or a
`{region: code}` mapping — see *asymmetric conditions*), or a
{class}`~nekmeshpy.core.physical.PhysicalGroups` registry.

```python
from nekmeshpy import writer
writer.to_re2(mesh, "part.re2", groups={"wall": "W  ", "inlet": "v  "})
```

**No presets, no default for `to_re2`** — a name-to-code table is a statement
about specific geometry, so `to_re2` raises without one rather than guessing.
Viewer writers (`to_vtu`/`to_mesh`) accept `None` and auto-number instead.

Every writer takes the full output filename, extension included. `.re2`
element ids are written 1-based; all internal indices are 0-based.

## See also

- {doc}`howto` — these concepts applied to concrete geometries.
- {doc}`architecture` — why the toolkit / examples split exists.
- {doc}`../reference/index` — the full API.
