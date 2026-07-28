# Concepts

This page explains the ideas the toolkit is built on: the dimensional ladder of
mesh containers, the two tag systems that name regions and boundaries, the
section and hex-block factories, per-section smoothing, and how physical groups
map to export codes. It is the conceptual companion to the {doc}`getting-started`
tutorial and the {doc}`../reference/index`.

## The line → quad → hex ladder

NekMeshPy models geometry with a ladder of mesh containers, one per dimension,
each with 2 / 4 / 8 vertices per element:

| container | element | role |
|---|---|---|
| {class}`~nekmeshpy.linemesh.LineMesh` | line (2 pts) | 1-D boundary — a ring, an edge, a spine |
| {class}`~nekmeshpy.quadmesh.QuadMesh` | quad (4 pts) | 2-D cross-section / surface |
| {class}`~nekmeshpy.hexmesh.HexMesh` | hex (8 pts)  | 3-D all-hex volume |

Each container stores its coordinates as a **bare `(P,3)` NumPy array** on
`.points` (mutate in place with `mesh.points[:] = X`). There is **no `Point`
class** — a single point is just a `(3,)` array. All boundaries live honestly in
3-D: a `(N,2)` array is *rejected*, never padded to `z=0`.

`LineMesh` is the 1-D sibling of the surface/volume containers: a shared `(N,3)`
point array plus `(L,2)` `lines` connectivity that **can branch** (it is a mesh,
not a single ordered path). Open vs closed is a **topological property**
(`is_open` / `is_closed`), not a subclass — the factories set it:

- `LineMesh.open` — a consecutive chain (default).
- `LineMesh.loop` — a chain that wraps back to the start.
- `LineMesh.circle(radius, n, center=…, normal=…)` — a closed ring placed in the
  plane with the given `normal` (default `+z`).
- `LineMesh.from_segments` — chain unordered segments into the largest closed
  loop (or `None`).

The ordered ops (`resample`, `resample_spline`, `align_to`, `radial_match`,
`split_by_fraction`, `.length`) treat the points in index order as a path/loop.

{class}`~nekmeshpy.trimesh.TriMesh` sits alongside as the **input surface** for
the vessel pipeline; its algorithms (cotan Laplacian, Dirichlet solve, boundary
loops) live in {mod}`nekmeshpy.trimesh.ops` (aliased `nekmeshpy.trisurf`).

## The two tag systems

Both tag systems **propagate up the ladder** (line → quad → hex) on `extrude` /
`loft`, and both are no-ops when untagged.

### `element_tags` — dense, per-element (region / material)

A dense per-element string array (`""` = untagged), one tag per line / quad / hex.
Carried by `resample` / `align_to` / `radial_match`, copied by the section
factories onto the section edges/quads and thence onto the hex faces/hexes.
`element_group_tags` is the sorted unique non-empty set.

### `boundary_tags` — sparse, parallel with `boundaries`

A sparse string array parallel with `boundaries` `(Nbc,2)`. At each level the
second column means one dimension's "side":

- `LineMesh`: `[elem id, side ∈ {1,2}]` → local end **point** `s-1`.
- `QuadMesh`: `[quad id, side ∈ {1..4}]` → local **edge** `EDGE_POINTS[s-1]`.
- `HexMesh`: `[elem id, face ∈ {1..6}]` → local **face**.

On `extrude`, a line's end-point tags become quad boundary **edges**, then hex
boundary **faces**. `boundary_group_tags` is the sorted unique set. See
`examples/flow_past_cylinder.py`.

### Tag at the lowest level; upper overrides lower

The guiding rule is **tag at the lowest level** — every section-wall tag can
originate on the `LineMesh` input (the circle / loop / arc / edge), which every
section factory reads:

- `ogrid` / `annulus` read the loop's per-line tags,
- `half_ogrid` reads the arc's per-segment tags,
- `structured` reads each edge's uniform tag.

The factories' scalar / mapping args (`wall_tag`, `inner_tag`, `outer_tag`,
`boundary_tags[side]`) are **overrides**: a non-empty arg replaces the line-level
tag for that wall/side (**upper overrides lower**); an empty/absent arg falls
through to it (and a present-but-empty `boundary_tags[side]` / `NO_BOUNDARY`
suppresses the side). The end caps of a sweep (`first_tag` / `last_tag`) are
named at the hex level because no lower level exists for them.

## Section factories (`QuadMesh` classmethods)

Sections fill a boundary with quads. All build **natively in 3-D** — nothing is
projected to a plane, so a boundary placed in any plane, or a genuinely curvy /
non-planar boundary, is filled in place with its true shape.

| factory | fills |
|---|---|
| {meth}`~nekmeshpy.quadmesh.QuadMesh.structured` | transfinite (Coons) grid over a surface bounded by 4 open `LineMesh` edges; resolution comes **from the edges' own points** (no resampling — opposite edges must match counts), so graded edges give a graded grid; each side is named from its edge's uniform tag |
| {meth}`~nekmeshpy.quadmesh.QuadMesh.ogrid` | butterfly O-grid inside a closed `LineMesh` loop (no collapsed centre); the outer ring is named from the loop's per-line tags |
| {meth}`~nekmeshpy.quadmesh.QuadMesh.half_ogrid` | half-disc O-grid split along a spine; the wall is named from the arc's per-segment tags |
| {meth}`~nekmeshpy.quadmesh.QuadMesh.annulus` | ring O-grid between an inner and outer closed loop (a body inside a far-field box), paired **by index** (equal point counts — align a coarse box loop first with `outer.radial_match(inner)`) |
| {meth}`~nekmeshpy.quadmesh.QuadMesh.extrude` / {meth}`~nekmeshpy.quadmesh.QuadMesh.loft` | sweep/stack a `LineMesh` **one dimension down** into a quad strip (mirrors the `HexMesh` versions) |
| {meth}`~nekmeshpy.quadmesh.QuadMesh.from_grid` | structured `(ni+1,nj+1)` quad grid from a point array; `element_tag` fills the dense per-quad tags |

`ogrid` / `annulus` build a straight-chord initial guess and rely on
`smoothing_method="conduction"` to relax the interior harmonically onto the curved
surface spanned by the fixed boundary ring; `structured` / `half_ogrid` blend the
3-D edge points directly. (`ogrid` / `half_ogrid` are ICEM/Pointwise terms kept
deliberately; everything else follows gmsh vocabulary.)

## Hex-block factories (`HexMesh` classmethods)

| factory | builds |
|---|---|
| {meth}`~nekmeshpy.hexmesh.HexMesh.extrude` | sweep one section along a straight axis (gmsh Extrude + Layers + Recombine) |
| {meth}`~nekmeshpy.hexmesh.HexMesh.loft` | recombine a stack of pre-positioned conformal profiles — the general case behind `extrude` |
| {meth}`~nekmeshpy.hexmesh.HexMesh.annulus` | fill the 3-D shell between two **closed `QuadMesh` surfaces**, paired **by index** (build one from the other's points, e.g. `sphere = R*normalize(cube.points)` on `cube.quads`) |
| {meth}`~nekmeshpy.hexmesh.HexMesh.merge` | stitch blocks, welding coincident **boundary** points only |
| {meth}`~nekmeshpy.hexmesh.HexMesh.from_grid` | structured `i×j×k` block; `face_tags` maps a side `x_min`…`z_max` to a boundary name |

`HexMesh` is **immutable by construction** (no incremental building). `extrude` /
`loft` are shared-point by construction (conformal slices → index arithmetic, no
weld); `merge` is the one place coincident seam points are coordinate-welded.

### The explicit-initial layer convention

Layer counts are set by a **normalized-position array**, not a count + grading
pair — one convention shared by every layered factory (`extrude`'s `layers`; the
`radial` of `ogrid` / `half_ogrid` / `annulus`). Values strictly increase in
`[0, 1]` with the initial position **explicit**: the first value is the near cap /
inner ring (`0` for a full span flush with the body) and the last is `1`, so
`array.size - 1` layers span `array[0]..1`. Use `uniform_spacing(k)` for uniform,
`geometric_spacing(k, ratio)` for graded (`ratio > 1` clusters toward the wall),
or `numpy.linspace(a, 1, k + 1)` to start at `a`. Both helpers live in
{mod}`nekmeshpy.model.fields`.

## Per-section smoothing

Cross-section interior nodes are repositioned on a single `QuadMesh` *before*
extrusion, via {func}`nekmeshpy.quadmesh.smoothing.set_section_smoothing` (registry
`SECTION_METHODS`; extend with `@register_section_smoothing("name")`). Built-ins:

- `bilinear` / `none` — algebraic, radially-graded blend (the default; near
  no-op).
- `conduction` — harmonic (Laplace) relaxation onto a curved boundary ring.
- `winslow` — elliptic (Winslow) smoothing.

Each factory takes an optional `smoothing_method=`. There is **no** HexMesh-level
smoothing registry; the constrained volume untangle/polish is the separate
{func}`nekmeshpy.hexmesh.smoothing.smooth`.

## Physical groups & export

Boundaries are identified by a plain **name** throughout construction. The mapping
of each name to a Nek BC code / integer id is supplied only at **export**, via the
`groups=` argument to the exporters:

- a plain `{name: nek_code}` dict,
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
