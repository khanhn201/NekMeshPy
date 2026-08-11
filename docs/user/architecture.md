# Architecture

NekMeshPy is split into **two layers**:

- `nekmeshpy/` — a **toolkit of composable primitives**: pure data containers plus
  free-function operations. No geometry-specific meshers.
- `examples/` — concrete meshers as **flat gmsh-style scripts** (constants at top,
  assign to a `mesh` global, export). No mesher classes by design — a mesher *is*
  its script. The suite runs them via `runpy.run_path` and inspects the `mesh`
  global, so examples double as integration tests.

## Data flow

```
example scripts (examples/):  carotid.py / femoral.py   circular_pipe.py / rectangular_pipe.py   flow_past_*.py
                                     │ compose the toolkit primitives below │
                                     ▼                                      ▼
TriMesh ──▶ QuadMesh cross-section slices ──hexmesh.extrude/sweep/loft/annulus/merge/from_grid──▶ HexMesh
(surface;    (points + connectivity)                                          (points + hexes;
 trimesh.ops:                                                                 weld, topology)
 cotan Laplacian,                                                                 │
 Dirichlet solve,         ┌──────────────────────────────────────────────────────┤
 boundary loops)          ▼               ▼            ▼               ▼          ▼
             quadmesh.smoothing/  quadmesh|hexmesh   model.topology  io.export  io.viz
             hexmesh.smoothing    .quality           (watertight/    (re2/vtu/  (plot)
                   (reposition)   (scaled Jac)       conformal)      mesh, meshio)
```

## Containers are pure data; operations are free functions

Everything that acts on a finished mesh is a **free function** taking the
container as its first argument — `io.export`, `io.viz`, `model.topology`, and the
per-type modules beside their container in each `<type>/` package:

- `hexmesh.quality` + `quadmesh.quality` — scaled-Jacobian metrics,
- `trimesh.ops` — surface ops (reached as `nekmeshpy.trimesh.ops`),
- `hexmesh.smoothing` / `quadmesh.smoothing` — smoothing.

Don't add heavy methods to the containers; add a function in the right module. The
public API is re-exported from `nekmeshpy/__init__.py`, so `from nekmeshpy import
...` is stable regardless of file layout — keep `__all__` and the imports in sync.

## Package layout

**One top-level package per mesh type** — each holds its container plus the code
that acts on it (smoothing / quality / surface ops):

| package | responsibility |
|---|---|
| `linemesh/` | `LineMesh`: `(N,3)` points + branching `(L,2)` lines (**required**) + `(L,order-1,3)` `interior`; pure container in `linemesh.py`; operations as **free functions** in per-rung namespaces (`linemesh.assemble` / `.shape` / `.morph` / `.query`), split by arity and rung delta — `assemble.py` (n-ary `loft`/`loft_fn`/`loft_spline`/`merge`, plus their inverse `select`/`remove`/`components`), `morph.py` (`blend` + the unary `translate`/`rotate`/`scale`/`transform`/`reverse`), `tag.py` (`retag_element`/`retag_point` — the vocabulary only, Δ 0 on the *tags* where `morph` is Δ 0 on the *geometry*), `query.py` (reads), plus shape factories in `shape.py` (`circle`/`rectangle`) and `shape.py` (`line`/`arc`; `shape` also carries `arclength_fractions` and `sweep_fractions`, which return a plain array rather than a mesh — respectively the `loft_fn` grading that spaces nodes evenly by arc length, and the sweep stations that land a node **exactly on every junction** of a piecewise path, each subinterval subdivided on its own at a target element length so a curvature jump is never straddled by an element. Both are explicit caller steps, since no factory resamples), each meshed exactly (no resampling; `_in_plane_axes` planar-frame helper in `_plane.py`). `loft` is the only connectivity-authoring entry point (`loop=False` chain / `loop=True` ring); `loft_fn` is `loft` with the points **evaluated** from a caller-supplied parametrization on the whole node lattice — corners *and* high-order interiors — at the `fractions` handed in, and takes the same `loop` flag |
| `trimesh/` | `TriMesh` container + `trimesh/ops.py` surface ops (reached as `nekmeshpy.trimesh.ops`) |
| `quadmesh/` | `QuadMesh`: the shared-edge `line_mesh` + `quad`/`orient` incidence + `(Q,(order-1)**2,3)` `interior`, with `points`/`quads` derived views; pure container + `from_corners` in `quadmesh.py`; operations as **free functions** in per-rung namespaces — `assemble.py` (n-ary `loft`/`loft_fn`/`loft_spline`/`merge` — `loft_fn` is `loft` with the profiles **evaluated** from a caller-supplied `f(t) -> LineMesh` at every node level of the sweep lattice, not just the corner levels, which is what makes a swept curved surface exact at `order > 1`; it hands them to `loft` through its `sweep_nodes` argument; `loft_spline` **fits** those same intermediates with a cubic spline through the whole stack, interpolating the profiles given, for when there is no parametrization to evaluate), `lift.py` (`extrude`/`sweep`/`sweep_path`/`annulus`/`from_grid` — `sweep` carries one profile along a curved path by a moving frame, the curved generalization of `extrude`, and `sweep_path` is the same driven by a `SpacePath` so it asks for an element length rather than a station array; `annulus` owns no shape model, so it sits here beside its `HexMesh` sibling rather than with the region fills), `morph.py` (`blend`, `reindex` — one section's geometry relabelled onto another's numbering, the exact pairing `blend` needs across independently built sections — and `place_on_path`, plus the unary placements `translate`/`rotate`/`scale`/`transform`), `lower.py` (`boundary_mesh`, Δ −1: the boundary as a `LineMesh`), `tag.py` (`retag_element`/`retag_edge`), `query.py` (reads, plus `plane_normal`), `ports.py` (`Port`: a section plus the outward direction and axis point it cannot state about itself, so the joins one rung up can check rather than guess), region fills in `shape.py` (`structured`/`rectangle`/`ogrid`/`half_ogrid`/`quadrant_ogrid`/`spined_ogrid`, exposed as `quadmesh.region`, which also carries `spine_fractions` and `quadrant_seam_fractions`, returning plain arrays rather than meshes — the samplings a `spined_ogrid` spine and a `quadrant_ogrid` seam must carry, since the factories mesh them exactly and never resample) and closed surfaces in `shape.py` (`box`/`sphere`/`half_box`/`hemisphere`, exposed as `quadmesh.surface`), plus `tri_patch` — the curved triangle between three surface curves, as the three-Coons split about a tip, shared `_elevate`/`_apply_smoothing`/`_check_boundary` in `_helpers.py`; + `quadmesh/smoothing.py` (registry) + `quadmesh/quality.py` (per-quad scaled-Jacobian, corner + order-N) |
| `hexmesh/` | `HexMesh`: the shared-face `quad_mesh` + `hex`/`orient` incidence + `(E,(order-1)**3,3)` `interior`, with `points`/`hexes` derived views; pure container + `from_corners` in `hexmesh.py`; operations as **free functions** in per-rung namespaces — `assemble.py` (n-ary `loft`/`loft_fn`/`loft_spline`/`merge` — `loft_fn` is `loft` with the sections **evaluated** from `f(t) -> QuadMesh` at every node level of the sweep lattice, handed to `loft` through its `sweep_nodes` argument; `loft_spline` **fits** them instead, by a cubic spline through the whole stack), `lift.py` (`extrude`/`sweep`/`sweep_path`/`annulus`/`from_grid`/`adapter`/`bridge` — `sweep` carries one section along a *curved* path by a moving frame from `core/frames.py`, placing it rigidly at each station rather than offsetting points, and delegates to the same `sweep_nodes` assembly; `adapter` and `bridge` join two sections whose node patterns differ, slightly and greatly respectively, with both end faces bit-exact), `lower.py` (`boundary_mesh`, Δ −1: a block's boundary as a `QuadMesh` carrying the block's own nodes), `morph.py` (`blend` + the unary placements `translate`/`rotate`/`scale`/`transform`), `tag.py` (`retag_element`/`retag_face` — renaming a face tag to `NO_TAG` drops its rows, which is how a name welded shut into an interior plane is retired), `query.py` (reads, topology, `report`, `tag_report`, `weld`); + `hexmesh/smoothing.py` (untangle/polish) + `hexmesh/quality.py` (per-hex scaled-Jacobian, corner + order-N) |

**`core/`** — mesh model, groups, metrics, sizing:

| module | responsibility |
|---|---|
| `core/mesh.py` | `Mesh`: generic shared-point model + meshio bridge |
| `core/physical.py` | `PhysicalGroup` / `PhysicalGroups`: name ↔ tag ↔ Nek BC code registry |
| `core/topology.py` | watertight / manifold / connectivity + hanging-point checks; the `TopologyReport` NamedTuple they return (the type discriminates a hex report from a surface one, so there is no `kind` field) |
| `core/quality.py` | the schema both `quality` modules share: the `QualitySummary` NamedTuple and `POOR_THRESHOLD`, which names both its `n_poor` field and the `poor (<…)` report line so the two cannot drift. Imports **no container** |
| `core/fields.py` | sizing `Field`s + 1-D graded distributions + `gll_nodes` / `lagrange_derivative_matrix` / `validate_layers` |
| `core/interp.py` | order-N numeric kernel over GLL reference nodes: `tensor_nodes`, `corner_indices`, `subdivide_element`, `coons_grid`, `blend_ho`, the edge/face slot tables, `scaled_jacobian_ho` |
| `core/measure.py` | the rung-agnostic metric kernel: `Bounds`, and one GLL quadrature over a `(E,(order+1)**dim,3)` node block serving `linemesh.length` / `quadmesh.area` / `hexmesh.volume` and every `centroid`. The node block is its only input, so the corner (linear) and curved order-N readings of a mesh are the *same* code on two different blocks — and the rule for taking the quadrature high enough to integrate an order-N Jacobian exactly lives here rather than at three call sites. Imports **no container** |
| `core/affine.py` | the affine maps behind the rung-preserving `translate` / `rotate` / `scale` / `mirror`: `translation` / `rotation` / `scaling` / `reflection` build a `(matrix, offset)` pair, `apply` maps any coordinate table whose trailing axis is the 3 components. `reflection` is the one with determinant −1, so it is the one its caller must pair with a re-winding. Imports **no container** |
| `core/frames.py` | the moving-frame machinery behind `quadmesh.sweep` / `hexmesh.sweep`: `tangents` (finite-difference; pass an analytic derivative when the end stations matter), the frame generators `fixed_up` / `parallel_transport` (RMF by double reflection) / `frenet`, `plane_frame` (a profile's own authored frame from an SVD best-fit plane), and `sweep_placements`, which composes them into one `(matrix, offset)` per station and pins the frame field's free roll so station 0 lands the section as authored. Imports **no container** |
| `core/paths.py` | the declarative 2-D turtle walk (`turtle_path`: a table of straights and arcs, C1 by construction, with an **analytic** tangent and the exact arc-length parametrization), and `embed`, which lifts one onto a plane in space into a `SpacePath`. The origin enters the centerline and *not* the tangent — a tangent is a direction, and translating it tilts every frame along a sweep. Imports **no container** |
| `core/surfaces.py` | curves carried as their **parametrization on a surface** rather than as points: `SurfaceCurve` plus `ruled` / `blend` / `reverse` / `shift` / `reparam` / `node` / `segment` / `spoke`. Interpolating two such curves has to happen in parameter space — a chord between two points of a cylinder dips inside it, so a point-space lerp leaves every intermediate station proud of the wall. Imports **no container** |
| `core/conform.py` | the topology / orientation / reconciliation engine behind the B-rep: `unique_edges`, `unique_faces` / `canonical_faces` / `hex_corners_from_faces`, `unique_rows` and `locate_rows` (the fast row dedup and id-set lookup both rest on), `entity_tol`, the `scatter_*`/`gather_*` pair, the `conformal_*` walks, D4 helpers. Imports **no container** — everything crosses as plain arrays |

**`io/`** — export and visualization: `io/export.py`
(`to_re2` / `to_vtu` / `to_mesh` / `to_meshio` / `write`), `io/viz.py`
(`plot`).

**`_typing.py`** — shared numpy dtype aliases, plus the `SmoothingMethod` literal
every section factory's `smoothing_method=` takes (it lived in `quadmesh/shape.py`,
which made the region fills the definition site of a name the whole toolkit spells);
see {doc}`conventions`.
