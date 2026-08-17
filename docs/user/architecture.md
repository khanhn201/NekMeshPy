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
example scripts (examples/):  carotid.py / femoral.py   circular_pipe.py / backward_facing_step.py   flow_past_*.py
                                     │ compose the toolkit primitives below │
                                     ▼                                      ▼
TriMesh ──▶ QuadMesh cross-section slices ──hexmesh.extrude/sweep/loft/annulus/merge/from_grid──▶ HexMesh
(surface;    (points + connectivity)                                          (points + hexes;
 trimesh.ops:                                                                 weld, topology)
 cotan Laplacian,                                                                 │
 Dirichlet solve,         ┌──────────────────────────────────────────────────────┤
 boundary loops)          ▼               ▼            ▼               ▼          ▼
             quadmesh.smoothing/  quadmesh|hexmesh   core.topology   io.export  io.viz
             hexmesh.smoothing    .quality           (watertight/    (re2/vtu/  (plot)
                   (relax)        (scaled Jac)       conformal)      mesh, meshio)
```

## Containers are pure data; operations are free functions

Everything acting on a finished mesh is a **free function** taking the container
as its first argument — `io.export`, `io.viz`, `core.topology`, and the per-type
modules beside their container (`hexmesh.quality`/`quadmesh.quality`,
`trimesh.ops`, `hexmesh.smoothing`/`quadmesh.smoothing`).

Don't add heavy methods to the containers; add a function in the right module. The
public API is re-exported from `nekmeshpy/__init__.py`, so `from nekmeshpy import
...` is stable regardless of file layout — keep `__all__` and the imports in sync.

## Package layout

**One top-level package per mesh type**, each holding its container plus the code
that acts on it. Operations live in sibling modules split by arity and rung delta
(see {doc}`concepts` for the full module-by-responsibility breakdown):

| package | container | own storage |
|---|---|---|
| `linemesh/` | `LineMesh` | `(N,3)` points + `(L,2)` lines + `(L,order-1,3)` interior |
| `quadmesh/` | `QuadMesh` | shared-edge `line_mesh` + `(Q,4)` `quads`/`orient` + `(Q,(order-1)²,3)` interior |
| `hexmesh/` | `HexMesh` | shared-face `quad_mesh` + `(E,6)` `hexes`/`orient` + `(E,(order-1)³,3)` interior |
| `trimesh/` | `TriMesh` | surface container; ops in `trimesh/ops.py` |

Each container also exposes derived `points`/`corners` views (see {doc}`concepts`).
Siblings: `assemble.py` (`loft`/`loft_fn`/`loft_spline`/`merge` and their inverse
`select`/`remove`/`components`), `lift.py` (rung-raising: `extrude`/`sweep`/
`annulus`/`from_grid`, plus hex-only `adapter`/`bridge`), `lower.py`
(`boundary_mesh`, Δ −1), `morph.py` (`blend`, `translate`/`rotate`/`scale`/
`transform`/`mirror`), `tag.py` (`retag_*` — vocabulary only, geometry untouched),
`query.py` (reads, plus rung-specific extras like hex `topology`/`report`/`weld`),
`shape.py` (factories: quad region fills/closed surfaces, line `circle`/`rectangle`/
`arc`), and per-package `smoothing.py`/`quality.py`.

**`core/`** — mesh model, groups, metrics, sizing. All modules below import **no
container** except `mesh.py` and `topology.py`:

| module | responsibility |
|---|---|
| `core/mesh.py` | `Mesh`: generic shared-point model + meshio bridge |
| `core/physical.py` | `PhysicalGroup`/`PhysicalGroups`: name ↔ tag ↔ Nek BC code registry |
| `core/topology.py` | watertight/manifold/connectivity + hanging-point checks; returns a `TopologyReport` |
| `core/quality.py` | shared `QualitySummary` NamedTuple + `POOR_THRESHOLD` used by both quality modules |
| `core/fields.py` | sizing `Field`s, graded 1-D distributions, `gll_nodes`/`lagrange_derivative_matrix`/`validate_layers` |
| `core/interp.py` | order-N kernel over GLL reference nodes: `tensor_nodes`, `corner_indices`, `subdivide_element`, `coons_grid`, `blend_ho`, `scaled_jacobian_ho` |
| `core/measure.py` | one GLL quadrature over a node block, behind every rung's `length`/`area`/`volume`/`centroid` — corner and curved readings are the same code on different blocks |
| `core/affine.py` | affine maps behind `translate`/`rotate`/`scale`/`mirror`; `reflection` has det −1, so its caller must pair it with a re-winding |
| `core/frames.py` | moving-frame machinery behind `sweep`: `tangents`, frame generators (`fixed_up`/`parallel_transport`/`frenet`/`plane_frame`), `sweep_placements` |
| `core/paths.py` | `turtle_path` (C1, analytic tangent + arc-length param) and `embed`, lifting one onto a `SpacePath` |
| `core/surfaces.py` | `SurfaceCurve` + `ruled`/`blend`/`reverse`/`shift`/`reparam` — curves carried as parametrization, since a point-space lerp between two curves on a cylinder dips inside it |
| `core/conform.py` | topology/orientation engine behind the B-rep: `unique_edges`/`unique_faces`/`canonical_faces`, `entity_tol`, `scatter_*`/`gather_*`, `conformal_*` walks, D4 helpers |

**`io/`** — `io/export.py` (`to_re2`/`to_vtu`/`to_mesh`/`to_meshio`/`write`),
`io/viz.py` (`plot`).

**`_typing.py`** — shared numpy dtype aliases; see {doc}`conventions`.
