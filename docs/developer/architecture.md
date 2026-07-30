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
example scripts (examples/):  bifurcation.py   circular_pipe.py / rectangular_pipe.py   flow_past_*.py
                                     │ compose the toolkit primitives below │
                                     ▼                                      ▼
TriMesh ──▶ QuadMesh cross-section slices ──HexMesh.extrude/loft/annulus/merge/from_grid──▶ HexMesh
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
| `linemesh/` | `LineMesh`: `(N,3)` points + branching `(L,2)` lines (**required**) + `(L,order-1,3)` `interior`; core `loft`/`open`/`loop`/`blend`/`merge`/`from_segments` constructors in `linemesh.py`, shape factories as **free functions** in `_closed.py` (`circle`/`rectangle`) and `_open.py` (`line`/`arc`) bound onto the class in the package `__init__` (setattr), each meshed exactly (no resampling; `_in_plane_axes` planar-frame helper in `_plane.py`) |
| `trimesh/` | `TriMesh` container + `trimesh/ops.py` surface ops (reached as `nekmeshpy.trimesh.ops`) |
| `quadmesh/` | `QuadMesh`: the shared-edge `LineMesh` + `quad`/`flip` incidence + `(Q,(order-1)**2,3)` `interior`, with `points`/`quads` derived views; core `from_corners`/`from_grid`/`merge`/`blend`/`extrude`/`loft` in `quadmesh.py`, region-fill factories as **free functions** in `_open.py` (`structured`/`rectangle`/`ogrid`/`half_ogrid`/`spined_ogrid`/`annulus`) and closed-surface factories in `_closed.py` (`box`/`sphere`/`half_box`/`hemisphere`), both bound onto the class in the package `__init__` (setattr), shared `_elevate`/`_apply_smoothing`/`_check_boundary` in `_helpers.py`; + `quadmesh/smoothing.py` (registry) + `quadmesh/quality.py` (per-quad scaled-Jacobian, corner + order-N) |
| `hexmesh/` | `HexMesh`: the shared-face `QuadMesh` + `hex`/`face_orient` incidence + `(E,(order-1)**3,3)` `interior`, with `points`/`hexes` derived views; immutable container + factories + `hexmesh/smoothing.py` (untangle/polish) + `hexmesh/quality.py` (per-hex scaled-Jacobian, corner + order-N) |

**`model/`** — mesh model, groups, metrics, sizing:

| module | responsibility |
|---|---|
| `model/mesh.py` | `Mesh`: generic shared-point model + meshio bridge |
| `model/physical.py` | `PhysicalGroup` / `PhysicalGroups`: name ↔ tag ↔ Nek BC code registry |
| `model/topology.py` | watertight / manifold / connectivity + hanging-point checks |
| `model/fields.py` | sizing `Field`s + 1-D graded distributions + `gll_nodes` / `lagrange_derivative_matrix` / `validate_layers` / `reject_loop_caps` |
| `model/interp.py` | order-N numeric kernel over GLL reference nodes: `tensor_nodes`, `corner_indices`, `subdivide_element`, `coons_grid`, `blend_ho`, the edge/face slot tables, `scaled_jacobian_ho` |
| `model/conform.py` | the topology / orientation / reconciliation engine behind the B-rep: `unique_edges`, `unique_faces` / `canonical_faces` / `hex_corners_from_faces`, `entity_tol`, the `scatter_*`/`gather_*` pair, the `conformal_*` walks, D4 helpers. Imports **no container** — everything crosses as plain arrays |

**`io/`** — export and visualization: `io/export.py`
(`to_re2` / `to_vtu` / `to_mesh` / `to_meshio` / `write`), `io/viz.py`
(`plot`), `io/templates/` (`.rea` header/footer templates).

**`_typing.py`** — shared numpy dtype aliases; see {doc}`conventions`.
