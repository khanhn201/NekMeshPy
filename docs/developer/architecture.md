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
| `linemesh/` | `LineMesh`: `(N,3)` points + branching `(L,2)` lines; core `open`/`loop`/`from_segments` constructors in `linemesh.py`, shape factories as **free functions** in `_closed.py` (`circle`/`rectangle`) and `_open.py` (`line`) bound onto the class in the package `__init__` (setattr), each meshed exactly (no resampling; `_in_plane_axes` planar-frame helper in `_plane.py`) |
| `trimesh/` | `TriMesh` container + `trimesh/ops.py` surface ops (reached as `nekmeshpy.trimesh.ops`) |
| `quadmesh/` | `QuadMesh` container + core `from_grid`/`merge`/`extrude`/`loft` in `quadmesh.py`, region-fill factories as **free functions** in `_open.py` (`structured`/`rectangle`/`ogrid`/`half_ogrid`/`annulus`) and closed-surface factories in `_closed.py` (`box`/`sphere`), both bound onto the class in the package `__init__` (setattr), shared validation in `_helpers.py`; + `quadmesh/smoothing.py` (registry) + `quadmesh/quality.py` (per-quad scaled-Jacobian) |
| `hexmesh/` | `HexMesh` immutable container + factories + `hexmesh/smoothing.py` (untangle/polish) + `hexmesh/quality.py` (per-hex scaled-Jacobian) |

**`model/`** — mesh model, groups, metrics, sizing:

| module | responsibility |
|---|---|
| `model/mesh.py` | `Mesh`: generic shared-point model + meshio bridge |
| `model/physical.py` | `PhysicalGroup` / `PhysicalGroups`: name ↔ tag ↔ Nek BC code registry |
| `model/topology.py` | watertight / manifold / connectivity + hanging-point checks |
| `model/fields.py` | sizing `Field`s + 1-D graded distributions |

**`io/`** — export and visualization: `io/export.py`
(`to_re2` / `to_vtu` / `to_mesh` / `to_meshio` / `write`), `io/viz.py`
(`plot`), `io/templates/` (`.rea` header/footer templates).

**`_typing.py`** — shared numpy dtype aliases; see {doc}`conventions`.
