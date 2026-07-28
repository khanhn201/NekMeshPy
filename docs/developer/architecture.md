# Architecture

NekMeshPy is deliberately split into **two layers**:

- `nekmeshpy/` is a **toolkit of composable primitives** — pure data containers
  plus free-function operations. It contains *no* geometry-specific meshers.
- `examples/` holds the concrete meshers as **flat, gmsh-style scripts** (constants
  at the top, top-to-bottom code, assign to a `mesh` global, export). There are no
  mesher classes by design — a bifurcation/pipe mesher *is* its script. The test
  suite executes these scripts via `runpy.run_path` and inspects the `mesh` global,
  so examples double as integration tests.

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
             hexmesh.smoothing    .quality           (watertight/    (re2/vtk/  (plot)
                   (reposition)   (scaled Jac)       conformal)      mesh, meshio)
```

## Containers are pure data; operations are free functions

**Containers are pure data; everything that acts on a finished mesh is a free
function** taking the container as its first argument — `io.export`, `io.viz`,
`model.topology`, and the per-type modules that live beside their container in the
top-level `<type>/` package:

- `hexmesh.quality` + `quadmesh.quality` — scaled-Jacobian metrics,
- `trimesh.ops` — surface ops (aliased as `nekmeshpy.trisurf`),
- `hexmesh.smoothing` / `quadmesh.smoothing` — the two smoothing modules.

Don't add heavy methods to the containers; add a function in the right
`<type>` / `model` / `io` module. The public API is re-exported from the top level
(`nekmeshpy/__init__.py`), so `from nekmeshpy import ...` is stable regardless of
internal file layout — keep `__all__` and the imports in sync when adding or
removing public names.

## Package layout

**One top-level package per mesh type** — each holds its container plus the
per-type code that acts on it (smoothing / quality / surface ops):

| package | responsibility |
|---|---|
| `linemesh/` | `LineMesh`: 1-D mesh sibling — `(N,3)` points + `(L,2)` `lines` (can branch); `open`/`loop`/`circle`/`from_segments` factories; resample/spline/align/split/radial_match (private planar-frame helpers in `linemesh/_plane.py`) |
| `trimesh/` | `TriMesh` surface container + `trimesh/ops.py` surface ops (aliased `nekmeshpy.trisurf`) |
| `quadmesh/` | `QuadMesh` section container + `quadmesh/smoothing.py` (section-smoothing registry) + `quadmesh/quality.py` (per-quad scaled-Jacobian) |
| `hexmesh/` | `HexMesh` immutable hex container + factories + `hexmesh/smoothing.py` (constrained untangle/polish) + `hexmesh/quality.py` (per-hex scaled-Jacobian) |

**`model/`** — mesh model, groups, metrics, sizing:

| module | responsibility |
|---|---|
| `model/mesh.py` | `Mesh`: generic shared-point model + meshio bridge |
| `model/physical.py` | `PhysicalGroup` / `PhysicalGroups`: name ↔ tag ↔ Nek BC code registry |
| `model/topology.py` | watertight / manifold / connectivity + hanging-point checks |
| `model/fields.py` | sizing `Field`s + 1-D graded distributions |

**`io/`** — export and visualization: `io/export.py`
(`to_re2` / `to_vtk` / `to_mesh` / `to_meshio` / `write`), `io/viz.py`
(`plot`), `io/templates/` (`.rea` header/footer templates).

**`_typing.py`** — shared numpy dtype aliases; see {doc}`conventions`.
