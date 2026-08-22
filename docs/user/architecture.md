# Architecture

NekMeshPy is split into **two layers**:

- `nekmeshpy/` — a **toolkit of composable primitives**: pure data containers plus
  free-function operations. No geometry-specific meshers.
- `examples/` — concrete meshers as **flat gmsh-style scripts** (constants at top,
  assign to a `mesh` global, export). No mesher classes by design — a mesher *is*
  its script. The suite runs them via `runpy.run_path` and inspects the `mesh`
  global, so examples double as integration tests.

The one carve-out is `examples/tjunction_lib.py`: a parametrized T-junction builder
imported by `cob_tjunction.py` and `quadrant_pipe_tjunction.py`. It is shared *between
examples*, not promoted into the toolkit — a junction is geometry-specific, which is
exactly what `nekmeshpy/` does not hold.

## Data flow

```
example scripts (examples/):  carotid.py / femoral.py   circular_pipe.py / backward_facing_step.py
                              cob_tjunction.py / chimera*.py (via tjunction_lib.py)   flow_past_*.py
                                     │ compose the toolkit primitives below │
                                     ▼                                      ▼
TriMesh ─┬─▶ QuadMesh cross-section slices ─hexmesh.extrude/sweep/loft/annulus/merge/attach─▶ HexMesh
(surface; │   (a section, one rung down)                                      (hexes over shared
 trimesh  │                                                                    faces; weld, topology)
 .ops:    └─▶ TetMesh ──▶ P1 conduction solve ──▶ level-set stations ──┘           │
 cotan Lap.,  (tetmesh.ops.tet_mesh, gmsh)   (solve_dirichlet/seam_fields)         │
 Dirichlet,               ┌───────────────────────────────────────────────────────┤
 boundary loops)          ▼               ▼            ▼               ▼          ▼
             quadmesh.smoothing/  quadmesh|hexmesh   core.topology   io.writer  io.viz
             hexmesh.smoothing    .quality           (watertight/    (re2/vtu/  (plot)
             (relax / untangle)   (scaled Jac,       overlap-free/   fld/vtp,
                                   order_scan)       conformal)      meshio)
```

`io/reader.py` closes the loop the other way: `read_rea` parses a `.rea` back into a
`HexMesh`.

## Containers are pure data; operations are free functions

Everything acting on a finished mesh is a **free function** taking the container
as its first argument — `io.writer`, `io.viz`, `core.topology`, and the per-type
modules beside their container (`hexmesh.quality`/`quadmesh.quality`,
`trimesh.ops`, `tetmesh.ops`, `hexmesh.smoothing`/`quadmesh.smoothing`).

Don't add heavy methods to the containers; add a function in the right module.

Each subpackage's own `__all__` is the operational API — `from nekmeshpy import
hexmesh; hexmesh.attach(...)`. The top-level `nekmeshpy/__init__.py` re-exports only
the containers and the rung-agnostic types (`Mesh`, `ElementTags`, `PhysicalGroup(s)`,
the `Field` family, the section-smoothing registry, and the `topology`/`fields`/
`smoothing`/`viz`/`writer` modules). Names collide across rungs by design (each has
its own `loft`, `merge`, `attach`), which is why there is no flat namespace above the
per-rung ones.

## Package layout

**One top-level package per mesh type**, each holding its container plus the code
that acts on it. Operations live in sibling modules split by arity and rung delta
(see {doc}`concepts` for the full module-by-responsibility breakdown):

| package | container | own storage |
|---|---|---|
| `pointmesh/` | `PointMesh` | `(N,3)` points + an `ElementTags` over point ids — the ladder's bottom rung |
| `linemesh/` | `LineMesh` | shared-point `point_mesh` + `(L,2)` lines + `(L,order-1,3)` interior |
| `quadmesh/` | `QuadMesh` | shared-edge `line_mesh` + `(Q,4)` `quads`/`orient` + `(Q,(order-1)²,3)` interior |
| `hexmesh/` | `HexMesh` | shared-face `quad_mesh` + `(E,6)` `hexes`/`orient` + `(E,(order-1)³,3)` interior |
| `trimesh/` | `TriMesh` | input surface container; ops in `trimesh/ops.py` |
| `tetmesh/` | `TetMesh` | linear tet volume; ops in `tetmesh/ops.py` |

`trimesh` and `tetmesh` are deliberately thin and sit **off** the ladder: a tet mesh
here exists to solve a field or walk a volume, never to be exported. The gmsh
dependency lives inside the single function `tetmesh.ops.tet_mesh`, imported at call
time (`pip install .[mesh]`); everything else in `tetmesh.ops` is the conduction
machinery `examples/femoral.py` cuts its stations from (`tet_laplacian`,
`solve_dirichlet`, `seam_fields`, `leg_label`).

Each ladder container also exposes derived `points`/`corners` views (see
{doc}`concepts`). Siblings, per rung:

| module | holds |
|---|---|
| `assemble.py` | `loft`/`loft_fn`/`loft_spline`, the two welds `merge`/`attach` (+ `Seam`), and their inverse `select`/`remove`/`components` |
| `lift.py` | rung-raising: `extrude`/`sweep`/`sweep_path`/`annulus`/`from_grid`, plus hex-only `adapter`/`bridge` |
| `lower.py` | `boundary_mesh` (Δ −1) — quad and hex only |
| `morph.py` | `blend`, `translate`/`rotate`/`scale`/`transform`/`mirror`, `offset`; quad also `reindex`/`place_on_path`, line also `reverse` |
| `tag.py` | `retag_*` (rename a vocabulary, geometry untouched) **and** the authoring bridges `quadmesh.tag_edges` / `hexmesh.tag_faces` |
| `query.py` | reads: measures, `bounds`/`centroid`, `element_blocks`, `tagged_edges`/`tagged_faces`, plus hex `topology_report`/`report`/`tag_report` |
| `shape.py` | factories: quad region fills and closed surfaces, line `line`/`arc`/`circle`/`rectangle`/`on_surface` and the fraction helpers, hex `tetra` |

Only `quadmesh/` and `hexmesh/` carry a `smoothing.py` and a `quality.py`.
`quadmesh/ports.py` is the odd sibling: a `Port` is a section plus the two facts a bare
section cannot state about itself — which way it faces and where its axis is — so it
lives at the *section* rung, though its consumers are the hex-rung connectors
`adapter`/`bridge`.

**`core/`** — mesh model, groups, metrics, sizing. All modules below import **no
container** except `mesh.py` and `topology.py`:

| module | responsibility |
|---|---|
| `core/mesh.py` | `Mesh`: generic shared-point model + meshio bridge |
| `core/tags.py` | `ElementTags`: the one sparse name→ids table every rung's `element_tags` (and so every rung-above's side tags) is; plus `sweep_element_tags`/`sweep_cap_tags`/`welded_element_tags` |
| `core/physical.py` | `PhysicalGroup`/`PhysicalGroups`: name ↔ tag ↔ Nek BC code registry |
| `core/topology.py` | watertight/manifold/connectivity + hanging-point checks (`TopologyReport`), plus the geometric `count_overlapping_pairs`/`is_overlap_free` |
| `core/quality.py` | shared `QualitySummary` + `POOR_THRESHOLD`, and the sampling schema `OrderScan`/`SCAN_ORDER`/`SCAN_BUDGET` both quality modules read |
| `core/fields.py` | sizing `Field`s, graded 1-D distributions (`uniform_`/`geometric_`/`symmetric_spacing`, `validate_layers`), `gll_nodes`/`lagrange_derivative_matrix` |
| `core/interp.py` | order-N kernel over GLL reference nodes: `tensor_nodes`, `corner_indices`, `subdivide_element`, `coons_grid`, `blend_ho`, `resample_block`, `scaled_jacobian` |
| `core/stations.py` | rung-agnostic sweep-station machinery behind every `loft`/`sweep`: `split_evaluated`, `refined_lattice`, `spline_levels`, `sweep_lattice` |
| `core/measure.py` | one GLL quadrature over a node block, behind every rung's `length`/`area`/`volume`/`centroid` — corner and curved readings are the same code on different blocks |
| `core/affine.py` | affine maps behind `translate`/`rotate`/`scale`/`mirror`; `reflection` has det −1, so its caller must pair it with a re-winding |
| `core/frames.py` | moving-frame machinery behind `sweep`: `tangents`, frame generators (`fixed_up`/`parallel_transport`/`frenet`/`plane_frame`), `spin`, `sweep_placements` |
| `core/paths.py` | `walk` — a 3-D turtle of `line`/`arc`/`helix` moves into a `Path` (C1, analytic tangent, arc-length param, and the moving frame the walk carried) |
| `core/surfaces.py` | `SurfaceCurve` + `ruled`/`blend`/`reverse`/`shift`/`reparam` — curves carried as parametrization, since a point-space lerp between two curves on a cylinder dips inside it |
| `core/conform.py` | two engines behind the B-rep. **Topology/orientation**: `unique_edges`/`unique_faces`/`canonical_faces`, `entity_tol`, `scatter_*`/`gather_*`, `conformal_*` walks, D4 helpers. **Welding**: `bbox_scale`, `coincident_clusters`, `weld_points`/`weld_pairs`, `MAX_WELD_FRACTION`, `fuse_entities`, `renumber_map` |

**`io/`** — `io/writer.py` (`to_re2`/`to_fld`/`to_vtu`/`to_mesh`/`to_meshio`/`write`,
plus `line_to_vtu`/`quad_to_vtu`/`boundary_to_vtp` for the lower rungs and the docs
gallery), `io/reader.py` (`read_rea`), `io/viz.py` (`plot`).

**`_typing.py`** — shared numpy dtype aliases; see {doc}`conventions`.
