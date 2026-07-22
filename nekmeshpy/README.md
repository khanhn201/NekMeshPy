# NekMeshPy

An object-oriented, extensible **all-hex mesher** with Nek5000/NekRS export.
It began as a port of the surface pipeline of a bifurcation hex-mesh generator
(originally MATLAB/Octave) and has grown a gmsh-style generic core: a shared-node
mesh model, named physical groups, pluggable meshing algorithms and interior
strategies, sizing fields, quality metrics, meshio I/O, and a CLI.

The bifurcation vessel mesher is now *one* algorithm (`BifurcationMesher`) built
on that core; `TransfiniteBlock` is a second, and new algorithms plug in through
the same `HexAlgorithm` contract.

## Install

```bash
pip install -e .              # core (numpy, scipy)
pip install -e ".[all]"       # + matplotlib, meshio, PyYAML, pytest
```

## Usage

### Command line (`nekmesh`)

```bash
nekmesh mesh --interior winslow --out vessel --format re2,vtk,vtu
nekmesh mesh --config case.yaml            # parameters from YAML/JSON
nekmesh pipe --shape circular    --radius 0.5 --length 5 --n-axial 40 --out pipe
nekmesh pipe --shape rectangular --width 2 --height 1 --length 6 --out duct
nekmesh quality vessel.vtu --histogram     # scaled-Jacobian report
nekmesh info    vessel.vtu                 # points / cells / groups
nekmesh convert vessel.vtu vessel.msh      # meshio format conversion
```

Also `python -m nekmeshpy` runs the default case.

### Python

`HexMesh`/`TriMesh` are pure data containers; the algorithms that act on a
finished mesh are free functions in dedicated modules (`io.export`,
`model.quality`, `ops.trisurf`, `ops.smoothing`, `io.viz`), each taking the
mesh/surface as first argument:

```python
from nekmeshpy import Config, BifurcationMesher, export, quality

cfg = Config()
cfg.interior_method = "winslow"        # bilinear | harmonic | harmonic3d | winslow
hexmesh = BifurcationMesher(cfg).run() # returns a HexMesh

export.to_re2(hexmesh, "vessel")        # native Nek5000/NekRS (.re2 + .rea)
export.write(hexmesh, "vessel.vtu")     # anything meshio supports
mesh = export.to_mesh(hexmesh)          # shared-node Mesh (points + cells + groups)
print(quality.summary(*hexmesh.weld()[:2]))
```

`Config` is a dataclass: instantiate and edit fields, or load/validate:

```python
cfg = Config.from_file("case.yaml"); cfg.validate()
```

Generic primitives, using the same assembly / export / quality machinery:

```python
from nekmeshpy import CircularPipe, RectangularPipe, TransfiniteBlock, ConstantField, export

# all-hex O-grid circular pipe (no collapsed cell at the centre)
export.to_re2(CircularPipe(radius=0.5, length=5.0, n_axial=40,
             n_side=6, n_radial=4, radial_grading=1.15).run(), "pipe")

# structured rectangular duct, swept along an arbitrary axis
export.to_re2(RectangularPipe(width=2.0, height=1.0, length=6.0, nx=16, ny=8,
                n_axial=48, axis=(0, 0, 1)).run(), "duct")

# a transfinite corner-defined block
corners = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]]
export.to_re2(TransfiniteBlock(corners, size_field=ConstantField(0.1)).run(), "block")
```

Every algorithm is also reachable by name through the registry (gmsh-style):

```python
from nekmeshpy import export, make
export.to_re2(make("circular_pipe", radius=0.5, length=5.0, n_axial=40).run(), "pipe")
```

Runnable versions of these live in [`examples/`](../examples).

## Architecture

```
                         HexAlgorithm (Protocol)  ──registry──▶  nekmesh CLI
                          ├─ BifurcationMesher (surface pipeline)
                          └─ TransfiniteBlock  (structured primitive)
                                     │ .run()
                                     ▼
TriMesh ──seam fields──▶ CutSurface ──rings──▶ OGridLeg ──slices──▶ HexMesh
(surface data;           (legs, seam         (QuadMesh      (hex data container:
 ops in trisurf:          rings, per-leg      cross-          extrude, tag, weld,
 cotan Laplacian,         fields)             sections)       finalize)
 isocontours,                                                     │
 projection)              ┌───────────────────────────────────────┤
                          ▼            ▼           ▼               ▼
                   interior /    quality      export (to_re2 /  viz (plot)
                   smoothing     (scaled Jac) to_vtk / to_mesh /
                   (reposition)               write, meshio)
```

### Package layout

Modules are grouped into role-based subpackages; the whole public API is
re-exported from the top level, so `from nekmeshpy import ...` is unaffected.

**`geometry/`** — pure geometric data containers

| module | responsibility |
|---|---|
| `geometry/polyline.py` | `Polyline`/`Arc`/`Ring` value objects (resample, spline, align, split, chain) |
| `geometry/trimesh.py` | `TriMesh`: pure surface container (`xyz`, `tri`, cotan-Laplacian cache) |
| `geometry/quadmesh.py` | `QuadMesh`: plain cross-section container (nodes, quads, `wall_edges`) |
| `geometry/hexmesh.py` | `HexMesh`: pure hex container — extrude/tag, weld, finalize, connectivity views |

**`model/`** — mesh model, groups, metrics, sizing

| module | responsibility |
|---|---|
| `model/mesh.py` | `Mesh`: generic shared-node model (points, cells, point/cell sets); meshio bridge |
| `model/physical.py` | `PhysicalGroup`/`PhysicalGroups`: name ↔ tag ↔ Nek BC code registry |
| `model/quality.py` | scaled-Jacobian metrics, summary, histogram, report (mesh-agnostic) |
| `model/fields.py` | sizing `Field`s (constant/linear/distance/min) + 1-D graded distributions |

**`ops/`** — operations acting on containers

| module | responsibility |
|---|---|
| `ops/trisurf.py` | surface ops on a `TriMesh`: cotan Laplacian, Dirichlet solve, boundary loops, isocontours, projection |
| `ops/interior.py` | interior-strategy registry (`register_interior`, `set_interior`) + harmonic/winslow impls |
| `ops/smoothing.py` | `smooth(mesh, surface, opts)`: untangle + polish with wall projection |

**`io/`** — export and visualization

| module | responsibility |
|---|---|
| `io/export.py` | `HexMesh` export free functions: `to_re2`/`to_vtk`/`to_mesh`/`to_meshio`/`write`, `summary` |
| `io/viz.py` | `plot(mesh, cfg)`: matplotlib rendering of the tagged boundary |

**`algorithms/`** — registry + mesh generators

| module | responsibility |
|---|---|
| `algorithms/registry.py` | `HexAlgorithm` Protocol + algorithm registry (`register_algorithm`, `make`, `available`) |
| `algorithms/bifurcation.py` | `BifurcationMesher` orchestrator |
| `algorithms/cutsurface.py` | `CutSurface`: cut into legs, seam smoothing, per-leg fields, conformal seam rings |
| `algorithms/ogrid.py` | `half_ogrid` section mesher + `OGridLeg`: fine rings → two stacks of `QuadMesh` slices |
| `algorithms/blocks.py` | `TransfiniteBlock` structured-hex primitive |
| `algorithms/pipes.py` | `CircularPipe` (swept O-grid disc) / `RectangularPipe` (structured duct) |

**top level** — configuration, CLI, data

| module | responsibility |
|---|---|
| `config.py` | `Config` dataclass; YAML/JSON load/save; `validate()`; `flux_tag_for` |
| `cli.py` | `nekmesh` command-line entry point |
| `data/` | bundled `car` case (`car.vtx`, `car.tri`) |
| `templates/` | `.rea` header/footer templates for the Nek exporter |

### Extension points

- **Interior strategy** — `@register_interior("name")` a `fn(hexmesh, twall, **opts)`.
- **Algorithm** — `@register_algorithm("name")` a class with `.run() -> HexMesh`
  (see `algorithms/blocks.py` / `algorithms/pipes.py`; extrude a stack of `QuadMesh` cross-section
  slices — sharing connectivity and `wall_edges` — into a hex stack with
  `HexMesh.add_extruded_section`, which returns the element-id grid so you can
  `tag_face` interior caps yourself).
- **Physical groups** — build a `PhysicalGroups` and pass `HexMesh(groups=...)` to
  rename tags / set Nek BC codes without touching the exporter.
- **Sizing** — subclass `Field`; feed it to `TransfiniteBlock(size_field=...)`.

## Conventions

- Triangle/vertex indices are **0-based** internally (input `.tri` is 1-based,
  converted on load; `.re2` element ids are written 1-based).
- `HexMesh.elements` is `(N,8,3)` in Nek node order and is the canonical,
  byte-exact export source; `HexMesh.boundaries` is `(Nbc,3)` =
  `[element id (0-based), face (1–6), tag]`. `weld()` / `export.to_mesh()` build
  the shared-node view on demand.
- Progress goes through the `nekmeshpy` logger (the CLI / `__main__` configure it).

## Validation

Verified against the reference MATLAB/Octave implementation on the bundled `car`
case (default parameters): the assembled + interior + smoothed mesh matches to
~2.7e-13 (the residual is `scipy.spsolve` vs MATLAB backslash). The exported
`.re2` boundaries are identical and the `.rea`/`.vtk` files are **byte-identical**
to the reference. All three interior methods reproduce their expected quality
(bilinear 0.250/0.838, harmonic 0.184/0.861, winslow 0.367/0.872 min/mean scaled
Jacobian).

`tests/` pins all of the above as a golden-regression suite (`pytest`) — the
reference outputs are frozen in `tests/golden/`, so every refactor stays exact.
The `weld` and `cotan_laplacian` routines are vectorized but were kept
**bit-identical** (weld preserves the element-major insertion-order node
labelling, since the sparse solves and accept/reject smoothing branches are not
permutation-invariant in floating point).

## Development

```bash
pip install -e ".[all,dev]"    # + ruff, mypy, pytest
ruff check nekmeshpy tests examples
mypy                            # type-checks the public-API modules
pytest                          # golden-regression + algorithm tests
```

CI (`.github/workflows/ci.yml`) runs the linter and type-checker plus the test
suite on Python 3.9–3.12. The package ships a `py.typed` marker; the public-API
modules (`algorithms/registry`, `config`, `model/fields`, `model/physical`,
`model/quality`, `algorithms/pipes`) are fully annotated and checked by mypy,
while the numeric internals (`geometry/hexmesh`, `geometry/trimesh`, …) are
annotated opportunistically.

## Scope / notes

- Only the **surface** method is ported (`cfg.method='surface'`); the volumetric
  pipeline is out of scope.
- meshio writers for HDF-based formats (`.xdmf`) additionally need `h5py`.
- Not yet done (future work): package rename, Sphinx docs, full type coverage of
  the numeric internals, KDTree-accelerated projection and O(n) segment chaining
  (left as exact brute-force to preserve byte-identity), curved/high-order
  elements, and multi-block assembly (`HexMesh.merge`).
```
