# NekMeshPy

An object-oriented, all-hex meshing **toolkit** with Nek5000/NekRS export.
It began as a port of the surface pipeline of a bifurcation hex-mesh generator
(originally MATLAB/Octave) and has been generalized into composable primitives: a
shared-point mesh model, named physical groups, `HexMesh` factories, interior /
smoothing / surface operations, sizing fields, quality + topology checks, and
meshio I/O.

Concrete geometry meshers — the bifurcation vessel pipeline, straight pipes, a
transfinite block — are **built on this toolkit and live in
[`examples/`](../examples)**, not in the library itself.

## Install

```bash
pip install -e .              # core (numpy, scipy)
pip install -e ".[all]"       # + matplotlib, meshio, pytest
```

## Usage

The library is driven from Python — there is no config file or command-line
tool. The mesh containers are pure data; the operations that act on a finished
mesh are free functions in dedicated modules (`io.export`, `model.quality`,
`model.topology`, `ops.trisurf`, `ops.smoothing`, `io.viz`), each taking the
mesh/surface as first argument.

### Building a mesh from the toolkit

```python
from nekmeshpy import HexMesh, QuadMesh, export
from nekmeshpy.model.fields import uniform_spacing

# sweep one QuadMesh section along a straight axis into a hex block, then stitch:
section = QuadMesh.ogrid(boundary, n_side=6, radial=uniform_spacing(4),
                         wall_name="wall")
block_a = HexMesh.extrude(section, axis=(0, 0, 1), length=5.0,
                          layers=uniform_spacing(40),   # 40 planes 0->1 (initial explicit)
                          first_cap="inlet", last_cap="outlet")
# (or HexMesh.loft(slices, ...) to recombine a stack of pre-positioned profiles)
mesh  = HexMesh.merge([block_a, block_b])       # welds coincident seam points

# boundaries are named at build time; map each name -> Nek BC code at export
codes = {"wall": "W  ", "inlet": "v  ", "outlet": "O  "}
export.to_re2(mesh, "part", groups=codes)   # native Nek5000/NekRS (.re2 + .rea)
export.write(mesh, "part.vtu", groups=codes)  # anything meshio supports
export.to_mesh(mesh, groups=codes)          # shared-point Mesh (points + cells + groups)
print(mesh.quality_summary())

assert mesh.is_watertight()             # closed, leak-tight boundary, single body
assert mesh.is_conforming()             # no hanging-point / T-junction interfaces
print(mesh.topology_report())           # faces, components, open edges, hanging points
```

`HexMesh` factories: `extrude(section, axis=…, length=…, layers=…, …)` sweeps one
`QuadMesh` section along a straight axis (gmsh Extrude + Layers + Recombine);
`loft(slices, ...)` recombines a stack of conformal `QuadMesh` profiles into a hex
block (CAD loft; the general case behind `extrude`); `merge([...])` stitches
blocks, welding coincident boundary points; `from_grid(P, face_tags=...)` builds a
structured `i×j×k` block (`face_tags` maps a side `x_min`…`z_max` to a boundary
name). Boundaries are identified by a plain **name** throughout; the mapping of
each name to a Nek BC code / integer id is supplied only at export.

### Complete meshers (examples)

Full geometry meshers built on the toolkit live in [`examples/`](../examples)
and are runnable from the repo root:

```bash
PYTHONPATH=. python examples/bifurcation.py        # bifurcation vessel (car case)
PYTHONPATH=. python examples/circular_pipe.py      # all-hex O-grid pipe
PYTHONPATH=. python examples/rectangular_pipe.py   # structured duct
```

## Architecture

```
example scripts (examples/):  bifurcation.py   circular_pipe.py / rectangular_pipe.py   transfinite_block.py
                                     │ compose the toolkit primitives below │
                                     ▼                                      ▼
TriMesh ──▶ QuadMesh cross-section slices ──HexMesh.extrude/loft/merge/from_grid──▶ HexMesh
(surface;    (points + connectivity)                                          (points + hexes;
 ops.trisurf:                                                                 weld, topology)
 cotan Laplacian,                                                                 │
 Dirichlet solve,         ┌──────────────────────────────────────────────────────┤
 boundary loops)          ▼            ▼            ▼               ▼              ▼
                   ops.interior/  model.quality  model.topology  io.export     io.viz
                   ops.smoothing  (scaled Jac)   (watertight/    (re2/vtk/     (plot)
                   (reposition)                   conformal)      mesh, meshio)
```

### Package layout

Modules are grouped into role-based subpackages; the whole public API is
re-exported from the top level, so `from nekmeshpy import ...` is unaffected.

**`geometry/`** — pure geometric data containers

| module | responsibility |
|---|---|
| `geometry/curve.py` | `Curve` (open) / `CurveLoop` (closed): typed wrapper over an `(N,3)` array (`.points`) with resample/spline/align/split/chain |
| `geometry/trimesh.py` | `TriMesh`: pure surface container (`points`, `tris`, cotan-Laplacian cache) |
| `geometry/quadmesh.py` | `QuadMesh`: plain cross-section container (points, quads, `boundaries`) |
| `geometry/hexmesh.py` | `HexMesh`: immutable hex container + factories `extrude`/`merge`/`from_grid`, connectivity views |

**`model/`** — mesh model, groups, metrics, sizing

| module | responsibility |
|---|---|
| `model/mesh.py` | `Mesh`: generic shared-point model (points, cells, point/cell sets); meshio bridge |
| `model/physical.py` | `PhysicalGroup`/`PhysicalGroups`: name ↔ tag ↔ Nek BC code registry |
| `model/quality.py` | scaled-Jacobian metrics, summary, histogram, report (mesh-agnostic) |
| `model/topology.py` | watertight / manifold / connectivity + hanging-point (T-junction) checks for hex volumes and tri surfaces (mesh-agnostic) |
| `model/fields.py` | sizing `Field`s (constant/linear/distance/min) + 1-D graded distributions |

**`ops/`** — operations acting on containers

| module | responsibility |
|---|---|
| `ops/trisurf.py` | surface ops on a `TriMesh`: cotan Laplacian, Dirichlet solve, boundary loops, isocontours, projection |
| `ops/interior.py` | per-section interior-strategy registry (`register_section_interior`, `set_section_interior`) + conduction/winslow impls |
| `ops/smoothing.py` | `smooth(mesh, surface, *, smooth_iters=…, …)`: untangle + polish with wall projection |

**`io/`** — export and visualization

| module | responsibility |
|---|---|
| `io/export.py` | `HexMesh` export free functions: `to_re2`/`to_vtk`/`to_mesh`/`to_meshio`/`write`, `summary` |
| `io/viz.py` | `plot(mesh, names=…)`: matplotlib rendering of the named boundary faces |

**Quad section factories** — `QuadMesh` classmethods that fill a boundary
(alongside the `HexMesh` factories). All boundary inputs are 3-D `(N,3)`
coordinates, and every factory builds **natively in 3-D** (nothing is projected to
a plane): a loop or edge set placed in any plane -- or a genuinely **curvy /
non-planar** boundary -- is filled in place with its true shape, never flattened to
`xy`.  For a strongly curved boundary pass `interior_method="conduction"` to
`ogrid`/`annulus` so the interior is relaxed harmonically onto the curved surface
spanned by the fixed boundary ring:

| factory | fills |
|---|---|
| `QuadMesh.structured(edges, boundary_names=…)` | transfinite (Coons) grid over a surface bounded by 4 edge curves; resolution and node distribution come **from the edges' own points** (no resampling — opposite edges must match counts), so graded edges (`Curve.resample` at clustered fractions) give a graded/near-wall grid; `boundary_names={"bottom"/"right"/"top"/"left": name}` tags outer sides at build time |
| `QuadMesh.ogrid(loop, n_side, radial, wall_name=…)` | butterfly O-grid inside a closed loop (built in 3-D, no collapsed centre; the wall ring is the loop resampled by arc length to `4*n_side` points, so a curvy loop keeps its shape); `wall_name` tags the outer ring at build time |
| `QuadMesh.half_ogrid(arc, spine, radial, wall_name=…)` | half-disc O-grid split along a spine; `wall_name` tags the arc wall at build time |
| `QuadMesh.annulus(inner, outer, radial, inner_name=…, outer_name=…)` | ring O-grid between an inner and an outer closed loop (a body inside a far-field box), blended in 3-D and paired **by index** (equal point counts — align a coarse box loop first with `outer.radial_match(inner)`); `inner_name` tags the body, `outer_name` tags the whole outer ring (a single string). To split a far field into distinct named sides, merge one structured patch per side (cf. `examples/flow_past_cylinder.py`) rather than tagging in the primitive |

Layer counts are set by a **normalized-position array**, not a count + grading
pair — a single **explicit-initial** convention shared by every layered factory:
`HexMesh.extrude`'s **`layers`** and the **`radial`** of `QuadMesh.ogrid` /
`half_ogrid` / `annulus`. Strictly increasing values in `[0, 1]` with the initial
position **explicit** — the first value is the near cap / inner ring / block
perimeter (`0` for a full span flush with the body, or e.g. `0.5` to sweep only the
far half of `length`) and the last is `1` (the outer/far face) — so `array.size - 1`
layers span `array[0]..1`. Pass `uniform_spacing(k)` for uniform,
`geometric_spacing(k, ratio)` for a graded distribution (`ratio > 1` clusters toward
the inner body / wall), or `numpy.linspace(a, 1, k + 1)` to start at `a`.

Both helpers live in `nekmeshpy.model.fields`; `uniform_spacing(k)` is shorthand for
`geometric_spacing(k, 1.0)` — `k+1` positions in `[0, 1]` including both endpoints.

Each takes an optional `interior_method=` (`conduction`/`winslow`/`bilinear`;
`None` = raw fill) that repositions interior points via
`ops.interior.set_section_interior`.

**`_typing.py`** — shared numpy array type aliases (`FloatArray` / `IntArray` /
`BoolArray`, plus the shape-documentation aliases `Point` / `Vec3` for a single
`(3,)` location / direction and `PointArray` for a `(P,3)` array of point
coordinates) used for array annotations package-wide (see *Conventions*).

**`templates/`** — `.rea` header/footer templates for the Nek exporter.

### Geometry meshers (`examples/`)

Concrete meshers are flat, gmsh-style **scripts** built on the toolkit and living
outside the library (edit the constants at the top and re-run):

| path | what it builds |
|---|---|
| `examples/bifurcation.py` | vessel surface pipeline: seam fields → cut into legs → O-grid legs (`half_ogrid`) → `extrude`/`merge` → smooth (uses `data/car.{vtx,tri}`) |
| `examples/circular_pipe.py` | a "butterfly" O-grid disc swept along an axis |
| `examples/rectangular_pipe.py` | a structured rectangular duct |
| `examples/transfinite_block.py` | eight corners → trilinear grid → `HexMesh.from_grid` |
| `examples/backward_facing_step.py` | external flow: merged structured rectangles (sides named at build time) swept along the span |
| `examples/flow_past_{cylinder,plate}.py` | external flow: four `from_grid` wedge patches around a body (circle / thin ellipse) welded with `merge`, body + each far-field box side named at build time, swept along the span |
| `examples/flow_past_half_cylinder.py` | external flow: structured section with a semicircular-bump floor, sides named at build time, swept along the span |
| `examples/flow_past_{sphere,hemisphere}.py` | external flow: cubed-sphere (half-)shell of `from_grid` patches (per-patch `face_tags`) welded with `merge` |

### Extension points

- **Interior strategy** — `@register_section_interior("name")` a `fn(qm, **opts)`
  that repositions one `QuadMesh` cross-section's interior points in place.
- **New geometry** — write a script (like those in `examples/`) that builds
  `QuadMesh` cross-sections with `QuadMesh.ogrid` / `structured` / `half_ogrid`
  and composes the `HexMesh` factories: `extrude(section, axis=…, length=…,
  layers=…, …)` to sweep one section along a straight axis, `loft(slices, …)` to
  recombine pre-positioned profiles, `merge([...])` to stitch blocks,
  `from_grid(P, face_tags=…)` for a structured block.
- **New cross-section** — add a `QuadMesh` factory classmethod that fills a
  boundary loop with quads (returns a `QuadMesh` with `boundaries`).
- **External-flow domains** — name the boundaries **as you build**, so the tags ride
  through construction (no post-hoc boundary detection): tag the section outer edges
  with `QuadMesh.structured(boundary_names=…)` / `QuadMesh.annulus(inner_name=…,
  outer_name=…)` (they propagate onto the swept side faces via `loft`/`extrude`),
  name the sweep end caps with `loft(…, first_cap=…, last_cap=…)`, and tag structured
  blocks side-by-side with `from_grid(P, face_tags=…)`. Leave a face welded away by
  `merge` **untagged** (`NO_BOUNDARY` / an omitted side) so merge stays a plain
  concatenate with no stale interior tag. See the `examples/flow_past_*.py` scripts.
- **Physical groups** — build with plain boundary **names**, then pass `groups=`
  to the exporters (`to_re2`/`to_vtk`/`to_mesh`/…) to map each name to a Nek BC
  code / integer id: a `{name: nek_code}` dict, a `PhysicalGroups` (use a preset
  such as `PhysicalGroups.nek_default()` for byte-exact codes), or `None` to
  auto-number the mesh's distinct names.
- **Sizing** — subclass `Field`; feed it to a size-field-aware mesher.

## Conventions

- Triangle/vertex indices are **0-based** internally (input `.tri` is 1-based,
  converted on load; `.re2` element ids are written 1-based).
- Coordinates are plain NumPy arrays everywhere — there is no `Point` class. `Curve`/`CurveLoop`
  are typed wrappers over a single `(N,3)` array exposed as `curve.points` (**3-D input is
  required — a `(N,2)` array is rejected, not padded to `z=0`**); the mesh containers
  (`TriMesh`/`QuadMesh`/`HexMesh`/`Mesh`) likewise store a `(P,3)` array `points` (mutate with
  `mesh.points[:] = X`). A single point is just a `(3,)` array (e.g. a row of `.points`).
- `HexMesh` is **immutable by construction** (no incremental building): build it
  with a factory (`HexMesh.extrude` / `loft` / `merge` / `from_grid`) or the array
  constructor `HexMesh(points, hexes, boundaries, boundary_names)`. `extrude`/`loft`
  are shared-point by construction (conformal slices → index arithmetic, no weld);
  `merge` is the one place coincident seam points are coordinate-welded (one
  explicit pass). It stores `points` + `hexes` `(N,8)` integer connectivity in Nek
  point order; `boundaries` is `(Nbc,2)` = `[element id (0-based), face (1–6)]`
  with a parallel `boundary_names` `(Nbc,)` naming each tagged face. `weld()`
  returns `(points, hexes, n_points)`; the exporters
  expand to per-element coordinates via `points[hexes]`. Coordinates may still be
  repositioned in place (smoothing/interior); topology is fixed.
- Array annotations use the dtype aliases from `nekmeshpy._typing` — `FloatArray`
  (coordinates/real data), `IntArray` (connectivity/indices), `BoolArray` (masks) —
  not a bare `np.ndarray` (see *Development*). `Point` / `Vec3` / `PointArray` alias
  `FloatArray` to document a single `(3,)` location / direction / a `(P,3)` coordinate
  array (not shape-enforced by mypy).
- Progress goes through the `nekmeshpy` logger (your script configures it).

## Validation

Interior repositioning runs **per-slice** inside the O-grid mesher (each
cross-section's interior is solved with its wall arc and flat spine held fixed),
so the assembled bifurcation coordinates no longer match the original MATLAB
assembled solve; the goldens are a frozen self-snapshot instead. The exported
`.re2` boundary block and the `.rea` file (no coordinates) are **byte-exact**,
`.re2` coordinates match to `1e-12`, and `.vtk` is byte-identical; export stays
deterministic. The three interior methods on the bundled `car` case (default
parameters) give bilinear 0.248/0.821, conduction 0.227/0.818, winslow
0.253/0.794 min/mean scaled Jacobian.

`tests/` pins all of the above as a golden-regression suite (`pytest`) — the
reference outputs are frozen in `tests/golden/`. Each leg-half is built
shared-point by `HexMesh.loft` (conformal profiles → index arithmetic) and the
six halves are stitched by `HexMesh.merge` (one coincident-point weld at the
seams), so the exported mesh is genuinely conforming (`topology.hex_report`:
watertight **and** conformal, no floating-point cracks at shared points).

## Development

```bash
pip install -e ".[all,dev]"    # + ruff, mypy, pytest
ruff check nekmeshpy tests examples
mypy                            # type-checks the whole nekmeshpy package
pytest                          # golden-regression + algorithm tests
```

CI (`.github/workflows/ci.yml`) runs the linter and type-checker plus the test
suite on Python 3.9–3.12. The package ships a `py.typed` marker and every
module — public API and numeric internals alike (`geometry/hexmesh`,
`geometry/trimesh`, `ops/`, `model/`, …) — is fully annotated and checked
by mypy with `disallow_untyped_defs` **and** `disallow_any_generics` enabled.
Array values use the dtype-parametrized aliases in `nekmeshpy._typing`
(`FloatArray` = `NDArray[np.float64]` for coordinates/real data, `IntArray` =
`NDArray[np.int64]` for connectivity/indices, `BoolArray` for masks) rather than
a bare `np.ndarray` (an implicit `NDArray[Any]`), which `disallow_any_generics`
rejects; use an explicit `NDArray[...]` for any other dtype. `Point`, `Vec3` and
`PointArray` alias `FloatArray` to flag a single `(3,)` location / direction, or a
`(P,3)` array of point coordinates (vs `(N,)` scalar data); since numpy typing has
no static shape checking they document intent only and mypy treats them as
interchangeable with `FloatArray`.

## Scope / notes

- Only the **surface** method is ported (`method='surface'`); the volumetric
  pipeline is out of scope.
- meshio writers for HDF-based formats (`.xdmf`) additionally need `h5py`.
- Not yet done (future work): package rename, Sphinx docs, KDTree-accelerated
  projection and O(n) segment chaining (left as exact brute-force to preserve
  byte-identity), curved/high-order elements, and multi-block assembly
  (`HexMesh.merge`).
```
