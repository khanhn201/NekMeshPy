# NekMeshPy

[![docs](https://img.shields.io/badge/docs-github%20pages-blue)](https://khanhn201.github.io/NekMeshPy/)

An all-hex meshing **toolkit** with Nek5000/NekRS export: composable primitives
— a shared-point mesh model, named physical groups, `HexMesh` factories,
smoothing / surface operations, sizing fields, quality + topology checks, and
meshio I/O. Concrete meshers (bifurcation vessel, pipes, external-flow domains)
are built on the toolkit and live in [`examples/`](examples), not the library.

## Documentation

📖 **Full documentation: <https://khanhn201.github.io/NekMeshPy/>**

- **[Getting started](https://khanhn201.github.io/NekMeshPy/user/getting-started.html)** — install, build your first mesh, export.
- **[Concepts](https://khanhn201.github.io/NekMeshPy/user/concepts.html)** — the line→quad→hex ladder, tags, section factories, smoothing.
- **[How-to recipes](https://khanhn201.github.io/NekMeshPy/user/howto.html)** — O-grid pipe, external flow, sphere shell, structured duct.
- **[API reference](https://khanhn201.github.io/NekMeshPy/reference/)** — every public module, class, and function.
- **[Architecture](https://khanhn201.github.io/NekMeshPy/user/architecture.html)** — how the toolkit is laid out and why.
- **[Conventions](https://khanhn201.github.io/NekMeshPy/user/conventions.html)** — typing, naming, and the invariants the code holds to.

## Install

```bash
pip install -e .              # core (numpy, scipy)
pip install -e ".[all]"       # + matplotlib, meshio, pytest
```

## Quick start

Driven from Python — no config file or CLI. Mesh containers are pure data;
operations on a finished mesh are free functions taking the mesh first.

```python
from nekmeshpy import HexMesh, LineMesh, QuadMesh, export

# Tag the wall on the boundary loop; the tag rides up line -> quad -> hex.
boundary = linemesh.circle(radius=1.0, n=24, element_tags=["wall"] * 24)

# Fill the loop with an O-grid section, then sweep it into a hex block.
# radial / layers count *cells*: an int is n uniform layers; pass an explicit
# array of normalized positions (geometric_spacing(n, r), ...) to grade them.
section = quadmesh.ogrid(boundary, n_side=6, radial=4)
block   = hexmesh.extrude(section, axis=(0, 0, 1), length=5.0, layers=40,
                          first_tag="inlet", last_tag="outlet")

# Boundaries are named at build time; map each name -> Nek BC code at export.
codes = {"wall": "W  ", "inlet": "v  ", "outlet": "O  "}
export.to_re2(block, "pipe.re2", groups=codes) # native Nek5000/NekRS binary mesh
export.write(block, "pipe.vtu", groups=codes)  # anything meshio supports

assert block.is_watertight()                   # closed, leak-tight, single body
assert block.is_conforming()                   # no hanging-point / T-junction faces
print(block)                       # <HexMesh 5945 points, 5280 hexes, order 1, ...>
print(block.quality_summary().min) # a QualitySummary NamedTuple, not a dict
```

Every container has a `__repr__` that names its size, order and tag vocabulary, and the
report-returning functions hand back **NamedTuples** rather than dicts —
`quality_summary()` a `QualitySummary` (`n_elements`/`min`/`max`/`mean`/`median`/
`n_inverted`/`n_poor`), `topology_report()` a `TopologyReport` — so a typo in
`stats.n_inverted` is a `mypy` error instead of a `KeyError` at runtime.

## Examples

Full meshers live in [`examples/`](examples), run from the repo root:

```bash
PYTHONPATH=. python examples/bifurcation.py        # bifurcation vessel (car case)
PYTHONPATH=. python examples/circular_pipe.py      # all-hex O-grid pipe
PYTHONPATH=. python examples/rectangular_pipe.py   # structured duct
PYTHONPATH=. python examples/flow_past_cylinder.py # external flow around a body
```

See the [how-to recipes](https://khanhn201.github.io/NekMeshPy/user/howto.html)
for a guided tour of each.

### Sweeping with `loft`

`loft` is one primitive at three dimensions — `linemesh.loft` (each profile a single
point, the rungs *are* the lines), `quadmesh.loft` (profiles are `LineMesh`es) and
`hexmesh.loft` (profiles are `QuadMesh`es) — with `extrude` the straight special case
at each rung, and `linemesh.loft` itself the only thing that authors 1-D
connectivity (`loop=False` chain / `loop=True` ring).

Pass `loop=True` for a **periodic** sweep: the last profile joins back to the first,
so `M` profiles give `M` layers and the seam is a genuine shared entity (no
duplicated layer, no free boundary in the sweep direction) — a torus surface from
revolved rings, a solid torus from revolved discs. A closed sweep has no near/far
cap, so `first_tag` / `last_tag` are rejected with a `ValueError` and no cap boundary
rows are emitted.

```python
sections = [...]                                   # rings revolved about an axis
torus    = quadmesh.loft(sections, loop=True)      # closed surface, zero free edges
```

### Bending along a path with `sweep`

`loft` asks you to position every profile. When each station is the **same section**
carried along a curve — a pipe through a 90° elbow, a U-turn, a coil —
`quadmesh.sweep` / `hexmesh.sweep` do the placing, from a moving frame:

```python
disc = quadmesh.ogrid(linemesh.circle(0.1, 20, order=2), n_side=5, radial=3)
bend = hexmesh.sweep(disc, path, np.linspace(0.0, 1.0, 21),   # path: (K,) -> (K,3)
                     origin=(0, 0, 0),          # the section's reference point
                     tangent=dpath, orientation="fixed", up=(0, 1, 0))
```

The section is placed **rigidly** (`p ↦ path(t) + R(t) @ p_local`), never offset
point-by-point — which is the whole point, because through a bend of radius `Rb` the
outboard wall traverses `Rb + d` and the inboard `Rb - d`, so neither follows the
centreline. The intermediate GLL stations are evaluated too, so the bend is exact at
`order > 1` along the sweep as well as around the section — there is no `order=` here,
because a rigid placement cannot change the section's own order. `origin=` is
**required**: it is the section's reference point, the one that rides the path, and it
has no safe default — an O-grid disc's centroid is *not* its centre, so the obvious
guess gave a quietly off-axis block. `orientation=` names a *mode* and nothing else —
`"transport"` (rotation-minimizing, right on a non-planar path), `"fixed"` with an
`up=`, or `"frenet"` (present, but undefined on a straight run) — and a per-station up
field is a `(K,3)` `up=` alongside `orientation="fixed"`, not something passed as the
mode. Whichever you pick, station 0 lands the section exactly as authored. `loop=True`
closes the sweep on the *identical* first placement, so a solid torus welds exactly.

A path assembled from pieces (straights and arcs) has curvature that **jumps** at each
junction, and an element straddling one is fitted across two different geometries —
a visible kink in the wall. `linemesh.sweep_fractions(breaks, total_length, target)`
returns the stations that avoid it: each interval between consecutive junctions is
subdivided on its own at roughly `target`, so every junction reappears in the output
bit-for-bit instead of being approached by a global `linspace`. Like
`linemesh.arclength_fractions` and `quadmesh.spine_fractions` it is a helper, not a
factory — it returns a plain array of `fractions`, because the sweep meshes exactly at
the stations it is given. See `examples/serpentine_pipe.py`.

### High-order (order-N) elements

The order is declared **once, at the bottom of the ladder**. A factory that builds its
points from nothing takes an optional `order=N` (default `1`) — `linemesh.circle` /
`arc` / `line` / `rectangle` / `loft`, `quadmesh.box` / `sphere` / `rectangle` /
`from_grid`, `hexmesh.from_grid`; everything that takes a *mesh* in (`ogrid`,
`structured`, `annulus`, `extrude`, `sweep`, `blend`, `loft`, `merge`) has no `order=`
at all and inherits it from its inputs, rejecting a mismatch loudly. So `order=` is set
on the boundary loop and rides all the way up. (`quadmesh.loft_fn` /
`hexmesh.loft_fn` keep an `order: int | None = None` because they *evaluate*
profiles rather than receive them; `None` means the profiles' own.) At order `N` each
element carries `(N+1)` Gauss–Lobatto–Legendre nodes per parametric direction (line
`N+1`, quad `(N+1)²`, hex `(N+1)³`). `.re2` export stays linear (corners only — Nek's
re2 has no high-order format yet), so a mesh exports byte-identically at any order, while `.vtu`
emits VTK Lagrange cells (68 / 70 / 72) that ParaView and VisIt render as curved
geometry. To hand the curved geometry to Nek itself, use `export.to_fld`, which writes
the Nek field format (`<prefix>0.f00001`, `fields="X"`) — that one *does* store the
full `lx1³` GLL block per element.

**The B-rep ladder is the storage.** There is no per-element node block anywhere and
no `.curved` facade: each container holds the rung below it plus what it privately
owns — `LineMesh` (`points`, `lines`, `interior (L,N-1,3)`); `QuadMesh` (a `lines`
*`LineMesh` of the shared edges* + `quad`/`flip` incidence + `interior
(Q,(N-1)²,3)`); `HexMesh` (a `quads` *`QuadMesh` of the shared faces* + `hex` /
`face_orient` incidence + `interior (E,(N-1)³,3)`). `points` / `quads` / `hexes` are
**derived read-only views** over it, so corner consistency is structural and
`mesh.points[:] = X` propagates everywhere for free. Conformality is likewise
structural: a shared edge or face is *one stored object* referenced by every incident
element, resolved by corner ids rather than a coordinate search
(`nekmeshpy.core.conform`). The conformal walks
`conform.conformal_line`/`_quad`/`_hex` flatten it on demand into `(nodes, conn_ho)`
— the high-order analog of `points` + `quads` — and that is what the `.vtu` writer and
the order-N quality metrics (`mesh.scaled_jacobian(high_order=True)`) read.

**Curved geometry is not automatic.** Factories that own an analytic shape place the
extra nodes on it — `linemesh.circle` / `linemesh.arc` on the exact arc,
`linemesh.loft_fn` on any analytic parametrization you hand it (it calls your callable on
the whole node lattice, corners *and* interiors),
`quadmesh.loft_fn` / `hexmesh.loft_fn` doing the same one and two rungs up along the
*sweep* (your callable returns a `LineMesh` profile / `QuadMesh` section and is called at
every node level, not just the corner levels),
`quadmesh.sweep` / `hexmesh.sweep` carrying **one** profile along a curved path by a moving
frame (a bent pipe from one O-grid disc),
`quadmesh.sphere` / `quadmesh.hemisphere` projecting every node onto the exact sphere —
the region fills (`ogrid` / `half_ogrid` / `quadrant_ogrid` / `structured`) carry their input walls'
curvature into the interior as well as onto the wall, and the combinators (`extrude` /
`blend` / `loft` / `annulus`) carry that curvature up the ladder. Anything built from an explicit point array
(`linemesh.loft`, `from_grid`) has only those points to go on and
straight-subdivides between them: high order in storage, linear in geometry — so pass
`linemesh.loft_fn` a closed form rather than sampling it into an array. A plain
`quadmesh.loft` is the same trap along its *sweep* direction — exact profiles still give a
surface that is straight between them — so reach for `loft_fn` or `sweep` (or hand
`loft` the intermediate profiles as `sweep_nodes=`) when the sweep path is curved. Order-N
smoothing is not implemented — a repositioning smoother raises `NotImplementedError`
above order 1 rather than degrading silently.

```python
loop  = linemesh.circle(radius=2.0, n=8, order=5)   # 6 GLL nodes / arc, on the circle
export.line_to_vtu(loop, "arc.vtu")                 # VTK_LAGRANGE_CURVE (XML)
```

See [`examples/high_order_curve.py`](examples/high_order_curve.py),
[`high_order_quad.py`](examples/high_order_quad.py), and
[`high_order_hex.py`](examples/high_order_hex.py).

## Development

```bash
pip install -e ".[all,dev]"          # + ruff, mypy, pytest
ruff check nekmeshpy tests examples
mypy                                 # type-checks the whole nekmeshpy package
python -m pytest                     # golden-regression + algorithm tests
```

CI runs all three on Python 3.9–3.12. `tests/golden/` freezes the output of
`examples/bifurcation.py` — coordinates to `1e-12`, connectivity/cell types/boundary
tags byte-for-byte — so a golden diff from a refactor is a bug.

## Roadmap

- [ ] Periodic boundary
- [ ] Function based operations
- [ ] Rework tagging system
- [ ] Rework smoothing
- [x] High-order / curved elements — `order=N` on the factories (GLL nodes, curved
  `.vtu` + Nek field file; `.re2` stays linear). See
  [`examples/high_order_*.py`](examples).
- [ ] Solid–fluid conjugate mesh
