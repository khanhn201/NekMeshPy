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
- **[Developer guide](https://khanhn201.github.io/NekMeshPy/developer/architecture.html)** — architecture, extension points, conventions, contributing.

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
from nekmeshpy.model.fields import uniform_spacing

# Tag the wall on the boundary loop; the tag rides up line -> quad -> hex.
boundary = LineMesh.circle(radius=1.0, n=24, element_tags=["wall"] * 24)

# Fill the loop with an O-grid section, then sweep it into a hex block.
section = QuadMesh.ogrid(boundary, n_side=6, radial=uniform_spacing(4))
block   = HexMesh.extrude(section, axis=(0, 0, 1), length=5.0,
                          layers=uniform_spacing(40),
                          first_tag="inlet", last_tag="outlet")

# Boundaries are named at build time; map each name -> Nek BC code at export.
codes = {"wall": "W  ", "inlet": "v  ", "outlet": "O  "}
export.to_re2(block, "pipe", groups=codes)     # native Nek5000/NekRS (.re2 + .rea)
export.write(block, "pipe.vtu", groups=codes)  # anything meshio supports

assert block.is_watertight()                   # closed, leak-tight, single body
assert block.is_conforming()                   # no hanging-point / T-junction faces
print(block.quality_summary())
```

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

### High-order (order-N) elements

Every factory takes an optional `order=N`: each element then carries `(N+1)`
Gauss–Lobatto–Legendre nodes per parametric direction (line `N+1`, quad `(N+1)²`,
hex `(N+1)³`), placed on the true geometry — a circle's arc nodes on the exact
circle, a shell's inner wall on the exact sphere. Corner connectivity stays
authoritative, so `.re2` export is unchanged (linear corners — Nek's re2 has no
high-order format yet) while `.vtu` export emits VTK Lagrange cells for curved
rendering. `order` defaults to `1` (plain linear elements). The XML `.vtu` writers
render Lagrange cells reliably in ParaView and VisIt.

The high-order nodes are stored **conformally**, decomposed by topology into shared
edge/face entities plus per-element interiors (module `nekmeshpy.model.conform`): two
elements meeting on an edge or face resolve to the *same* nodes, decided by corner ids
(not a coordinate search). `mesh.to_conformal()` returns `(nodes, conn)` — one global
node array with dense per-element connectivity, the high-order analog of
`points` + `quads`; the tables are also readable via `.edges`/`.edge_nodes` and (hex)
`.faces`/`.face_nodes`.

```python
loop  = LineMesh.circle(radius=2.0, n=8, order=5)   # 6 GLL nodes / arc, on the circle
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

CI runs all three on Python 3.9–3.12. See the
[contributing guide](https://khanhn201.github.io/NekMeshPy/developer/contributing.html)
for the workflow and the golden-regression invariant.

## Roadmap

- [ ] Periodic boundary
- [x] High-order / curved elements — `order=N` on the factories (GLL nodes, curved
  `.vtu`; `.re2` stays linear). See [`examples/high_order_*.py`](examples).
- [ ] Solid–fluid conjugate mesh
