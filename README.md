# NekMeshPy

[![docs](https://img.shields.io/badge/docs-github%20pages-blue)](https://khanhn201.github.io/NekMeshPy/)

Conformal high order all-hex meshing.

**[Gallery](https://khanhn201.github.io/NekMeshPy/user/gallery.html)**

## Documentation

📖 **Full documentation: <https://khanhn201.github.io/NekMeshPy/>**

- **[Getting started](https://khanhn201.github.io/NekMeshPy/user/getting-started.html)** — install, build your first mesh, export.
- **[Concepts](https://khanhn201.github.io/NekMeshPy/user/concepts.html)** — the line→quad→hex ladder, tags, section factories, smoothing, high-order geometry.
- **[How-to recipes](https://khanhn201.github.io/NekMeshPy/user/howto.html)** — O-grid pipe, external flow, sphere shell, structured duct.
- **[API reference](https://khanhn201.github.io/NekMeshPy/reference/)** — every public module, class, and function.
- **[Architecture](https://khanhn201.github.io/NekMeshPy/user/architecture.html)** — how the toolkit is laid out and why.
- **[Conventions](https://khanhn201.github.io/NekMeshPy/user/conventions.html)** — typing, naming, and the invariants the code holds to.

## Install

```bash
pip install -e .              # core (numpy, scipy)
pip install -e ".[all]"       # + matplotlib, meshio, pytest, gmsh
```

## Quick start

```python
from nekmeshpy import writer, linemesh, quadmesh, hexmesh

# Tag the wall on the boundary loop; the tag rides up line -> quad -> hex.
boundary = linemesh.circle(radius=1.0, n=24, element_tag="wall")

# Fill the loop with an O-grid section, then sweep it into a hex block.
# radial / layers count *cells*: an int is n uniform layers; pass an explicit
# array of normalized positions (geometric_spacing(n, r), ...) to grade them.
section = quadmesh.ogrid(boundary, n_side=6, radial=4)
block   = hexmesh.extrude(section, axis=(0, 0, 1), length=5.0, layers=40,
                          first_tag="inlet", last_tag="outlet")

# Boundaries are named at build time; map each name -> Nek BC code at export.
codes = {"wall": "W  ", "inlet": "v  ", "outlet": "O  "}
writer.to_re2(block, "pipe.re2", groups=codes) # native Nek5000/NekRS binary mesh
writer.write(block, "pipe.vtu", groups=codes)  # anything meshio supports

assert hexmesh.is_watertight(block)            # closed, leak-tight, single body
assert hexmesh.is_conforming(block)            # no hanging-point / T-junction faces
print(block)                                # <HexMesh 5945 points, 5280 hexes, order 1, ...>
print(hexmesh.quality_summary(block).min)   # a QualitySummary NamedTuple, not a dict
```

For sweeping (`loft`/`sweep`) and high-order geometry, see
[Concepts](https://khanhn201.github.io/NekMeshPy/user/concepts.html).

## Examples

Full meshers live in [`examples/`](examples), run from the repo root:

```bash
PYTHONPATH=. python examples/carotid.py            # carotid vessel (car case)
PYTHONPATH=. python examples/femoral.py            # femoral-style T-junction, surface and all
PYTHONPATH=. python examples/circular_pipe.py      # all-hex O-grid pipe
PYTHONPATH=. python examples/flow_past_cylinder.py # external flow around a body
```

See the [how-to recipes](https://khanhn201.github.io/NekMeshPy/user/howto.html)
for a guided tour of each.

## Development

```bash
pip install -e ".[all,dev]"          # + ruff, mypy, pytest
ruff check nekmeshpy tests examples
mypy                                 # type-checks the whole nekmeshpy package
python -m pytest                     # golden-regression + algorithm tests
```

CI runs lint and mypy once on 3.12 and the test run on 3.9–3.12; `docs.yml` builds
the docs site separately. `examples/femoral.py` ships and is maintained but is not
part of that test run at all — it tet-meshes with gmsh, costs 316 s cold, and gmsh's
own non-determinism makes its element-quality check unreliable regardless (see
`CLAUDE.md`). Run it directly to try it: `PYTHONPATH=. python examples/femoral.py`.
`tests/golden/` freezes the output of `examples/carotid.py` — coordinates to
`1e-12`, connectivity/cell types/boundary tags byte-for-byte — so a golden diff
from a refactor is a bug.

## Roadmap

- [x] Periodic boundary
- [ ] Rework smoothing
- [ ] GUI?
- [ ] Paving algorithm / advancing front?
- [ ] Polyhedron meshing/midpoint subdivision?
- [ ] Hyperbolic meshing?
