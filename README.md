# NekMeshPy

[![docs](https://img.shields.io/badge/docs-github%20pages-blue)](https://khanhn201.github.io/NekMeshPy/)

An object-oriented, all-hex meshing **toolkit** with Nek5000/NekRS export. It
began as a port of the surface pipeline of a bifurcation hex-mesh generator
(originally MATLAB/Octave) and has been generalized into composable primitives:
a shared-point mesh model, named physical groups, `HexMesh` factories,
smoothing / surface operations, sizing fields, quality + topology checks, and
meshio I/O. Concrete geometry meshers — the bifurcation vessel pipeline,
straight pipes, external-flow domains — are **built on this toolkit and live in
[`examples/`](examples)**, not in the library itself.

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

The library is driven from Python — there is no config file or command-line
tool. Mesh containers are pure data; operations that act on a finished mesh are
free functions taking the mesh as their first argument.

```python
from nekmeshpy import HexMesh, LineMesh, QuadMesh, export
from nekmeshpy.model.fields import uniform_spacing

# Tag the wall at the lowest level — on the boundary loop itself — so the tag
# rides up the ladder (line -> quad -> hex) through construction:
boundary = LineMesh.circle(radius=1.0, n=24, element_tags=["wall"] * 24)

# Fill the loop with a butterfly O-grid section, then sweep it into a hex block:
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

Full geometry meshers built on the toolkit live in [`examples/`](examples) and
run from the repo root:

```bash
PYTHONPATH=. python examples/bifurcation.py        # bifurcation vessel (car case)
PYTHONPATH=. python examples/circular_pipe.py      # all-hex O-grid pipe
PYTHONPATH=. python examples/rectangular_pipe.py   # structured duct
PYTHONPATH=. python examples/flow_past_cylinder.py # external flow around a body
```

See the [how-to recipes](https://khanhn201.github.io/NekMeshPy/user/howto.html)
for a guided tour of each.

## Development

```bash
pip install -e ".[all,dev]"          # + ruff, mypy, pytest
ruff check nekmeshpy tests examples
mypy                                 # type-checks the whole nekmeshpy package
MPLBACKEND=Agg python -m pytest      # golden-regression + algorithm tests
```

CI runs the linter, type-checker, and test suite on Python 3.9–3.12. See the
[contributing guide](https://khanhn201.github.io/NekMeshPy/developer/contributing.html)
for the full workflow and the golden-regression invariant.

## Roadmap

- [ ] Periodic boundary
- [ ] High-order / curved elements
- [ ] Solid–fluid conjugate mesh
