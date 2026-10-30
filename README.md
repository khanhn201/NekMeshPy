# NekMeshPy

[![docs](https://img.shields.io/badge/docs-github%20pages-blue)](https://khanhn201.github.io/NekMeshPy/)

Conformal high order all-hex meshing.

**[Gallery](https://khanhn201.github.io/NekMeshPy/user/gallery.html)**

## Documentation

📖 **Full documentation: <https://khanhn201.github.io/NekMeshPy/>**

- **[Getting started](https://khanhn201.github.io/NekMeshPy/user/getting-started.html)** — install, build your first mesh, export.
- **[How-to recipes](https://khanhn201.github.io/NekMeshPy/user/howto.html)** — O-grid pipe, external flow, sphere shell, structured duct.
- **[API reference](https://khanhn201.github.io/NekMeshPy/reference/)** — every public module, class, and function.

## Install

NekMeshPy needs Python >= 3.9. Its only required dependencies are `numpy>=1.21` and `scipy>=1.7`.

```bash
git clone https://github.com/khanhn201/NekMeshPy.git
cd NekMeshPy
```

Either run straight from the checkout:

```bash
PYTHONPATH=. python examples/flow_past_hemisphere.py
```

or install it as a library (editable, so the checkout stays the source):

```bash
pip install -e .
python examples/flow_past_hemisphere.py
```

## Quick start

```python
from nekmeshpy import BoundaryConditions, writer, linemesh, quadmesh, hexmesh

n_side = 6
boundary = linemesh.circle(radius=0.5, n=4*n_side, element_tag="wall")
section = quadmesh.ogrid(boundary, n_side=n_side, radial=4)
mesh = hexmesh.extrude(section, axis=(0, 0, 1), length=5.0, layers=40,
                       first_tag="inlet", last_tag="outlet")
print(hexmesh.report(mesh))

VEL_BC = {"wall": ("W  ", 1), "inlet": ("v  ", 2), "outlet": ("O  ", 3)}  # (cbc, boundaryID)
TEMP_BC = {"wall": ("I  ", 1), "inlet": ("t  ", 2), "outlet": ("O  ", 3)}
BC = BoundaryConditions(velocity=VEL_BC, temperature=TEMP_BC)
writer.to_re2(mesh, "case.re2", bc=BC)         # native Nek5000/NekRS binary mesh
writer.to_fld(mesh, "case.f00000")             # contains high order nodes
writer.to_vtu(mesh, "case.vtu", bc=VEL_BC) # ParaView / VisIt (XML VTK)
```

## Examples

Full meshers live in [`examples/`](https://github.com/khanhn201/NekMeshPy/tree/main/examples), run from the repo root:

```bash
python examples/circular_pipe.py
python examples/flow_past_hemisphere.py
```

## Development

Optional extras of the library (declared in `pyproject.toml`):

| install | adds | needed for |
|---|---|---|
| `pip install -e ".[plot]"` | matplotlib | plotting in the examples / docs |
| `pip install -e ".[io]"` | meshio | reading the written `.vtu` back in Python (tests, scripts) |
| `pip install -e ".[mesh]"` | gmsh | `examples/femoral.py` (on Linux the gmsh wheel may also want `libGLU`) |
| `pip install -e ".[all]"` | matplotlib, meshio, pytest, gmsh | running every example and the whole test suite |
| `pip install -e ".[all,dev]"` | + ruff, mypy | development (what the CI lint job installs) |
| `pip install -e ".[docs]"` | sphinx, myst-parser, furo, sphinx-copybutton, matplotlib, meshio | building the docs |

```bash
pip install -e ".[all,dev]"          # once
ruff check nekmeshpy tests examples
mypy                                 # type-checks the whole nekmeshpy package
python -m pytest                     # algorithm tests
```

## Build and view the docs

The build needs network access: Sphinx fetches the Python/NumPy/SciPy inventories, and the gallery loads vtk.js from a CDN when a page is opened.

```bash
pip install -e ".[docs]"                                       # once
python docs/_ext/gen_viewer_assets.py                          # gallery .vtp assets
sphinx-build -b html -n -W --keep-going docs docs/_build/html  # same flags as CI
python -m http.server -d docs/_build/html 8000                 # open http://localhost:8000
```

## Roadmap

- [ ] Rework smoothing
- [ ] GUI?
- [ ] Paving algorithm / advancing front?
- [ ] Polyhedron meshing/midpoint subdivision?
- [ ] Hyperbolic meshing?
