# Getting started

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

## Build a mesh

```python
from nekmeshpy import BoundaryConditions, writer, linemesh, quadmesh, hexmesh

n_side = 6
boundary = linemesh.circle(radius=0.5, n=4*n_side, element_tag="wall")
section = quadmesh.ogrid(boundary, n_side=n_side, radial=4)
mesh = hexmesh.extrude(section, axis=(0, 0, 1), length=5.0, layers=40,
                       first_tag="inlet", last_tag="outlet")
print(hexmesh.report(mesh))
```

## Export it

Tag names map to Nek BC codes at export.

```python
VEL_BC = {"wall": ("W  ", 1), "inlet": ("v  ", 2), "outlet": ("O  ", 3)}  # (cbc, boundaryID)
TEMP_BC = {"wall": ("I  ", 1), "inlet": ("t  ", 2), "outlet": ("O  ", 3)}
BC = BoundaryConditions(velocity=VEL_BC, temperature=TEMP_BC)
writer.to_re2(mesh, "case.re2", bc=BC) # native Nek5000/NekRS binary mesh
writer.to_fld(mesh, "case.f00000")             # contains high order nodes
writer.to_vtu(mesh, "case.vtu", bc=VEL_BC) # ParaView / VisIt (XML VTK)
```

## Visualize it

It is recommended to export and open the `.vtu` file with your favorite visualizer
(Paraview/Visit) and reload the file as you mesh to the `.vtu` file.

## Use high order nodes in Nek5000/NekRS
`.re2` only store the linear mesh. Higher order nodes can be exported through fld file

```python
writer.to_re2(mesh, "case.re2", bc=VEL_BC)   # topology + BCs, linear
writer.to_fld(mesh, "case.f00000")               # contains high order nodes
```

`to_fld` writes the `X` field — the mesh's high-order GLL nodes.
Point the solver's `.par` restart at it so Nek5000/RS reads the
high-order nodes:

```
# Nek5000/NekRS .par file
[GENERAL]
startFrom = case.f00000
```

Restarting a run with velocity while keeping the high-order geometry

```
# Nek5000 .par file
[GENERAL]
startFrom = "checkpoint1.f00000 upt,case.f00000 x"

# NekRS .par file
[GENERAL]
startFrom = "checkpoint1.f00000+u+p+t,case.f00000+x"
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
