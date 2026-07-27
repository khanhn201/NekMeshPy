# NekMeshPy

An object-oriented, all-hex meshing **toolkit** with Nek5000/NekRS export.
The library is composable primitives; concrete geometry meshers live in
[`examples/`](examples).

## TODO
- [ ] Periodic boundary

## Install

```bash
pip install -e .              # core (numpy, scipy)
pip install -e ".[all]"       # + matplotlib, meshio, pytest
```

## Quick start

Build a hex mesh from the toolkit primitives — a shared point pool, `HexMesh`
factories (`extrude` / `loft` / `merge` / `from_grid`), quality, topology, and export:

```python
from nekmeshpy import HexMesh, QuadMesh, export
from nekmeshpy.model.fields import uniform_spacing

# sweep one QuadMesh section along a straight axis:
section = QuadMesh.ogrid(boundary, n_side=6, radial=uniform_spacing(4))
# layers gives the section-plane positions in [0,1] (initial explicit): here
# 40 uniform layers 0->1; e.g. numpy.linspace(0.5, 1, 21) to sweep only the far half
mesh = HexMesh.extrude(section, axis=(0, 0, 1), length=5.0,
                       layers=uniform_spacing(40),
                       first_cap_tag=1, last_cap_tag=2, wall_tag=3)
# ...or loft a stack of pre-positioned cross-section profiles:
# mesh = HexMesh.loft(slices, first_cap_tag=1, last_cap_tag=2, wall_tag=3)
assert mesh.is_watertight() and mesh.is_conforming()
export.to_re2(mesh, "part")                   # native Nek5000/NekRS (.re2 + .rea)
export.write(mesh, "part.vtu")                # anything meshio supports
print(mesh.quality_summary())
```

Complete geometry meshers built on the toolkit are runnable scripts in
[`examples/`](examples) — e.g. the bifurcation vessel mesher:

```bash
PYTHONPATH=. python examples/bifurcation.py     # the bundled `car` case
PYTHONPATH=. python examples/circular_pipe.py
```

Full documentation — the architecture diagram, module reference, extension
points, and conventions — is in [`nekmeshpy/README.md`](nekmeshpy/README.md).

## Layout

```
nekmeshpy/                                          the toolkit (library)
├── geometry/    points  curve  trimesh  quadmesh  hexmesh  (data containers)
│                (QuadMesh.structured / ogrid / half_ogrid = quad section factories)
├── model/       mesh  physical  quality  topology  fields    (model + metrics)
├── ops/         trisurf  interior  smoothing               (operations on containers)
├── io/          export  viz                                (export + visualization)
└── templates/   .rea header/footer for the Nek exporter
examples/                                           flat gmsh-style meshing scripts
├── bifurcation.py        vessel surface pipeline (car case)
├── circular_pipe.py  rectangular_pipe.py  transfinite_block.py
└── data/                 bundled car surface (car.vtx, car.tri)
```

The toolkit's public API is re-exported from the top level, so
`from nekmeshpy import ...` is stable regardless of the internal layout.

## Development

```bash
pip install -e ".[all,dev]"    # + ruff, mypy, pytest
ruff check nekmeshpy tests examples
mypy                           # type-checks the whole nekmeshpy package
pytest                         # golden-regression + algorithm tests
```

Numerics are pinned by a golden-regression suite: `tests/` freezes the exported
`.re2`/`.rea`/`.vtk` outputs in `tests/golden/` (`.rea` and the `.re2` boundary
block byte-exact, `.re2` coordinates to `1e-12`, `.vtk` byte-identical), so every
refactor stays exact. CI (`.github/workflows/ci.yml`) runs the linter,
type-checker, and test suite on Python 3.9–3.12.
