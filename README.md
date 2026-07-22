# NekMeshPy

An object-oriented, extensible **all-hex mesher** with Nek5000/NekRS export.

It began as a port of the surface pipeline of a bifurcation hex-mesh generator
(originally MATLAB/Octave) and has grown a gmsh-style generic core: a shared-node
mesh model, named physical groups, pluggable meshing algorithms and interior
strategies, sizing fields, quality metrics, meshio I/O, and a CLI. The
bifurcation vessel mesher is now *one* algorithm (`BifurcationMesher`);
`TransfiniteBlock` and the pipe/duct builders are others, all plugging in through
the same `HexAlgorithm` contract.

## Install

```bash
pip install -e .              # core (numpy, scipy)
pip install -e ".[all]"       # + matplotlib, meshio, PyYAML, pytest
```

## Quick start

Command line:

```bash
nekmesh mesh --interior winslow --out vessel --format re2,vtk,vtu
nekmesh pipe --shape circular --radius 0.5 --length 5 --n-axial 40 --out pipe
nekmesh quality vessel.vtu --histogram
```

Python:

```python
from nekmeshpy import Config, BifurcationMesher, export, quality

hexmesh = BifurcationMesher(Config()).run()   # returns a HexMesh
export.to_re2(hexmesh, "vessel")              # native Nek5000/NekRS (.re2 + .rea)
export.write(hexmesh, "vessel.vtu")           # anything meshio supports
print(quality.summary(*hexmesh.weld()[:2]))
```

Runnable scripts live in [`examples/`](examples). Full documentation — the
architecture diagram, module reference, extension points, and conventions — is
in [`nekmeshpy/README.md`](nekmeshpy/README.md).

## Layout

```
nekmeshpy/
├── geometry/    polyline  trimesh  quadmesh  hexmesh   (pure data containers)
├── model/       mesh  physical  quality  fields        (mesh model + metrics)
├── ops/         trisurf  interior  smoothing           (operations on containers)
├── io/          export  viz                            (export + visualization)
├── algorithms/  registry  bifurcation  cutsurface      (registry + generators)
│                ogrid  blocks  pipes
├── config.py  cli.py  __main__.py                      (config + CLI)
├── data/        bundled `car` case
└── templates/   .rea header/footer for the Nek exporter
```

The entire public API is re-exported from the top level, so
`from nekmeshpy import ...` is stable regardless of the internal layout.

## Development

```bash
pip install -e ".[all,dev]"    # + ruff, mypy, pytest
ruff check nekmeshpy tests examples
mypy                           # type-checks the public-API modules
pytest                         # golden-regression + algorithm tests
```

Numerics are pinned **byte-identical** to the reference MATLAB/Octave
implementation: `tests/` freezes the exported `.re2`/`.rea`/`.vtk` outputs in
`tests/golden/`, so every refactor stays exact. CI
(`.github/workflows/ci.yml`) runs the linter, type-checker, and test suite on
Python 3.9–3.12.

## License

MIT
