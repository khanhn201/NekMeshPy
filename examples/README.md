# Examples

Runnable scripts demonstrating the NekMeshPy API. Install the package first
(`pip install -e .` from the repo root), then run any script from the repo root:

```bash
python examples/circular_pipe.py       # all-hex O-grid circular pipe
python examples/rectangular_pipe.py    # structured rectangular duct
python examples/bifurcation.py         # the bifurcation vessel surface mesher
```

Each writes native Nek5000/NekRS `.re2`/`.rea` plus a `.vtk` for viewing in
ParaView.

## The same via the CLI

```bash
nekmesh pipe --shape circular    --radius 0.5 --length 5 --n-axial 40 \
             --n-radial 4 --radial-grading 1.15 --out circular_pipe
nekmesh pipe --shape rectangular --width 2 --height 1 --length 6 \
             --nx 16 --ny 8 --n-axial 48 --out rectangular_pipe --format re2,vtk,vtu
```

## The same via the algorithm registry (gmsh-style)

```python
from nekmeshpy import make, export
mesh = make("circular_pipe", radius=0.5, length=5.0, n_axial=40).run()
export.to_re2(mesh, "pipe")
```
