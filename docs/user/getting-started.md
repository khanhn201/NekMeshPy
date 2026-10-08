# Getting started

## Quick start
```bash
git clone https://github.com/khanhn201/NekMeshPy.git
cd NekMeshPy
PYTHONPATH=. python examples/flow_past_hemisphere.py
```

## Build a mesh

```python
from nekmeshpy import writer, linemesh, quadmesh, hexmesh

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
startFrom = "checkpoint1.f00000 v,case.f00000 x"

# NekRS .par file
[GENERAL]
startFrom = "checkpoint1.f00000+v,case.f00000+x"
```
