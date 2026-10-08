# NekMeshPy

A high-order all-hex meshing toolkit.

```python
from nekmeshpy import writer, linemesh, quadmesh, hexmesh

n_side = 6
boundary = linemesh.circle(radius=0.5, n=4*n_side, element_tag="wall")
section = quadmesh.ogrid(boundary, n_side=n_side, radial=4)
mesh = hexmesh.extrude(section, axis=(0, 0, 1), length=5.0, layers=40,
                       first_tag="inlet", last_tag="outlet")
print(hexmesh.report(mesh))

VEL_BC = {"wall": ("W  ", 1), "inlet": ("v  ", 2), "outlet": ("O  ", 3)}  # (cbc, boundaryID)
TEMP_BC = {"wall": ("I  ", 1), "inlet": ("t  ", 2), "outlet": ("O  ", 3)}
BC = BoundaryConditions(velocity=VEL_BC, temperature=TEMP_BC)
writer.to_re2(mesh, "case.re2", bc=BC) # native Nek5000/NekRS binary mesh
writer.to_fld(mesh, "case.f00000")             # contains high order nodes
writer.to_vtu(mesh, "case.vtu", bc=VEL_BC) # ParaView / VisIt (XML VTK)
```

## Where to go next

- **{doc}`user/getting-started`**
- **{doc}`user/howto`**
- **{doc}`user/gallery`**
- **{doc}`reference/index`**

```{toctree}
:hidden:
:caption: Guide

user/getting-started
user/howto
user/gallery
```

```{toctree}
:hidden:
:caption: Reference

reference/index
```
