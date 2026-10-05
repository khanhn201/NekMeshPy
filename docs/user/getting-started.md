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
GROUPS = {"wall": "W  ", "inlet": "v  ", "outlet": "O  "}
THERMAL = {"wall": "I  ", "inlet": "t  ", "outlet": "O  "}
writer.to_re2(mesh, "case.re2", groups=GROUPS, thermal=THERMAL) # native Nek5000/NekRS binary mesh
writer.to_fld(mesh, "case.f00000")             # contains high order nodes
writer.to_vtu(mesh, "case.vtu", groups=GROUPS) # ParaView / VisIt (XML VTK)
```

## Visualize it

It is recommended to export and open the `.vtu` file with your favorite visualizer
(Paraview/Visit) and reload the file as you mesh to the `.vtu` file.

## Use it in Nek5000/NekRS
### Boundary condition
Currently, `writer.to_re2` export the 3 characters array `cbc` to `.re2` file.

To convert to `boundaryID`, do it in usrdat2:
```
      subroutine usrdat2

      include 'SIZE'
      include 'TOTAL'
      integer e,f
      parameter (lt=lx1*ly1*lz1*lelt)

      n = nx1*ny1*nz1*nelt

      do iel=1,nelt
      do ifc=1,2*ndim
         if (cbc(ifc,iel,1) .eq. 'W  ') boundaryID(ifc,iel) = 1
         if (cbc(ifc,iel,1) .eq. 'v  ') boundaryID(ifc,iel) = 2
         if (cbc(ifc,iel,1) .eq. 'O  ') boundaryID(ifc,iel) = 3
      enddo
      enddo

      do iel=1,nelt
      do ifc=1,2*ndim
         if (cbc(ifc,iel,2) .eq. 'I  ') boundaryIDt(ifc,iel) = 1
         if (cbc(ifc,iel,2) .eq. 't  ') boundaryIDt(ifc,iel) = 2
         if (cbc(ifc,iel,2) .eq. 'O  ') boundaryIDt(ifc,iel) = 3
      enddo
      enddo

      return
      end
```


### High order nodes
`.re2` only store the linear mesh. Higher order nodes can be exported through fld file

```python
writer.to_re2(mesh, "case.re2", groups=GROUPS)   # topology + BCs, linear
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
