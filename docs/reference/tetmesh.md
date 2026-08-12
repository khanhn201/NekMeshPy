# `tetmesh`

A linear tetrahedral volume-mesh container and its volumetric algorithms (reached as
`nekmeshpy.tetmesh.ops`), mirroring `trimesh`'s flat shape one dimension up. Unlike the
point/line/quad/hex ladder, a `TetMesh` carries no order, no interior nodes and no
tags: it exists to solve a field or walk a volume, not to be exported -- a mesher's
actual output is the hex mesh built *around* what the solve decided.

gmsh generation ({func}`tetmesh.ops.tet_mesh <nekmeshpy.tetmesh.ops.tet_mesh>`) is the
only place in the toolkit that imports `gmsh`, and it is deferred to call time, so
importing `nekmeshpy` never requires it (`pip install .[mesh]`).

## `tetmesh`

```{eval-rst}
.. automodule:: nekmeshpy.tetmesh
   :members:
```

## `tetmesh.tetmesh`

```{eval-rst}
.. automodule:: nekmeshpy.tetmesh.tetmesh
   :members:
```

## `tetmesh.ops`

```{eval-rst}
.. automodule:: nekmeshpy.tetmesh.ops
   :members:
```
