# `linemesh`

The 1-D mesh sibling of `QuadMesh` / `HexMesh`: a `(N,3)` point array plus
branching `(L,2)` line connectivity, open or closed.

```{eval-rst}
.. automodule:: nekmeshpy.linemesh
   :members:
```

## Factories

The operations that take no mesh are reached through these namespaces rather than
bound onto the class -- `linemesh.shape.circle(r, n)`, not `LineMesh.circle(r, n)`.

```{eval-rst}
.. automodule:: nekmeshpy.linemesh.assemble
   :members:

.. automodule:: nekmeshpy.linemesh.shape
   :members:

.. automodule:: nekmeshpy.linemesh.morph
   :members: blend
```
