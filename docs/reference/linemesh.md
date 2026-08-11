# `linemesh`

The 1-D mesh sibling of `QuadMesh` / `HexMesh`: a `(N,3)` point array plus
branching `(L,2)` line connectivity, open or closed.

```{eval-rst}
.. automodule:: nekmeshpy.linemesh
   :members:
```

## Operations

Every operation is a free function in one of these namespaces --
`linemesh.circle(...)`, not `LineMesh.circle(...)`.

```{eval-rst}
.. automodule:: nekmeshpy.linemesh.assemble
   :members:

.. automodule:: nekmeshpy.linemesh.shape
   :members:

.. automodule:: nekmeshpy.linemesh.morph
   :members:

.. automodule:: nekmeshpy.linemesh.query
   :members:

.. automodule:: nekmeshpy.linemesh.tag
   :members:
```
