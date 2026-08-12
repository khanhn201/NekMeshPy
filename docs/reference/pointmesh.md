# `pointmesh`

The ladder's bottom rung: an `(N,3)` point array and the tags on those points. It
carries no interior, no order and no connectivity — a point *is* an element — and there
are deliberately no factories or shape model here.

It exists so the rule that holds between every other pair of rungs — **a mesh's side
tags are its rung-below's element tags** — holds at the bottom too. What a
{class}`LineMesh <nekmeshpy.linemesh.LineMesh>` reads as its `point_tags` is this
container's own `element_tags`.

```{eval-rst}
.. automodule:: nekmeshpy.pointmesh
   :members:
```

## `pointmesh.pointmesh`

```{eval-rst}
.. automodule:: nekmeshpy.pointmesh.pointmesh
   :members:
```

## Operations

```{eval-rst}
.. automodule:: nekmeshpy.pointmesh.morph
   :members:

.. automodule:: nekmeshpy.pointmesh.tag
   :members:
```
