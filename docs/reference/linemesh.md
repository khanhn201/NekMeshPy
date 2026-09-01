# `linemesh`

The 1-D mesh sibling of `QuadMesh` / `HexMesh`: a `(N,3)` point array plus
branching `(L,2)` line connectivity, open or closed.

```{eval-rst}
.. autoclass:: nekmeshpy.linemesh.LineMesh
   :members:
   :show-inheritance:
```

## Operations

Every operation lives in exactly one of the namespaces below, grouped by what it
does to the mesh — but each is also re-exported flat on `linemesh` itself, so
`linemesh.circle(...)` and `linemesh.shape.circle(...)` are the *same* function,
not two. Prefer the flat form in code (`linemesh.circle`, not `LineMesh.circle` —
there are no methods); the grouping below exists to help you find the right
operation, not to document it twice. See
[Architecture](../user/architecture.md#package-layout) for how a new operation is
placed into one of these.

### `linemesh.assemble` — n-ary: builds a new numbering (`loft`/`loft_fn`/`loft_spline`/`merge`/`attach`) or runs it backward (`select`/`remove`/`components`)

```{eval-rst}
.. automodule:: nekmeshpy.linemesh.assemble
   :members:
```

### `linemesh.shape` — factories: own a shape model, exact at any order

```{eval-rst}
.. automodule:: nekmeshpy.linemesh.shape
   :members:
```

### `linemesh.morph` — Δ0: geometry at the same rung — `blend`, `reverse`, `offset`, the affine placements `translate`/`rotate`/`scale`/`transform`/`mirror`, and `transform_fn` (a non-affine warp)

```{eval-rst}
.. automodule:: nekmeshpy.linemesh.morph
   :members:
```

### `linemesh.query` — read-only: `bounds`/`centroid`/`length`, boundary queries, `element_blocks`

```{eval-rst}
.. automodule:: nekmeshpy.linemesh.query
   :members:
```

### `linemesh.tag` — vocabulary only: renames the tag tables, geometry untouched

```{eval-rst}
.. automodule:: nekmeshpy.linemesh.tag
   :members:
```
