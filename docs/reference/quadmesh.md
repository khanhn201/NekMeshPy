# `quadmesh`

The quad cross-section container, its factories, the smoothing registry, and
per-quad quality metrics.

```{eval-rst}
.. autoclass:: nekmeshpy.quadmesh.QuadMesh
   :members:
   :show-inheritance:

.. autodata:: nekmeshpy.quadmesh.NO_TAG
```

## `quadmesh.smoothing`

```{eval-rst}
.. automodule:: nekmeshpy.quadmesh.smoothing
   :members:
```

## `quadmesh.quality`

`scaled_jacobian` here is **not** the same function as `quadmesh.scaled_jacobian`
below, despite the shared name: this one takes raw `(points, quads)` and reads
corners only; the one re-exported on `quadmesh` (defined in `quadmesh.query`) takes
a `QuadMesh` and can read the curved order-N block via `high_order=True`. Call
through `quadmesh.scaled_jacobian`/`quadmesh.quality_summary` unless you are
working with bare arrays.

```{eval-rst}
.. automodule:: nekmeshpy.quadmesh.quality
   :members:
```

## Operations

Every operation lives in exactly one of the namespaces below, grouped by what it
does to the mesh — but each is also re-exported flat on `quadmesh` itself, so
`quadmesh.circle(...)` and `quadmesh.shape.circle(...)` are the *same* function,
not two. Prefer the flat form in code (`quadmesh.circle`, not `QuadMesh.circle` —
there are no methods); the grouping below exists to help you find the right
operation, not to document it twice. See
[Architecture](../user/architecture.md#package-layout) for how a new operation is
placed into one of these.

### `quadmesh.assemble` — n-ary: builds a new numbering (`loft`/`loft_fn`/`loft_spline`/`merge`) or runs it backward (`select`/`remove`/`components`)

```{eval-rst}
.. automodule:: nekmeshpy.quadmesh.assemble
   :members:
```

### `quadmesh.lift` — Δ+1: `extrude`/`sweep`/`sweep_path`/`annulus`/`from_grid` onto `HexMesh`

```{eval-rst}
.. automodule:: nekmeshpy.quadmesh.lift
   :members:
```

### `quadmesh.shape` — factories: region fills (`ogrid`/`quadrant_ogrid`/`spined_ogrid`/…) and closed surfaces (`box`/`sphere`/…), exact at any order

```{eval-rst}
.. automodule:: nekmeshpy.quadmesh.shape
   :members:
```

### `quadmesh.ports` — `Port`: a section plus the outward direction and axis point it cannot state about itself, for the rung above to check a join against rather than guess

```{eval-rst}
.. automodule:: nekmeshpy.quadmesh.ports
   :members:
```

### `quadmesh.lower` — Δ−1: `boundary_mesh`, the boundary one rung down as a `LineMesh`

```{eval-rst}
.. automodule:: nekmeshpy.quadmesh.lower
   :members:
```

### `quadmesh.morph` — Δ0: geometry at the same rung — `blend`, `reindex`, `place_on_path`, and the affine placements `translate`/`rotate`/`scale`/`transform`/`mirror`

```{eval-rst}
.. automodule:: nekmeshpy.quadmesh.morph
   :members:
```

### `quadmesh.query` — read-only: `bounds`/`centroid`/`area`, `quality_summary`/`scaled_jacobian` (mesh-level; see the note under `quadmesh.quality` above), `plane_normal`, boundary queries

```{eval-rst}
.. automodule:: nekmeshpy.quadmesh.query
   :members:
```

### `quadmesh.tag` — vocabulary only: renames the tag tables, geometry untouched

```{eval-rst}
.. automodule:: nekmeshpy.quadmesh.tag
   :members:
```
