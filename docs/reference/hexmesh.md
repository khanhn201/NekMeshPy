# `hexmesh`

The immutable all-hex volume container, its factories, the volume smoother, and
per-hex quality metrics.

```{eval-rst}
.. autoclass:: nekmeshpy.hexmesh.HexMesh
   :members:
   :show-inheritance:
```

## `hexmesh.smoothing`

```{eval-rst}
.. automodule:: nekmeshpy.hexmesh.smoothing
   :members:
```

## `hexmesh.quality`

`scaled_jacobian` here is **not** the same function as `hexmesh.scaled_jacobian`
below, despite the shared name: this one takes raw `(points, hexes)` and reads
corners only; the one re-exported on `hexmesh` (defined in `hexmesh.query`) takes a
`HexMesh` and can read the curved order-N block via `high_order=True` — the split
that matters most here, since a mesh reading `0 inverted` at the corners can still
be inverted once its curved nodes are read (see
[Concepts](../user/concepts.md#high-order-order-n-elements)). Call through
`hexmesh.scaled_jacobian`/`hexmesh.quality_summary` unless you are working with
bare arrays.

```{eval-rst}
.. automodule:: nekmeshpy.hexmesh.quality
   :members:
```

## Operations

Every operation lives in exactly one of the namespaces below, grouped by what it
does to the mesh — but each is also re-exported flat on `hexmesh` itself, so
`hexmesh.extrude(...)` and `hexmesh.lift.extrude(...)` are the *same* function, not
two. Prefer the flat form in code (`hexmesh.extrude`, not `HexMesh.extrude` — there
are no methods); the grouping below exists to help you find the right operation,
not to document it twice. See
[Architecture](../user/architecture.md#package-layout) for how a new operation is
placed into one of these.

### `hexmesh.assemble` — n-ary: builds a new numbering (`loft`/`loft_fn`/`loft_spline`/`merge`) or runs it backward (`select`/`remove`/`components`)

```{eval-rst}
.. automodule:: nekmeshpy.hexmesh.assemble
   :members:
```

### `hexmesh.lift` — Δ+1: `extrude`/`sweep`/`sweep_path`/`annulus`/`from_grid`, plus the hex-only seam joins `adapter`/`bridge`

```{eval-rst}
.. automodule:: nekmeshpy.hexmesh.lift
   :members:
```

### `hexmesh.shape` — factories: own a shape model, exact at any order (`tetra`, the curvilinear-tetrahedron octant split)

```{eval-rst}
.. automodule:: nekmeshpy.hexmesh.shape
   :members:
```

### `hexmesh.lower` — Δ−1: `boundary_mesh`, a block's boundary one rung down as a `QuadMesh` carrying the block's own nodes

```{eval-rst}
.. automodule:: nekmeshpy.hexmesh.lower
   :members:
```

### `hexmesh.morph` — Δ0: geometry at the same rung — `blend` and the affine placements `translate`/`rotate`/`scale`/`transform`/`mirror`

```{eval-rst}
.. automodule:: nekmeshpy.hexmesh.morph
   :members:
```

### `hexmesh.query` — read-only: `bounds`/`centroid`/`volume`, `quality_summary`/`scaled_jacobian` (mesh-level; see the note under `hexmesh.quality` above), topology (`report`/`tag_report`/`is_watertight`/`is_conforming`/`weld`), boundary queries

```{eval-rst}
.. automodule:: nekmeshpy.hexmesh.query
   :members:
```

### `hexmesh.tag` — vocabulary only: renames the tag tables, geometry untouched (`retag_face` to `NO_TAG` drops the row, retiring a name welded into an interior plane)

```{eval-rst}
.. automodule:: nekmeshpy.hexmesh.tag
   :members:
```
