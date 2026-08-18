# `core`

The mesh-agnostic layer: the shared-point `Mesh`, physical groups, topology
checks, and sizing fields. The order-N kernel — `core.interp` (numerics over GLL
reference nodes) and `core.conform` (topology / orientation / reconciliation
behind the B-rep) — is internal; see
[Concepts](../user/concepts.md#high-order-order-n-elements).

## `core.mesh`

```{eval-rst}
.. automodule:: nekmeshpy.core.mesh
   :members:
```

## `core.physical`

```{eval-rst}
.. automodule:: nekmeshpy.core.physical
   :members:
```

## `core.topology`

```{eval-rst}
.. automodule:: nekmeshpy.core.topology
   :members:
```

## `core.quality`

```{eval-rst}
.. automodule:: nekmeshpy.core.quality
   :members:
```

## `core.measure`

```{eval-rst}
.. automodule:: nekmeshpy.core.measure
   :members:
```

## `core.affine`

```{eval-rst}
.. automodule:: nekmeshpy.core.affine
   :members:
```

## `core.frames`

```{eval-rst}
.. automodule:: nekmeshpy.core.frames
   :members:
```

## `core.paths`

`Move`, `UpSpec` and `Orientation` — the union of the three move types, what a sweep
accepts for `up`, and the frame-generator names — are excluded below: they are `typing`
aliases, and autodoc renders those through `typing`'s own docstring, which does not
survive `-n`. `line`, `arc` and `helix` below document the move forms themselves.

```{eval-rst}
.. automodule:: nekmeshpy.core.paths
   :members:
   :exclude-members: Move, Orientation, UpSpec
```

## `core.surfaces`

`SurfaceMap` — a `(K,2) -> (K,3)` callable — is excluded below for the same
reason as `paths.Move`: it is a `typing` alias, and autodoc renders those through
`typing`'s own docstring, which does not survive `-n`.

```{eval-rst}
.. automodule:: nekmeshpy.core.surfaces
   :members:
   :exclude-members: SurfaceMap
```

## `core.fields`

```{eval-rst}
.. automodule:: nekmeshpy.core.fields
   :members:
```

## `core.tags`

```{eval-rst}
.. automodule:: nekmeshpy.core.tags
   :members:
```
