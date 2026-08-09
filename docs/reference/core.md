# `model`

The mesh-agnostic model: the shared-point `Mesh`, physical groups, topology
checks, and sizing fields. The order-N kernel — `model.interp` (numerics over GLL
reference nodes) and `model.conform` (topology / orientation / reconciliation
behind the B-rep) — is internal; see
[Concepts](../user/concepts.md#high-order-order-n-elements).

## `model.mesh`

```{eval-rst}
.. automodule:: nekmeshpy.core.mesh
   :members:
```

## `model.physical`

```{eval-rst}
.. automodule:: nekmeshpy.core.physical
   :members:
```

## `model.topology`

```{eval-rst}
.. automodule:: nekmeshpy.core.topology
   :members:
```

## `model.quality`

```{eval-rst}
.. automodule:: nekmeshpy.core.quality
   :members:
```

## `model.affine`

```{eval-rst}
.. automodule:: nekmeshpy.core.affine
   :members:
```

## `model.frames`

```{eval-rst}
.. automodule:: nekmeshpy.core.frames
   :members:
```

## `model.paths`

`Move` — one row of a move table, `("line", length, 0.0)` or `("arc", radius,
signed_degrees)` — is excluded below: it is a `typing.Tuple` alias, and autodoc
renders those through `typing`'s own docstring, which does not survive `-n`. The
module docstring describes the two forms.

```{eval-rst}
.. automodule:: nekmeshpy.core.paths
   :members:
   :exclude-members: Move
```

## `model.surfaces`

`SurfaceMap` — a `(K,2) -> (K,3)` callable — is excluded below for the same
reason as `paths.Move`: it is a `typing` alias, and autodoc renders those through
`typing`'s own docstring, which does not survive `-n`.

```{eval-rst}
.. automodule:: nekmeshpy.core.surfaces
   :members:
   :exclude-members: SurfaceMap
```

## `model.fields`

```{eval-rst}
.. automodule:: nekmeshpy.core.fields
   :members:
```

## `model.tags`

```{eval-rst}
.. automodule:: nekmeshpy.core.tags
   :members:
```
