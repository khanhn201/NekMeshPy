# `model`

The mesh-agnostic model: the shared-point `Mesh`, physical groups, topology
checks, and sizing fields. The order-N kernel — `model.interp` (numerics over GLL
reference nodes) and `model.conform` (topology / orientation / reconciliation
behind the B-rep) — is internal; see
[Concepts](../user/concepts.md#high-order-order-n-elements).

## `model.mesh`

```{eval-rst}
.. automodule:: nekmeshpy.model.mesh
   :members:
```

## `model.physical`

```{eval-rst}
.. automodule:: nekmeshpy.model.physical
   :members:
```

## `model.topology`

```{eval-rst}
.. automodule:: nekmeshpy.model.topology
   :members:
```

## `model.quality`

```{eval-rst}
.. automodule:: nekmeshpy.model.quality
   :members:
```

## `model.affine`

```{eval-rst}
.. automodule:: nekmeshpy.model.affine
   :members:
```

## `model.frames`

```{eval-rst}
.. automodule:: nekmeshpy.model.frames
   :members:
```

## `model.fields`

```{eval-rst}
.. automodule:: nekmeshpy.model.fields
   :members:
```

## `model.tags`

```{eval-rst}
.. automodule:: nekmeshpy.model.tags
   :members:
```
