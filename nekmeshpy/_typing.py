"""Shared numpy array type aliases used across the package.

Dtype-parametrized ``NDArray`` aliases replace bare ``np.ndarray`` (rejected by
``mypy``'s ``disallow_any_generics``): ``FloatArray`` for real data, ``IntArray``
for connectivity/indices, ``BoolArray`` for masks, ``StrArray`` for labels.

``Point``, ``Vec3`` and ``PointArray`` are shape-documentation aliases of
``FloatArray``: ``Point`` marks a single ``(3,)`` location, ``Vec3`` a single
``(3,)`` direction, and ``PointArray`` **any** array of point coordinates whose
trailing axis is the 3 spatial components -- with any leading shape, so ``(P,3)``,
``(L,order-1,3)``, ``(ni+1,nj+1,3)`` and ``(E,6,(order-1)**2,3)`` are all
``PointArray``.  The concrete shape belongs in the docstring of the field or
parameter; the alias deliberately does not encode it.  numpy has no static shape
checking, so all three document intent only.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
StrArray = NDArray[np.str_]         # per-element string labels (boundary names)

# shape-documentation aliases (not shape-enforced by mypy)
Point = NDArray[np.float64]        # a single (3,) location
Vec3 = NDArray[np.float64]         # a single (3,) direction/vector
PointArray = NDArray[np.float64]   # point coordinates: any leading shape + trailing 3

__all__ = ["FloatArray", "IntArray", "BoolArray", "StrArray",
           "Point", "Vec3", "PointArray"]
