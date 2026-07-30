"""Shared numpy array type aliases used across the package.

Dtype-parametrized ``NDArray`` aliases replace bare ``np.ndarray`` (rejected by
``mypy``'s ``disallow_any_generics``): ``FloatArray`` for real data, ``IntArray``
for connectivity/indices, ``BoolArray`` for masks, ``StrArray`` for labels.

``Point``, ``Vec3`` and ``PointArray`` are shape-documentation aliases of
``FloatArray`` marking a ``(3,)`` location, a ``(3,)`` vector, or a ``(P,3)``
point array.  numpy has no static shape checking, so they document intent only.
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
PointArray = NDArray[np.float64]   # a (P,3) array of point coordinates
CurvedBlock = NDArray[np.float64]  # per-element high-order nodes (E,(N+1)^d,3)

__all__ = ["FloatArray", "IntArray", "BoolArray", "StrArray",
           "Point", "Vec3", "PointArray", "CurvedBlock"]
