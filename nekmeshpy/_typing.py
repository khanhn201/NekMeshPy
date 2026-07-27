"""Shared numpy array type aliases used across the package.

The library annotates array-valued data with dtype-parametrized
:data:`numpy.typing.NDArray` aliases rather than a bare ``np.ndarray`` so the
element type is part of the signature: :data:`FloatArray` for coordinates and
other real-valued data, :data:`IntArray` for connectivity / index arrays, and
:data:`BoolArray` for masks.  ``mypy`` is run with ``disallow_any_generics``, so
a bare ``np.ndarray`` (an implicit ``NDArray[Any]``) is an error -- use one of
these aliases, or an explicit ``NDArray[...]`` for other dtypes.

:data:`Point`, :data:`Vec3` and :data:`PointArray` are **shape-documentation**
aliases of :data:`FloatArray`: they mark a parameter that is a single ``(3,)``
location (``Point``), a single ``(3,)`` direction/vector (``Vec3``), or a
``(P,3)`` array of point coordinates (``PointArray``, as opposed to ``(N,)``
scalar data).  numpy's type system has no static shape checking, so these are
interchangeable with ``FloatArray`` to ``mypy`` -- they document intent only,
they do not enforce the shape.
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

__all__ = ["FloatArray", "IntArray", "BoolArray", "StrArray",
           "Point", "Vec3", "PointArray"]
