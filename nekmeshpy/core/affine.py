"""Affine maps behind the rung-preserving ``translate`` / ``rotate`` / ``scale``."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import numpy as np

from .._typing import FloatArray, Point, PointArray, Vec3

#: The world origin -- default centre for :func:`rotation` / :func:`scaling`.
ORIGIN: Point = np.array([0.0, 0.0, 0.0])
#: The ``+z`` axis -- default rotation axis, matching the sweep factories' ``axis=``.
Z_AXIS: Vec3 = np.array([0.0, 0.0, 1.0])

# An affine map: ``(matrix, offset)``, with ``matrix=None`` meaning pure translation.
# This is a *runtime* assignment, not an annotation, so ``from __future__ import
# annotations`` does not defer it: spelling the union ``FloatArray | None`` here would
# evaluate PEP 604 eagerly and raise ``TypeError`` on the Python 3.9 CI leg.
Affine = tuple[Optional[FloatArray], Vec3]


def apply(P: PointArray, matrix: FloatArray | None, offset: Vec3) -> PointArray:
    """Map every coordinate of ``P`` (any leading shape, trailing axis 3) through the
    affine pair, returning a new array of the same shape.  With ``matrix=None`` this
    is the exact translation ``P + offset``."""
    A: PointArray = np.asarray(P, dtype=float)
    if matrix is None:
        out: PointArray = A + offset
        return out
    out = A @ np.asarray(matrix, dtype=float).T + offset
    return out


def translation(vector: Vec3 | Sequence[float]) -> Affine:
    """The affine map that shifts by ``vector`` ``(3,)``."""
    v: Vec3 = np.asarray(vector, dtype=float).reshape(-1)
    if v.shape != (3,):
        raise ValueError("translate: vector must be a (3,) displacement, got %s"
                         % (v.shape,))
    return None, v


def scaling(factor: float | Vec3 | Sequence[float],
            center: Point | Sequence[float] = ORIGIN) -> Affine:
    """The affine map that scales about ``center`` by ``factor`` -- a scalar (uniform)
    or a ``(3,)`` per-axis vector.  A zero or negative component is rejected: it would
    collapse or invert every element."""
    f: FloatArray = np.asarray(factor, dtype=float).reshape(-1)
    if f.shape == (1,):
        f = np.repeat(f, 3)
    if f.shape != (3,):
        raise ValueError(
            "scale: factor must be a scalar or a (3,) per-axis vector, got %s"
            % (np.shape(factor),))
    if not np.all(f > 0.0):
        raise ValueError("scale: every factor must be positive (got %s); a zero or "
                         "negative factor collapses or inverts every element" % (f,))
    return np.diag(f), _fixed_point_offset(np.diag(f), center)


def rotation(angle: float, axis: Vec3 | Sequence[float] = Z_AXIS,
             center: Point | Sequence[float] = ORIGIN) -> Affine:
    """The affine map that rotates by ``angle`` **radians** about the line through
    ``center`` with direction ``axis`` (right-handed; ``axis`` need not be normalized).
    """
    k: Vec3 = np.asarray(axis, dtype=float).reshape(-1)
    if k.shape != (3,):
        raise ValueError("rotate: axis must be a (3,) direction, got %s" % (k.shape,))
    n = float(np.linalg.norm(k))
    if n == 0.0:
        raise ValueError("rotate: axis must be non-zero")
    k = k / n
    c, s = np.cos(float(angle)), np.sin(float(angle))
    K: FloatArray = np.array([[0.0, -k[2], k[1]],
                              [k[2], 0.0, -k[0]],
                              [-k[1], k[0], 0.0]])
    R: FloatArray = c * np.eye(3) + s * K + (1.0 - c) * np.outer(k, k)
    return R, _fixed_point_offset(R, center)


def reflection(normal: Vec3 | Sequence[float],
               point: Point | Sequence[float] = ORIGIN) -> Affine:
    """The affine map that mirrors through the plane with unit ``normal`` (need not be
    normalized) passing through ``point`` -- the Householder matrix ``I - 2 n n^T``.

    Its determinant is ``-1``, so it turns every element inside out: this is the map
    ``mirror`` applies to the coordinates, *paired* with a re-winding of the
    connectivity.  Handing it to ``transform`` alone leaves an inverted mesh."""
    n: Vec3 = np.asarray(normal, dtype=float).reshape(-1)
    if n.shape != (3,):
        raise ValueError("mirror: normal must be a (3,) direction, got %s" % (n.shape,))
    norm = float(np.linalg.norm(n))
    if norm == 0.0:
        raise ValueError("mirror: normal must be non-zero -- it names a plane")
    n = n / norm
    M: FloatArray = np.eye(3) - 2.0 * np.outer(n, n)
    return M, _fixed_point_offset(M, point)


def _fixed_point_offset(matrix: FloatArray,
                        center: Point | Sequence[float]) -> Vec3:
    """The offset that keeps ``center`` fixed under ``matrix``: ``c - matrix @ c``."""
    c: Point = np.asarray(center, dtype=float).reshape(-1)
    if c.shape != (3,):
        raise ValueError("center must be a (3,) point, got %s" % (c.shape,))
    off: Vec3 = c - matrix @ c
    return off
