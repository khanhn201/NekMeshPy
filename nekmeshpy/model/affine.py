"""Affine maps behind the rung-preserving ``translate`` / ``rotate`` / ``scale``.

An affine map is a pair ``(matrix, offset)`` sending a point ``p`` to
``p @ matrix.T + offset``.  The builders here return that pair; :func:`apply` is the
only place a coordinate table is actually touched, and it works on **any** array whose
trailing axis is the 3 spatial components -- ``(P,3)`` ``points``, ``(L,order-1,3)``
line interiors, ``(Q,(order-1)**2,3)`` quad interiors, ``(E,(order-1)**3,3)`` hex
interiors.  That is what lets one definition of "rotate by this angle about this axis"
serve all three rungs of the ladder: a rigid map moves every node of an element by the
same rule, so the high-order state rides along with the corners and the element stays
exactly as curved as it was.

Like ``model/conform.py`` this module **imports no container** -- everything crosses as
plain arrays.

A pure translation carries ``matrix=None`` rather than the identity.  That is not an
optimization: ``apply`` then adds the offset without a matmul, so translating by a
vector is bit-exact (and translating by ``0`` is a strict no-op), which is what keeps
``extrude`` byte-identical.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .._typing import FloatArray, Point, PointArray, Vec3

#: The world origin -- default centre for :func:`rotation` / :func:`scaling`.
ORIGIN: Point = np.array([0.0, 0.0, 0.0])
#: The ``+z`` axis -- default rotation axis, matching the sweep factories' ``axis=``.
Z_AXIS: Vec3 = np.array([0.0, 0.0, 1.0])

# An affine map: ``(matrix, offset)``, with ``matrix=None`` meaning pure translation.
Affine = tuple[FloatArray | None, Vec3]


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
    ``center`` with direction ``axis`` (right-handed; ``axis`` need not be
    normalized).  Rodrigues' formula, so the matrix is orthogonal and the map is
    rigid -- lengths, angles and element quality are preserved exactly."""
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


def _fixed_point_offset(matrix: FloatArray,
                        center: Point | Sequence[float]) -> Vec3:
    """The offset that keeps ``center`` fixed under ``matrix``: ``c - matrix @ c``."""
    c: Point = np.asarray(center, dtype=float).reshape(-1)
    if c.shape != (3,):
        raise ValueError("center must be a (3,) point, got %s" % (c.shape,))
    off: Vec3 = c - matrix @ c
    return off
