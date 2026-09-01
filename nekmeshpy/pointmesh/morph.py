"""Fixed-arity, rung-preserving ``PointMesh`` operations (delta 0).

A point has no interior and no connectivity to re-wind, so this is the affine family
alone -- :func:`linemesh.morph <nekmeshpy.linemesh.morph>` and the rungs above build
their own ``transform`` / ``translate`` / ... on top of these for the shared points,
then map their private ``interior`` nodes through the same affine pair.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .._typing import FloatArray, Point, PointArray, Vec3
from ..core import affine
from .pointmesh import PointMesh

# An arbitrary non-affine node map, (N, 3) -> (N, 3) -- what transform_fn and its
# rung-above counterparts hand every stored node block, the curved-warp counterpart
# of an affine (matrix, offset).
NodeMap = Callable[[PointArray], PointArray]


def _mapped(arr: PointArray, fn: NodeMap) -> PointArray:
    """``fn`` applied to an ``(..., 3)`` node block, shape preserved.  An empty block
    (order 1 has no ``interior``) passes straight through without calling ``fn``."""
    a: PointArray = np.asarray(arr, dtype=float)
    if a.size == 0:
        return a
    out: PointArray = np.asarray(fn(a.reshape(-1, 3)), dtype=float)
    if out.shape != (a.size // 3, 3):
        raise ValueError(
            "transform_fn: fn must map (N, 3) -> (N, 3); got %s for N = %d"
            % (out.shape, a.size // 3))
    return out.reshape(a.shape)


def _affine(mesh: PointMesh, matrix: FloatArray | None, offset: Vec3) -> PointMesh:
    """Map every coordinate of ``mesh`` through the affine pair ``(matrix, offset)``."""
    return PointMesh(affine.apply(mesh.points, matrix, offset), mesh.element_tags)


def transform(mesh: PointMesh, matrix: FloatArray,
              offset: Vec3 | Sequence[float] = affine.ORIGIN) -> PointMesh:
    """The points mapped through the affine ``p @ matrix.T + offset``."""
    return _affine(mesh, np.asarray(matrix, dtype=float).reshape(3, 3),
                   np.asarray(offset, dtype=float).reshape(3))


def transform_fn(mesh: PointMesh, fn: NodeMap) -> PointMesh:
    """The points mapped through an arbitrary ``fn`` -- ``transform`` for a warp that
    is not affine.  ``fn`` is called once with the whole ``(N, 3)`` point array and
    must return ``(N, 3)``.  Tags ride through verbatim.

    A point has no orientation and no signed measure, so -- like :func:`mirror` at this
    rung -- this is *only* the coordinate map: an ``fn`` that folds space is not caught
    here, though :func:`quadmesh.morph.transform_fn
    <nekmeshpy.quadmesh.morph.transform_fn>` and the hex rung above do warn once there
    is a Jacobian to check."""
    return PointMesh(_mapped(mesh.points, fn), mesh.element_tags)


def translate(mesh: PointMesh, vector: Vec3 | Sequence[float]) -> PointMesh:
    """The points shifted rigidly by ``vector`` ``(3,)``.  Bit-exact: the offset is
    added without a matmul, so translating by ``0`` returns the identical
    coordinates."""
    return _affine(mesh, *affine.translation(vector))


def rotate(mesh: PointMesh, angle: float,
           axis: Vec3 | Sequence[float] = affine.Z_AXIS,
           center: Point | Sequence[float] = affine.ORIGIN) -> PointMesh:
    """The points rotated by ``angle`` **radians** about the line through ``center``
    along ``axis`` (right-handed, ``axis`` need not be normalized)."""
    return _affine(mesh, *affine.rotation(angle, axis, center))


def scale(mesh: PointMesh, factor: float | Vec3 | Sequence[float],
          center: Point | Sequence[float] = affine.ORIGIN) -> PointMesh:
    """The points scaled about ``center`` by ``factor`` -- a scalar (uniform) or a
    ``(3,)`` per-axis vector.  Every factor must be positive."""
    return _affine(mesh, *affine.scaling(factor, center))


def mirror(mesh: PointMesh, normal: Vec3 | Sequence[float],
           point: Point | Sequence[float] = affine.ORIGIN) -> PointMesh:
    """The points reflected through the plane with ``normal`` through ``point``.

    A point has no orientation to re-wind, so unlike the rungs above this is *only*
    the coordinate map -- the same exception :func:`linemesh.morph.mirror
    <nekmeshpy.linemesh.morph.mirror>` documents for the line rung, one rung further
    down."""
    return _affine(mesh, *affine.reflection(normal, point))


__all__ = [
    "NodeMap",
    "mirror",
    "rotate",
    "scale",
    "transform",
    "transform_fn",
    "translate",
]
