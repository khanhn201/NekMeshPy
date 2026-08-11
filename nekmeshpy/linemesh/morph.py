"""Fixed-arity, rung-preserving ``LineMesh`` operations (delta 0)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .._typing import (
    FloatArray,
    IntArray,
    Point,
    PointArray,
    Vec3,
)
from ..core import affine
from ..core.tags import PointTags
from .linemesh import LineMesh
from .query import boundary_points


def blend(a: LineMesh, b: LineMesh,
          fractions: FloatArray | Sequence[float]) -> list[LineMesh]:
    """Linearly morph between two conformal profiles ``a`` and ``b`` (equal point count
    and identical ``lines`` connectivity -- which is exactly what makes both open or
    both closed), one profile per fraction ``t`` with points ``(1-t)*a + t*b`` -- so
    ``t=0`` reproduces ``a`` and ``t=1`` reproduces ``b``.  Each profile returned carries
    ``a``'s ``point_tags`` and empty ``element_tags``."""
    A: PointArray = np.asarray(a.points, dtype=float).reshape(-1, 3)
    B: PointArray = np.asarray(b.points, dtype=float).reshape(-1, 3)
    if A.shape[0] != B.shape[0]:
        raise ValueError(
            "blend: profiles must have equal point counts (got %d, %d); build "
            "one from the other's points so they pair by index"
            % (A.shape[0], B.shape[0]))
    if not np.array_equal(a.lines, b.lines):
        raise ValueError(
            "blend: profiles must share identical connectivity (paired by index)")
    if a.order != b.order:
        raise ValueError("blend: profiles must have the same order (got %d, %d)"
                         % (a.order, b.order))
    out: list[LineMesh] = []
    for t in np.asarray(fractions, dtype=float).ravel():
        # the private interiors take the same lerp as the corners (which ride in
        # the blended points); at order 1 both interiors are empty, so this is a
        # no-op and the result equals the plain point blend.
        ia: PointArray = (1.0 - t) * a.interior + t * b.interior
        out.append(LineMesh((1.0 - t) * A + t * B, a.lines, ia,
                            point_tags=a.point_tags))
    return out


def _reverse_relabel(mesh: LineMesh) -> IntArray:
    """The point relabel :func:`reverse` walks the curve backwards through.

    An **open** chain reverses onto ``i -> N-1-i``: the last point becomes the first.
    A **closed** one cannot use that map -- it also rotates the ring by one, leaving
    line ``k`` spanning points ``k-1 -> k`` where it spanned ``k -> k+1``.  Every
    factory that meshes a loop *exactly* reads line ``k`` as the segment leaving point
    ``k``, so the rotation shifts its per-segment tags by one at order 1 and mis-pairs
    its high-order nodes above it.  Pinning point ``0`` instead (``i -> (N-i) mod N``)
    reverses the traversal and leaves the ring canonical.

    Both maps are involutions, so ``reverse`` remains its own inverse either way."""
    n = mesh.points.shape[0]
    i: IntArray = np.arange(n, dtype=np.int64)
    if n and boundary_points(mesh).size == 0:      # no degree-1 end: nothing to put last
        return (n - i) % n
    return n - 1 - i


def reverse(mesh: LineMesh) -> LineMesh:
    """The same curve traversed the other way, by relabelling rather than re-placing --
    so the high-order nodes ride along on the true geometry instead of being
    straight-subdivided between the reversed points.

    An open chain sends point ``i`` to ``N-1-i``.  A **closed** one pins point ``0`` and
    sends ``i`` to ``(N-i) mod N`` instead: the first map would also rotate the ring, so
    line ``k`` would come back spanning ``k-1 -> k`` where it spanned ``k -> k+1``, and
    every factory that meshes a loop *exactly* reads line ``k`` as the segment leaving
    point ``k``.  Both maps are involutions, so ``reverse`` is its own inverse either
    way."""
    L = mesh.lines.shape[0]
    # relabel, then restore each line's canonical start->end direction: reverse the
    # element order and swap the two endpoints of every line.
    rmap = _reverse_relabel(mesh)
    lines: IntArray = rmap[mesh.lines][::-1, ::-1]
    b = mesh.point_tags
    bnd = PointTags(L - 1 - b.elements, 3 - b.sides, b.tags).ordered()
    return LineMesh(np.ascontiguousarray(mesh.points[rmap]),
                    np.ascontiguousarray(lines),
                    np.ascontiguousarray(mesh.interior[::-1, ::-1, :]),
                    bnd,
                    mesh.element_tags.renumber(
                        (L - 1 - np.arange(L, dtype=np.int64))),
)


def _affine(mesh: LineMesh, matrix: FloatArray | None, offset: Vec3) -> LineMesh:
    """Map every coordinate of ``mesh`` through the affine pair ``(matrix, offset)``."""
    return LineMesh(affine.apply(mesh.points, matrix, offset), mesh.lines,
                    affine.apply(mesh.interior, matrix, offset),
                    point_tags=mesh.point_tags,
                    element_tags=mesh.element_tags)


def transform(mesh: LineMesh, matrix: FloatArray,
              offset: Vec3 | Sequence[float] = affine.ORIGIN) -> LineMesh:
    """A new curve with every node mapped through the affine ``p @ matrix.T + offset``.
    """
    return _affine(mesh, np.asarray(matrix, dtype=float).reshape(3, 3),
                   np.asarray(offset, dtype=float).reshape(3))


def translate(mesh: LineMesh, vector: Vec3 | Sequence[float]) -> LineMesh:
    """A new curve shifted rigidly by ``vector`` ``(3,)``.  Bit-exact: the offset is
    added without a matmul, so translating by ``0`` returns the identical
    coordinates."""
    return _affine(mesh, *affine.translation(vector))


def rotate(mesh: LineMesh, angle: float,
           axis: Vec3 | Sequence[float] = affine.Z_AXIS,
           center: Point | Sequence[float] = affine.ORIGIN) -> LineMesh:
    """A new curve rotated by ``angle`` **radians** about the line through
    ``center`` along ``axis`` (right-handed, ``axis`` need not be normalized).  The
    map is rigid, so element quality is unchanged."""
    return _affine(mesh, *affine.rotation(angle, axis, center))


def scale(mesh: LineMesh, factor: float | Vec3 | Sequence[float],
          center: Point | Sequence[float] = affine.ORIGIN) -> LineMesh:
    """A new curve scaled about ``center`` by ``factor`` -- a scalar (uniform) or a
    ``(3,)`` per-axis vector.  Every factor must be positive."""
    return _affine(mesh, *affine.scaling(factor, center))


def mirror(mesh: LineMesh, normal: Vec3 | Sequence[float],
           point: Point | Sequence[float] = affine.ORIGIN) -> LineMesh:
    """A new curve reflected through the plane with ``normal`` through ``point``.

    This is the one rung where a mirror is *only* the coordinate map.  The rungs above
    pair it with a re-winding, because a reflection has determinant ``-1`` and a quad or
    a hex has a signed Jacobian to put back the right way round; a line element has no
    such sign, and the region fills take their orientation from the loop they are handed
    rather than from its traversal direction, so a reflected loop fills exactly as its
    original did.  What *does* invert is the direction of travel -- compose with
    :func:`reverse` if a caller depends on it.

    Use it to build the half you can mesh and recover the whole:
    ``linemesh.merge([half, mirror(half, normal=(1, 0, 0))])``."""
    return _affine(mesh, *affine.reflection(normal, point))


__all__ = [
    "blend",
    "mirror",
    "reverse",
    "rotate",
    "scale",
    "transform",
    "translate",
]
