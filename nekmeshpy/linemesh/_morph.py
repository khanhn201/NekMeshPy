"""Fixed-arity, rung-preserving ``LineMesh`` operations (delta 0).

Two arities live here.  **Binary**: ``blend`` morphs between two index-paired
profiles.  **Unary**: ``translate`` / ``rotate`` / ``scale`` / ``transform`` place a
finished curve, and ``reverse`` flips its traversal direction.  All but ``reverse``
change only coordinates -- ``a``'s ``lines`` connectivity and ``point_tags`` ride
through verbatim, so the input's numbering *is* the output's
and nothing is re-derived (the placements keep ``element_tags`` too; ``blend`` leaves
them for the consuming factory).  ``reverse`` is the mirror image: it moves no
coordinate and instead relabels the index space with the bijection ``i -> N-1-i``,
which is still rung-preserving and still invents no numbering.

Free functions assigned into the :class:`~nekmeshpy.LineMesh` class body (see
``linemesh.py``) -- the binary ``blend`` wrapped in ``staticmethod``, the unary
placements bare, so ``lm.translate(v)`` reads as it should.  Internal toolkit code
imports them from here directly.  Each builds its result with ``type(mesh)`` rather
than naming ``LineMesh``: the container imports this module, so a runtime import of
it here would be a cycle.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from .._typing import (
    FloatArray,
    IntArray,
    Point,
    PointArray,
    Vec3,
)
from ..model import affine
from ..model.tags import PointTags

if TYPE_CHECKING:                    # the container imports us, so this cannot be
    from .linemesh import LineMesh  # a runtime import -- annotations only


def blend(a: LineMesh, b: LineMesh,
          fractions: FloatArray | Sequence[float]) -> list[LineMesh]:
    """Linearly morph between two conformal profiles ``a`` and ``b`` (equal point
    count and identical ``lines`` connectivity -- which is exactly what makes
    both open or both closed), one profile per fraction
    ``t`` with points ``(1-t)*a + t*b`` -- so ``t=0`` reproduces ``a`` and ``t=1``
    reproduces ``b``.  Each result carries ``a``'s connectivity and ``point_tags``
    (positional BC markers follow the morph); per-element
    ``element_tags`` are left for the consuming factory/``loft`` to assign, so a
    blended stack feeds straight into ``loft`` or a section factory.  This is the
    profile-positioning step behind ``annulus`` (and any morphing sweep).

    High-order profiles morph too: ``a``/``b`` share the same
    order ``N`` (so their private per-line ``interior`` nodes pair by index) and
    each result carries the blended interior ``(1-t)*a.interior + t*b.interior`` --
    the same lerp the corners get from the blended ``points``, so the result stays
    corner-consistent by construction and a high-order blended stack feeds ``loft``
    unchanged.  At order 1 ``interior`` is empty and the result is byte-identical
    to the plain linear morph."""
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
        out.append(type(a)((1.0 - t) * A + t * B, a.lines, ia,
                           point_tags=a.point_tags, order=a.order))
    return out


def reverse(mesh: LineMesh) -> LineMesh:
    """The same curve traversed the other way: point ``i`` becomes point ``N-1-i``.

    Every coordinate is carried over unchanged -- this is a **relabelling**, not a
    move -- so the result is the identical geometry with the opposite orientation.
    That includes the high-order state: each line's private ``interior`` flips on
    **both** axes (element order and, within an element, node order), which is what
    keeps a reversed arc on its true arc.  Reversing by hand as
    ``LineMesh.loft(mesh.points[::-1])`` is the trap this exists to close: ``loft``
    with no explicit ``interior`` re-fills it with straight GLL chords, silently
    discarding the curve at ``order > 1``.

    ``element_tags`` reverse with their lines; ``point_tags`` are remapped (line
    ``l`` -> ``L-1-l``, side ``s`` -> ``3-s``, since each line's two endpoints swap)
    and re-sorted, so a tagged end point stays on the same physical point.

    The relabel is a bijection of the *existing* index space, not a new one, so this
    is a rung-preserving morph rather than an ``_assemble`` operation -- and it
    applies to any connectivity, branching or cyclic, not just a simple chain.  It
    is 1-D only: a quad or hex has no single traversal direction to flip (the
    analogous operation there is an orientation flip, which is a different thing)."""
    n = mesh.points.shape[0]
    L = mesh.lines.shape[0]
    # relabel r(i) = n-1-i, then restore each line's canonical start->end direction:
    # reverse the element order and swap the two endpoints of every line.
    lines: IntArray = (n - 1 - mesh.lines)[::-1, ::-1]
    b = mesh.point_tags
    bnd = PointTags(L - 1 - b.elements, 3 - b.sides, b.tags).ordered()
    return type(mesh)(np.ascontiguousarray(mesh.points[::-1]),
                      np.ascontiguousarray(lines),
                      np.ascontiguousarray(mesh.interior[::-1, ::-1, :]),
                      bnd,
                      mesh.element_tags.renumber(
                          (L - 1 - np.arange(L, dtype=np.int64))),
                      order=mesh.order)


def _affine(mesh: LineMesh, matrix: FloatArray | None, offset: Vec3) -> LineMesh:
    """Map every coordinate of ``mesh`` through the affine pair ``(matrix, offset)``.

    A ``LineMesh`` owns exactly two coordinate tables -- its ``points`` and its
    per-line private ``interior`` -- and both take the *same* map, so a curved
    element keeps its shape and its endpoints stay its corners.  Everything else
    (connectivity, element and point tags) is topology and rides through untouched."""
    return type(mesh)(affine.apply(mesh.points, matrix, offset), mesh.lines,
                      affine.apply(mesh.interior, matrix, offset),
                      point_tags=mesh.point_tags,
                      element_tags=mesh.element_tags, order=mesh.order)


def transform(mesh: LineMesh, matrix: FloatArray,
              offset: Vec3 | Sequence[float] = affine.ORIGIN) -> LineMesh:
    """A new curve with every node mapped through the affine ``p @ matrix.T +
    offset``.  The general case behind :func:`translate` / :func:`rotate` /
    :func:`scale`; reach for it for a map they do not name (a shear, a mirror, a
    pre-composed matrix)."""
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
