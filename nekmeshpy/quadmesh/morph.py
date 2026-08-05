"""Fixed-arity, rung-preserving ``QuadMesh`` operations (delta 0).

Two arities live here.  **Binary**: ``blend`` morphs between two index-paired
sections.  **Unary**: ``translate`` / ``rotate`` / ``scale`` / ``transform`` place a
finished section.  All of them change only coordinates -- ``a``'s ``quad`` / ``flip``
incidence and ``edge_tags`` ride through verbatim, so the input's
numbering *is* the output's and nothing is re-derived (the unary placements keep
``element_tags`` too; ``blend`` leaves them for the consuming ``loft``).  Both
delegate their corner and shared-edge half one rung down to
:mod:`nekmeshpy.linemesh.morph`.

Free functions bound onto :class:`QuadMesh <nekmeshpy.quadmesh.quadmesh.QuadMesh>` by ``quadmesh/__init__.py``
(the binary ``blend`` as a ``staticmethod``, the unary placements as instance
methods); internal toolkit code imports them from here directly rather than through
the bound ``QuadMesh.<name>`` sugar.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .._typing import (
    FloatArray,
    Point,
    PointArray,
    Vec3,
)
from ..linemesh.morph import _affine as line_affine
from ..linemesh.morph import blend as line_blend
from ..model import affine
from .quadmesh import QuadMesh


def blend(a: QuadMesh, b: QuadMesh,
          fractions: FloatArray | Sequence[float]) -> list[QuadMesh]:
    """Linearly morph between two conformal sections ``a`` and ``b`` (identical
    ``quads``, equal point count), one section per fraction ``t`` with points
    ``(1-t)*a + t*b`` -- ``t=0`` reproduces ``a``, ``t=1`` reproduces ``b``.  Each
    result carries ``a``'s ``quads`` and ``edge_tags``
    (positional BC markers follow the morph); per-quad ``element_tags`` are left
    for the consuming ``loft`` caps to assign, so a blended stack lofts directly.
    This is the profile-positioning step behind ``HexMesh.annulus``.

    The morph is delegated one rung **down the B-rep ladder**: the shared corners
    and the shared edge-interior nodes are exactly the edge ``LineMesh``, so
    :func:`linemesh.morph.blend <nekmeshpy.linemesh.morph.blend>` produces the blended
    edge mesh and this method only lerps what a quad owns privately -- its
    per-quad ``interior`` -- while keeping ``a``'s ``quad`` / ``flip`` incidence
    verbatim."""
    A: PointArray = np.asarray(a.points, dtype=float).reshape(-1, 3)
    B: PointArray = np.asarray(b.points, dtype=float).reshape(-1, 3)
    if A.shape[0] != B.shape[0]:
        raise ValueError(
            "blend: sections must have equal point counts (got %d, %d); build "
            "one from the other's points so they pair by index"
            % (A.shape[0], B.shape[0]))
    if not np.array_equal(a.quads, b.quads):
        raise ValueError(
            "blend: sections must share identical connectivity (paired by index)")
    if a.order != b.order:
        raise ValueError("blend: sections must have the same order "
                         "(got %d, %d)" % (a.order, b.order))
    # identical connectivity => identical edge tables, so a's and b's shared edge
    # nodes and private interiors already pair one-for-one and each morphs with the
    # same lerp the corners get from the blended points.  The shared corners *are*
    # the edge LineMesh's points and the shared edge nodes *are* its interior, so
    # that whole half of the morph is one ``LineMesh.blend`` of the rung below; the
    # result reuses ``a``'s per-quad edge indices / flips verbatim -- a blend is a
    # pure point-space morph, so nothing is re-derived.  At order 1 both entity
    # tables are empty and this is exactly the plain point blend.
    ho = a.order > 1
    ai, bi = a.interior, b.interior
    fr: FloatArray = np.asarray(fractions, dtype=float).ravel()
    return [QuadMesh(lm, a.quad, a.flip,
                (1.0 - t) * ai + t * bi if ho else None,
                edge_tags=a.edge_tags, order=a.order)
            for t, lm in zip(fr, line_blend(a.lines, b.lines, fr))]


def _affine(mesh: QuadMesh, matrix: FloatArray | None, offset: Vec3) -> QuadMesh:
    """Map every coordinate of ``mesh`` through the affine pair ``(matrix, offset)``.

    Composed downward like every other rung: the shared corners *are* the edge
    ``LineMesh``'s points and the shared edge nodes *are* its ``interior``, so that
    whole half is one ``LineMesh`` map, and only the per-quad private ``interior``
    is mapped here.  ``quad`` / ``flip`` incidence rides through verbatim -- an
    affine map is a pure point-space placement, so nothing is re-derived."""
    return QuadMesh(line_affine(mesh.lines, matrix, offset), mesh.quad, mesh.flip,
                    affine.apply(mesh.interior, matrix, offset),
                    edge_tags=mesh.edge_tags,
                    element_tags=mesh.element_tags, order=mesh.order)


def transform(mesh: QuadMesh, matrix: FloatArray,
              offset: Vec3 | Sequence[float] = affine.ORIGIN) -> QuadMesh:
    """A new section with every node mapped through the affine ``p @ matrix.T +
    offset``.  The general case behind :func:`translate <nekmeshpy.quadmesh.morph.translate>` / :func:`rotate <nekmeshpy.quadmesh.morph.rotate>` /
    :func:`scale <nekmeshpy.quadmesh.morph.scale>`; reach for it for a map they do not name (a shear, a mirror, a
    pre-composed matrix)."""
    return _affine(mesh, np.asarray(matrix, dtype=float).reshape(3, 3),
                   np.asarray(offset, dtype=float).reshape(3))


def translate(mesh: QuadMesh, vector: Vec3 | Sequence[float]) -> QuadMesh:
    """A new section shifted rigidly by ``vector`` ``(3,)``.  Bit-exact: the offset
    is added without a matmul, so translating by ``0`` returns the identical
    coordinates -- which is what lets ``extrude`` place its slices through here."""
    return _affine(mesh, *affine.translation(vector))


def rotate(mesh: QuadMesh, angle: float,
           axis: Vec3 | Sequence[float] = affine.Z_AXIS,
           center: Point | Sequence[float] = affine.ORIGIN) -> QuadMesh:
    """A new section rotated by ``angle`` **radians** about the line through
    ``center`` along ``axis`` (right-handed, ``axis`` need not be normalized).  The
    map is rigid, so element quality is unchanged -- this is how a stack of revolved
    profiles is placed for a swept ``loft``."""
    return _affine(mesh, *affine.rotation(angle, axis, center))


def scale(mesh: QuadMesh, factor: float | Vec3 | Sequence[float],
          center: Point | Sequence[float] = affine.ORIGIN) -> QuadMesh:
    """A new section scaled about ``center`` by ``factor`` -- a scalar (uniform) or a
    ``(3,)`` per-axis vector.  Every factor must be positive."""
    return _affine(mesh, *affine.scaling(factor, center))
