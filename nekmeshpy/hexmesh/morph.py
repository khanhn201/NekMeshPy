"""Fixed-arity, rung-preserving ``HexMesh`` operations (delta 0).

Two arities live here.  **Binary**: ``blend`` morphs between two index-paired blocks.
**Unary**: ``translate`` / ``rotate`` / ``scale`` / ``transform`` place a finished
block.  All of them change only coordinates -- ``a``'s ``hex`` / ``face_orient``
incidence and ``face_tags`` ride through verbatim, so the input's
numbering *is* the output's and nothing is re-derived (the unary placements keep
``element_tags`` too; ``blend`` leaves them for the caller).  Both delegate their
corner, shared-edge and shared-face part one rung down to
:mod:`nekmeshpy.quadmesh.morph`.

Free functions bound onto :class:`HexMesh <nekmeshpy.hexmesh.hexmesh.HexMesh>` by ``hexmesh/__init__.py`` (the
binary ``blend`` as a ``staticmethod``, the unary placements as instance methods);
internal toolkit code imports them from here directly rather than through the bound
``HexMesh.<name>`` sugar.
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
from ..model import affine
from ..quadmesh.morph import _affine as quad_affine
from ..quadmesh.morph import blend as quad_blend
from .hexmesh import HexMesh


def blend(a: HexMesh, b: HexMesh,
          fractions: FloatArray | Sequence[float]) -> list[HexMesh]:
    """Linearly morph between two conformal blocks ``a`` and ``b`` (identical
    ``hexes``, equal point count), one block per fraction ``t`` with points
    ``(1-t)*a + t*b`` -- ``t=0`` reproduces ``a``, ``t=1`` reproduces ``b``.  Each
    result carries ``a``'s ``hexes`` and ``face_tags``
    (positional BC markers follow the morph); per-hex ``element_tags`` are left
    for the caller to assign.  The 3-D sibling of
    :func:`QuadMesh.blend <nekmeshpy.quadmesh.morph.blend>`.

    The morph is delegated one rung **down the B-rep ladder**: the shared corners,
    shared edge nodes and shared face nodes are exactly the shared-face
    ``QuadMesh`` (whose own corners and edge nodes are in turn its edge
    ``LineMesh``), so
    :func:`QuadMesh.blend <nekmeshpy.quadmesh.morph.blend>` produces the blended
    face mesh and this method only lerps what a hex owns privately -- its per-hex
    ``interior`` -- while keeping ``a``'s ``hex`` / ``face_orient`` incidence
    verbatim."""
    A: PointArray = np.asarray(a.points, dtype=float).reshape(-1, 3)
    B: PointArray = np.asarray(b.points, dtype=float).reshape(-1, 3)
    if A.shape[0] != B.shape[0]:
        raise ValueError(
            "blend: blocks must have equal point counts (got %d, %d); build one "
            "from the other's points so they pair by index"
            % (A.shape[0], B.shape[0]))
    if not np.array_equal(a.hexes, b.hexes):
        raise ValueError(
            "blend: blocks must share identical connectivity (paired by index)")
    if a.order != b.order:
        raise ValueError("blend: blocks must have the same order (got %d, %d)"
                         % (a.order, b.order))
    # identical connectivity => identical edge / face tables, so a's and b's shared
    # edge nodes, shared face nodes and private interiors already pair one-for-one
    # and each morphs with the same lerp the corners get from the blended points.
    # The shared corners, shared edge nodes and shared face nodes *are* the
    # shared-face QuadMesh, so that whole part of the morph is one
    # ``QuadMesh.blend`` of the rung below (which in turn delegates the corners and
    # edge nodes to ``LineMesh.blend``); the result reuses ``a``'s per-hex face
    # indices and D4 codes verbatim -- a blend is a pure point-space morph, so
    # nothing is re-derived.  At order 1 all three tables are empty and this is
    # exactly the plain point blend.
    ho = a.order > 1
    ai, bi = a.interior, b.interior
    fr: FloatArray = np.asarray(fractions, dtype=float).ravel()
    return [HexMesh(faces, a.hex, a.face_orient,
                (1.0 - t) * ai + t * bi if ho else None,
                a.face_tags)
            for t, faces in zip(fr, quad_blend(a.quads, b.quads, fr))]


def _affine(mesh: HexMesh, matrix: FloatArray | None, offset: Vec3) -> HexMesh:
    """Map every coordinate of ``mesh`` through the affine pair ``(matrix, offset)``.

    Composed downward like every other rung: the shared corners, shared edge nodes
    and shared face nodes *are* the shared-face ``QuadMesh``, so that whole part is
    one ``QuadMesh`` map (which in turn delegates to ``LineMesh``), and only the
    per-hex private ``interior`` is mapped here.  ``hex`` / ``face_orient``
    incidence rides through verbatim."""
    return HexMesh(quad_affine(mesh.quads, matrix, offset), mesh.hex,
                   mesh.face_orient, affine.apply(mesh.interior, matrix, offset),
                   mesh.face_tags, mesh.element_tags,
)


def transform(mesh: HexMesh, matrix: FloatArray,
              offset: Vec3 | Sequence[float] = affine.ORIGIN) -> HexMesh:
    """A new block with every node mapped through the affine ``p @ matrix.T +
    offset``.  The general case behind :func:`translate <nekmeshpy.hexmesh.morph.translate>` / :func:`rotate <nekmeshpy.hexmesh.morph.rotate>` /
    :func:`scale <nekmeshpy.hexmesh.morph.scale>`; reach for it for a map they do not name (a shear, a mirror, a
    pre-composed matrix).  A matrix with negative determinant mirrors the block and
    therefore inverts every element -- check
    :func:`scaled_jacobian <nekmeshpy.hexmesh.query.scaled_jacobian>` after one."""
    return _affine(mesh, np.asarray(matrix, dtype=float).reshape(3, 3),
                   np.asarray(offset, dtype=float).reshape(3))


def translate(mesh: HexMesh, vector: Vec3 | Sequence[float]) -> HexMesh:
    """A new block shifted rigidly by ``vector`` ``(3,)``.  Bit-exact: the offset is
    added without a matmul, so translating by ``0`` returns the identical
    coordinates."""
    return _affine(mesh, *affine.translation(vector))


def rotate(mesh: HexMesh, angle: float,
           axis: Vec3 | Sequence[float] = affine.Z_AXIS,
           center: Point | Sequence[float] = affine.ORIGIN) -> HexMesh:
    """A new block rotated by ``angle`` **radians** about the line through ``center``
    along ``axis`` (right-handed, ``axis`` need not be normalized).  The map is
    rigid, so element quality is unchanged -- place a block this way and
    :func:`merge <nekmeshpy.hexmesh.assemble.merge>` it onto its neighbour."""
    return _affine(mesh, *affine.rotation(angle, axis, center))


def scale(mesh: HexMesh, factor: float | Vec3 | Sequence[float],
          center: Point | Sequence[float] = affine.ORIGIN) -> HexMesh:
    """A new block scaled about ``center`` by ``factor`` -- a scalar (uniform) or a
    ``(3,)`` per-axis vector.  Every factor must be positive."""
    return _affine(mesh, *affine.scaling(factor, center))

__all__ = [
    "blend",
    "rotate",
    "scale",
    "transform",
    "translate",
]
