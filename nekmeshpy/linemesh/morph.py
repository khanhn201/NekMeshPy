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
from ..core import affine, conform
from ..core.interp import _element_tangents
from ..pointmesh import PointMesh
from .linemesh import LineMesh
from .query import boundary_points, element_blocks


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
        out.append(LineMesh(PointMesh((1.0 - t) * A + t * B, a.point_tags),
                            a.lines, ia))
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
    # a point tag rides its point through the relabel -- new point j is old point
    # rmap[j], which is exactly what ``gather`` means.  Nothing to remap side-wise:
    # the tag names the point, not one line's view of it.
    return LineMesh(PointMesh(np.ascontiguousarray(mesh.points[rmap]),
                              mesh.point_tags.gather(rmap)),
                    np.ascontiguousarray(lines),
                    np.ascontiguousarray(mesh.interior[::-1, ::-1, :]),
                    mesh.element_tags.renumber(
                        (L - 1 - np.arange(L, dtype=np.int64))),
)


def _affine(mesh: LineMesh, matrix: FloatArray | None, offset: Vec3) -> LineMesh:
    """Map every coordinate of ``mesh`` through the affine pair ``(matrix, offset)``."""
    return LineMesh(PointMesh(affine.apply(mesh.points, matrix, offset),
                              mesh.point_tags),
                    mesh.lines, affine.apply(mesh.interior, matrix, offset),
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


def offset_shift(dirs: PointArray, conn_ho: IntArray, n_nodes: int, distance: float,
                 crease: float) -> PointArray:
    """``(n_nodes, 3)`` displacement for an offset, the way a CAD offset resolves it.

    Averaging a node's incident normals and stepping ``distance`` along the mean is right
    only where the surface is smooth.  Across a crease it is simply wrong: the mean
    bisects the two normals, so the perpendicular distance to each incident face comes out
    ``distance * cos(theta/2)`` and the layer pinches -- at a square corner it keeps only
    71% of its thickness.

    A CAD offset never moves points at all: it offsets each *face* along its own normal
    and re-intersects the neighbours, so every face keeps its full ``distance``.  The
    discrete form of that is one equation per incident face, ``n_i . u = distance``,
    solved in the least-squares sense -- one normal gives ``distance * n`` back, two give
    the miter, three the trihedral corner.

    The other half of what CAD does is treat tangent-continuous faces as *one* surface and
    only miter across genuine edges, and that half matters just as much: mitering every
    facet of a discretized cylinder would push its nodes out to ``R + distance/cos(pi/n)``
    rather than ``R + distance``.  So incident normals within ``crease`` radians of the
    node's mean are first collapsed into a single plane, and only what is left over is
    intersected."""
    flat: IntArray = conn_ho.ravel()
    n: PointArray = dirs.reshape(-1, 3)
    total: PointArray = np.zeros((n_nodes, 3))
    np.add.at(total, flat, n)
    mag = np.linalg.norm(total, axis=1)
    unit: PointArray = total / np.where(mag > 0.0, mag, 1.0)[:, None]

    # the sharpest departure of any incident face from the node's own mean normal
    worst: FloatArray = np.ones(n_nodes)
    np.minimum.at(worst, flat, np.sum(n * unit[flat], axis=1))
    shift: PointArray = distance * unit

    sharp: IntArray = np.flatnonzero(worst < np.cos(crease))
    if sharp.size == 0:
        return shift
    order: IntArray = np.argsort(flat, kind="stable")
    srt: IntArray = flat[order]
    lo: IntArray = np.asarray(np.searchsorted(srt, sharp, "left"), dtype=np.int64)
    hi: IntArray = np.asarray(np.searchsorted(srt, sharp, "right"), dtype=np.int64)
    cos_c = float(np.cos(crease))
    for k, node in enumerate(sharp.tolist()):
        planes: list[PointArray] = []
        for v in n[order[lo[k]:hi[k]]]:
            for j, acc in enumerate(planes):
                if float(v @ (acc / np.linalg.norm(acc))) > cos_c:
                    planes[j] = acc + v
                    break
            else:
                planes.append(v.astype(float).copy())
        N: PointArray = np.stack([p / np.linalg.norm(p) for p in planes])
        shift[node] = np.linalg.lstsq(N, np.full(N.shape[0], distance), rcond=None)[0]
    return shift


def _auto_plane_normal(points: PointArray) -> Vec3:
    """Least-squares plane normal of a point cloud (the smallest right singular vector
    of the centred points), used when :func:`offset` is not told which plane its curve
    lies in."""
    c = points.mean(axis=0)
    n: Vec3 = np.linalg.svd(points - c)[2][2]
    return n


def offset(mesh: LineMesh, distance: float,
           normal: Vec3 | Sequence[float] | None = None,
           crease: float = np.deg2rad(30.0)) -> LineMesh:
    """A new curve displaced by ``distance`` along its in-plane normal -- the tangent
    (computed from the underlying GLL nodes, so high-order interior nodes follow the
    same rule as corners) rotated a quarter turn within the plane given by ``normal``.

    ``normal`` need not be supplied for a planar curve: it defaults to the curve's own
    least-squares plane. At a point shared by more than one line (a chain's interior
    corners, or a junction where several lines meet), the offset direction is the
    average of every incident line's own direction at that point, renormalized -- so a
    private high-order node (touched by exactly one line) is simply displaced by its
    own element's direction.

    This is the building block for skinning: loft a curve and its ``offset`` copy to
    get a thin, perpendicular boundary-layer quad strip."""
    order = mesh.order
    n: Vec3 = (np.asarray(normal, dtype=float).reshape(3) if normal is not None
              else _auto_plane_normal(mesh.points))
    n = n / np.linalg.norm(n)

    curved = element_blocks(mesh)                          # (L, order+1, 3)
    (tangent,) = _element_tangents(curved, order, 1)        # (L, order+1, 3)
    dirs = np.cross(n[None, None, :], tangent)
    dirs = dirs / np.linalg.norm(dirs, axis=2, keepdims=True)

    nodes, conn_ho = conform.conformal_line(mesh.points, mesh.lines, mesh.interior, order)
    moved = nodes + offset_shift(dirs, conn_ho, nodes.shape[0], distance, crease)
    p = mesh.points.shape[0]
    new_points = moved[:p]
    new_interior = moved[p:].reshape(mesh.lines.shape[0], order - 1, 3)
    return LineMesh(PointMesh(new_points, mesh.point_tags), mesh.lines, new_interior,
                    element_tags=mesh.element_tags)


__all__ = [
    "blend",
    "mirror",
    "offset",
    "reverse",
    "rotate",
    "scale",
    "transform",
    "translate",
]
