"""Fixed-arity, rung-preserving ``QuadMesh`` operations (delta 0)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

from .._typing import (
    FloatArray,
    IntArray,
    Point,
    PointArray,
    Vec3,
)
from ..core import affine, conform, frames
from ..core.interp import _element_tangents
from ..core.paths import SpacePath
from ..linemesh import LineMesh
from ..linemesh.morph import _affine as line_affine
from ..linemesh.morph import blend as line_blend
from ..linemesh.morph import offset_shift
from ..pointmesh import PointMesh
from .quadmesh import QuadMesh
from .query import element_blocks


def blend(a: QuadMesh, b: QuadMesh,
          fractions: FloatArray | Sequence[float]) -> list[QuadMesh]:
    """Linearly morph between two conformal sections ``a`` and ``b`` (identical
    ``quads``, equal point count), one section per fraction ``t`` with points ``(1-t)*a
    + t*b`` -- ``t=0`` reproduces ``a``, ``t=1`` reproduces ``b``.  Each section returned
    carries ``a``'s ``edge_tags`` and empty ``element_tags``."""
    A: PointArray = np.asarray(a.points, dtype=float).reshape(-1, 3)
    B: PointArray = np.asarray(b.points, dtype=float).reshape(-1, 3)
    if A.shape[0] != B.shape[0]:
        raise ValueError(
            "blend: sections must have equal point counts (got %d, %d); build "
            "one from the other's points so they pair by index"
            % (A.shape[0], B.shape[0]))
    if not np.array_equal(a.corners, b.corners):
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
    # the edge tags ride the blended ``lines`` themselves -- ``LineMesh.blend`` keeps
    # ``a``'s point tags but drops its element tags, so they are put back here
    return [QuadMesh(LineMesh(lm.point_mesh, lm.lines, lm.interior,
                              a.line_mesh.element_tags),
                     a.quads, a.orient,
                     (1.0 - t) * ai + t * bi if ho else None)
            for t, lm in zip(fr, line_blend(a.line_mesh, b.line_mesh, fr))]


def _affine(mesh: QuadMesh, matrix: FloatArray | None, offset: Vec3) -> QuadMesh:
    """Map every coordinate of ``mesh`` through the affine pair ``(matrix, offset)``."""
    return QuadMesh(line_affine(mesh.line_mesh, matrix, offset), mesh.quads, mesh.orient,
                    affine.apply(mesh.interior, matrix, offset),
                    element_tags=mesh.element_tags)


def transform(mesh: QuadMesh, matrix: FloatArray,
              offset: Vec3 | Sequence[float] = affine.ORIGIN) -> QuadMesh:
    """A new section with every node mapped through the affine ``p @ matrix.T +
    offset``."""
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
    """A new section rotated by ``angle`` **radians** about the line through ``center``
    along ``axis`` (right-handed, ``axis`` need not be normalized)."""
    return _affine(mesh, *affine.rotation(angle, axis, center))


def scale(mesh: QuadMesh, factor: float | Vec3 | Sequence[float],
          center: Point | Sequence[float] = affine.ORIGIN) -> QuadMesh:
    """A new section scaled about ``center`` by ``factor`` -- a scalar (uniform) or a
    ``(3,)`` per-axis vector.  Every factor must be positive."""
    return _affine(mesh, *affine.scaling(factor, center))


def _rewind(mesh: QuadMesh) -> QuadMesh:
    """The same section with every quad traversed the other way round -- corners
    ``(c0,c1,c2,c3)`` become ``(c0,c3,c2,c1)``.

    In B-rep storage that is the edge columns reversed with every traversal bit
    toggled; the shared edges themselves are untouched, since an edge row is canonical
    (min corner first) and knows nothing of the winding of the quads on it.  The private
    interior transposes with the local frame.  The edge **tags** need no fixing at
    all: they name the shared edges, which are exactly what re-winding leaves alone --
    where a ``(quad, side)`` table had to ride each row to ``5 - side``."""
    n = mesh.order - 1
    perm: IntArray = (np.arange(n * n, dtype=np.int64).reshape(n, n).T.ravel()
                      if n else np.zeros(0, dtype=np.int64))
    return QuadMesh(mesh.line_mesh, mesh.quads[:, ::-1], ~mesh.orient[:, ::-1],
                    mesh.interior[:, perm, :], mesh.element_tags)


def mirror(mesh: QuadMesh, normal: Vec3 | Sequence[float],
           point: Point | Sequence[float] = affine.ORIGIN) -> QuadMesh:
    """A new section reflected through the plane with ``normal`` through ``point``, with
    every quad **re-wound** so it keeps the orientation it was authored with.

    A reflection has determinant ``-1``, so it inverts every element: the bare
    ``transform`` of a reflection matrix hands back a section whose scaled Jacobian has
    flipped sign throughout.  ``mirror`` pairs the coordinate map with the winding fix,
    so the result measures exactly as its original did and lifts to a hex block the
    right way out.

    The symmetric-domain idiom -- mesh the half you can, recover the whole:
    ``quadmesh.merge([half, mirror(half, normal=(1, 0, 0))])``, which welds along the
    symmetry plane the two now share."""
    return _rewind(_affine(mesh, *affine.reflection(normal, point)))


def offset(mesh: QuadMesh, distance: float,
           crease: float = np.deg2rad(30.0)) -> QuadMesh:
    """A new section displaced by ``distance`` along its own surface normal -- the
    cross product of the two parametric tangents (computed from the underlying GLL
    nodes, so high-order interior nodes follow the same rule as corners). Consistent
    CCW winding across the section gives every quad's normal the same sign, so no
    reference direction is needed.

    At a point shared by more than one quad (a corner, or a shared-edge node), the
    offset direction is the average of every incident quad's own normal at that node,
    renormalized -- so a quad's private interior node (touched by no other quad) is
    simply displaced by its own element's normal.

    This is the building block for skinning: loft a section and its ``offset`` copy to
    get a thin, perpendicular boundary-layer hex shell."""
    order = mesh.order
    curved = element_blocks(mesh)                          # (Q, (order+1)**2, 3)
    t_u, t_v = _element_tangents(curved, order, 2)
    dirs = np.cross(t_u, t_v)
    dirs = dirs / np.linalg.norm(dirs, axis=2, keepdims=True)

    lm = mesh.line_mesh
    nodes, conn_ho = conform.conformal_quad(
        mesh.points, mesh.corners, mesh.quads, mesh.orient, lm.interior,
        mesh.interior, order)
    moved = nodes + offset_shift(dirs, conn_ho, nodes.shape[0], distance, crease)
    p = mesh.points.shape[0]
    ne = lm.lines.shape[0]
    k = order - 1
    new_points = moved[:p]
    new_edge_interior = moved[p:p + ne * k].reshape(ne, k, 3)
    new_interior = moved[p + ne * k:].reshape(mesh.quads.shape[0], k * k, 3)
    new_lines = LineMesh(PointMesh(new_points, lm.point_tags), lm.lines,
                         new_edge_interior, lm.element_tags)
    return QuadMesh(new_lines, mesh.quads, mesh.orient, new_interior, mesh.element_tags)


def reindex(structure: QuadMesh, target: QuadMesh,
            sigma: IntArray | Sequence[int]) -> QuadMesh:
    """``target``'s own geometry, reached through ``structure``'s own index labels:
    point ``i`` takes ``target``'s point ``sigma[i]``, and every shared-edge and
    per-quad interior node follows its relabelled corners."""
    if not (np.array_equal(structure.quads, target.quads)
            and np.array_equal(structure.orient, target.orient)):
        raise ValueError(
            "reindex: structure and target must share identical quad/flip incidence; "
            "they are two samplings of one recipe, not two different meshes")
    if not np.array_equal(structure.line_mesh.lines, target.line_mesh.lines):
        raise ValueError(
            "reindex: structure and target must share identical edge connectivity")
    s: IntArray = np.asarray(sigma, dtype=np.int64).ravel()
    n = structure.points.shape[0]
    if s.shape != (n,):
        raise ValueError("reindex: sigma must have one entry per point (%d), got %s"
                         % (n, s.shape))
    if not np.array_equal(np.sort(s), np.arange(n, dtype=np.int64)):
        raise ValueError(
            "reindex: sigma is not a permutation of the point ids -- the two patterns "
            "do not pair one-for-one, so relabelling would drop or duplicate a node")

    # Edges: structure's edge e runs sigma[u] -> sigma[v]; find target's edge on that
    # same unordered pair and copy its interior, reversed when the two traverse it the
    # other way.  Both sides are lexsorted on the sorted pair and paired positionally,
    # which needs no packed key -- and so has no bound on the point count.
    te: IntArray = np.asarray(target.line_mesh.lines, dtype=np.int64)
    se: IntArray = s[np.asarray(structure.line_mesh.lines, dtype=np.int64)]
    tidx = conform.locate_rows(te, se, who="reindex", what="edge")
    rev = te[tidx, 0] != se[:, 0]
    new_ei: PointArray = np.asarray(target.line_mesh.interior, dtype=float)[tidx].copy()
    new_ei[rev] = new_ei[rev][:, ::-1]
    new_lines = LineMesh(PointMesh(target.points[s],
                                   target.line_mesh.point_tags.gather(s)),
                         structure.line_mesh.lines, new_ei,
                         target.line_mesh.element_tags)

    # Quads: match on the relabelled corner *set*, which is orientation-free, so the
    # two pair however each happens to be wound.
    qidx = conform.locate_rows(np.asarray(target.corners, dtype=np.int64),
                                s[np.asarray(structure.corners, dtype=np.int64)],
                                who="reindex", what="quad")
    new_qi: PointArray = np.asarray(target.interior, dtype=float)[qidx]

    return QuadMesh(new_lines, structure.quads, structure.orient, new_qi,
                    target.element_tags)


def place_on_path(section: QuadMesh, path: SpacePath,
                  fractions: FloatArray | Sequence[float], *,
                  origin: Point | Sequence[float] | None = None,
                  orientation: Literal["transport", "fixed", "frenet"] = "transport",
                  up: Vec3 | Sequence[float] | PointArray | None = None,
                  twist: float = 0.0,
                  close_twist: bool = True,
                  normal: Vec3 | Sequence[float] | None = None,
                  loop: bool = False) -> list[QuadMesh]:
    """Where :func:`sweep <nekmeshpy.hexmesh.lift.sweep_path>` **would** put ``section``
    at each of ``fractions``, without building the block: one rigidly placed copy per
    station, through the same :func:`frames.sweep_placements
    <nekmeshpy.core.frames.sweep_placements>` the sweep itself uses."""
    P: PointArray = path.centerline(np.asarray(fractions, dtype=float).ravel())
    T: PointArray = path.tangent(np.asarray(fractions, dtype=float).ravel())
    places = frames.sweep_placements(
        section.points, P, orientation=orientation, up=up, twist=twist,
        close_twist=close_twist, loop=loop, origin=origin, normal=normal,
        path_tangents=T)
    return [transform(section, M, o) for M, o in places]


__all__ = [
    "blend",
    "mirror",
    "offset",
    "place_on_path",
    "reindex",
    "rotate",
    "scale",
    "transform",
    "translate",
]
