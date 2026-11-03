"""High-order rebuild: take the vertices a *linear* smoothing produced and give the mesh its
curved edge and face nodes back, on the target geometry.

The smoother (``smooth``) moves element vertices only -- by decision, until a high-order
smoother is robust.  What it hands back is a straight-sided mesh; the geometry an order-``N``
solver sees is the curved one, so the nodes between the vertices have to be rebuilt and
re-projected afterwards.  Three steps, each simple on purpose:

1. **Carry.**  Every node of an element keeps its *offset from the straight-sided block*:
   ``x_new = M(new corners) + (x_old - M(old corners))`` with ``M`` the multilinear map.  A node
   therefore moves with its element (a rigid corner move translates the whole curved element) and
   keeps the sagitta it had.  Edge nodes depend only on their edge's two corners, so shared edges
   stay conforming; elements whose corners did not move are bit-for-bit untouched.
2. **Assemble** the shared edge / private face tables back from the per-element blocks, with the
   toolkit's own ``entities_from_blocks`` (the path ``refine`` uses); edge and point tags ride through.
3. **Project.**  On the *assembled* shared entities (never per element, so a shared edge cannot be
   projected two ways): an edge whose endpoints lie on one curve goes onto it; an edge or a face whose
   vertices lie on one surface -- frozen vertices allowed, they are on it already -- goes onto it.
   Projection is the target's own retraction (``constrain``), the one the smoother snapped with.

Honest limits: a node projected radially from a chord is not at its Gauss-Lobatto *arc* position
(the spacing along a strongly curved edge is only approximately GLL); the carried sagitta is that of
the *old* geometry, which is right for small vertex moves and is overwritten by the projection wherever
there is a target; and nothing here makes a high-order element valid -- the report samples the
Jacobian on a finer lattice and says what it finds.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ..._typing import BoolArray, FloatArray, IntArray, PointArray
from ...core import conform, interp
from ...core.smoother.blocks import carry_blocks, scatter_blocks
from ...linemesh import LineMesh
from ...pointmesh import PointMesh
from .._helpers import entities_from_blocks
from ..quadmesh import QuadMesh
from ..query import element_blocks, scaled_jacobian
from . import constrain as cn

__all__ = ["HighOrderReport", "linearize", "rebuild_high_order", "node_ids", "with_blocks", "carry"]


def linearize(mesh: QuadMesh) -> QuadMesh:
    """The straight-sided order-1 mesh with the same corners, connectivity and tags -- what the
    linear smoother works on.  A new mesh; ``mesh`` is untouched."""
    lm = mesh.line_mesh
    new_lm = LineMesh(PointMesh(np.array(lm.points, copy=True), lm.point_mesh.element_tags),
                      lm.lines, None, lm.element_tags)
    return QuadMesh(new_lm, mesh.quads, mesh.orient, None, mesh.element_tags)


class HighOrderReport(NamedTuple):
    """What the rebuild did and what it left behind."""

    order: int
    check_order: int
    n_edge_nodes_projected: int
    n_face_nodes_projected: int
    #: Largest distance of a to-be-projected node from its target, before and after projection.
    max_distance_before: float
    max_distance_after: float
    #: Largest displacement of any node of an element whose corners did not move (must be 0).
    max_untouched_move: float
    #: Sampled scaled Jacobian of the input and of the rebuilt mesh on the ``check_order`` lattice.
    sj_min_input: float
    sj_min_carried: float
    sj_min_rebuilt: float
    n_inverted_sampled: int
    notes: tuple[str, ...]

    def __str__(self) -> str:
        head = ("high order %d: projected %d edge-interior + %d face-interior nodes (distance to target "
                "%.2e -> %.2e); untouched elements moved %.1e; scaled J sampled at order %d: input %.4f, "
                "carried %.4f, rebuilt %.4f, %d inverted"
                % (self.order, self.n_edge_nodes_projected, self.n_face_nodes_projected,
                   self.max_distance_before, self.max_distance_after, self.max_untouched_move,
                   self.check_order, self.sj_min_input, self.sj_min_carried, self.sj_min_rebuilt,
                   self.n_inverted_sampled))
        return head + "".join("\n  - " + n for n in self.notes)


def _weights(order: int) -> FloatArray:
    """``((order+1)**2, 4)`` multilinear weights of the four corners at the GLL lattice."""
    W: FloatArray = np.empty(((order + 1) ** 2, 4))
    for j in range(4):
        c = np.zeros((4, 3))
        c[j, 0] = 1.0
        W[:, j] = interp.subdivide_element(c, order, 2)[:, 0]
    return W


def _classify(resolved: cn.Resolved, a: IntArray, b: IntArray | None = None) -> tuple[IntArray, IntArray]:
    """For each entity (an edge ``a-b``, or a face given as a ``(n,4)`` array in ``a``): the surface index
    it lies on (``-1`` none) and the curve index (``-1`` none).  A surface needs every vertex ON_SURFACE of
    one surface or FROZEN, with at least one ON_SURFACE; a curve needs every vertex ON_CURVE of one curve."""
    v = a if b is None else np.stack([a, b], axis=1)
    kinds, tgts = resolved.kind[v], resolved.target[v]
    surf = np.full(v.shape[0], -1, dtype=np.int64)
    curve = np.full(v.shape[0], -1, dtype=np.int64)
    is_s = kinds == cn.ON_SURFACE
    ok_s = (is_s | (kinds == cn.FROZEN)) & True
    first_s = np.where(is_s, tgts, -1).max(axis=1)
    same = np.where(is_s, tgts == first_s[:, None], True).all(axis=1)
    sel = ok_s.all(axis=1) & is_s.any(axis=1) & same
    surf[sel] = first_s[sel]
    is_c = kinds == cn.ON_CURVE
    first_c = np.where(is_c, tgts, -1).max(axis=1)
    sel_c = is_c.all(axis=1) & (tgts == first_c[:, None]).all(axis=1)
    curve[sel_c] = first_c[sel_c]
    return surf, curve


def rebuild_high_order(mesh: QuadMesh, new_points: PointArray, resolved: cn.Resolved | None = None, *,
                       order: int | None = None, project: bool = True,
                       check_order: int | None = None) -> tuple[QuadMesh, HighOrderReport]:
    """The curved mesh of ``order`` (default ``mesh.order``) whose corners are ``new_points`` -- see the
    module docstring.  ``mesh`` is the mesh to take the curvature from (a straight-sided order-1 mesh
    works: it is then *lifted* to ``order`` and projected); ``new_points`` ``(P, 3)`` are the smoothed
    vertices, aligned with ``mesh.points``; ``resolved`` (from :func:`constrain.resolve`) says which
    vertices lie on which target, ``None`` skips the projection.  A **new** mesh and a report."""
    P0 = np.asarray(mesh.points, dtype=float)
    P1 = np.asarray(new_points, dtype=float)
    if P1.shape != P0.shape:
        raise ValueError("rebuild_high_order: new_points must be %s, got %s" % (P0.shape, P1.shape))
    n0 = mesh.order
    n = n0 if order is None else int(order)
    if n < n0:
        raise ValueError("rebuild_high_order: order %d is below the mesh's own order %d; lowering would "
                         "discard the very nodes that make it curved" % (n, n0))
    if n < 1:
        raise ValueError("rebuild_high_order: order must be >= 1")
    corners = mesh.corners
    blocks_old = element_blocks(mesh)
    if n > n0:
        blocks_old = interp.resample_block(blocks_old, n0, n, 2)
    W = _weights(n)
    c_old, c_new = P0[corners], P1[corners]
    lin_old = np.einsum("mc,qck->qmk", W, c_old)
    lin_new = np.einsum("mc,qck->qmk", W, c_new)
    blocks = lin_new + (blocks_old - lin_old)                          # 1. carry
    untouched = (c_old == c_new).all(axis=(1, 2))
    blocks[untouched] = blocks_old[untouched]
    notes: list[str] = []
    if n > n0:
        notes.append("lifted from order %d to %d by a change of nodal basis (no geometry invented)" % (n0, n))

    lm, elem_edges, flip, interior = entities_from_blocks(blocks, corners, P1, n)        # 2. assemble
    if mesh.line_mesh.element_tags:
        match = conform.locate_rows(lm.lines, mesh.line_mesh.lines, what="edge")
        edge_tags = mesh.line_mesh.element_tags.renumber(match, lm.n_lines)
    else:
        edge_tags = mesh.line_mesh.element_tags
    pm = PointMesh(np.array(P1, copy=True), mesh.line_mesh.point_mesh.element_tags)
    edge_nodes = np.array(lm.interior, copy=True)
    face_nodes = np.array(interior, copy=True)
    carried = QuadMesh(LineMesh(pm, lm.lines, edge_nodes if n > 1 else None, edge_tags),
                       elem_edges, flip, face_nodes if n > 1 else None, mesh.element_tags)

    n_e = n_f = 0
    d_before = d_after = 0.0
    if project and resolved is not None and n > 1:                                        # 3. project
        es, ec = _classify(resolved, lm.lines[:, 0], lm.lines[:, 1])
        fs, _ = _classify(resolved, corners)
        for t in np.unique(ec[ec >= 0]):                       # curves win on an edge
            m = ec == t
            X = edge_nodes[m].reshape(-1, 3)
            Y = resolved.curves[t].project(X)
            d_before = max(d_before, float(np.linalg.norm(Y - X, axis=1).max()))
            edge_nodes[m] = Y.reshape(-1, n - 1, 3)
            d_after = max(d_after, float(np.linalg.norm(resolved.curves[t].project(Y) - Y, axis=1).max()))
            n_e += int(m.sum()) * (n - 1)
        for t in np.unique(es[(es >= 0) & (ec < 0)]):
            m = (es == t) & (ec < 0)
            X = edge_nodes[m].reshape(-1, 3)
            Y = resolved.surfaces[t].project(X)
            d_before = max(d_before, float(np.linalg.norm(Y - X, axis=1).max()))
            edge_nodes[m] = Y.reshape(-1, n - 1, 3)
            d_after = max(d_after, float(np.linalg.norm(resolved.surfaces[t].project(Y) - Y, axis=1).max()))
            n_e += int(m.sum()) * (n - 1)
        for t in np.unique(fs[fs >= 0]):
            m = fs == t
            X = face_nodes[m].reshape(-1, 3)
            Y = resolved.surfaces[t].project(X)
            d_before = max(d_before, float(np.linalg.norm(Y - X, axis=1).max()))
            face_nodes[m] = Y.reshape(-1, (n - 1) ** 2, 3)
            d_after = max(d_after, float(np.linalg.norm(resolved.surfaces[t].project(Y) - Y, axis=1).max()))
            n_f += int(m.sum()) * (n - 1) ** 2
        if not (n_e or n_f):
            notes.append("no edge or face lies on a single target: nothing to project")
    elif project and resolved is None:
        notes.append("no constraints given: the carried nodes are not projected")

    out = QuadMesh(LineMesh(pm, lm.lines, edge_nodes if n > 1 else None, edge_tags),
                   elem_edges, flip, face_nodes if n > 1 else None, mesh.element_tags)
    co = max(n, 8) if check_order is None else int(check_order)
    sj_in = scaled_jacobian(mesh, order=max(co, n0))
    sj_carried = scaled_jacobian(carried, order=co) if n > 1 else scaled_jacobian(carried, order=1)
    sj_out = scaled_jacobian(out, order=co) if n > 1 else scaled_jacobian(out, order=1)
    moved = float(np.abs(out_blocks_move(out, mesh, untouched, n)).max()) if untouched.any() else 0.0
    rep = HighOrderReport(n, co, n_e, n_f, d_before, d_after, moved, float(sj_in.min()),
                          float(sj_carried.min()), float(sj_out.min()), int((sj_out <= 0).sum()),
                          tuple(notes))
    return out, rep


def out_blocks_move(out: QuadMesh, mesh: QuadMesh, untouched: BoolArray, order: int) -> FloatArray:
    """Displacement of every node of the elements whose corners did not move, between the input's
    blocks (resampled to ``order`` if needed) and the rebuilt ones."""
    b_out = element_blocks(out)
    b_in = element_blocks(mesh)
    if order > mesh.order:
        b_in = interp.resample_block(b_in, mesh.order, order, 2)
    return np.asarray(b_out[untouched] - b_in[untouched])


def _shadow(mesh: QuadMesh, flat: FloatArray) -> QuadMesh:
    """``mesh`` with its stored nodes -- points, then edge interiors, then quad interiors, in that order --
    replaced by the rows of ``flat`` ``(n, 3)``; the incidence, orientation and tags are the mesh's own."""
    lm = mesh.line_mesh
    nP, nE = lm.points.shape[0], lm.interior.shape[0] * lm.interior.shape[1]
    pts = np.array(flat[:nP])
    edge = np.array(flat[nP:nP + nE]).reshape(lm.interior.shape)
    inner = np.array(flat[nP + nE:]).reshape(mesh.interior.shape)
    new_lm = LineMesh(PointMesh(pts, lm.point_mesh.element_tags), lm.lines, edge if mesh.order > 1 else None,
                      lm.element_tags)
    return QuadMesh(new_lm, mesh.quads, mesh.orient, inner if mesh.order > 1 else None, mesh.element_tags)


def _flat(mesh: QuadMesh) -> FloatArray:
    lm = mesh.line_mesh
    return np.concatenate([lm.points, lm.interior.reshape(-1, 3), mesh.interior.reshape(-1, 3)])


def node_ids(mesh: QuadMesh) -> IntArray:
    """``(Q, (order+1)**2)`` the index of every block node in the flat table ``points, edge interiors, quad
    interiors`` -- read by running :func:`~nekmeshpy.quadmesh.query.element_blocks` on the ids themselves, so the
    storage's own orientation conventions are used and none re-derived."""
    flat = _flat(mesh)
    ids = np.zeros_like(flat)
    ids[:, 0] = np.arange(flat.shape[0])
    return np.rint(element_blocks(_shadow(mesh, ids))[:, :, 0]).astype(np.int64)


def with_blocks(mesh: QuadMesh, blocks: FloatArray, points: PointArray | None = None) -> QuadMesh:
    """A **new** mesh of the same order, connectivity and tags whose node blocks are ``blocks``
    ``(Q, (order+1)**2, 3)``.  Each shared node is stored once -- an element's own value, the blocks of a
    conformal mesh agreeing -- so conformality stays structural; ``points`` (when given) are the vertices exactly."""
    flat = scatter_blocks(node_ids(mesh), np.asarray(blocks, float), _flat(mesh))
    if points is not None:
        flat[:mesh.n_points] = points
    return _shadow(mesh, flat)


def carry(mesh: QuadMesh, moved: QuadMesh) -> QuadMesh:
    """``mesh`` (curved) with its nodes carried to the vertices of ``moved`` (a mesh of the same connectivity, e.g.
    the linear shadow a native stage smoothed): each node follows the displacement of its element's corners,
    interpolated bilinearly (:func:`~nekmeshpy.core.smoother.blocks.carry_blocks`) -- no projection onto a target."""
    d = np.asarray(moved.points, float) - np.asarray(mesh.points, float)
    new = carry_blocks(element_blocks(mesh), d[mesh.corners], mesh.order, 2)
    return with_blocks(mesh, new, np.asarray(moved.points, float))
