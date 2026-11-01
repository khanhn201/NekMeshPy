"""Quad smoother kernels: the corner shape metric, its objective and analytic gradient.

A quad mesh is a *surface in 3-D with 2-D topology*, so every quantity here is read off
the 3x2 corner Jacobian ``J = [a b]`` of a quad corner -- ``a`` and ``b`` the two edge
vectors leaving it, in CCW order -- and never assumes the mesh is planar.  The maths
is set out below.

Shape metric (``d = 2``; ``||J||_F^2 = |a|^2 + |b|^2``, ``||J^+||_F^2 = tr (J^T J)^-1``)::

    phi = ||J||_F^2 ||J^+||_F^2 / d^2 = (|a|^2 + |b|^2)^2 / (4 |a x b|^2)  >= 1

with equality only for a *square* corner.  ``phi`` cannot see orientation (a flipped square
still scores 1), so inversion is a separate, signed term: the scaled Jacobian
``s = (a x b) . n / (|a||b|)`` against a reference normal ``n`` per quad, penalised by
``kappa * max(0, eps - s)``.  The objective is the **sum** over quads of the corner mean
of ``(phi - 1) + penalty``; a vertex's gradient involves only its incident quads, so it
does not depend on the element count (the paper's ``tau = 1000 E`` on a mean objective is
exactly ``kappa = 1000`` on this sum).

Linear only: these read corners, never high-order nodes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np

from ..._typing import BoolArray, FloatArray, IntArray, PointArray
from ...core.smoother import descent, metric
from ...core.smoother import quality as _quality
from ...core.smoother.metric import DELTA, EPS, KAPPA, CornerMetrics
from ...core.smoother.quality import ShapeStats
from ...core.smoother.schedule import (
    Registry,
    Schedule,
    StageLog,
    suggest_schedule,
    untangle_schedule,
    user_schedule,
)
from ...linemesh import LineMesh
from ...pointmesh import PointMesh
from ..quadmesh import QuadMesh
from ..query import boundary_edge_ids
from .constrain import (
    CORNER_DEG,
    FREE,
    FROZEN,
    TANGENT,
    Resolved,
    boundary_adjacency,
    boundary_corners,
    constrain_direction,
    snap,
    vertex_length,
    vertex_normals,
)

__all__ = ["KAPPA", "EPS", "CornerMetrics", "corner_vectors", "reference_normals",
           "DELTA", "corner_metrics", "objective", "gradient", "with_points", "Quality", "quality",
           "failing_elements", "StageLog", "laplace", "winslow", "optimize", "run_schedule", "user_schedule", "untangle_schedule",
           "suggest_schedule", "ShapeStats", "shape_stats", "quad_neighbours", "size_ratios", "Schedule",
           "register_method", "methods", "REGISTRY", "tmop"]


def corner_vectors(points: PointArray, corners: IntArray) -> tuple[PointArray, PointArray]:
    """``(a, b)``, each ``(Q, 4, 3)``: at every quad corner the edge to the next corner
    ``a`` and the edge to the previous corner ``b`` (CCW, so ``a x b`` points along the
    quad's own normal)."""
    P = points[corners]
    return np.roll(P, -1, axis=1) - P, np.roll(P, 1, axis=1) - P


def reference_normals(points: PointArray, corners: IntArray) -> PointArray:
    """``(Q, 3)`` unit normals, one per quad: the normalised sum of its four corner
    ``a x b`` -- the quad's own orientation, read once from the *input* mesh.  Use a
    geometry's true normals instead where the quads sit on a known surface."""
    a, b = corner_vectors(points, corners)
    n = np.cross(a, b).sum(axis=1)
    length = np.linalg.norm(n, axis=1, keepdims=True)
    return np.asarray(n / np.where(length > 0, length, 1.0))


def corner_metrics(points: PointArray, corners: IntArray, normals: PointArray, *,
                   delta: float = 0.0) -> CornerMetrics:
    """``phi`` and the signed scaled Jacobian at every corner of every quad.  ``delta > 0``
    returns the barrier-free ``phi_d = K c / s_d^2`` (``K = (A+B)^2 / 4AB``) the optimizer
    descends on; the default is the exact metric the report prints."""
    return metric.corner_metrics(corner_vectors(points, corners), normals, delta=delta)


def objective(points: PointArray, corners: IntArray, normals: PointArray, *,
              kappa: float = KAPPA, eps: float = EPS, delta: float = 0.0,
              weights: FloatArray | None = None) -> float:
    """``F = sum_q (1/4) sum_c [ w_q (phi - 1) + kappa max(0, eps - s) ]`` -- ``0`` only for an
    all-square, all-positive mesh.  ``weights`` ``(Q,)`` scale the *shape* term per quad (default 1):
    ``0`` makes a quad's shape irrelevant (a region whose size is only a constraint on its neighbour)
    while the inversion penalty still protects it.  Divide by the quad count for the mean the report shows."""
    m = corner_metrics(points, corners, normals, delta=delta)
    return metric.objective_value(m, 2, kappa=kappa, eps=eps, weights=weights)


def gradient(points: PointArray, corners: IntArray, normals: PointArray, *,
             kappa: float = KAPPA, eps: float = EPS, delta: float = 0.0,
             weights: FloatArray | None = None) -> PointArray:
    """Analytic ``dF/dx`` ``(P, 3)`` with the reference normals held fixed (the maths is in
    :mod:`nekmeshpy.core.smoother.metric`).  A corner's vertex receives ``-(da + db)``, its next
    vertex ``da``, its previous ``db``."""
    a, b = corner_vectors(points, corners)
    ga, gb = metric.corner_gradient((a, b), normals, kappa=kappa, eps=eps, delta=delta, weights=weights)
    return metric.assemble_gradient(points.shape[0], corners,
                                    (np.roll(corners, -1, axis=1), np.roll(corners, 1, axis=1)), (ga, gb))


def with_points(mesh: QuadMesh, points: PointArray) -> QuadMesh:
    """A **new** linear mesh with the shared corners moved to ``points`` ``(P, 3)``;
    connectivity, element tags, edge tags and point tags ride through verbatim and
    ``mesh`` is untouched (smoothing never writes through ``mesh.points``)."""
    if mesh.order != 1:
        raise ValueError("smoothing moves vertices only; got an order-%d mesh -- smooth the "
                         "linear mesh and rebuild the high-order nodes afterwards"
                         % mesh.order)
    pts = np.asarray(points, dtype=float)
    if pts.shape != mesh.points.shape:
        raise ValueError("with_points: expected %s points, got %s"
                         % (mesh.points.shape, pts.shape))
    lm = mesh.line_mesh
    new_lm = LineMesh(PointMesh(pts.copy(), lm.point_mesh.element_tags), lm.lines,
                      lm.interior, lm.element_tags)
    return QuadMesh(new_lm, mesh.quads, mesh.orient, mesh.interior, mesh.element_tags)


class Quality(NamedTuple):
    """One honest reading of a mesh.  Per-quad values are the *worst corner* of each quad
    (``sj`` its minimum, ``phi`` its maximum); the ``q`` fields are the tail the user watches:
    the value below which the worst 1%, 5% and 10% of quads fall."""

    n_quads: int
    n_inverted: int            # quads with a corner whose signed scaled Jacobian is <= 0
    sj_min: float
    sj_mean: float
    sj_median: float
    sj_q01: float
    sj_q05: float
    sj_q10: float
    phi_mean: float
    phi_max: float
    objective_mean: float      # F / n_quads, the sum objective's mean

    def __str__(self) -> str:
        return ("%d quads: %d inverted | scaled J min %.4f mean %.4f median %.4f "
                "worst 1%%/5%%/10%% %.4f/%.4f/%.4f | phi mean %.4f max %.3e | F/quad %.4f"
                % (self.n_quads, self.n_inverted, self.sj_min, self.sj_mean, self.sj_median,
                   self.sj_q01, self.sj_q05, self.sj_q10, self.phi_mean, self.phi_max,
                   self.objective_mean))


def quality(points: PointArray, corners: IntArray, normals: PointArray, *,
            kappa: float = KAPPA, eps: float = EPS) -> Quality:
    """:class:`Quality` of the linear mesh ``(points, corners)`` against ``normals``."""
    m = corner_metrics(points, corners, normals)
    return Quality(*_quality.reduce_corners(m, objective(points, corners, normals, kappa=kappa, eps=eps)))


def failing_elements(points: PointArray, corners: IntArray, normals: PointArray, *,
                     sj_target: float = EPS, short_edge: float = 0.1) -> BoolArray:
    """``(Q,)`` quads the local stages should work on: a corner with signed scaled Jacobian
    below ``sj_target``, or an edge shorter than ``short_edge`` times the median edge."""
    a, _ = corner_vectors(points, corners)
    return _quality.failing_mask(corner_metrics(points, corners, normals), np.linalg.norm(a, axis=2),
                                 sj_target=sj_target, short_edge=short_edge)


# Reference corner coordinates of the CCW bilinear quad, and the edge k -> (axis, side) map:
# edge 0 is s=-1, edge 1 is r=+1, edge 2 is s=+1, edge 3 is r=-1.
_RS = np.array([-1.0, 1.0, 1.0, -1.0])
_SS = np.array([-1.0, -1.0, 1.0, 1.0])


def _shrunk_corners(P: PointArray, corners: IntArray, bnd: BoolArray, w: float) -> PointArray:
    """Every quad's four corners re-evaluated from its own bilinear map at ``+-w`` in each
    reference axis -- the element shrinks toward its centre -- except that an axis whose
    side is a boundary edge keeps that side at ``+-1``, so a boundary corner is not pulled
    off the boundary: it only shrinks along it."""
    q = corners.shape[0]
    r_lo = np.where(bnd[:, 3], -1.0, -w)
    r_hi = np.where(bnd[:, 1], 1.0, w)
    s_lo = np.where(bnd[:, 0], -1.0, -w)
    s_hi = np.where(bnd[:, 2], 1.0, w)
    r = np.where(_RS[None, :] < 0, r_lo[:, None], r_hi[:, None])        # (Q,4) reference r of each new corner
    s = np.where(_SS[None, :] < 0, s_lo[:, None], s_hi[:, None])
    wt = 0.25 * (1.0 + r[:, :, None] * _RS[None, None, :]) * (1.0 + s[:, :, None] * _SS[None, None, :])
    out = np.einsum("qij,qjk->qik", wt, P[corners])
    assert out.shape == (q, 4, 3)
    return np.asarray(out)


def _unit_rows(v: PointArray) -> PointArray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.asarray(v / np.where(n > 0, n, 1.0))


def laplace(mesh: QuadMesh, resolved: Resolved, *, passes: int = 3, w: float = 0.97,
            normals: PointArray | None = None, elements: BoolArray | None = None,
            max_sj_drop: float = 0.05, kappa: float = KAPPA, eps: float = EPS) -> tuple[QuadMesh, StageLog]:
    """Element-shrink Laplace smoothing (the user's scheme, ported to quads).

    Each pass shrinks every quad toward its own centre by ``w`` (``0.97`` conservative; ``1``
    is a no-op), keeping boundary sides unshrunk across their own normal so boundary vertices
    only slide *along* the boundary -- which is what stops a plain average cancelling them --
    then averages each shared vertex over its incident quads and snaps constrained vertices
    back onto their targets with a weight ramping ``i/passes -> 1`` (the last pass is a full
    snap).  ``elements`` (a ``(Q,)`` mask) restricts the update to the vertices of those quads,
    the *local* variant for failing patches.

    A ``FREE`` boundary vertex (``resolve(..., boundary="free")``) takes the proposed move as is -- the 1-D
    Laplacian of the boundary polygon, which shrinks a convex boundary.  A ``TANGENT`` one
    (``boundary="tangent"``) keeps only the component of its move along the tangent of the circle through it
    and its two boundary neighbours (:func:`~.constrain.circle_tangent`, read from the current positions), so
    tension cannot pull the boundary in onto its chords; corners are frozen by ``resolve``.

    A pass is accepted only if it creates no new inverted quad and does not lower the minimum
    scaled Jacobian by more than ``max_sj_drop`` (a Laplace pass trades a little worst case for
    evenness; a larger loss is refused); the first refused pass rolls back to the last accepted
    iterate and stops the stage, saying why.  Frozen vertices never move.  Returns a **new**
    mesh and the :class:`StageLog`; ``mesh`` is untouched.  Linear meshes only."""
    if not 0.0 < w <= 1.0:
        raise ValueError("laplace: w must be in (0, 1], got %r" % (w,))
    if passes < 1:
        raise ValueError("laplace: passes must be >= 1, got %r" % (passes,))
    corners = mesh.corners
    P0 = np.asarray(mesh.points, float)
    nrm = reference_normals(P0, corners) if normals is None else np.asarray(normals, float)
    bnd = boundary_edge_ids(mesh)[mesh.quads]
    mult = np.bincount(corners.ravel(), minlength=mesh.n_points).astype(float)
    free: BoolArray | None = None
    if elements is not None:
        sel = np.asarray(elements, bool)
        if sel.shape != (mesh.n_quads,):
            raise ValueError("laplace: elements must be a (%d,) mask" % mesh.n_quads)
        free = np.zeros(mesh.n_points, bool)
        free[corners[sel].ravel()] = True
    method = "laplace" if elements is None else "laplace_local"
    before = quality(P0, corners, nrm, kappa=kappa, eps=eps)
    if elements is not None and not sel.any():
        return mesh, StageLog(method, passes, 0, "no_elements", "no quad selected", before,
                              before, ())
    tangent = resolved.kind == TANGENT
    P = P0.copy()
    cur = before
    history: list[tuple[int, int, float, float]] = []
    stopped, reason, done = "completed", "all %d passes accepted" % passes, 0
    for i in range(1, passes + 1):
        sh = _shrunk_corners(P, corners, bnd, w)
        avg = metric.scatter(mesh.n_points, corners, sh) / mult[:, None]
        if free is not None:
            avg[~free] = P[~free]
        if tangent.any():                    # a TANGENT vertex keeps only its move along the boundary
            avg[tangent] = P[tangent] + constrain_direction(avg - P, P, resolved)[tangent]
        cand = snap(avg, resolved, weight=i / passes)
        got = quality(cand, corners, nrm, kappa=kappa, eps=eps)
        if got.n_inverted > cur.n_inverted or got.sj_min < cur.sj_min - max_sj_drop:
            stopped = "rolled_back"
            reason = ("pass %d refused: inverted %d -> %d, min scaled J %.4f -> %.4f "
                      "(allowed drop %.3g)" % (i, cur.n_inverted, got.n_inverted, cur.sj_min,
                                               got.sj_min, max_sj_drop))
            break
        P, cur, done = cand, got, i
        history.append((i, cur.n_inverted, cur.sj_min, cur.objective_mean))
    if done < passes:
        # a rolled-back stage ends on a partial-weight snap: finish it, geometry comes first
        P = snap(P, resolved, 1.0)
        cur = quality(P, corners, nrm, kappa=kappa, eps=eps)
        if cur.n_inverted > history[-1][1] if history else cur.n_inverted > before.n_inverted:
            reason += "; the closing full snap added inverted quads (%d)" % cur.n_inverted
    return with_points(mesh, P), StageLog(method, passes, done, stopped, reason, before, cur,
                                          tuple(history))


def _chart(n: int) -> FloatArray:
    """``(2n, 2)`` logical positions of ``[e_0..e_{n-1}, d_0..d_{n-1}]`` in the rhombus chart of valence ``n``:
    the ``n`` incident quads are congruent rhombi of angle ``2 pi / n`` at the vertex (``n = 4`` is the unit grid)."""
    th = 2.0 * np.pi * np.arange(n) / n
    e = np.stack([np.cos(th), np.sin(th)], axis=1)
    return np.asarray(np.concatenate([e, e + np.roll(e, -1, axis=0)]))


def _ls_weights(n: int) -> FloatArray:
    """``(5, 2n)``: rows ``x_xi, x_eta, x_xixi, x_xieta, x_etaeta`` as linear functionals of ``x_k - x_v``, the
    least-squares quadratic Taylor fit over the chart of valence ``n``."""
    p = _chart(n)
    A = np.stack([p[:, 0], p[:, 1], 0.5 * p[:, 0] ** 2, p[:, 0] * p[:, 1], 0.5 * p[:, 1] ** 2], axis=1)
    return np.asarray(np.linalg.pinv(A))


def _ninepoint_weights() -> FloatArray:
    """The standard 9-point stencil on ``[E, N, W, S, NE, NW, SW, SE]`` (differences from the vertex)."""
    W = np.zeros((5, 8))
    W[0, 0], W[0, 2] = 0.5, -0.5
    W[1, 1], W[1, 3] = 0.5, -0.5
    W[2, 0] = W[2, 2] = 1.0
    W[3, 4:] = [0.25, -0.25, 0.25, -0.25]
    W[4, 1] = W[4, 3] = 1.0
    return W


class _Stencil:
    """The rings of one mesh, read once: interior vertices grouped by valence with their ordered one-ring, the
    boundary chains, and a colouring of the "shares a quad" graph for Gauss-Seidel."""

    def __init__(self, mesh: QuadMesh) -> None:
        C = np.asarray(mesh.corners, dtype=np.int64)
        n = self.n = int(mesh.n_points)
        inc: list[list[tuple[int, int]]] = [[] for _ in range(n)]
        for q in range(C.shape[0]):
            for k in range(4):
                inc[int(C[q, k])].append((q, k))
        nb = boundary_adjacency(mesh)
        self.bnd: BoolArray = np.zeros(n, bool)
        self.bnd[list(nb)] = True
        ids = np.array(sorted(nb), dtype=np.int64)
        ok = np.array([len(nb[int(i)]) == 2 for i in ids], bool)
        self.chain_ids = ids[ok]                                     # boundary vertices with two boundary neighbours
        self.chain_nb = np.array([nb[int(i)] for i in self.chain_ids], dtype=np.int64).reshape(-1, 2)
        corner = boundary_corners(np.asarray(mesh.points, float), nb, CORNER_DEG,
                                  vertex_normals(np.asarray(mesh.points, float), C))
        self.chain_corner = np.array([int(i) in corner for i in self.chain_ids], dtype=bool)
        self.pinch = ids[~ok]                                        # the others are never moved
        groups: dict[int, list[tuple[int, list[int]]]] = {}
        fallback: list[int] = []                                     # interior vertices whose fan does not close
        for v in range(n):
            if self.bnd[v] or not inc[v]:
                continue
            ring = self._ring(inc[v], C)
            if ring is None or len(inc[v]) < 3:
                fallback.append(v)
            else:
                groups.setdefault(len(inc[v]), []).append((v, ring))
        self.groups = [(val, np.array([g[0] for g in gs], np.int64), np.array([g[1] for g in gs], np.int64),
                        _ninepoint_weights() if val == 4 else _ls_weights(val)) for val, gs in sorted(groups.items())]
        self.fallback = np.array(fallback, np.int64)
        adj: list[set[int]] = [set() for _ in range(n)]
        for q in C:
            for a in q:
                adj[int(a)].update(int(b) for b in q if b != a)
        self.adj = adj
        col: IntArray = -np.ones(n, np.int64)
        for v in np.argsort(-np.array([len(a) for a in adj]), kind="stable"):
            used = {col[u] for u in adj[v] if col[u] >= 0}
            c = 0
            while c in used:
                c += 1
            col[v] = c
        self.colours = [np.flatnonzero(col == c) for c in range(int(col.max()) + 1)]

    @staticmethod
    def _ring(inc: list[tuple[int, int]], C: IntArray) -> list[int] | None:
        """``[e_0..e_{n-1}, d_0..d_{n-1}]`` (edge neighbours, then the opposite corners) CCW around the vertex, or
        None if the fan of quads does not close."""
        nxt = {int(C[q, (k + 1) % 4]): (q, k) for q, k in inc}
        q, k = inc[0]
        es, ds = [], []
        for _ in range(len(inc)):
            es.append(int(C[q, (k + 1) % 4]))
            ds.append(int(C[q, (k + 2) % 4]))
            prv = int(C[q, (k - 1) % 4])
            if prv not in nxt:
                return None
            q, k = nxt[prv]
        if (q, k) != inc[0] or len(set(es)) != len(es):
            return None
        return es + ds


def _winslow_moves(st: _Stencil, X: PointArray, mask: BoolArray) -> PointArray:
    """``(P, 3)`` unconstrained Winslow move ``x* - x`` at the vertices in ``mask`` (zero elsewhere).

    Interior vertex: ``x* = x + sum_k w_k (x_k - x) / sum_k w_k`` with ``w_k = alpha W_xixi,k - 2 beta W_xieta,k +
    gamma W_etaeta,k``, ``alpha, beta, gamma`` read from the current positions.  Boundary vertex: the 1-D rule along
    its chain, ``x* = (x_prev + x_next) / 2`` (a ``TANGENT`` vertex has it cut to the boundary tangent by
    :func:`constrain_direction`); a corner of the input boundary (:func:`boundary_corners`) is held."""
    d = np.zeros_like(X)
    for _, vs, nbr, W in st.groups:
        sel = mask[vs]
        if not sel.any():
            continue
        vs, nbr = vs[sel], nbr[sel]
        D = X[nbr] - X[vs][:, None, :]                               # (m, 2n, 3)
        G = np.einsum("jk,mkc->mjc", W[:2], D)
        xi, eta = G[:, 0], G[:, 1]
        al = np.einsum("mc,mc->m", eta, eta)
        be = np.einsum("mc,mc->m", xi, eta)
        ga = np.einsum("mc,mc->m", xi, xi)
        w = al[:, None] * W[2] - 2.0 * be[:, None] * W[3] + ga[:, None] * W[4]
        den = w.sum(axis=1)
        good = den > 1e-12 * (al + ga + 1e-300)
        mv = np.einsum("mk,mkc->mc", w, D) / np.where(good, den, 1.0)[:, None]
        mv[~good] = D[~good, :nbr.shape[1] // 2].mean(axis=1)        # a collapsed ring: the plain neighbour mean
        d[vs] = mv
    for v in st.fallback[mask[st.fallback]]:
        d[v] = X[list(st.adj[v])].mean(axis=0) - X[v]
    sel = mask[st.chain_ids]
    ids, nb, held = st.chain_ids[sel], st.chain_nb[sel], st.chain_corner[sel]
    if ids.size:
        a, b = X[nb[:, 0]], X[nb[:, 1]]
        mv = 0.5 * (a + b) - X[ids]
        mv[held] = 0.0
        d[ids] = mv
    return d


def _vertex_normals(corners: IntArray, nrm: PointArray, n: int) -> PointArray:
    """Unit normal per vertex: the mean of its quads' reference normals."""
    v = np.zeros((n, 3))
    for k in range(4):
        np.add.at(v, corners[:, k], nrm)
    return _unit_rows(v)


def winslow(mesh: QuadMesh, resolved: Resolved, *, sweeps: int = 200, omega: float = 1.0, tol: float = 1e-6,
            normals: PointArray | None = None, elements: BoolArray | None = None,
            max_sj_drop: float = 0.05, retries: int = 3,
            kappa: float = KAPPA, eps: float = EPS) -> tuple[QuadMesh, StageLog]:
    """Winslow (inverse-Laplace) smoothing: an alternative to :func:`laplace`.

    Asks the logical coordinates to be harmonic in physical space, i.e. per Cartesian component
    ``alpha x_xixi - 2 beta x_xieta + gamma x_etaeta = 0`` with ``alpha = |x_eta|^2``, ``beta = x_xi . x_eta``,
    ``gamma = |x_xi|^2``.  Each sweep is one multicolour Gauss-Seidel pass of the per-vertex update (Picard:
    ``alpha, beta, gamma`` from the current positions): the 9-point stencil at a valence-4 vertex, a least-squares
    stencil on a local rhombus chart at any other valence, the 1-D rule along the chain at a boundary vertex.
    The move is relaxed by ``omega`` and then handled exactly as :func:`optimize` handles a step:
    :func:`constrain_direction` (frozen: none, curve: its tangent, surface: its tangent plane; ``FREE`` vertices
    of a non-planar mesh keep the tangent plane of their reference normal), then a full :func:`snap`.
    A ``FREE`` boundary vertex takes the 1-D chain move as is and a ``TANGENT`` one only its part along the
    boundary tangent (:func:`laplace`); a corner is held either way.  ``elements`` restricts the
    update to the vertices of those quads.

    A sweep that creates an inverted quad, or lowers the minimum scaled Jacobian by more than ``max_sj_drop``
    **below the best any accepted sweep reached** (below the last accepted one while the mesh still has
    an inverted quad, whose minimum is not a level to defend), is retried with ``omega`` halved up to ``retries`` times, then
    refused: the stage stops at the last accepted iterate.  Measuring against the best, not the last, is what
    catches a slow drift (a closed surface under many sweeps).  The stage also stops once the largest move falls
    below ``tol`` edge lengths.  ``StageLog.passes_done`` counts accepted sweeps.  Returns a **new** mesh and the
    :class:`StageLog`; ``mesh`` is untouched.  Linear meshes only."""
    if not 0.0 < omega <= 1.0:
        raise ValueError("winslow: omega must be in (0, 1], got %r" % (omega,))
    if sweeps < 1:
        raise ValueError("winslow: sweeps must be >= 1, got %r" % (sweeps,))
    corners = np.asarray(mesh.corners, dtype=np.int64)
    P0 = np.asarray(mesh.points, float)
    nrm = reference_normals(P0, corners) if normals is None else np.asarray(normals, float)
    method = "winslow" if elements is None else "winslow_local"
    before = quality(P0, corners, nrm, kappa=kappa, eps=eps)
    movable = resolved.kind != FROZEN
    if elements is not None:
        sel = np.asarray(elements, bool)
        if sel.shape != (mesh.n_quads,):
            raise ValueError("winslow: elements must be a (%d,) mask" % mesh.n_quads)
        if not sel.any():
            return mesh, StageLog(method, sweeps, 0, "no_elements", "no quad selected", before, before, ())
        in_sel: BoolArray = np.zeros(mesh.n_points, bool)
        in_sel[corners[sel].ravel()] = True
        movable = movable & in_sel
    st = _Stencil(mesh)
    movable[st.pinch] = False
    free = resolved.kind == FREE
    vn = None if np.allclose(nrm, nrm[0], atol=1e-9) else _vertex_normals(corners, nrm, mesh.n_points)
    X = snap(P0, resolved, 1.0)
    cur = quality(X, corners, nrm, kappa=kappa, eps=eps)
    best = cur.sj_min
    history: list[tuple[int, int, float, float]] = []
    stopped, reason, done = "completed", "all %d sweeps accepted" % sweeps, 0

    def sweep(X: PointArray, om: float) -> tuple[PointArray, float]:
        ell = vertex_length(X, corners)
        Y = X.copy()
        r = 0.0
        for cl in st.colours:
            m: BoolArray = np.zeros(mesh.n_points, bool)
            m[cl] = True
            m &= movable
            if not m.any():
                continue
            d = _winslow_moves(st, Y, m)
            if vn is not None:
                on = free & m
                d[on] -= np.einsum("ij,ij->i", d[on], vn[on])[:, None] * vn[on]
            d = constrain_direction(d, Y, resolved)
            d[~m] = 0.0
            r = max(r, float((np.linalg.norm(d[m], axis=1) / ell[m]).max()))
            Y = snap(Y + om * d, resolved, 1.0)
        return Y, r

    for i in range(1, sweeps + 1):
        om = omega
        for _ in range(retries + 1):
            Y, r = sweep(X, om)
            got = quality(Y, corners, nrm, kappa=kappa, eps=eps)
            floor = (best if cur.n_inverted == 0 else cur.sj_min) - max_sj_drop
            if got.n_inverted <= cur.n_inverted and got.sj_min >= floor:
                break
            om *= 0.5
        else:
            stopped = "rolled_back"
            reason = ("sweep %d refused after %d omega halvings: inverted %d -> %d, min scaled J %.4f -> %.4f "
                      "(allowed floor %.4f)" % (i, retries, cur.n_inverted, got.n_inverted, cur.sj_min,
                                                  got.sj_min, floor))
            break
        X, cur, done = Y, got, i
        best = max(best, cur.sj_min)
        history.append((i, cur.n_inverted, cur.sj_min, cur.objective_mean))
        if r < tol:
            stopped, reason = "completed", "converged: largest move %.1e edge lengths (tol %.0e)" % (r, tol)
            break
    return with_points(mesh, X), StageLog(method, sweeps, done, stopped, reason, before, cur, tuple(history))


def _eval(P: PointArray, corners: IntArray, normals: PointArray, kappa: float,
          eps: float, delta: float = 0.0,
          weights: FloatArray | None = None) -> tuple[float, int]:
    """``(F, n_inverted)`` over the given quads in one pass over the corner metrics."""
    m = corner_metrics(P, corners, normals, delta=delta)
    return metric.evaluate(m, 2, kappa=kappa, eps=eps, weights=weights)


def optimize(mesh: QuadMesh, resolved: Resolved, *, iters: int = 20,
             normals: PointArray | None = None, elements: BoolArray | None = None,
             alpha_max: float = 1.0, restart: str = "halve", kappa: float = KAPPA,
             eps: float = EPS, delta: float = DELTA,
             tol_step: float = 1e-4, tol_rel: float = 1e-6,
             weights: FloatArray | None = None) -> tuple[QuadMesh, StageLog]:
    """Condition-number optimizer: nonlinear conjugate gradient on :func:`objective`
    (Fletcher-Reeves), with the mesh's constraints enforced **every step**.

    Everything is measured in one unit, the local edge length ``l_v`` (shortest incident edge,
    frozen at stage start): the variables are ``u_v = dx_v / l_v``, the search direction is
    normalised so its largest move is one ``l_v``, and the line-search variable ``alpha`` is
    that move in edge lengths, capped at ``alpha_max``.  Nothing depends on the element count
    or the model's units; the objective is a *sum*, so the gradient does not either.

    Each step: take the analytic gradient, project it with the targets' own normals/tangents
    (``constrain_direction``), form the CG direction, and line-search ``alpha`` on
    ``F(snap(x + alpha l s))`` -- the objective **after** the snap, so the search sees what the
    mesh will actually become.  A trial that would add an inverted quad is never accepted, and the stage is refused
    if it ends with more inverted quads than it began with.  The mesh may *start* inverted and be
    repaired: the penalty term is what pulls an inverted
    corner back, which is the reason this method exists next to :func:`laplace`.

    The descent runs on the **barrier-free** metric (``delta``, see :data:`DELTA`): the exact
    ``phi`` is a wall at a corner's zero Jacobian, which a mesh that starts inverted could never
    cross.  The numbers the log reports are the exact metric.  ``delta=0`` descends on the exact
    one (cannot untangle).

    ``weights`` ``(Q,)`` scale each quad's shape term (see :func:`objective`).

    Stops on ``iters``, a step below ``tol_step`` edge lengths, a relative drop of ``F`` below
    ``tol_rel``, or a mean excess below ``1e-10`` (already ideal).  ``restart="halve"`` halves
    ``beta`` on every second iteration (the user's tuned variant), ``"none"`` is plain
    Fletcher-Reeves.  ``elements`` restricts the work to the vertices of those quads and the
    quads touching them (the *local* variant).  Returns a **new** mesh and the
    :class:`StageLog`.  Linear meshes only."""
    if iters < 1:
        raise ValueError("optimize: iters must be >= 1, got %r" % (iters,))
    if restart not in ("halve", "none"):
        raise ValueError("optimize: restart must be 'halve' or 'none', got %r" % (restart,))
    corners = mesh.corners
    P0 = np.asarray(mesh.points, float)
    nrm = reference_normals(P0, corners) if normals is None else np.asarray(normals, float)
    method = "optimize" if elements is None else "optimize_local"
    before = quality(P0, corners, nrm, kappa=kappa, eps=eps)
    use = np.ones(corners.shape[0], bool)
    movable = resolved.kind != FROZEN
    if elements is not None:
        sel = np.asarray(elements, bool)
        if sel.shape != (mesh.n_quads,):
            raise ValueError("optimize: elements must be a (%d,) mask" % mesh.n_quads)
        if not sel.any():
            return mesh, StageLog(method, iters, 0, "no_elements", "no quad selected", before,
                                  before, ())
        vmask: BoolArray = np.zeros(mesh.n_points, bool)
        vmask[corners[sel].ravel()] = True
        movable = movable & vmask
        use = vmask[corners].any(axis=1)             # the quads the movable vertices touch
    cs, ns = corners[use], nrm[use]
    ws = None if weights is None else np.asarray(weights, float)[use]
    ell = vertex_length(P0, corners)[:, None]
    P = snap(P0, resolved, 1.0) if elements is None else P0.copy()
    res = descent.minimize(
        P, evaluate=lambda Q: _eval(Q, cs, ns, kappa, eps, delta, ws),
        gradient=lambda X: gradient(X, cs, ns, kappa=kappa, eps=eps, delta=delta, weights=ws),
        project=lambda d, X: constrain_direction(d, X, resolved), snap=lambda X: snap(X, resolved, 1.0),
        summary=lambda X: _quality.history_row(quality(X, corners, nrm, kappa=kappa, eps=eps)),
        ell=ell, movable=movable, tangent=resolved.kind == TANGENT, n_used=max(int(use.sum()), 1),
        iters=iters, alpha_max=alpha_max, restart=restart, tol_step=tol_step, tol_rel=tol_rel)
    P, stopped, reason, done, history = res.P, res.stopped, res.reason, res.done, res.history
    P = snap(P, resolved, 1.0)
    after = quality(P, corners, nrm, kappa=kappa, eps=eps)
    if after.n_inverted > before.n_inverted:
        # the objective fell but more quads are inverted than we started with: refuse the stage
        return mesh, StageLog(method, iters, 0, "rolled_back",
                              "ended with %d inverted quads vs %d at the start; stage refused"
                              % (after.n_inverted, before.n_inverted), before, before, ())
    return with_points(mesh, P), StageLog(method, iters, done, stopped, reason, before, after,
                                          tuple(history))


def tmop(mesh: QuadMesh, resolved: Resolved, *, normals: PointArray | None = None,
         designed: Any = None, **knobs: Any) -> tuple[QuadMesh, StageLog]:
    """The MFEM / TMOP stage (``("tmop", {...})`` in a schedule), a thin wrapper over
    :func:`nekmeshpy.core.smoother.mfem.tmop`: the quads of a **planar** mesh, ``resolved`` mapped to essential
    components (frozen: all; axis-aligned lines: slide; anything curved: held), the element tags as
    attributes -- ``designed`` names the tags (or a ``(Q,)`` mask) of elements that keep their own shape --
    and ``knobs`` its ``target`` / ``metric`` / ``solver`` / ``iters`` / ... .  Needs PyMFEM, imported only when
    this runs.  Returns a **new** linear mesh and the :class:`StageLog`; ``mesh`` is untouched."""
    from ...core.smoother import mfem as _mfem

    corners = mesh.corners
    P0 = np.asarray(mesh.points, float)
    nrm = reference_normals(P0, corners) if normals is None else np.asarray(normals, float)
    attr, dmask = _mfem.regions(mesh.element_tags.to_dense(), designed)
    P, log = _mfem.tmop(P0, corners, resolved, attributes=attr, designed=dmask,
                        quality=lambda X: quality(X, corners, nrm, kappa=KAPPA, eps=EPS), **knobs)
    return with_points(mesh, P), log


# -- schedules and the method registry ------------------------------------------------------------
#: The stage names :func:`run_schedule` knows; :func:`register_method` adds one.
REGISTRY = Registry()
for _name, _fn, _local, _w in (("laplace", laplace, False, False), ("laplace_local", laplace, True, False),
                               ("winslow", winslow, False, False), ("winslow_local", winslow, True, False),
                               ("optimize", optimize, False, True), ("optimize_local", optimize, True, True)):
    REGISTRY.register(_name, _fn, local=_local, weights=_w)
REGISTRY.register("tmop", tmop)               # needs PyMFEM, imported when the stage runs


def register_method(name: str, fn: Callable[..., tuple[QuadMesh, StageLog]], *, local: bool = False,
                    weights: bool = False, replace: bool = False) -> None:
    """Make ``fn`` available to :func:`run_schedule` as the stage ``name`` -- the hook another backend plugs
    in by.  ``fn(mesh, resolved, normals=..., **knobs)`` must return a **new** mesh and a :class:`StageLog`,
    leave ``mesh`` untouched and honour ``resolved``; ``local=True`` hands it ``elements=`` (the failing quads),
    ``weights=True`` the schedule's ``weights=``.  An existing name is refused unless ``replace=True``."""
    REGISTRY.register(name, fn, local=local, weights=weights, replace=replace)


def methods() -> list[str]:
    """The stage names :func:`run_schedule` knows, sorted."""
    return REGISTRY.names()


def run_schedule(mesh: QuadMesh, resolved: Resolved, schedule: Schedule, *,
                 normals: PointArray | None = None, sj_target: float = EPS,
                 weights: FloatArray | None = None,
                 record: bool = False) -> tuple[QuadMesh, list[StageLog]]:
    """Run the stages in order, each on the previous stage's result, with the reference
    normals read once from the *input* (or ``normals``) so orientation is never re-guessed.
    ``*_local`` stages pick their quads with :func:`failing_elements` at the moment they run.
    A stage is any name in the registry (:func:`methods`).
    Returns the final mesh and one :class:`StageLog` per stage; ``mesh`` is untouched.  ``record=True``
    keeps a copy of the vertex positions after every stage in ``StageLog.points`` (frame ``0`` is the input
    itself), which is what :func:`nekmeshpy.quadmesh.smoother.frames.write_frames` plays back."""
    nrm = reference_normals(mesh.points, mesh.corners) if normals is None else normals
    return REGISTRY.run(mesh, resolved, schedule,
                        failing=lambda m: failing_elements(m.points, m.corners, nrm, sj_target=sj_target),
                        weights=weights, common={"normals": nrm}, record=record)


# -- shape readings ---------------------------------------------------------------------------
def quad_neighbours(corners: IntArray) -> IntArray:
    """``(M, 2)`` pairs of quads that share an edge (an edge borne by more than two quads yields the
    consecutive pairs)."""
    c = np.asarray(corners)
    e = np.stack([c, np.roll(c, -1, axis=1)], axis=2).reshape(-1, 2)          # (4Q, 2) edge of quad row//4
    key = np.sort(e, axis=1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    ks = key[order]
    same = (ks[1:] == ks[:-1]).all(axis=1)
    a, b = order[:-1][same] // 4, order[1:][same] // 4
    return np.stack([a, b], axis=1).astype(np.int64)


def size_ratios(points: PointArray, corners: IntArray) -> FloatArray:
    """``(M,)`` larger-over-smaller ratio of ``sqrt(area)`` for every pair of edge-sharing quads."""
    P = np.asarray(points, dtype=float)
    Q = P[corners]
    area = 0.5 * np.linalg.norm(np.cross(Q[:, 2] - Q[:, 0], Q[:, 3] - Q[:, 1]), axis=1)
    length = np.sqrt(np.maximum(area, 1e-300))
    return _quality.pair_ratios(length, quad_neighbours(corners))


def shape_stats(points: PointArray, corners: IntArray) -> ShapeStats:
    """:class:`ShapeStats` of the linear mesh ``(points, corners)``."""
    a, b = corner_vectors(points, corners)
    edge = np.linalg.norm(a, axis=2)                                   # (Q, 4) edge k -> k+1
    return _quality.shape_stats(edge, _quality.corner_angles(a, b), size_ratios(points, corners))
