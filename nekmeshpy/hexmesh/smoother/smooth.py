"""Hex smoother kernels: the corner shape metric, its objective and analytic gradient, the
element-shrink Laplace, the optimizer, and the schedules that string them together.

Every quantity is read off the 3x3 corner Jacobian ``J = [a b c]`` of a hex corner -- ``a``,
``b``, ``c`` the three edge vectors leaving it, ordered right-handed (the order
:func:`nekmeshpy.hexmesh.quality.corner_scaled_jacobian` uses), so ``det J > 0`` on a valid
corner.  With ``A = |a|^2``, ``B = |b|^2``, ``C = |c|^2`` and ``g = det J = (a x b) . c``:

Signed scaled Jacobian::

    s = g / (|a| |b| |c|)  in [-1, 1],  1 only for three orthogonal edges

Shape metric (Knupp's mean-ratio shape measure ``eta = |J|_F^2 / (3 det J^(2/3))`` to the power
``d = 3``)::

    phi = eta^3 = ((A + B + C) / 3)^3 / g^2 = K / s^2,   K = ((A + B + C) / 3)^3 / (A B C) >= 1

``phi >= 1`` with equality only for a *cube* corner: ``K = 1`` needs three equal edges (AM-GM)
and ``s^2 = 1`` three orthogonal ones.  It is the exact continuation of the quad metric, which
is ``eta^2 = (A + B)^2 / (4 |a x b|^2) = K / s^2`` at ``d = 2``: invariant to scale, rotation and
translation, a ``1/s^2`` wall at a zero Jacobian, and orientation-blind -- so inversion is the
same separate, signed penalty ``kappa * max(0, eps - s)``, and the barrier-free
``phi_d = K c / s_d^2`` (``s_d = (s + sqrt(s^2 + 4 delta^2)) / 2``, ``c = s_d(1)^2``) carries over
unchanged, as do :data:`KAPPA`, :data:`EPS` and :data:`DELTA`.  The condition-number form
``|J|_F^2 |J^-1|_F^2 / 9`` would also score 1 only at a cube, but its gradient needs the adjugate
and it no longer reduces to the quad metric; ``eta^3`` keeps the two smoothers one family.

The objective is the **sum** over hexes of the corner mean (``1/8``) of
``w (phi - 1) + kappa max(0, eps - s)``; a vertex's gradient involves only its incident hexes.

Linear only: these read corners, never high-order nodes.  A curved mesh is smoothed as its
straight-sided shadow (:func:`nekmeshpy.hexmesh.smoother.highorder.linearize`); rebuilding its
curved nodes afterwards is not available yet.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import numpy as np

from ..._typing import BoolArray, FloatArray, IntArray, PointArray
from ...quadmesh.smoother import smooth as _qs
from ...quadmesh.smoother.smooth import DELTA, EPS, KAPPA, _regularised
from ..hexmesh import HexMesh
from ..query import boundary_face_ids
from .constrain import (
    FROZEN,
    TANGENT,
    Resolved,
    constrain_direction,
    snap,
    vertex_length,
)

__all__ = ["KAPPA", "EPS", "DELTA", "CornerMetrics", "corner_vectors", "corner_metrics", "objective",
           "gradient", "with_points", "Quality", "quality", "failing_elements", "StageLog", "laplace",
           "optimize", "Method", "register_method", "methods", "run_schedule", "user_schedule",
           "untangle_schedule", "suggest_schedule", "ShapeStats", "shape_stats", "hex_neighbours",
           "size_ratios", "Schedule"]

#: Floor on ``s^2`` that keeps a collapsed corner finite instead of ``inf``/``nan``.
_S2_FLOOR = 1e-14

#: Corner ``k`` -> ``[k, a-end, b-end, c-end]``: the three edge neighbours of each Nek-order hex
#: corner, ordered so ``(a x b) . c > 0`` on a valid hex (``hexmesh.quality``'s own table).
_CN = np.array([[0, 1, 3, 4], [1, 2, 0, 5], [2, 3, 1, 6], [3, 0, 2, 7],
                [4, 7, 5, 0], [5, 4, 6, 1], [6, 5, 7, 2], [7, 6, 4, 3]], dtype=np.int64)

#: The 12 hex edges as Nek-order corner pairs.
_EDGES = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
                   [0, 4], [1, 5], [2, 6], [3, 7]], dtype=np.int64)


class CornerMetrics(NamedTuple):
    """Per-corner readings, each ``(E, 8)`` in the hex's own Nek corner order."""

    #: ``phi >= 1``, ``1`` at a cube corner; blind to orientation.
    phi: FloatArray
    #: Signed scaled Jacobian in ``[-1, 1]``; ``<= 0`` is inverted or degenerate.
    sj: FloatArray


def corner_vectors(points: PointArray, hexes: IntArray) -> tuple[PointArray, PointArray, PointArray]:
    """``(a, b, c)``, each ``(E, 8, 3)``: the three right-handed edge vectors at every hex corner."""
    P = np.asarray(points, float)
    H = np.asarray(hexes, dtype=np.int64)
    X = P[H[:, _CN.ravel()]].reshape(H.shape[0], 8, 4, 3)
    o = X[:, :, 0]
    return X[:, :, 1] - o, X[:, :, 2] - o, X[:, :, 3] - o


def _terms(points: PointArray, hexes: IntArray) -> tuple[
        PointArray, PointArray, PointArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """The shared per-corner pieces: ``a, b, c, A, B, C, L = |a||b||c|, s``."""
    a, b, c = corner_vectors(points, hexes)
    A = np.einsum("ekj,ekj->ek", a, a)
    B = np.einsum("ekj,ekj->ek", b, b)
    C = np.einsum("ekj,ekj->ek", c, c)
    L = np.sqrt(A * B * C)
    g = np.einsum("ekj,ekj->ek", np.cross(a, b), c)
    # a corner with a zero-length edge has no direction: its signed Jacobian is 0 -- counted as
    # inverted, never NaN, so the gates see it
    s = np.where(L > 0.0, g / np.where(L > 0.0, L, 1.0), 0.0)
    return a, b, c, A, B, C, L, s


def _shape_factor(A: FloatArray, B: FloatArray, C: FloatArray) -> FloatArray:
    """``K = ((A + B + C) / 3)^3 / (A B C) >= 1``; ``inf`` on a zero-length edge."""
    with np.errstate(divide="ignore", invalid="ignore"):
        K = ((A + B + C) / 3.0) ** 3 / (A * B * C)
    return np.asarray(np.where(np.isfinite(K), K, np.inf))


def _sd(s: FloatArray, delta: float) -> tuple[FloatArray, float, FloatArray]:
    """``(s_d, c, ds_d/ds)``: the regularised sine for ``delta > 0`` (see the module docstring),
    else ``s`` itself with ``|s|`` floored at ``sqrt(_S2_FLOOR)`` (derivative ``0`` there)."""
    if delta > 0.0:
        return _regularised(s, delta)
    small = s * s < _S2_FLOOR
    return (np.where(small, np.sqrt(_S2_FLOOR), s), 1.0,
            np.where(small, 0.0, 1.0))


def corner_metrics(points: PointArray, hexes: IntArray, *, delta: float = 0.0) -> CornerMetrics:
    """``phi`` and the signed scaled Jacobian at every corner of every hex.  ``delta > 0`` returns
    the barrier-free ``phi_d = K c / s_d^2`` the optimizer descends on; the default is the exact
    metric the report prints."""
    _, _, _, A, B, C, _, s = _terms(points, hexes)
    K = _shape_factor(A, B, C)
    sd, cst, _ = _sd(s, delta)
    with np.errstate(invalid="ignore"):
        phi = K * cst / sd ** 2
    return CornerMetrics(phi=np.where(np.isnan(phi), np.inf, phi), sj=s)


def objective(points: PointArray, hexes: IntArray, *, kappa: float = KAPPA, eps: float = EPS,
              delta: float = 0.0, weights: FloatArray | None = None) -> float:
    """``F = sum_e (1/8) sum_c [ w_e (phi - 1) + kappa max(0, eps - s) ]`` -- ``0`` only for an
    all-cube, all-positive mesh.  ``weights`` ``(E,)`` scale the *shape* term per hex (default 1;
    ``0`` makes a hex's shape irrelevant while the inversion penalty still protects it).  Divide
    by the hex count for the mean the report shows."""
    m = corner_metrics(points, hexes, delta=delta)
    w = 1.0 if weights is None else np.asarray(weights, float)[:, None]
    return float(0.125 * (w * (m.phi - 1.0) + kappa * np.maximum(0.0, eps - m.sj)).sum())


def gradient(points: PointArray, hexes: IntArray, *, kappa: float = KAPPA, eps: float = EPS,
             delta: float = 0.0, weights: FloatArray | None = None) -> PointArray:
    """Analytic ``dF/dx`` ``(P, 3)``.

    Per corner, with ``S = A + B + C`` and ``L = |a||b||c|``::

        ds/da = (b x c) / L - s a / A        (b: c x a, B;  c: a x b, C)
        dK/da = K (6 / S - 2 / A) a
        dphi/da = c [ dK/da / s_d^2 - 2 K s_d' ds/da / s_d^3 ]

    (``delta = 0``: ``s_d = s``, ``c = s_d' = 1``).  The corner's vertex receives
    ``-(da + db + dc)`` and the three edge ends ``da``, ``db``, ``dc``.  The penalty branch is on
    only where ``s < eps``."""
    a, b, c, A, B, C, L, s = _terms(points, hexes)
    K = _shape_factor(A, B, C)
    S = A + B + C
    Lx = np.where(L > 0.0, L, 1.0)[..., None]
    ds = [np.cross(b, c) / Lx - (s / np.where(A > 0, A, 1.0))[..., None] * a,
          np.cross(c, a) / Lx - (s / np.where(B > 0, B, 1.0))[..., None] * b,
          np.cross(a, b) / Lx - (s / np.where(C > 0, C, 1.0))[..., None] * c]
    sd, cst, dsd = _sd(s, delta)
    S2 = (sd ** 2)[..., None]
    S3 = (sd ** 3)[..., None]
    Kc = K[..., None]
    on = (s < eps)[..., None]
    w = 1.0 if weights is None else np.asarray(weights, float)[:, None, None]
    H = np.asarray(hexes, dtype=np.int64)
    nv = np.asarray(points).shape[0]
    out = np.zeros((nv, 3))
    total = np.zeros_like(a)
    for j, (v, X) in enumerate(((a, A), (b, B), (c, C))):
        dK = Kc * (6.0 / S - 2.0 / np.where(X > 0, X, 1.0))[..., None] * v
        dphi = cst * (dK / S2 - 2.0 * Kc * dsd[..., None] * ds[j] / S3)
        gj = 0.125 * (w * dphi - kappa * on * ds[j])
        total += gj
        out += _qs._scatter(nv, H[:, _CN[:, j + 1]], gj)
    return np.asarray(out - _qs._scatter(nv, H[:, _CN[:, 0]], total))


def _require_linear(mesh: HexMesh) -> None:
    """Refuse a curved mesh, naming the route: the native smoother is linear."""
    if mesh.order != 1:
        raise ValueError("the native hex smoother moves vertices only and works on linear meshes; got "
                         "an order-%d mesh -- smooth its straight-sided shadow (smoother.linearize(mesh)); "
                         "its curved nodes are not rebuilt afterwards" % mesh.order)


def with_points(mesh: HexMesh, points: PointArray) -> HexMesh:
    """A **new** linear mesh with the shared corners moved to ``points`` ``(P, 3)``; connectivity,
    element tags and face tags ride through verbatim and ``mesh`` is untouched."""
    _require_linear(mesh)
    pts = np.asarray(points, dtype=float)
    if pts.shape != mesh.points.shape:
        raise ValueError("with_points: expected %s points, got %s" % (mesh.points.shape, pts.shape))
    return HexMesh(_qs.with_points(mesh.quad_mesh, pts), mesh.hexes, mesh.orient, None,
                   mesh.element_tags)


class Quality(NamedTuple):
    """One honest reading of a linear hex mesh.  Per-hex values are the *worst corner* of each
    hex (``sj`` its minimum, ``phi`` its maximum); the ``q`` fields are the values below which the
    worst 1%, 5% and 10% of hexes fall."""

    n_hexes: int
    n_inverted: int            # hexes with a corner whose signed scaled Jacobian is <= 0
    sj_min: float
    sj_mean: float
    sj_median: float
    sj_q01: float
    sj_q05: float
    sj_q10: float
    phi_mean: float
    phi_max: float
    objective_mean: float      # F / n_hexes, the sum objective's mean

    def __str__(self) -> str:
        return ("%d hexes: %d inverted | scaled J min %.4f mean %.4f median %.4f "
                "worst 1%%/5%%/10%% %.4f/%.4f/%.4f | phi mean %.4f max %.3e | F/hex %.4f"
                % (self.n_hexes, self.n_inverted, self.sj_min, self.sj_mean, self.sj_median,
                   self.sj_q01, self.sj_q05, self.sj_q10, self.phi_mean, self.phi_max,
                   self.objective_mean))


def quality(points: PointArray, hexes: IntArray, *, kappa: float = KAPPA, eps: float = EPS) -> Quality:
    """:class:`Quality` of the linear hex mesh ``(points, hexes)`` (corner readings)."""
    m = corner_metrics(points, hexes)
    sj = m.sj.min(axis=1)
    phi = m.phi.max(axis=1)
    q = np.asarray(np.quantile(sj, [0.01, 0.05, 0.10])) if sj.size else np.zeros(3)
    f = float(0.125 * (m.phi - 1.0 + kappa * np.maximum(0.0, eps - m.sj)).sum())
    return Quality(int(sj.size), int((sj <= 0).sum()), float(sj.min(initial=np.inf)),
                   float(sj.mean()) if sj.size else 0.0, float(np.median(sj)) if sj.size else 0.0,
                   float(q[0]), float(q[1]), float(q[2]),
                   float(phi.mean()) if phi.size else 0.0, float(phi.max(initial=0.0)),
                   f / max(sj.size, 1))


def failing_elements(points: PointArray, hexes: IntArray, *, sj_target: float = EPS,
                     short_edge: float = 0.1) -> BoolArray:
    """``(E,)`` hexes the local stages should work on: a corner with signed scaled Jacobian below
    ``sj_target``, or an edge shorter than ``short_edge`` times the median edge."""
    sj = corner_metrics(points, hexes).sj.min(axis=1)
    edge = _edge_lengths(points, hexes)
    return np.asarray((sj < sj_target) | (edge.min(axis=1) < short_edge * np.median(edge)))


def _edge_lengths(points: PointArray, hexes: IntArray) -> FloatArray:
    """``(E, 12)`` edge lengths of every hex."""
    X = np.asarray(points, float)[np.asarray(hexes, dtype=np.int64)]
    return np.asarray(np.linalg.norm(X[:, _EDGES[:, 0]] - X[:, _EDGES[:, 1]], axis=2))


class StageLog(NamedTuple):
    """What one smoothing stage did -- the part of the round report this stage owns."""

    method: str
    passes_requested: int
    passes_done: int
    stopped: str               # "completed" | "converged" | "rolled_back" | "no_elements"
    reason: str
    before: Quality
    after: Quality
    #: Per pass ``(pass, n_inverted, sj_min, objective_mean)`` of the *accepted* iterates.
    history: tuple[tuple[int, int, float, float], ...]
    #: The vertex positions after this stage, only when ``run_schedule(..., record=True)`` asked for them.
    points: PointArray | None = None


# Reference coordinates of the 8 Nek-order corners.  Local face f (0-based) is
# 0: s=-1, 1: r=+1, 2: s=+1, 3: r=-1, 4: t=-1, 5: t=+1 (HexMesh.FACE_POINTS).
_RS = np.array([-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0])
_SS = np.array([-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0])
_TS = np.array([-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0])


def _shrunk_corners(P: PointArray, hexes: IntArray, bnd: BoolArray, w: float) -> PointArray:
    """Every hex's eight corners re-evaluated from its own trilinear map at ``+-w`` in each
    reference axis -- the element shrinks toward its centre -- except that an axis whose side is a
    boundary face keeps that side at ``+-1``, so a boundary corner is not pulled off the boundary:
    it only shrinks within it (the 3-D form of the quad rule)."""
    lo_hi = [(np.where(bnd[:, i_lo], -1.0, -w), np.where(bnd[:, i_hi], 1.0, w))
             for i_lo, i_hi in ((3, 1), (0, 2), (4, 5))]
    ref = [np.where(R[None, :] < 0, lo[:, None], hi[:, None]) for R, (lo, hi) in zip((_RS, _SS, _TS), lo_hi)]
    wt = 0.125 * ((1.0 + ref[0][:, :, None] * _RS[None, None, :])
                  * (1.0 + ref[1][:, :, None] * _SS[None, None, :])
                  * (1.0 + ref[2][:, :, None] * _TS[None, None, :]))
    return np.asarray(np.einsum("eij,ejk->eik", wt, P[hexes]))


def _elements_mask(mesh: HexMesh, elements: BoolArray | None, who: str) -> BoolArray | None:
    if elements is None:
        return None
    sel = np.asarray(elements, bool)
    if sel.shape != (mesh.n_hexes,):
        raise ValueError("%s: elements must be a (%d,) mask" % (who, mesh.n_hexes))
    return sel


def laplace(mesh: HexMesh, resolved: Resolved, *, passes: int = 3, w: float = 0.97,
            elements: BoolArray | None = None, max_sj_drop: float = 0.05, kappa: float = KAPPA,
            eps: float = EPS) -> tuple[HexMesh, StageLog]:
    """Element-shrink Laplace smoothing, the quad scheme in 3-D.

    Each pass shrinks every hex toward its own centre by ``w`` in its logical coordinates
    (``0.97`` conservative; ``1`` is a no-op), keeping a boundary face's side unshrunk across its
    own normal so boundary vertices only slide *within* the boundary, then averages each shared
    vertex over its incident hexes' shrunk corners and snaps constrained vertices back onto their
    targets with a weight ramping ``i/passes -> 1``.  ``elements`` (an ``(E,)`` mask) restricts the
    update to the vertices of those hexes, the *local* variant.

    A ``FREE`` boundary vertex (``resolve(..., boundary="free")``) takes the proposed move as is, which shrinks a
    convex boundary.  A ``TANGENT`` one (``boundary="tangent"``) keeps only the part of its move in its boundary
    patch's tangent plane (a face vertex) or along the circle through it and its two chain neighbours (an edge
    vertex); corners are frozen by ``resolve``.

    A pass is accepted only if it creates no new inverted hex and does not lower the minimum
    scaled Jacobian by more than ``max_sj_drop``; the first refused pass rolls back to the last
    accepted iterate and stops the stage, saying why.  Returns a **new** mesh and the
    :class:`StageLog`; ``mesh`` is untouched.  Linear meshes only."""
    if not 0.0 < w <= 1.0:
        raise ValueError("laplace: w must be in (0, 1], got %r" % (w,))
    if passes < 1:
        raise ValueError("laplace: passes must be >= 1, got %r" % (passes,))
    _require_linear(mesh)
    hexes = mesh.corners
    P0 = np.asarray(mesh.points, float)
    bnd = boundary_face_ids(mesh)[mesh.hexes]
    mult = np.bincount(hexes.ravel(), minlength=mesh.n_points).astype(float)
    sel = _elements_mask(mesh, elements, "laplace")
    method = "laplace" if sel is None else "laplace_local"
    before = quality(P0, hexes, kappa=kappa, eps=eps)
    free: BoolArray | None = None
    if sel is not None:
        if not sel.any():
            return mesh, StageLog(method, passes, 0, "no_elements", "no hex selected", before, before, ())
        free = np.zeros(mesh.n_points, bool)
        free[hexes[sel].ravel()] = True
    tangent = resolved.kind == TANGENT
    P = P0.copy()
    cur = before
    history: list[tuple[int, int, float, float]] = []
    stopped, reason, done = "completed", "all %d passes accepted" % passes, 0
    for i in range(1, passes + 1):
        sh = _shrunk_corners(P, hexes, bnd, w)
        avg = _qs._scatter(mesh.n_points, hexes, sh) / mult[:, None]
        if free is not None:
            avg[~free] = P[~free]
        if tangent.any():                    # a TANGENT vertex keeps only its tangential move
            avg[tangent] = P[tangent] + constrain_direction(avg - P, P, resolved)[tangent]
        cand = snap(avg, resolved, weight=i / passes)
        got = quality(cand, hexes, kappa=kappa, eps=eps)
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
        cur = quality(P, hexes, kappa=kappa, eps=eps)
        ref = history[-1][1] if history else before.n_inverted
        if cur.n_inverted > ref:
            reason += "; the closing full snap added inverted hexes (%d)" % cur.n_inverted
    return with_points(mesh, P), StageLog(method, passes, done, stopped, reason, before, cur,
                                          tuple(history))


def _eval(P: PointArray, hexes: IntArray, kappa: float, eps: float, delta: float,
          weights: FloatArray | None) -> tuple[float, int]:
    """``(F, n_inverted)`` over the given hexes in one pass over the corner metrics."""
    m = corner_metrics(P, hexes, delta=delta)
    w = 1.0 if weights is None else np.asarray(weights, float)[:, None]
    f = float(0.125 * (w * (m.phi - 1.0) + kappa * np.maximum(0.0, eps - m.sj)).sum())
    return f, int((m.sj.min(axis=1) <= 0).sum())


def optimize(mesh: HexMesh, resolved: Resolved, *, iters: int = 20, elements: BoolArray | None = None,
             alpha_max: float = 1.0, restart: str = "halve", kappa: float = KAPPA, eps: float = EPS,
             delta: float = DELTA, tol_step: float = 1e-4, tol_rel: float = 1e-6,
             weights: FloatArray | None = None) -> tuple[HexMesh, StageLog]:
    """Shape optimizer: nonlinear conjugate gradient on :func:`objective` (Fletcher-Reeves), with
    the mesh's constraints enforced **every step** -- the quad optimizer on the hex metric.

    Everything is measured in the local edge length ``l_v`` (shortest incident hex edge, frozen at
    stage start): the search direction is normalised so its largest move is one ``l_v``, and the
    line-search variable ``alpha`` is that move in edge lengths, capped at ``alpha_max``.  Each
    step projects the analytic gradient with the targets' own normals/tangents
    (:func:`~nekmeshpy.hexmesh.smoother.constrain.constrain_direction`), forms the CG direction and
    line-searches ``F(snap(x + alpha l s))`` -- the objective after the snap.  A trial that adds an
    inverted hex is never accepted, and the stage is refused if it ends with more inverted hexes
    than it began with.  The descent runs on the barrier-free metric (``delta``; ``0`` is the exact
    one, which cannot untangle); the log reports the exact metric.

    Stops on ``iters``, a step below ``tol_step`` edge lengths, a relative drop of ``F`` below
    ``tol_rel``, or a mean excess below ``1e-10``.  ``restart="halve"`` halves ``beta`` on every
    second iteration, ``"none"`` is plain Fletcher-Reeves.  ``elements`` restricts the work to the
    vertices of those hexes and the hexes touching them.  ``weights`` ``(E,)`` scale each hex's
    shape term.  Returns a **new** mesh and the :class:`StageLog`.  Linear meshes only."""
    if iters < 1:
        raise ValueError("optimize: iters must be >= 1, got %r" % (iters,))
    if restart not in ("halve", "none"):
        raise ValueError("optimize: restart must be 'halve' or 'none', got %r" % (restart,))
    _require_linear(mesh)
    hexes = mesh.corners
    P0 = np.asarray(mesh.points, float)
    sel = _elements_mask(mesh, elements, "optimize")
    method = "optimize" if sel is None else "optimize_local"
    before = quality(P0, hexes, kappa=kappa, eps=eps)
    use = np.ones(hexes.shape[0], bool)
    movable = resolved.kind != FROZEN
    if sel is not None:
        if not sel.any():
            return mesh, StageLog(method, iters, 0, "no_elements", "no hex selected", before, before, ())
        vmask: BoolArray = np.zeros(mesh.n_points, bool)
        vmask[hexes[sel].ravel()] = True
        movable = movable & vmask
        use = vmask[hexes].any(axis=1)
    hs = hexes[use]
    ws = None if weights is None else np.asarray(weights, float)[use]
    ell = vertex_length(P0, hexes)[:, None]
    P = snap(P0, resolved, 1.0) if sel is None else P0.copy()
    tangent = resolved.kind == TANGENT
    f, ninv = _eval(P, hs, kappa, eps, delta, ws)
    n_used = max(int(use.sum()), 1)
    history: list[tuple[int, int, float, float]] = []
    g_old = s = None
    alpha_prev = 0.1
    stopped, reason, done = "completed", "all %d iterations accepted" % iters, 0
    for it in range(1, iters + 1):
        if f / n_used < 1e-10:
            stopped, reason = "converged", "mean excess below 1e-10 (already ideal)"
            break
        gx = gradient(P, hs, kappa=kappa, eps=eps, delta=delta, weights=ws)
        gx = constrain_direction(gx, P, resolved)
        gx[~movable] = 0.0
        g = ell * gx                                     # gradient in u = dx / l
        gg = float((g * g).sum())
        if gg == 0.0:
            stopped, reason = "converged", "projected gradient is zero"
            break
        if s is None or g_old is None:
            s = -g
        else:
            beta = gg / float((g_old * g_old).sum())
            if restart == "halve" and it % 2 == 0:
                beta *= 0.5
            s = -g + beta * s
            if float((g * s).sum()) >= 0.0:              # lost descent: restart on steepest descent
                s = -g
        top = float(np.linalg.norm(s, axis=1).max())
        dirx = ell * s / top                             # largest move = 1 local edge per unit alpha
        if tangent.any():                                # a CG mix of older tangents: re-cut to today's
            dirx[tangent] = constrain_direction(dirx, P, resolved)[tangent]

        def trial(a: float, P: PointArray = P, dirx: PointArray = dirx,
                  ninv: int = ninv) -> tuple[float, int, PointArray]:
            Q = snap(P + a * dirx, resolved, 1.0)
            ff, nn = _eval(Q, hs, kappa, eps, delta, ws)
            return (ff if nn <= ninv else np.inf), nn, Q

        # bracket by doubling from the last accepted step, then refine by golden section
        a = min(alpha_prev * 2.0, alpha_max)
        pts = [(0.0, f)]
        while True:
            fa = trial(a)[0]
            pts.append((a, fa))
            if (fa > pts[-2][1] and len(pts) > 2) or a >= alpha_max or not np.isfinite(fa) and len(pts) > 2:
                break
            if fa > f and len(pts) == 2:                 # first trial already worse: shrink
                a *= 0.25
                pts = [(0.0, f)]
                if a < tol_step * 0.1:
                    break
                continue
            a = min(a * 2.0, alpha_max)
        best = min(pts, key=lambda t: t[1])
        i = pts.index(best)
        lo = pts[max(i - 1, 0)][0]
        hi = pts[min(i + 1, len(pts) - 1)][0]
        gr = 0.5 * (np.sqrt(5.0) - 1.0)
        for _ in range(12):
            if hi - lo < 1e-3 * max(hi, 1e-12):
                break
            x1, x2 = hi - gr * (hi - lo), lo + gr * (hi - lo)
            if trial(x1)[0] < trial(x2)[0]:
                hi = x2
            else:
                lo = x1
        cand = [(trial(x)[0], x) for x in (best[0], 0.5 * (lo + hi))]
        f_new, a_best = min(cand)
        if not np.isfinite(f_new) or a_best <= 0.0 or f_new >= f:
            stopped, reason = "converged", "no step lowers the objective (stalled at %.3e)" % f
            break
        _, ninv_new, Pn = trial(a_best)
        rel = (f - f_new) / max(abs(f), 1e-300)
        P, f, ninv = Pn, f_new, ninv_new
        g_old, alpha_prev, done = g, max(a_best, 1e-3), it
        q = quality(P, hexes, kappa=kappa, eps=eps)
        history.append((it, q.n_inverted, q.sj_min, q.objective_mean))
        if a_best < tol_step:
            stopped, reason = "converged", "step %.2e edge lengths below %.0e" % (a_best, tol_step)
            break
        if rel < tol_rel:
            stopped, reason = "converged", "relative drop %.1e below %.0e" % (rel, tol_rel)
            break
    P = snap(P, resolved, 1.0)
    after = quality(P, hexes, kappa=kappa, eps=eps)
    if after.n_inverted > before.n_inverted:
        return mesh, StageLog(method, iters, 0, "rolled_back",
                              "ended with %d inverted hexes vs %d at the start; stage refused"
                              % (after.n_inverted, before.n_inverted), before, before, ())
    return with_points(mesh, P), StageLog(method, iters, done, stopped, reason, before, after,
                                          tuple(history))


# -- schedules and the method registry -------------------------------------------------------
#: A schedule is an ordered list of ``(method, knobs)``; the methods are names in the registry.
Schedule = Sequence[tuple[str, dict[str, Any]]]


class Method(NamedTuple):
    """One registered smoothing method: ``fn(mesh, resolved, **knobs) -> (mesh, StageLog)``.
    ``local``: :func:`run_schedule` hands it ``elements=`` (the failing hexes) when it runs;
    ``weights``: it accepts ``weights=`` and receives :func:`run_schedule`'s."""

    fn: Callable[..., tuple[HexMesh, StageLog]]
    local: bool
    weights: bool


_METHODS: dict[str, Method] = {
    "laplace": Method(laplace, False, False),
    "laplace_local": Method(laplace, True, False),
    "optimize": Method(optimize, False, True),
    "optimize_local": Method(optimize, True, True),
}


def register_method(name: str, fn: Callable[..., tuple[HexMesh, StageLog]], *, local: bool = False,
                    weights: bool = False, replace: bool = False) -> None:
    """Make ``fn`` available to :func:`run_schedule` as the stage ``name`` -- the hook another
    backend plugs in by.  ``fn(mesh, resolved, **knobs)`` must return a **new** mesh and a
    :class:`StageLog`, leave ``mesh`` untouched and honour ``resolved``.  An existing name is
    refused unless ``replace=True``."""
    if not name or not isinstance(name, str):
        raise ValueError("register_method: name must be a non-empty string, got %r" % (name,))
    if name in _METHODS and not replace:
        raise ValueError("register_method: %r is already registered (pass replace=True)" % name)
    if not callable(fn):
        raise TypeError("register_method: fn must be callable, got %r" % (fn,))
    _METHODS[name] = Method(fn, local, weights)


def methods() -> list[str]:
    """The stage names :func:`run_schedule` knows, sorted."""
    return sorted(_METHODS)


def user_schedule(outer: int = 10, n_lap: int = 3, n_opt: int = 20, *, w: float = 0.97,
                  final_local: bool = True) -> list[tuple[str, dict[str, Any]]]:
    """The quad smoother's tuned shape -- ``outer`` times ``laplace(n_lap)`` then
    ``optimize(n_opt)``, closed by two local Laplace passes.  A starting suggestion only."""
    sched: list[tuple[str, dict[str, Any]]] = []
    for _ in range(outer):
        sched.append(("laplace", {"passes": n_lap, "w": w}))
        sched.append(("optimize", {"iters": n_opt}))
    if final_local:
        sched += [("laplace_local", {"passes": 5, "w": w})] * 2
    return sched


def untangle_schedule(deltas: Sequence[float] = (1.0, 0.3, 0.1, 0.03, 0.01),
                      iters: int = 60) -> list[tuple[str, dict[str, Any]]]:
    """Optimizer stages with a shrinking barrier width: a wide ``delta`` lets a folded vertex be
    pulled through, each narrower one sharpens the metric back toward the exact one."""
    return [("optimize", {"iters": iters, "delta": d}) for d in deltas]


def suggest_schedule(start: Quality, *, outer: int = 10) -> list[tuple[str, dict[str, Any]]]:
    """The schedule to *propose* for a mesh in the state ``start``: :func:`user_schedule`, preceded
    by :func:`untangle_schedule` when any hex starts inverted.  A suggestion, never silently run."""
    base = user_schedule(outer=outer)
    return (untangle_schedule() + base) if start.n_inverted > 0 else base


def run_schedule(mesh: HexMesh, resolved: Resolved, schedule: Schedule, *, sj_target: float = EPS,
                 weights: FloatArray | None = None,
                 record: bool = False) -> tuple[HexMesh, list[StageLog]]:
    """Run the stages in order, each on the previous stage's result.  A stage is any name in the
    registry (:func:`methods`; :func:`register_method` adds one); *local* stages pick their hexes
    with :func:`failing_elements` at the moment they run.  Returns the final mesh and one
    :class:`StageLog` per stage; ``mesh`` is untouched.  ``record=True`` keeps a copy of the vertex
    positions after every stage in ``StageLog.points``.  Linear meshes only: a curved mesh is
    refused with the route to take (:func:`~nekmeshpy.hexmesh.smoother.highorder.linearize`)."""
    _require_linear(mesh)
    unknown = [m for m, _ in schedule if m not in _METHODS]
    if unknown:
        raise ValueError("unknown smoothing method(s) %s; registered: %s" % (unknown, methods()))
    logs: list[StageLog] = []
    cur = mesh
    for name, knobs in schedule:
        m = _METHODS[name]
        kw = dict(knobs)
        if m.local:
            kw["elements"] = failing_elements(cur.points, cur.corners, sj_target=sj_target)
        if m.weights and weights is not None:
            kw.setdefault("weights", weights)
        cur, log = m.fn(cur, resolved, **kw)
        logs.append(log._replace(points=np.array(cur.points, copy=True)) if record else log)
    return cur, logs


# -- shape readings ---------------------------------------------------------------------------
class ShapeStats(NamedTuple):
    """Edge- and size-based readings that complement :class:`Quality`: aspect ratio (longest over
    shortest edge of a hex), node spacing, the angles between the edges at a corner, and the
    **size jump between face-neighbours** (``size_ratio_*``: larger over smaller of the two sizes
    ``|mean corner det J|^(1/3)``, so ``1`` is a perfectly graded mesh)."""

    aspect_max: float
    aspect_p95: float
    spacing_min: float
    spacing_max: float
    angle_min_deg: float
    angle_max_deg: float
    size_ratio_max: float
    size_ratio_p95: float
    size_ratio_mean: float


def hex_neighbours(hexes: IntArray) -> IntArray:
    """``(M, 2)`` pairs of hexes that share a face."""
    H = np.asarray(hexes, dtype=np.int64)
    key = np.sort(H[:, HexMesh.FACE_POINTS].reshape(-1, 4), axis=1)
    order = np.lexsort(key.T[::-1])
    ks = key[order]
    same = (ks[1:] == ks[:-1]).all(axis=1)
    return np.stack([order[:-1][same] // 6, order[1:][same] // 6], axis=1).astype(np.int64)


def size_ratios(points: PointArray, hexes: IntArray) -> FloatArray:
    """``(M,)`` larger-over-smaller size ratio for every pair of face-sharing hexes."""
    a, b, c = corner_vectors(points, hexes)
    vol = np.abs(np.einsum("ekj,ekj->ek", np.cross(a, b), c).mean(axis=1))
    size = np.cbrt(np.maximum(vol, 1e-300))
    pr = hex_neighbours(hexes)
    if pr.size == 0:
        return np.ones(0)
    ra = size[pr[:, 0]] / size[pr[:, 1]]
    return np.asarray(np.maximum(ra, 1.0 / ra))


def shape_stats(points: PointArray, hexes: IntArray) -> ShapeStats:
    """:class:`ShapeStats` of the linear hex mesh ``(points, hexes)``."""
    edge = _edge_lengths(points, hexes)
    aspect = edge.max(axis=1) / np.maximum(edge.min(axis=1), 1e-300)
    a, b, c = corner_vectors(points, hexes)
    angs = []
    for u, v in ((a, b), (b, c), (c, a)):
        cosang = np.einsum("ekj,ekj->ek", u, v) / np.maximum(
            np.linalg.norm(u, axis=2) * np.linalg.norm(v, axis=2), 1e-300)
        angs.append(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))
    ang = np.stack(angs)
    sr = size_ratios(points, hexes)
    return ShapeStats(float(aspect.max(initial=1.0)), float(np.quantile(aspect, 0.95)) if aspect.size else 1.0,
                      float(edge.min(initial=np.inf)), float(edge.max(initial=0.0)),
                      float(ang.min(initial=180.0)), float(ang.max(initial=0.0)),
                      float(sr.max(initial=1.0)), float(np.quantile(sr, 0.95)) if sr.size else 1.0,
                      float(sr.mean()) if sr.size else 1.0)
