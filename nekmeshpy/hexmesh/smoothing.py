"""Constrained hex-mesh smoothing."""

from __future__ import annotations

import logging

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from .._typing import BoolArray, FloatArray, IntArray, PointArray
from ..linemesh import LineMesh
from ..pointmesh import PointMesh
from ..quadmesh import QuadMesh
from ..trimesh import TriMesh, ops
from . import quality
from .hexmesh import HexMesh
from .query import _unique_edges, classify_points

_log = logging.getLogger("nekmeshpy")


def smooth(
    mesh: HexMesh,
    surface: TriMesh,
    *,
    smooth_iters: int = 8,
    smooth_lambda: float = 0.5,
    wall: str = "wall",
    project_to_stl: bool = True,
    untangle_iters: int = 40,
    quality_floor: float = 0.2,
) -> HexMesh:
    """Constrained untangle + polish, keeping the ``wall``-named points on ``surface``
    and opening/cap points fixed."""
    if mesh.order > 1:
        raise NotImplementedError(
            "hexmesh.smoothing.smooth: cannot smooth an order-%d mesh (operates on "
            "corner nodes only; high-order smoothing is not implemented yet). Use "
            "order=1." % mesh.order)
    if smooth_iters <= 0:
        return mesh
    lam0 = smooth_lambda or 0.5
    proj = project_to_stl
    nUnt = untangle_iters or 40
    qfloor = quality_floor or 0.2
    Oxyz, Otri = surface.points, surface.tris

    # a copy: ``points`` is the mesh's own live array, and a smoother that wrote into
    # it would be the one operation in the toolkit that mutates its input
    X, HC, nu = mesh.points.copy(), mesh.corners, mesh.n_points
    he = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
                   [0, 4], [1, 5], [2, 6], [3, 7]], dtype=np.int64)
    E = _unique_edges(HC, he)
    A = sp.coo_matrix((np.ones(E.shape[0] * 2),
                       (np.concatenate([E[:, 0], E[:, 1]]),
                        np.concatenate([E[:, 1], E[:, 0]]))), shape=(nu, nu)).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1
    Avg = sp.diags(1.0 / deg) @ A
    adj = _adjacency_lists(E, nu)
    NH = _incidence_lists(HC, nu)
    is_wall, is_fixed = classify_points(mesh, wall)
    free = ~is_fixed
    wtri = _wall_tri_neighbourhoods(X, is_wall, Oxyz, Otri)

    sj = quality.corner_scaled_jacobian(X, HC)
    _log.info("smoothing: untangle<=%d, polish %d, lambda=%.2f  (points=%d, wall=%d, fixed=%d)",
              nUnt, smooth_iters, lam0, nu, int(np.sum(is_wall)), int(np.sum(is_fixed)))
    _log.info("  quality before: min scaled Jac=%.4f  mean=%.4f", np.min(sj), np.mean(sj))

    # stage 1: point-local untangle
    mn = float(np.min(sj))
    for _ in range(nUnt):
        if mn >= qfloor:
            break
        active = _active_points(quality.corner_scaled_jacobian(X, HC), HC, adj, free, qfloor)
        if active.size == 0:
            break
        for v in active:
            tgt = X[adj[v], :].mean(axis=0)
            els = NH[v]
            base = float(np.min(quality.corner_scaled_jacobian(X, HC[els, :])))
            bestq = base
            bestx = X[v, :].copy()
            for fr in (1.0, 0.7, 0.4, 0.15):
                cand = (1 - fr) * X[v, :] + fr * tgt
                if is_wall[v]:
                    cand = ops.project_to_surface(surface, cand[None, :], Otri[wtri[v], :])[0]
                xo = X[v, :].copy()
                X[v, :] = cand
                q = float(np.min(quality.corner_scaled_jacobian(X, HC[els, :])))
                X[v, :] = xo
                if q > bestq + 1e-12:
                    bestq = q
                    bestx = cand
            X[v, :] = bestx
        mn_new = float(np.min(quality.corner_scaled_jacobian(X, HC)))
        if mn_new <= mn + 1e-9:
            mn = mn_new
            break
        mn = mn_new

    # stage 2: global Jacobi polish
    mn = float(np.min(quality.corner_scaled_jacobian(X, HC)))
    for _ in range(smooth_iters):
        target = Avg @ X
        lam = lam0
        accepted = False
        for _bt in range(6):
            Xn = X.copy()
            Xn[free, :] = (1 - lam) * X[free, :] + lam * target[free, :]
            if proj and np.any(is_wall):
                Xn[is_wall, :] = ops.project_to_surface(surface, Xn[is_wall, :])
            mnn = float(np.min(quality.corner_scaled_jacobian(Xn, HC)))
            if mnn >= mn - 1e-9:
                X = Xn
                mn = mnn
                accepted = True
                break
            lam = lam / 2
        if not accepted:
            break

    if proj and np.any(is_wall):
        X[is_wall, :] = ops.project_to_surface(surface, X[is_wall, :])

    sj = quality.corner_scaled_jacobian(X, HC)
    _log.info("  quality after : min scaled Jac=%.4f  mean=%.4f", np.min(sj), np.mean(sj))
    if np.min(sj) <= 0:
        _log.warning("  %d element(s) still non-positive after smoothing",
                     int(np.sum(sj <= 0)))
    # mesh.order == 1 is enforced above, so every interior in the ladder is empty
    # and this only ever moves corners.
    qm, lm = mesh.quad_mesh, mesh.quad_mesh.line_mesh
    return HexMesh(
        QuadMesh(LineMesh(PointMesh(X, lm.point_tags), lm.lines, lm.interior,
                          lm.element_tags),
                qm.quads, qm.orient, qm.interior, qm.element_tags),
        mesh.hexes, mesh.orient, mesh.interior, mesh.element_tags)


# -- helpers (module-private) -------------------------------------------
def _csr_groups(rows: IntArray, cols: IntArray, n: int) -> list[IntArray]:
    """``[cols where rows == i]`` for every ``i``, built through a CSR matrix.

    The list-of-lists this replaces cost one Python iteration per entry, which for the
    point-to-hex incidence is eight per element -- 1.6M of them on a 200k-hex mesh, and
    measured 32x slower than assembling the same thing as a sparse matrix and slicing
    its ``indptr``."""
    m = sp.coo_matrix((np.ones(rows.shape[0], dtype=np.int8), (rows, cols)),
                      shape=(n, int(cols.max()) + 1 if cols.size else 1)).tocsr()
    ind, ptr = m.indices, m.indptr
    return [np.asarray(ind[ptr[i]:ptr[i + 1]], dtype=np.int64) for i in range(n)]


def _adjacency_lists(E: IntArray, nu: int) -> list[IntArray]:
    """Point -> the points sharing an edge with it."""
    return _csr_groups(np.concatenate([E[:, 0], E[:, 1]]),
                       np.concatenate([E[:, 1], E[:, 0]]), nu)


def _incidence_lists(HC: IntArray, nu: int) -> list[IntArray]:
    """Point -> the hexes carrying it."""
    return _csr_groups(np.asarray(HC, dtype=np.int64).ravel(),
                       np.repeat(np.arange(HC.shape[0], dtype=np.int64), 8), nu)


def _active_points(
    sj: FloatArray, HC: IntArray, adj: list[IntArray], free: BoolArray, qfloor: float,
) -> IntArray:
    bad = np.flatnonzero(sj < qfloor)
    if bad.size == 0:
        return np.array([], dtype=np.int64)
    seed = np.unique(HC[bad, :])
    mark: BoolArray = np.zeros(free.size, dtype=bool)
    mark[seed] = True
    for v in seed:
        mark[adj[v]] = True
    return np.flatnonzero(mark & free)


def _wall_tri_neighbourhoods(
    X: PointArray, is_wall: BoolArray, Vx: PointArray, T: IntArray,
) -> list[IntArray | None]:
    nv = Vx.shape[0]
    VT: list[IntArray] = ops._vertex_fans(T, nv)
    wall_ids: IntArray = np.flatnonzero(is_wall)
    # one batched query for every wall node, rather than an O(V) scan apiece
    nearest: IntArray = (cKDTree(Vx).query(X[wall_ids, :])[1] if wall_ids.size
                         else np.zeros(0, dtype=np.int64))
    wtri: list[IntArray | None] = [None] * is_wall.size
    for k, v in enumerate(wall_ids):
        nvtx = int(nearest[k])
        ring = np.unique(T[VT[nvtx], :])
        fan = [VT[u] for u in ring]
        wtri[v] = np.unique(np.concatenate(fan)) if fan else np.array([], dtype=np.int64)
    return wtri
