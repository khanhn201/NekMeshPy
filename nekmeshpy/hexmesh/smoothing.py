"""Constrained hex-mesh smoothing.

``smooth`` untangles and polishes the assembled hex mesh while keeping wall points
on the triangulated ``surface`` and opening/cap points fixed. Two stages: a
point-local untangle, then a back-tracked global Jacobi polish that never lowers
the minimum scaled Jacobian.
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.sparse as sp

from .._typing import BoolArray, FloatArray, IntArray, PointArray
from ..trimesh import TriMesh, ops
from . import quality
from .hexmesh import HexMesh
from .query import _unique_edges, classify_points, weld

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
    """Constrained untangle + polish, keeping the ``wall``-named points on
    ``surface`` and opening/cap points fixed. Runs up to ``untangle_iters``
    point-local sweeps (stopping once every element clears ``quality_floor``), then
    ``smooth_iters`` global polish sweeps (``<=0`` returns the mesh unchanged).

    Operates on the corner graph only, so an ``order > 1`` mesh is rejected --
    high-order smoothing is not implemented yet."""
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

    X, HC, nu = weld(mesh)
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

    sj = quality.scaled_jacobian(X, HC)
    _log.info("smoothing: untangle<=%d, polish %d, lambda=%.2f  (points=%d, wall=%d, fixed=%d)",
              nUnt, smooth_iters, lam0, nu, int(np.sum(is_wall)), int(np.sum(is_fixed)))
    _log.info("  quality before: min scaled Jac=%.4f  mean=%.4f", np.min(sj), np.mean(sj))

    # stage 1: point-local untangle
    mn = float(np.min(sj))
    for _ in range(nUnt):
        if mn >= qfloor:
            break
        active = _active_points(quality.scaled_jacobian(X, HC), HC, adj, free, qfloor)
        if active.size == 0:
            break
        for v in active:
            tgt = X[adj[v], :].mean(axis=0)
            els = NH[v]
            base = float(np.min(quality.scaled_jacobian(X, HC[els, :])))
            bestq = base
            bestx = X[v, :].copy()
            for fr in (1.0, 0.7, 0.4, 0.15):
                cand = (1 - fr) * X[v, :] + fr * tgt
                if is_wall[v]:
                    cand = ops.project_to_surface(surface, cand[None, :], Otri[wtri[v], :])[0]
                xo = X[v, :].copy()
                X[v, :] = cand
                q = float(np.min(quality.scaled_jacobian(X, HC[els, :])))
                X[v, :] = xo
                if q > bestq + 1e-12:
                    bestq = q
                    bestx = cand
            X[v, :] = bestx
        mn_new = float(np.min(quality.scaled_jacobian(X, HC)))
        if mn_new <= mn + 1e-9:
            mn = mn_new
            break
        mn = mn_new

    # stage 2: global Jacobi polish
    mn = float(np.min(quality.scaled_jacobian(X, HC)))
    for _ in range(smooth_iters):
        target = Avg @ X
        lam = lam0
        accepted = False
        for _bt in range(6):
            Xn = X.copy()
            Xn[free, :] = (1 - lam) * X[free, :] + lam * target[free, :]
            if proj and np.any(is_wall):
                Xn[is_wall, :] = ops.project_to_surface(surface, Xn[is_wall, :])
            mnn = float(np.min(quality.scaled_jacobian(Xn, HC)))
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

    sj = quality.scaled_jacobian(X, HC)
    _log.info("  quality after : min scaled Jac=%.4f  mean=%.4f", np.min(sj), np.mean(sj))
    if np.min(sj) <= 0:
        _log.warning("  %d element(s) still non-positive after smoothing",
                     int(np.sum(sj <= 0)))
    mesh.points[:] = X
    return mesh


# -- helpers (module-private) -------------------------------------------
def _adjacency_lists(E: IntArray, nu: int) -> list[IntArray]:
    adj: list[list[int]] = [[] for _ in range(nu)]
    for a, b in E:
        adj[a].append(b)
        adj[b].append(a)
    return [np.asarray(a, dtype=np.int64) for a in adj]


def _incidence_lists(HC: IntArray, nu: int) -> list[IntArray]:
    NH: list[list[int]] = [[] for _ in range(nu)]
    for e in range(HC.shape[0]):
        for k in range(8):
            NH[HC[e, k]].append(e)
    return [np.asarray(a, dtype=np.int64) for a in NH]


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
    VT_lists: list[list[int]] = [[] for _ in range(nv)]
    for e in range(T.shape[0]):
        VT_lists[T[e, 0]].append(e)
        VT_lists[T[e, 1]].append(e)
        VT_lists[T[e, 2]].append(e)
    VT: list[IntArray] = [np.asarray(a, dtype=np.int64) for a in VT_lists]
    wtri: list[IntArray | None] = [None] * is_wall.size
    for v in np.flatnonzero(is_wall):
        nvtx = int(np.argmin(np.sum((Vx - X[v, :]) ** 2, axis=1)))
        ring = np.unique(T[VT[nvtx], :])
        fan = [VT[u] for u in ring]
        wtri[v] = np.unique(np.concatenate(fan)) if fan else np.array([], dtype=np.int64)
    return wtri
