"""Constrained hex-mesh smoothing (free function on a :class:`HexMesh`).

:func:`smooth` untangles and polishes the assembled hex mesh while keeping wall
nodes on the triangulated ``surface`` and opening/cap nodes fixed.  It operates
on the welded shared-node view (:meth:`HexMesh.weld`) and is numerically
identical to the original verbatim implementation (two stages: node-local
untangle, then a back-tracked global Jacobi polish that never lowers the minimum
scaled Jacobian).
"""

import logging

import numpy as np
import scipy.sparse as sp

from ..model import quality
from . import trisurf

_log = logging.getLogger("nekmeshpy")


def smooth(mesh, surface, p):
    """Constrained untangle + polish, keeping the wall on ``surface`` (a
    TriMesh).  ``p`` has keys smooth_iters, smooth_lambda, tag_wall,
    project_to_stl (+ optional untangle_iters, quality_floor)."""
    mesh.finalize()
    if p.get("smooth_iters", 0) is None or p.get("smooth_iters", 0) <= 0:
        return mesh
    lam0 = p.get("smooth_lambda", 0.5) or 0.5
    twall = p.get("tag_wall", 1)
    proj = p.get("project_to_stl", True)
    nUnt = p.get("untangle_iters", 40) or 40
    qfloor = p.get("quality_floor", 0.2) or 0.2
    Oxyz, Otri = surface.xyz, surface.tri

    X, HC, nu = mesh.weld()
    he = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
                   [0, 4], [1, 5], [2, 6], [3, 7]], dtype=np.int64)
    E = mesh._unique_edges(HC, he)
    A = sp.coo_matrix((np.ones(E.shape[0] * 2),
                       (np.concatenate([E[:, 0], E[:, 1]]),
                        np.concatenate([E[:, 1], E[:, 0]]))), shape=(nu, nu)).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1
    Avg = sp.diags(1.0 / deg) @ A
    adj = _adjacency_lists(E, nu)
    NH = _incidence_lists(HC, nu)
    is_wall, is_fixed = mesh.classify_nodes(twall)
    free = ~is_fixed
    wtri = _wall_tri_neighbourhoods(X, is_wall, Oxyz, Otri)

    sj = quality.scaled_jacobian(X, HC)
    _log.info("smoothing: untangle<=%d, polish %d, lambda=%.2f  (nodes=%d, wall=%d, fixed=%d)",
              nUnt, p["smooth_iters"], lam0, nu, int(np.sum(is_wall)), int(np.sum(is_fixed)))
    _log.info("  quality before: min scaled Jac=%.4f  mean=%.4f", np.min(sj), np.mean(sj))

    # stage 1: node-local untangle
    mn = float(np.min(sj))
    for _ in range(nUnt):
        if mn >= qfloor:
            break
        active = _active_nodes(quality.scaled_jacobian(X, HC), HC, adj, free, qfloor)
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
                    cand = trisurf.project_to_surface(surface, cand[None, :], Otri[wtri[v], :])[0]
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
    for _ in range(p["smooth_iters"]):
        target = Avg @ X
        lam = lam0
        accepted = False
        for _bt in range(6):
            Xn = X.copy()
            Xn[free, :] = (1 - lam) * X[free, :] + lam * target[free, :]
            if proj and np.any(is_wall):
                Xn[is_wall, :] = trisurf.project_to_surface(surface, Xn[is_wall, :])
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
        X[is_wall, :] = trisurf.project_to_surface(surface, X[is_wall, :])

    sj = quality.scaled_jacobian(X, HC)
    _log.info("  quality after : min scaled Jac=%.4f  mean=%.4f", np.min(sj), np.mean(sj))
    if np.min(sj) <= 0:
        _log.warning("  %d element(s) still non-positive after smoothing",
                     int(np.sum(sj <= 0)))
    mesh._write_back(X, HC)
    return mesh


# -- helpers (module-private) -------------------------------------------
def _adjacency_lists(E, nu):
    adj = [[] for _ in range(nu)]
    for a, b in E:
        adj[a].append(b)
        adj[b].append(a)
    return [np.asarray(a, dtype=np.int64) for a in adj]


def _incidence_lists(HC, nu):
    NH = [[] for _ in range(nu)]
    for e in range(HC.shape[0]):
        for k in range(8):
            NH[HC[e, k]].append(e)
    return [np.asarray(a, dtype=np.int64) for a in NH]


def _active_nodes(sj, HC, adj, free, qfloor):
    bad = np.flatnonzero(sj < qfloor)
    if bad.size == 0:
        return np.array([], dtype=np.int64)
    seed = np.unique(HC[bad, :])
    mark = np.zeros(free.size, dtype=bool)
    mark[seed] = True
    for v in seed:
        mark[adj[v]] = True
    return np.flatnonzero(mark & free)


def _wall_tri_neighbourhoods(X, is_wall, Vx, T):
    nv = Vx.shape[0]
    VT = [[] for _ in range(nv)]
    for e in range(T.shape[0]):
        VT[T[e, 0]].append(e)
        VT[T[e, 1]].append(e)
        VT[T[e, 2]].append(e)
    VT = [np.asarray(a, dtype=np.int64) for a in VT]
    wtri = [None] * is_wall.size
    for v in np.flatnonzero(is_wall):
        nvtx = int(np.argmin(np.sum((Vx - X[v, :]) ** 2, axis=1)))
        ring = np.unique(T[VT[nvtx], :])
        fan = [VT[u] for u in ring]
        wtri[v] = np.unique(np.concatenate(fan)) if fan else np.array([], dtype=np.int64)
    return wtri
